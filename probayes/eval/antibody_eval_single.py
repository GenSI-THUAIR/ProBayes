import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import copy
import math
from tqdm.auto import tqdm
import functools
import os
import argparse
import pandas as pd
from copy import deepcopy

from probayes.dataset.pep_dataset import PepDataset

from probayes.utils.train import recursive_to

from probayes.modules.protein.writers import save_pdb

from probayes.utils.data import PaddingCollate

from probayes.core.utils import process_dic
from probayes.core.bfn_model_quat import BFNModel_quat
from ema_pytorch import EMA
from Bio.PDB import PDBExceptions
from probayes.data import all_atom
from probayes.modules.protein.constants import AA,resindex_to_ressymb
from probayes.data import residue_constants
from probayes.data import utils as du
from remote.PepGLAD.generate import save_data
from probayes.dataset.antibody_dataset import AntibodyDataset
from remote.PepGLAD.data.converter import list_blocks_to_pdb
from remote.PepGLAD.generate import overwrite_blocks
from remote.diffab.diffab.utils.data import apply_patch_to_tensor
ATOM_MASK = torch.tensor(residue_constants.restype_atom14_mask)
collate_fn = PaddingCollate(eight=False)
from probayes.utils.misc import load_config
from train_bfn import get_model
from probayes.utils.misc import seed_all
from remote.diffab.diffab.utils.transforms import get_transform
from remote.PepGLAD.data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from remote.PepGLAD.data.converter.list_blocks_to_pdb import list_blocks_to_pdb

import pickle as pkl
import json
from remote.dyMEAN.data.pdb_utils import AgAbComplex, VOCAB
# from remote.diffab.diffab.utils.protein import save_pdb
from remote.diffab.diffab.utils.transforms import MaskSingleCDR, MergeChains
from remote.diffab.diffab.utils.inference import get_residue_first_last
from remote.diffab.diffab.tools.eval.similarity import extract_reslist, entity_to_seq

from torchvision.transforms import Compose
import argparse


def item_to_batch(item, nums=32):
    data_list = [deepcopy(item) for i in range(nums)]
    return collate_fn(data_list)


def generate_antibody(ckpt_path, data, model, device, 
             num_steps, num_samples, 
             sample_bb,sample_ang,sample_seq,
             dataset, sample_mode,
             sc_pack='generated'):
    print('sample_bb:', sample_bb,'sample_ang:', sample_ang, 'sample_seq:', sample_seq)
    if (not sample_bb) and (not sample_seq) and (sample_ang):
        sc_pack = 'generated' # sidechain packing mode

    if (sample_bb) and (not sample_seq) and (sample_ang):
        sc_pack = 'generated' # folding/binding comformation generation mode
        
    print('sidechain packing mode', sc_pack)
    
    # sample the trajectory
    batch = recursive_to(data,device=device)
    
    traj = model.sample(batch, num_steps=num_steps,sample_bb=sample_bb,sample_ang=sample_ang,sample_seq=sample_seq,mode=sample_mode)
    final_state = recursive_to(traj[-1], device=device)
    
    if sc_pack == 'rosetta':
        bb_atoms = all_atom.to_atom37(trans=final_state['trans'], rots=final_state['rotmats'])[:, :, :3] 
        pos_ha = F.pad(bb_atoms, pad=(0,0,0,15-3), value=0.) # (32,L,A,3) pos14 A=14
        pos_new = torch.where(batch['generate_mask'][:,:,None,None],pos_ha,batch['pos_heavyatom']) 
        mask_bb_atoms = torch.zeros_like(batch['mask_heavyatom'])
        mask_bb_atoms[:,:,:3] = True
        mask_new = torch.where(batch['generate_mask'][:,:,None],mask_bb_atoms,batch['mask_heavyatom'])   
    elif sc_pack == 'generated':
        pos_ha = all_atom.compute_allatom(
                        bb_rigids=du.create_rigid(rots=final_state['rotmats'],trans=final_state['trans']),
                        angles=final_state['angles'],aatype=final_state['seqs'].clip(max=int(AA.UNK)),)

        pos_new = F.pad(pos_ha, pad=(0,0,0,15-14), value=0.)
        mask_bb_atoms = torch.zeros_like(batch['mask_heavyatom'])
        mask_bb_atoms[:,:,:14] = True
        mask_new = torch.where(batch['generate_mask'][:,:,None],mask_bb_atoms,batch['mask_heavyatom'])

    aa_gen = final_state['seqs']
    # create related directories
    save_path = os.path.join(ckpt_path,'results_single')
    # os.mkdir(save_path) if not os.path.exists(save_path) else None
    assert os.path.exists(save_path)
    os.mkdir(os.path.join(save_path,'references')) if not os.path.exists(os.path.join(save_path,'references')) else None
    os.mkdir(os.path.join(save_path,'candidates')) if not os.path.exists(os.path.join(save_path,'candidates')) else None
    
    batch = recursive_to(batch, device='cpu')
    jsons = []
    
    for i in range(len(batch['id'])):
        pdb_id = batch['id'][i]
        this_entry = dataset.entryid2entry[pdb_id]

        ref = dataset.__getitem__(i, do_transform=False) # TODO: this code is not correct for batch_idx > 1
        transform = Compose([
                MergeChains(),
            ])
        raw_ref = transform(ref)
        mask_transform =  Compose([
                MaskSingleCDR('H3', augmentation=False),
                MergeChains(),
        ])
        mask_ref = mask_transform(ref)
        
        # raw_ref = recursive_to(raw_ref, device=device)
        os.mkdir(os.path.join(save_path,'candidates',f'{pdb_id}')) if not os.path.exists(os.path.join(save_path,'candidates',f'{pdb_id}')) else None

        ref_complex_path = save_pdb(raw_ref,path=os.path.join(save_path,'references',f'{pdb_id}_ref.pdb'))
        
        aa_gen = aa_gen.cpu()
        pos_new = pos_new.cpu()
        mask_new = mask_new.cpu()
        
        res_mask = batch['res_mask'][i].gt(0)
        heavyatom_mask = batch['mask_heavyatom'][i].gt(0)
        gen_mask = batch['generate_mask'][i].gt(0)
        gt_aa = raw_ref['aa'][batch['patch_idx'][i].cpu()][gen_mask.cpu()]
        gen_aa = aa_gen[i][gen_mask]
        
        non_gen_mask =  (~gen_mask) & res_mask
        non_gen_gt_aa = raw_ref['aa'][batch['patch_idx'][i].cpu()][non_gen_mask.cpu()]
        non_gen_data_aa = aa_gen[i][non_gen_mask]
        assert (non_gen_data_aa == non_gen_gt_aa).all()
        
        # data_pos = (pos_new[i]+batch['origin'][i][None,None,:])
        # non_gen_pos = data_pos[non_gen_mask]
        # non_gen_ha_pos = non_gen_pos[heavyatom_mask[non_gen_mask],...]
        # non_gen_gt_ha_pos = raw_ref['pos_heavyatom'][batch['patch_idx'][i],...][non_gen_mask][heavyatom_mask[non_gen_mask],...]
        
        aa_complex = apply_patch_to_tensor(
            x_full=raw_ref['aa'], x_patch=aa_gen[i][res_mask], patch_idx=batch['patch_idx'][i][res_mask],
        )
        pos_complex = apply_patch_to_tensor(
            x_full=raw_ref['pos_heavyatom'], x_patch=(pos_new[i][res_mask,...]+batch['origin'][i][:3][None,None,:]), patch_idx=batch['patch_idx'][i][res_mask],
        )
        mask_complex = apply_patch_to_tensor(
            x_full=raw_ref['mask_heavyatom'], x_patch=mask_new[i][res_mask,...], patch_idx=batch['patch_idx'][i][res_mask],
        )

        gen_complex = {
                    'chain_nb':raw_ref['chain_nb'],'chain_id':raw_ref['chain_id'],
                    'resseq':raw_ref['resseq'],'icode':raw_ref['icode'],
                    'aa':aa_complex, 'mask_heavyatom':mask_complex, 
                    'pos_heavyatom':pos_complex,
        }
        gen_complex_path = save_pdb(gen_complex,path=os.path.join(save_path,'candidates',f'{pdb_id}_gen.pdb'))   
        infos = get_residue_first_last(data=mask_ref)
        jsons.append({'mod_pdb':gen_complex_path,
                    'ref_pdb':ref_complex_path,
                    'pdb': pdb_id,
                    "cdr_type": ["H3"],
                    "H": this_entry['H_chain'],
                    "L": this_entry['L_chain'],
                    'A': this_entry['ag_chains'],
                    'pmetric': 0.,
                    "residue_first": list(infos[0]),
                    "residue_last": list(infos[1]),
        })
        
    return jsons


def prepare_result_json_antibody(ckpt_path:str, model:BFNModel_quat, val_dataset:AntibodyDataset,
                          n_steps=200, sample_bb=True, sample_ang=True,
                          sample_seq=True, device='cuda',n_samples=1, dataset_pdb_dir=None, sample_mode=None,sc_pack='generated'):
    
    
    model.eval()
    if not os.path.exists(os.path.join(ckpt_path,'results_single')):
        os.mkdir(os.path.join(ckpt_path,'results_single'))
    json_path = os.path.join(ckpt_path, 'results_single/' 'summary.json')
    fout = open(json_path, 'w')
    
    test_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=False, collate_fn=PaddingCollate(), num_workers=8)

    for i, batch in enumerate(tqdm(test_loader, desc='Get Metrics', dynamic_ncols=True)):
        jsons = generate_antibody(ckpt_path, 
                    data=batch,model=model,device=device,num_samples=n_samples, 
                        num_steps=n_steps,sample_bb=sample_bb,sample_ang=sample_ang,sample_seq=sample_seq, 
                        dataset=val_dataset,sample_mode=sample_mode,sc_pack=sc_pack)

        for this_json in jsons:
            fout.write(json.dumps(this_json) + '\n')
    fout.close()
    
    return json_path

import os.path as osp
from remote.dyMEAN.cal_metrics import run_cal_metrics_antibody

def get_metrics_antibody(ckpt_dir, model, dataset, n_steps, sample_bb, sample_ang, sample_seq, device, n_samples, dataset_pdb_dir, sample_mode='end_back',sc_pack='generated'):
    # if relative path, convert to absolute path
    if not osp.isabs(ckpt_dir):
        ckpt_dir = osp.abspath(ckpt_dir)
    
    result_path = prepare_result_json_antibody(ckpt_dir, model, dataset, n_steps=n_steps,
                              sample_bb=sample_bb, sample_ang=sample_ang, sample_seq=sample_seq, 
                              device=device, n_samples=n_samples, dataset_pdb_dir=dataset_pdb_dir,sample_mode=sample_mode,sc_pack=sc_pack)
    metrics = run_cal_metrics_antibody(test_summary_path=result_path, num_workers=8)
    energy_metrics = run_dG_antibody(result_path, num_workers=8)
    metrics.update(energy_metrics)
    return metrics

from remote.diffab.diffab.tools.eval.energy import eval_interface_energy, EvalTask
from remote.diffab.diffab.tools.eval.total_energy import eval_total_energy

from remote.diffab.diffab.tools.relax.base import RelaxTask
from remote.diffab.diffab.tools.relax.run import pipeline_openmm_pyrosetta, pipeline_pyrosetta

import ray
import time

@ray.remote(num_cpus=1)
def evaluate(task):
    funcs = []
    funcs.append(eval_interface_energy)
    funcs.append(eval_total_energy)
    for f in funcs:
        task = f(task)
    return task

def run_relax_tasks(json_path):
    with open(json_path, 'r') as f:
        lines = f.readlines()
    relax_tasks = []
    
    for line in lines:
        task = json.loads(line)
        relax_tasks.append(
            RelaxTask(
            in_path=task['mod_pdb'],
            current_path=task['mod_pdb'],
            info=None,
            status='created',
            flexible_residue_first=task['residue_first'],
            flexible_residue_last=task['residue_last'],
            ))
        relax_tasks.append(
            RelaxTask(
            in_path=task['ref_pdb'],
            current_path=task['ref_pdb'],
            info=None,
            status='created',
            flexible_residue_first=task['residue_first'],
            flexible_residue_last=task['residue_last'],
            ))
    f.close()
    
    # relax_futures = [pipeline_openmm_pyrosetta.remote(t) for t in relax_tasks]
    relax_futures = [pipeline_pyrosetta.remote(t) for t in relax_tasks]
    if len(relax_futures) > 0:
        print(f'Submitted {len(relax_futures)} relax tasks.')
    while len(relax_futures) > 0:
        done_ids, relax_futures = ray.wait(relax_futures, num_returns=1)
        # done_task = ray.get(done_ids)
        for done_id in done_ids:
            done_task = ray.get(done_id)
            print(f'Remaining {len(relax_futures)}.')
        time.sleep(1.0)


from Bio import PDB


def run_dG_antibody(json_path, num_workers=8, run_relax=True):
    if run_relax:
        run_relax_tasks(json_path)
            
    # run eval tasks
    with open(json_path, 'r') as f:
        lines = f.readlines()
    
    tasks = []
    for line in lines:
        task = json.loads(line)
        gen_path = task['mod_pdb']
        fname, ext = os.path.splitext(gen_path)
        
        # use the rosetta relaxed
        in_path = f"{fname}_rosetta{ext}"
        ref_fname, ref_ext = os.path.splitext(task['ref_pdb'])
        relaxed_ref_path = f"{ref_fname}_rosetta{ref_ext}"
        assert os.path.exists(in_path), f"File {in_path} does not exist."
        
        parser = PDB.PDBParser(QUIET=True)
        ref_biopy_model = parser.get_structure(relaxed_ref_path, relaxed_ref_path)[0]
        _, cdrh3_reslist = entity_to_seq(extract_reslist(ref_biopy_model, residue_first=task['residue_first'], residue_last=task['residue_last']))
        
        tasks.append(EvalTask(
            in_path=in_path,
            ref_path=task['ref_pdb'],
            relaxed_ref_path=relaxed_ref_path,
            ab_chains=task['H'],
            info=None, # seems not used in eval_interface_energy
            structure=None,
            name=task['pdb'],
            method=None,
            cdr='H_CDR3', # only support H3 now
            residue_first=task['residue_first'],
            residue_last=task['residue_last'],
            cdrh3_reslist=cdrh3_reslist
        ))
    
    dg_json_path = json_path.replace('summary.json', 'dg_results.json')
    fout = open(dg_json_path, 'w') 
    dg_gens = []
    dg_refs = []
    ddg_gens = []
    gen_total_energy_sum = []
    ref_total_energy_sum = []
    
    futures = [evaluate.remote(t) for t in tasks]
    if len(futures) > 0:
        print(f'Submitted {len(futures)} tasks.')
    while len(futures) > 0:
        done_ids, futures = ray.wait(futures, num_returns=1)
        for done_id in done_ids:
            done_task = ray.get(done_id)
            print(f'Remaining {len(futures)}. Finished {done_task.in_path}, dG_gen {done_task.scores}, dG_ref {done_task.scores["dG_ref"]}')
            pdb_id = done_task.name.split('_')[0]
            if pdb_id in ['2ghw', '3uzq','3h3b','5d96', '4etq']:
                print(f'Skipping {pdb_id}...')
                continue
            
            done_task.scores.update({'pdb_id': done_task.name.split('_')[0]})
            done_task.scores.update({'cdrh3_seq': done_task.cdrh3_seq})
            
            gt_seq = entity_to_seq(extract_reslist(done_task.get_ref_biopython_model(), done_task.residue_first, done_task.residue_last))[0]
            assert gt_seq == done_task.cdrh3_seq, f"gt_seq: {gt_seq}, cdrh3_seq: {done_task.cdrh3_seq}"
            
            fout.write(json.dumps(done_task.scores) + '\n')
            dg_gens.append(done_task.scores['dG_gen'])
            dg_refs.append(done_task.scores['dG_ref'])
            ddg_gens.append(done_task.scores['ddG'])
            gen_total_energy_sum.append(done_task.scores['gen_total_energy_sum'])
            ref_total_energy_sum.append(done_task.scores['ref_total_energy_sum'])
        time.sleep(1.0)
    
    dg_gens = np.array(dg_gens)
    dg_refs = np.array(dg_refs)
    ddg_gens = np.array(ddg_gens)

    gen_total_energy_sum = np.array(gen_total_energy_sum)
    ref_total_energy_sum = np.array(ref_total_energy_sum)
    IMP = (ddg_gens < 0).sum() / len(ddg_gens)
    fout.close()
    metrics = {
        'dg_gen(mean)': dg_gens.mean(),
        'dg_gen(median)': np.median(dg_gens),
        'dg_ref': dg_refs.mean(),
        'ddg_gen(mean)': ddg_gens.mean(),
        'ddg_gen(median)': np.median(ddg_gens),
        'IMP': IMP,
        'gen_total_energy': np.mean(gen_total_energy_sum),
        'ref_total_energy': np.mean(ref_total_energy_sum),
    }
    return metrics

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str,default='./configs/learn_angle.yaml')
    args.add_argument('--device', type=str, default='cuda:0')
    args.add_argument('--ckpt_dir', type=str, default='logs/bfn_antibody[dev-723dcf1][07-14-15-22-00]_no_mixsc/checkpoints')
    args.add_argument('--n_test_items', type=int, default=1000)
    args.add_argument('--sample_mode', type=str, default='end_back')
    args.add_argument('--run_gen', type=bool, default=True)
    args.add_argument('--tag', type=str, default='single_eval')
    
    parser = args.parse_args()

    if parser.run_gen:
        # 将ckpt_dir下的yaml文件作为config_path
        files = os.listdir(parser.ckpt_dir)
        yaml_fname = [file for file in files if file.endswith('.yaml')][0]
        config, cfg_name = load_config(osp.join(parser.ckpt_dir, yaml_fname))
        device = parser.device
        
        dataset =  AntibodyDataset(structure_dir = config.dataset.test.structure_dir, 
                                dataset_dir = config.dataset.test.dataset_dir,
                                    name = config.dataset.test.name, 
                                    split_json_path= config.dataset.test.split_json_path,
                                    transform=get_transform(config.dataset.test.transform), 
                                    reset=config.dataset.test.reset)
        # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=PaddingCollate(eight=False), num_workers=4, pin_memory=True)
        ckpt_files = os.listdir(osp.join(parser.ckpt_dir, 'checkpoints'))
        pt_files = [file for file in ckpt_files if file.endswith('.pt')]
        assert len(pt_files) == 1
        ckpt_fname = pt_files[0]
        print('eval ckpt:',ckpt_fname)
        ckpt = torch.load(osp.join(parser.ckpt_dir, 'checkpoints',ckpt_fname), map_location=device)

        seed_all(114516)
        # Model
        model = get_model(config)
        model = model.to(device)
        if 'ema' in ckpt.keys() and ckpt['ema']!=None:
            model = EMA(
                model,
                beta = config.train.ema.decay,              # exponential moving average factor
                update_after_step = config.train.ema.update_after_step,    # only after this number of .update() calls will it start updating
                update_every = config.train.ema.update_every,          # how often to actually update, to save on compute (updates every 10th .update() call)
                forward_method_names=['sample']
            )
            model.load_state_dict(ckpt['ema'])
        else:
            model.load_state_dict(process_dic(ckpt['model']))
            
        model.eval()
        sc_pack = 'generated' 
        metrics = get_metrics_antibody(ckpt_dir=parser.ckpt_dir, 
                                model=model, dataset=dataset, 
                                n_steps=config.model.bfn.n_steps, 
                                sample_bb=config.model.sample_bb, 
                                sample_ang=config.model.sample_sc, 
                                sample_seq=config.model.sample_seq, 
                                device=device, 
                                n_samples=1,
                                dataset_pdb_dir=config.dataset.test.structure_dir,
                                sample_mode=parser.sample_mode,
                                sc_pack=sc_pack
                                )
        print('done')
        for key, value in metrics.items():
            print(f'{key}: {value}')
        
        # # 将 metrics 写入文件夹
        metrics_df = pd.DataFrame(metrics,index=[0])
        metrics_df.T.to_csv(osp.join(parser.ckpt_dir, f'metrics_{parser.tag}.csv'))
    else:
        json_path = osp.join(parser.ckpt_dir,  'results/','summary.json')
        # json_path = osp.join(parser.ckpt_dir, 'results/' 'summary.json')

        energy_metrics = run_dG_antibody(json_path, num_workers=8, run_relax=False)
        print(energy_metrics)
