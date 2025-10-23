import shutil
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

from tqdm.contrib.concurrent import process_map

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
from remote.diffab.diffab.utils.transforms import get_transform

import json
from probayes.eval.antibody_run_dG_multi import run_dG_antibody_multi
from remote.dyMEAN.data.pdb_utils import AgAbComplex, VOCAB
# from remote.diffab.diffab.utils.protein import save_pdb
from remote.diffab.diffab.utils.transforms import MaskSingleCDR, MergeChains
from remote.diffab.diffab.utils.inference import get_residue_first_last
from remote.diffab.diffab.tools.eval.similarity import extract_reslist, entity_to_seq
from remote.diffab.diffab.tools.eval.energy import EvalTask
from remote.diffab.diffab.tools.relax.base import RelaxTask
from remote.diffab.diffab.tools.relax.run import pipeline_pyrosetta

import ray
import time
from probayes.eval.antibody_eval_single import evaluate
from torchvision.transforms import Compose
from tqdm import trange
import argparse

def generate_antibody(ckpt_path, data, model, device, 
             num_steps, num_samples, 
             sample_bb,sample_ang,sample_seq,
             dataset, sample_mode, batch_id,
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
    save_path = os.path.join(ckpt_path,'results_multi') 

    # if exist then remove references and candidates
    if os.path.exists(os.path.join(save_path,'references')):
        print('delete references')
        shutil.rmtree(os.path.join(save_path,'references'))
    os.makedirs(os.path.join(save_path,'references'), exist_ok=True)
    if os.path.exists(os.path.join(save_path,'candidates')):
        shutil.rmtree(os.path.join(save_path,'candidates'))
    os.makedirs(os.path.join(save_path,'candidates'), exist_ok=True)

    
    batch = recursive_to(batch, device='cpu')
    jsons = []
    
    for i in trange(len(batch['id']),desc='Process Generated Batch'):
        pdb_id = batch['id'][i]
        this_entry = dataset.entryid2entry[pdb_id]

        # ref = dataset.__getitem__(i, do_transform=False) 
        ref = dataset.get_structure_bypdbid(pdb_id) # TODO：modify here
        
        transform = Compose([
                MergeChains(),
            ])
        raw_ref = transform(ref)

        gen_save_dir = os.path.join(save_path,'candidates',f'{pdb_id.split("_")[0]}')        
        # os.mkdir(gen_save_dir) if not os.path.exists(gen_save_dir) else None
        os.makedirs(gen_save_dir, exist_ok=True)

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

        # apply the generated patch to the reference structure
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
        gen_id = batch_id * len(batch['id']) + i
        gen_complex_path = save_pdb(gen_complex,path=os.path.join(gen_save_dir, f'{pdb_id}_gen_{gen_id}.pdb'))
        
        # get the cdr info
        mask_transform =  Compose([
                MaskSingleCDR('H3', augmentation=False),
                MergeChains(),
        ])
        mask_ref = mask_transform(ref)        
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

class RepeatDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, repeat_times):
        self.base_dataset = base_dataset
        self.repeat_times = repeat_times
        self.base_len = len(base_dataset)
    def __len__(self):
        return self.base_len * self.repeat_times
    def __getitem__(self, idx):
        base_idx = idx % self.base_len
        return self.base_dataset[base_idx]

def prepare_result_json_antibody(ckpt_path:str, model:BFNModel_quat, val_dataset:AntibodyDataset,
                          n_steps=200, sample_bb=True, sample_ang=True,
                          sample_seq=True, device='cuda',n_samples=1, dataset_pdb_dir=None, sample_mode=None,sc_pack='generated'):
    model.eval()
    if not os.path.exists(os.path.join(ckpt_path,'results_multi')):
        os.mkdir(os.path.join(ckpt_path,'results_multi'))
    json_path = os.path.join(ckpt_path, 'results_multi/' 'summary.json')
    fout = open(json_path, 'w')
    
    repeat_val_dataset = RepeatDataset(val_dataset, n_samples)
    # repeat_val_dataset = val_dataset # test the rest code 
    test_loader = DataLoader(repeat_val_dataset, batch_size=128,
                             shuffle=False, collate_fn=PaddingCollate(), num_workers=8)

    for i, batch in enumerate(tqdm(test_loader, desc='Get Metrics', dynamic_ncols=True)):
        jsons = generate_antibody(ckpt_path, 
                    data=batch,model=model,device=device,num_samples=n_samples, 
                        num_steps=n_steps,sample_bb=sample_bb,sample_ang=sample_ang,sample_seq=sample_seq, 
                        dataset=val_dataset,sample_mode=sample_mode,sc_pack=sc_pack, batch_id=i)

        for this_json in jsons:
            fout.write(json.dumps(this_json) + '\n')
    fout.close()
    
    return json_path

import os.path as osp
from remote.dyMEAN.cal_metrics_multi import run_cal_metrics_antibody as run_cal_metrics_antibody_multi

def get_metrics_antibody(ckpt_dir, model, dataset, n_steps, sample_bb, sample_ang, sample_seq, device, n_samples, dataset_pdb_dir, sample_mode='end_back',sc_pack='generated'):
    # if relative path, convert to absolute path
    if not osp.isabs(ckpt_dir):
        ckpt_dir = osp.abspath(ckpt_dir)
    
    result_path = prepare_result_json_antibody(ckpt_dir, model, dataset, n_steps=n_steps,
                              sample_bb=sample_bb, sample_ang=sample_ang, sample_seq=sample_seq, 
                              device=device, n_samples=n_samples, dataset_pdb_dir=dataset_pdb_dir,sample_mode=sample_mode,sc_pack=sc_pack)
    metrics = run_cal_metrics_antibody_multi(test_summary_path=result_path, num_workers=16)
    energy_metrics = run_dG_antibody_multi(result_path, num_workers=8)
    metrics.update(energy_metrics)
    return metrics


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str,default='./configs/learn_angle.yaml')
    args.add_argument('--device', type=str, default='cuda:7')
    args.add_argument('--ckpt_dir', type=str, default='logs/bfn_antibody[dev-723dcf1][07-14-15-22-00]_no_mixsc')
    args.add_argument('--sample_mode', type=str, default='end_back')
    args.add_argument('--run_gen', type=bool, default=True)
    args.add_argument('--tag', type=str, default='multi_eval')
    args.add_argument('--num_samples', type=int, default=1)
    
    parser = args.parse_args()
    ray.init(num_cpus=12)
    if parser.run_gen:
        # 将ckpt_dir下的yaml文件作为config_path
        files = os.listdir(parser.ckpt_dir)
        # add the parent directory of ckpt_dir
        # files += os.listdir(osp.dirname(parser.ckpt_dir))
        yaml_fname = [file for file in files if file.endswith('.yaml')][0]
        config, cfg_name = load_config(osp.join(parser.ckpt_dir, yaml_fname))
        device = parser.device
        
        dataset =  AntibodyDataset(structure_dir = config.dataset.test.structure_dir, 
                                dataset_dir = config.dataset.test.dataset_dir,
                                    name = config.dataset.test.name, 
                                    split_json_path= config.dataset.test.split_json_path,
                                    transform=get_transform(config.dataset.test.transform), 
                                    reset=config.dataset.test.reset)
        ckpt_files = os.listdir(osp.join(parser.ckpt_dir, 'checkpoints'))
        pt_files = [file for file in ckpt_files if file.endswith('.pt')]
        assert len(pt_files) == 1
        ckpt_fname = pt_files[0]
        print('eval ckpt:',ckpt_fname)
        ckpt = torch.load(osp.join(parser.ckpt_dir, 'checkpoints',ckpt_fname), map_location=device)

        
        # Model
        model = get_model(config)
        model = model.to(device)
        if 'ema' in ckpt.keys() and ckpt['ema']!=None:
            model = EMA(
                model,
                beta = config.train.ema.decay,              
                update_after_step = config.train.ema.update_after_step,    
                update_every = config.train.ema.update_every,          
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
                                n_samples=parser.num_samples,
                                dataset_pdb_dir=config.dataset.test.structure_dir,
                                sample_mode=parser.sample_mode,
                                sc_pack=sc_pack
                                )
        for key, value in metrics.items():
            print(f'{key}: {value}')
        
        # 将 metrics 写入文件夹
        metrics_df = pd.DataFrame(metrics,index=[0])
        metrics_df.T.to_csv(osp.join(parser.ckpt_dir, f'metrics_{parser.tag}.csv'))
    else:
        json_path = osp.join(parser.ckpt_dir, 'results/','summary.json')
        energy_metrics = run_dG_antibody_multi(json_path, num_workers=16, run_relax=False)
        metrics = run_cal_metrics_antibody_multi(test_summary_path=json_path, num_workers=16)
        print(energy_metrics)
