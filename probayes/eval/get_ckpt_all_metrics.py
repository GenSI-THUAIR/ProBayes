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
import Bio
from copy import deepcopy


from probayes.dataset.pep_dataset import PepDataset
from probayes.utils.misc import load_config
from probayes.utils.train import recursive_to

from probayes.utils.data import PaddingCollate

from probayes.core.utils import process_dic

import gc
import os.path as osp
from probayes.core.flow_model import FlowModel
from Bio.PDB import Structure, PDBParser

from probayes.utils.misc import seed_all
from probayes.utils.metrics import get_metrics, prepare_result_json, Evaluator
from probayes.utils.metrics import get_bind_site, get_second_stru
from probayes.eval.geometry import align_chains, diff_ratio, get_rmsd
from torch.utils.data import Subset
from train_pep import get_model
from ema_pytorch import EMA
from remote.PepGLAD.evaluation.dG.run import run_eval_tasks
from remote.PepGLAD.cal_metrics import run_cal_metrics
from remote.PepGLAD.cal_metrics import _get_gen_pdb, _get_ref_pdb
collate_fn = PaddingCollate(eight=False)
from collections import defaultdict
import argparse
import json
from Bio.SeqUtils import seq1

def get_all_metrics(ckpt_dir, model, dataset, n_steps, sample_bb, sample_ang, sample_seq, device, n_samples, dataset_pdb_dir, sample_mode='end_back',sc_pack='generated'):
    # if relative path, convert to absolute path
    if not osp.isabs(ckpt_dir):
        ckpt_dir = osp.abspath(ckpt_dir)
    
    result_path, metrics = prepare_result_json(ckpt_dir, model, dataset, n_steps=n_steps,
                              sample_bb=sample_bb, sample_ang=sample_ang, sample_seq=sample_seq, 
                              device=device, n_samples=n_samples, dataset_pdb_dir=dataset_pdb_dir,sample_mode=sample_mode,sc_pack=sc_pack)
    dG_metrics = run_eval_tasks(results=result_path,n_sample=n_samples,rfdiff_relax=False)
    pepglad_metrics = run_cal_metrics(results=result_path)
    probayes_metrics = get_probayes_metrics(result_path)
    
    metrics.update(dG_metrics)
    metrics.update(probayes_metrics)
    metrics.update(pepglad_metrics)
    return metrics


def get_probayes_metrics(result_path, filter_dG=None):
    # load results
    with open(result_path, 'r') as fin:
        lines = fin.read().strip().split('\n')
    
    # load dG filter
    if filter_dG is None:
        filter_func = lambda _id, n: True
    else:
        dG_results = json.load(open(filter_dG, 'r'))
        filter_func = lambda _id, n: dG_results[_id]['all'][str(n)] < 0
    
    id2items = {}
    for line in lines:
        item = json.loads(line)
        _id = item['id']
        if not filter_func(_id, item['number']):
            continue
        if _id not in id2items:
            id2items[_id] = []
        id2items[_id].append(item)
        
    pdb_ids = list(id2items.keys())
    results = defaultdict(list)
    result_dir = osp.dirname(result_path)
    for pdb_id in pdb_ids:
        items = id2items[pdb_id]
        items = sorted(items, key=lambda x: x['number'])
        id2items[pdb_id] = items
        
        pep_chain_id = items[0]['lig_chain']
        rec_chain_id = items[0]['rec_chain']
        ref_merge:Structure = PDBParser().get_structure(file=_get_ref_pdb(pdb_id, result_dir),id=pdb_id)[0]
        ref_pep_ch = [ch for ch in list(ref_merge.get_chains()) if ch.id == pep_chain_id][0]
        use_rosetta = False
        # reference bind site
        ref_bind_site = get_bind_site(ref_pep_ch, list(ref_merge.get_chains()), pep_chain_id)
        ref_sec_struc = get_second_stru(_get_ref_pdb(pdb_id, result_dir), pep_chain_id)
        for item in items:
            gen_struct = PDBParser().get_structure(file=_get_gen_pdb(pdb_id, item['number'], result_dir, use_rosetta),id=pdb_id)[0]
            # parse_pdb(path=_get_gen_pdb(pdb_id, item['number'], result_dir, item['rosetta']))[0]
            gen_pep_ch = [ch for ch in list(gen_struct.get_chains()) if ch.id == pep_chain_id][0]

            gen_bind_site = get_bind_site(gen_pep_ch, list(gen_struct.get_chains()), pep_chain_id)
            gen_sec_struc = get_second_stru(_get_gen_pdb(pdb_id, item['number'], result_dir, use_rosetta), pep_chain_id)

            ref_res, gen_res = align_chains(ref_pep_ch, gen_pep_ch)
            ref_seq = seq1("".join([residue.get_resname() for residue in ref_res]))
            sample_seq = seq1("".join([residue.get_resname() for residue in gen_res]))
            AAR = diff_ratio(ref_seq, sample_seq)
            BSR = len(gen_bind_site.intersection(ref_bind_site))/(len(ref_bind_site)+1e-10)
            SSR = (ref_sec_struc == gen_sec_struc).mean()
            noalign_Ca_RMSD, aligned_Ca_RMSD = get_rmsd(ref_pep_ch, gen_pep_ch)
            
            results['AAR'].append(AAR)
            results['BSR'].append(BSR)
            results['SSR'].append(SSR)
            results['aligned_CaRMSD'].append(aligned_Ca_RMSD)
            # results['noalign_CaRMSD'].append(noalign_Ca_RMSD)
    
    return {k:np.mean(v) for k, v in results.items()}


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--config', type=str,default='./configs/learn_angle.yaml')
    args.add_argument('--device', type=str, default='cpu')
    args.add_argument('--ckpt_dir', type=str, default='')
    # args.add_argument('--output', type=str, default='./results')
    args.add_argument('--num_steps', type=int, default=200)
    args.add_argument('--num_samples', type=int, default=40)
    args.add_argument('--sample_bb', type=bool, default=True)
    args.add_argument('--sample_ang', type=bool, default=True)
    args.add_argument('--sample_seq', type=bool, default=True)
    args.add_argument('--n_test_items', type=int, default=1000)
    args.add_argument('--sample_mode', type=str, default='end_back')
    args.add_argument('--tag', type=str, default='default')
    
    parser = args.parse_args()

    # 将ckpt_dir下的yaml文件作为config_path
    files = os.listdir(parser.ckpt_dir)
    yaml_fname = [file for file in files if file.endswith('.yaml')][0]
    config, cfg_name = load_config(osp.join(parser.ckpt_dir, yaml_fname))

    device = parser.device
    
    dataset = PepDataset(structure_dir = config.dataset.test.structure_dir, 
                         dataset_dir = config.dataset.test.dataset_dir,
                            name = config.dataset.test.name, transform=None, reset=config.dataset.test.reset)
    n_test_items = min(parser.n_test_items, len(dataset))
    dataset = Subset(dataset, range(n_test_items))
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=PaddingCollate(eight=False), num_workers=4, pin_memory=True)
    ckpt_files = os.listdir(osp.join(parser.ckpt_dir, 'checkpoints'))
    pt_files = [file for file in ckpt_files if file.endswith('.pt')]
    assert len(pt_files) == 1
    ckpt_fname = pt_files[0]
    print('eval ckpt:',ckpt_fname)
    ckpt = torch.load(osp.join(parser.ckpt_dir, 'checkpoints',ckpt_fname), map_location=device)

    seed_all(114514)
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
    sc_pack = 'generated' if 'sc_pack' not in config.model else config.model.sc_pack
    num_samples = 40 if 'n_gen_samples' not in config.dataset else config.dataset.n_gen_samples
    print('num_samples:', num_samples)
    metrics = get_all_metrics(parser.ckpt_dir, 
                              model, dataset, 
                              parser.num_steps, 
                              config.model.sample_bb, 
                              config.model.sample_sc, 
                              config.model.sample_seq, 
                              device, 
                              num_samples,
                              dataset_pdb_dir=config.dataset.test.structure_dir,
                              sample_mode=parser.sample_mode,
                              sc_pack=sc_pack
                              )
    print('done')
    for key, value in metrics.items():
        print(f'{key}: {value}')
        
    # # 将 metrics 写入文件夹
    metrics_df = pd.DataFrame(metrics,index=[0])
    metrics_df.T.to_csv(osp.join(parser.ckpt_dir, 'checkpoints',f'metrics_{parser.tag}.csv'))

