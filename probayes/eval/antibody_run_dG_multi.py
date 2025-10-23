import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from tqdm.auto import tqdm
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
from probayes.utils.misc import seed_all
from remote.diffab.diffab.utils.transforms import get_transform

import pickle as pkl
import json
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


import os.path as osp
from remote.dyMEAN.cal_metrics_multi import run_cal_metrics_antibody as run_cal_metrics_antibody_multi



def run_relax_tasks(json_path):
    with open(json_path, 'r') as f:
        lines = f.readlines()
    relax_tasks = []
    ref_dones = []
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
        if task['pdb'] not in ref_dones:
            relax_tasks.append(
                RelaxTask(
                in_path=task['ref_pdb'],
                current_path=task['ref_pdb'],
                info=None,
                status='created',
                flexible_residue_first=task['residue_first'],
                flexible_residue_last=task['residue_last'],
                ))
            ref_dones.append(task['pdb'])
    f.close()
    
    # relax_futures = [pipeline_openmm_pyrosetta.remote(t) for t in relax_tasks]
    relax_futures = [pipeline_pyrosetta.remote(t) for t in relax_tasks]
    if len(relax_futures) > 0:
        print(f'Submitted {len(relax_futures)} relax tasks.')
    p_bar = tqdm(total=len(relax_futures), desc='Relaxing')
    while len(relax_futures) > 0:
        done_ids, relax_futures = ray.wait(relax_futures, num_returns=1)
        # done_task = ray.get(done_ids)
        for done_id in done_ids:
            p_bar.update(1)
        time.sleep(1.0)


from Bio import PDB

def load_task(line):
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
    _, cdrh3_reslist = entity_to_seq(extract_reslist(ref_biopy_model, residue_first=task['residue_first'],residue_last=task['residue_last']))    
    return EvalTask(
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
    ) 
    
def run_dG_antibody_multi(json_path, num_workers=8, run_relax=True):
    if run_relax:
        run_relax_tasks(json_path)
    else:
        print('WARNING: Skipping relax...')
            
    # run eval tasks
    with open(json_path, 'r') as f:
        lines = f.readlines()
    # lines = lines[:100]

    tasks = process_map(load_task, lines, max_workers=num_workers, desc='Loading tasks', chunksize=1)
    
    dg_json_path = json_path.replace('summary.json', 'dg_results.json')
    fout = open(dg_json_path, 'w') 
    pdb_metrics = {}

    futures = [evaluate.remote(t) for t in tasks]
    
    if len(futures) > 0:
        print(f'Submitted {len(futures)} tasks.')
        
    p_bar = tqdm(total=len(futures), desc='Evaluating')
    
    while len(futures) > 0:
        done_ids, futures = ray.wait(futures, num_returns=1)
        for done_id in done_ids:
            done_task = ray.get(done_id)
            print(f'Remaining {len(futures)}. \
                Finished {done_task.in_path}, dG_gen {done_task.scores}, dG_ref {done_task.scores["dG_ref"]}')
            pdb_id = done_task.name.split('_')[0]
            if pdb_id in ['2ghw', '3uzq','3h3b','5d96', '4etq']:
                continue
            
            done_task.scores.update({'pdb_id': pdb_id})
            # sanity check
            gt_seq = entity_to_seq(extract_reslist(
                done_task.get_ref_biopython_model(), 
                done_task.residue_first, 
                done_task.residue_last))[0]
            assert gt_seq == done_task.cdrh3_seq, f"gt_seq: {gt_seq}, cdrh3_seq: {done_task.cdrh3_seq}"
            
            fout.write(json.dumps(done_task.scores) + '\n')
            pdb_metrics[pdb_id] = done_task.scores
            p_bar.update(1)
        time.sleep(1.0)
    
    fout.close()
    
    # 计算 dG, ddG, IMP, E_total
    dG_gens = np.array([np.median(pdb_metrics[pdb_id]['dG_gen']) for pdb_id in pdb_metrics])
    dG_refs = np.array([np.median(pdb_metrics[pdb_id]['dG_ref']) for pdb_id in pdb_metrics])
    ddG_gens = dG_gens - dG_refs
    E_total_gens = np.array([np.median(pdb_metrics[pdb_id]['gen_total_energy_sum']) for pdb_id in pdb_metrics])
    E_total_refs = np.array([np.median(pdb_metrics[pdb_id]['ref_total_energy_sum']) for pdb_id in pdb_metrics])
    IMP = (ddG_gens < 0).sum() / len(ddG_gens)
    metrics = {
        'dG_gen': dG_gens.mean(),
        'dG_ref': dG_refs.mean(),
        'ddG_gen': ddG_gens.mean(),
        'IMP': IMP,
        'E_total_gen': E_total_gens.mean(),
        'E_total_ref': E_total_refs.mean(),
    }
    return metrics


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    args.add_argument('--json_path', type=str, default='logs/bfn_antibody[dev-723dcf1][07-14-15-22-00]_no_mixsc/checkpoints')

    
    parser = args.parse_args()
    ray.init(num_cpus=12)
    energy_metrics = run_dG_antibody_multi(parser.json_path, num_workers=16, run_relax=False)
    print(energy_metrics)
