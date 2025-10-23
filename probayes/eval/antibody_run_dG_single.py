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
from probayes.data import residue_constants

from probayes.dataset.antibody_dataset import AntibodyDataset
ATOM_MASK = torch.tensor(residue_constants.restype_atom14_mask)
collate_fn = PaddingCollate(eight=False)
from probayes.utils.misc import load_config
from train_bfn import get_model
from probayes.utils.misc import seed_all
from remote.diffab.diffab.utils.transforms import get_transform
import json
from remote.diffab.diffab.tools.eval.similarity import extract_reslist, entity_to_seq

import argparse


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
    args.add_argument('--json_path', type=str, default='logs/bfn_antibody[dev-723dcf1][07-14-15-22-00]_no_mixsc/checkpoints')
    args.add_argument('--tag', type=str, default='single_eval_dG')
    
    parser = args.parse_args()
    
    energy_metrics = run_dG_antibody(parser.json_path, num_workers=8, run_relax=False)
    print(energy_metrics)
