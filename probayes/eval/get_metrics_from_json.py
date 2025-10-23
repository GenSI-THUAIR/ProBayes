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
from probayes.utils.data import PaddingCollate
import gc
import os.path as osp
from probayes.core.flow_model import FlowModel
from Bio.PDB import Structure, PDBParser

from probayes.utils.misc import seed_all
from probayes.utils.metrics import get_metrics, prepare_result_json, Evaluator
from remote.PepGLAD.evaluation.dG.run import run_eval_tasks
from remote.PepGLAD.cal_metrics import run_cal_metrics
from remote.PepGLAD.data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from remote.PepGLAD.data.converter.list_blocks_to_pdb import list_blocks_to_pdb
collate_fn = PaddingCollate(eight=False)
from collections import defaultdict
import argparse
import json
from Bio.SeqUtils import seq1
from tqdm import tqdm

def get_metrics_from_json(json_path,filter_dG=None, get_dG=False,get_pepglad=False,num_workers=1):
    # load results
    with open(json_path, 'r') as fin:
        lines = fin.read().strip().split('\n')
    
    # load dG filter
    if filter_dG is None:
        filter_func = lambda _id, n: True
    else:
        dG_results = json.load(open(filter_dG, 'r'))
        filter_func = lambda _id, n: dG_results[_id]['all'][str(n)] < 0
    
    # load by pdb id
    id2items = {}
    for line in lines:
        item = json.loads(line)
        _id = item['id']
        if not filter_func(_id, item['number']):
            continue
        if _id not in id2items:
            id2items[_id] = []
        id2items[_id].append(item)
    
    
    evaluator = Evaluator()    
    pdb_ids = list(id2items.keys())

    cnt = 0
    for pdb_id in tqdm(pdb_ids,desc='Processing'):
        items = id2items[pdb_id]
        items = sorted(items, key=lambda x: x['number'])
        id2items[pdb_id] = items
        gen_lig_paths = []
        ref_lig_path = None      
        
        # cnt += 1
        # if cnt == 3:
        #     break
          
        if 'ref_lig_path' not in items[0].keys():
            # for PepGLAD 
            for item in items:
                pdb_id = item['id']
                
                gen_pdb = item['gen_pdb']
                fname, ext = osp.splitext(osp.basename(gen_pdb))
                # use rosetta if exists
                # rosetta_pdb = osp.join(osp.dirname(gen_pdb), f'{fname}_rosetta{ext}')
                # if os.path.exists(rosetta_pdb):
                #     gen_pdb = rosetta_pdb
                #     print('use rosetta pdb')
                # else:
                #     print('fail')
                #     exit()
                rec_chain = item['rec_chain']
                lig_chain = item['lig_chain']
                ref_pdb = item['ref_pdb']
                number = item['number']
                
                rec_blocks, gen_lig_blocks = pdb_to_list_blocks(gen_pdb, selected_chains=[rec_chain, lig_chain])
                # 1bjr_pep_atom14_24.pdb
                gen_lig_fname = f'{pdb_id}_gen_lig_{number}.pdb'
                gen_lig_path = osp.join(osp.dirname(gen_pdb), gen_lig_fname)
                list_blocks_to_pdb([gen_lig_blocks], [lig_chain], gen_lig_path)
                gen_lig_paths.append(gen_lig_path)
                
                if ref_lig_path is None:
                    rec_blocks, ref_lig_blocks = pdb_to_list_blocks(ref_pdb, selected_chains=[rec_chain, lig_chain])
                    ref_lig_fname = f'{pdb_id}_lig_ref.pdb'
                    ref_lig_path = osp.join(osp.dirname(ref_pdb), ref_lig_fname)
                    list_blocks_to_pdb([ref_lig_blocks], [lig_chain], ref_lig_path)
                    
            
        else:
            if item['cand_lig_path'].endswith('ref_lig.pdb'): # a previous bug
                 for item in items:
                    gen_lig_paths.append(item['ref_lig_path'])
                    if ref_lig_path is None:
                        ref_lig_path = item['cand_lig_path']           
            else:
                for item in items:
                    gen_lig_paths.append(item['cand_lig_path'])
                    if ref_lig_path is None:
                        ref_lig_path = item['ref_lig_path']
                    
        evaluator.add_pair_path(ref_pep_path=ref_lig_path, gen_pep_path=gen_lig_paths)

    if get_dG:
        dG_metrics = run_eval_tasks(results=json_path,n_sample=10,rfdiff_relax=False)
        print(dG_metrics)
        metrics.update(dG_metrics)

    if get_pepglad:
        pepglad_metrics = run_cal_metrics(results=json_path, num_workers=num_workers, rosetta=False)
        print(pepglad_metrics)
        metrics.update(pepglad_metrics)

    metrics = evaluator.get_metrics()


    # probayes_metrics = get_probayes_metrics(json_path)
    return metrics


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--json_path', type=str,default='/data/wuhl/bfn4pep/rfdiffusion_results/results.jsonl')
    # store true
    args.add_argument('--get_dG', action='store_true')
    args.add_argument('--get_pepglad', action='store_true')
    args.add_argument('--device', type=str, default='cuda:0')
    args.add_argument('--num_workers', type=int, default=1)
    args.add_argument('--tag', type=str, default='from_json')
    
    parser = args.parse_args()

    device = parser.device

    json_path = parser.json_path
    metrics = get_metrics_from_json(json_path, get_dG=bool(parser.get_dG),get_pepglad=bool(parser.get_pepglad),num_workers=parser.num_workers)
    print('done')
    for key, value in metrics.items():
        print(f'{key}: {value}')
        
    # # 将 metrics 写入文件夹
    metrics_df = pd.DataFrame(metrics,index=[0]).T
    metrics_df.to_csv(osp.join(osp.dirname(json_path),f'metrics_{parser.tag}.csv'))

