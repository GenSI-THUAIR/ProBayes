#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from copy import deepcopy
from collections import defaultdict
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
import statistics
import numpy as np

from remote.dyMEAN.cal_metrics import cal_metrics

def main(args):
    # read summary file
    with open(args.summary, 'r') as fin:
        lines = fin.read().strip().split('\n')
    items = [json.loads(line) for line in lines]
    
    metric_inputs = []
    for sample in items:
        mod_pdb, ref_pdb, H, L, A, cdr_type = \
            sample['mod_pdb'], sample['ref_pdb'], sample['H'], sample['L'], sample['A'], sample['cdr_type']
        inputs = (mod_pdb, ref_pdb, H, L, A, cdr_type)
        metric_inputs.append(inputs)

    # multi-process or single-process evaluation
    if args.num_workers > 1:
        metrics = process_map(cal_metrics, metric_inputs, max_workers=args.num_workers, desc="Evaluating antibodies", chunksize=1)
    else:
        metrics = [cal_metrics(inputs) for inputs in tqdm(metric_inputs, desc="Evaluating antibodies")]
    
    # aggregate by pdb_id
    id2metrics = defaultdict(list)
    for metric in metrics:
        id2metrics[metric['pdb_id']].append(metric)
    
    log_metrics = defaultdict(list)
    
    for pdb_id, metrics in id2metrics.items():
        for name in metrics[0].keys():
            if name == 'pdb_id':
                continue
            metric_vals = [item[name] for item in metrics]
            metric_vals_median = np.median(metric_vals)
            metric_vals_mean = np.mean(metric_vals)
            log_metrics[name].append(metric_vals_median)
            log_metrics[f'{name}(mean)'].append(metric_vals_mean)
    
    return_metrics = defaultdict(float)
    for name in log_metrics.keys():
        return_metrics[name] = np.mean(log_metrics[name])
    print(return_metrics)
    return return_metrics

def run_cal_metrics_antibody(test_summary_path, num_workers=16):
    args = argparse.Namespace(summary=test_summary_path, num_workers=num_workers)
    return main(args)

def parse():
    parser = argparse.ArgumentParser(description='calculate metrics for multiple samples per receptor')
    parser.add_argument('--summary', type=str, help='Path to summary file (jsonl)', default='logs/bfn_antibody[dev-4b14445][05-11-01-23-53]_no_seq_sc_mask/results/summary.json')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers to use')
    return parser.parse_args()

if __name__ == '__main__':
    main(parse()) 