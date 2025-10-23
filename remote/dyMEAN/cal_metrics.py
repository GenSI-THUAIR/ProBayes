#!/usr/bin/python
# -*- coding:utf-8 -*-
import argparse
import json
import os
from time import time
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

import numpy as np

from remote.dyMEAN.data.pdb_utils import AgAbComplex
from remote.dyMEAN.evaluation.rmsd import compute_rmsd
from remote.dyMEAN.evaluation.tm_score import tm_score
from remote.dyMEAN.evaluation.lddt import lddt
from remote.dyMEAN.evaluation.dockq import dockq
from remote.dyMEAN.utils.relax import openmm_relax, rosetta_sidechain_packing
from remote.dyMEAN.utils.logger import print_log

from remote.dyMEAN.configs import CONTACT_DIST

from remote.ppflow.tools.relax.rosetta_packing import side_chain_packing

def cal_metrics(inputs):
    if len(inputs) == 6:
        mod_pdb, ref_pdb, H, L, A, cdr_type = inputs
        sidechain = False
    elif len(inputs) == 7:
        mod_pdb, ref_pdb, H, L, A, cdr_type, sidechain = inputs
    do_refine = False

    # sidechain = True
    # sidechain packing
    if sidechain:
        refined_pdb = mod_pdb[:-4] + '_sidechain.pdb'
        # mod_pdb = rosetta_sidechain_packing(mod_pdb, refined_pdb)
        side_chain_packing(mod_pdb, refined_pdb)

    # load complex
    if do_refine:
        refined_pdb = mod_pdb[:-4] + '_refine.pdb'
        pdb_id = os.path.split(mod_pdb)[-1]
        print(f'{pdb_id} started refining')
        start = time()
        mod_pdb = openmm_relax(mod_pdb, refined_pdb, excluded_chains=A)  # relax clashes
        print(f'{pdb_id} finished openmm relax, elapsed {round(time() - start)} s')
    mod_cplx = AgAbComplex.from_pdb(mod_pdb, H, L, A, skip_epitope_cal=True, numbering='imgt')
    ref_cplx = AgAbComplex.from_pdb(ref_pdb, H, L, A, skip_epitope_cal=False, numbering='imgt')

    results = {}
    cdr_type = [cdr_type] if type(cdr_type) == str else cdr_type

    # 1. AAR & CAAR
    # CAAR
    epitope = ref_cplx.get_epitope()
    is_contact = []
    if cdr_type is None:  # entire antibody
        gt_s = ref_cplx.get_heavy_chain().get_seq() + ref_cplx.get_light_chain().get_seq()
        pred_s = mod_cplx.get_heavy_chain().get_seq() + mod_cplx.get_light_chain().get_seq()
        # contact
        for chain in [ref_cplx.get_heavy_chain(), ref_cplx.get_light_chain()]:
            for ab_residue in chain:
                contact = False
                for ag_residue, _, _ in epitope:
                    dist = ab_residue.dist_to(ag_residue)
                    if dist < CONTACT_DIST:
                        contact = True
                is_contact.append(int(contact))
    else:
        gt_s, pred_s = '', ''
        for cdr in cdr_type:
            gt_cdr = ref_cplx.get_cdr(cdr)
            cur_gt_s = gt_cdr.get_seq()
            cur_pred_s = mod_cplx.get_cdr(cdr).get_seq()
            gt_s += cur_gt_s
            pred_s += cur_pred_s
            # contact
            cur_contact = []
            for ab_residue in gt_cdr:
                contact = False
                for ag_residue, _, _ in epitope:
                    dist = ab_residue.dist_to(ag_residue)
                    if dist < CONTACT_DIST:
                        contact = True
                cur_contact.append(int(contact))
            is_contact.extend(cur_contact)

            hit, chit = 0, 0
            for a, b, contact in zip(cur_gt_s, cur_pred_s, cur_contact):
                if a == b:
                    hit += 1
                    if contact == 1:
                        chit += 1
            results[f'AAR {cdr}'] = hit * 1.0 / len(cur_gt_s)
            results[f'CAAR {cdr}'] = chit * 1.0 / (sum(cur_contact) + 1e-10)

    if len(gt_s) != len(pred_s):
        print_log(f'Length conflict: {len(gt_s)} and {len(pred_s)}', level='WARN')
    hit, chit = 0, 0
    for a, b, contact in zip(gt_s, pred_s, is_contact):
        if a == b:
            hit += 1
            if contact == 1:
                chit += 1
    results['AAR'] = hit * 1.0 / len(gt_s)
    results['CAAR'] = chit * 1.0 / (sum(is_contact) + 1e-10)

    # 2. RMSD(CA) w/o align
    gt_x, pred_x = [], []
    for xl, c in zip([gt_x, pred_x], [ref_cplx, mod_cplx]):
        for chain in [c.get_heavy_chain(), c.get_light_chain()]:
            for i in range(len(chain)):
                xl.append(chain.get_ca_pos(i))
    assert len(gt_x) == len(pred_x), f'coordinates length conflict'
    gt_x, pred_x = np.array(gt_x), np.array(pred_x)
    results['RMSD(CA) aligned'] = compute_rmsd(gt_x, pred_x, aligned=False)
    # results['RMSD(CA)'] = compute_rmsd(gt_x, pred_x, aligned=True)
    if cdr_type is not None:
        for cdr in cdr_type:
            gt_cdr, pred_cdr = ref_cplx.get_cdr(cdr), mod_cplx.get_cdr(cdr)
            gt_x = np.array([gt_cdr.get_ca_pos(i) for i in range(len(gt_cdr))])
            pred_x = np.array([pred_cdr.get_ca_pos(i) for i in range(len(pred_cdr))])
            results[f'RMSD(CA) CDR{cdr}'] = compute_rmsd(gt_x, pred_x, aligned=True)
            results[f'RMSD(CA) CDR{cdr} aligned'] = compute_rmsd(gt_x, pred_x, aligned=False)

    # 3. TMscore
    results['TMscore'] = tm_score(mod_cplx.antibody, ref_cplx.antibody)

    # 4. LDDT
    score, _ = lddt(mod_cplx.antibody, ref_cplx.antibody)
    results['LDDT'] = score

    # 5. DockQ
    try:
        score = dockq(mod_cplx, ref_cplx, cdrh3_only=True) # consistent with HERN
    except Exception as e:
        print_log(f'Error in dockq: {e}, set to 0', level='ERROR')
        score = 0
    results['DockQ'] = score

    # print(f'{mod_cplx.get_id()}: {results}')
    results['pdb_id'] = mod_cplx.get_id().split('_')[0]
    return results


def main(args):
    with open(args.test_set, 'r') as fin:
        lines = fin.read().strip().split('\n')
    items = [json.loads(line) for line in lines]
    metric_inputs, pdbs = [], [item['pdb'] for item in items]
    pmets = []
    for item in items:
        keys = ['mod_pdb', 'ref_pdb', 'H', 'L', 'A', 'cdr_type']
        inputs = [item[key] for key in keys]
        if 'sidechain' in item:
            inputs.append(item['sidechain'])
        metric_inputs.append(inputs)
        pmets.append(item['pmetric'])

    if args.num_workers > 1:
        metrics = process_map(cal_metrics, metric_inputs, max_workers=args.num_workers)
    else:
        metrics = [cal_metrics(inputs) for inputs in tqdm(metric_inputs)]
        
    log_metrics = {}
    for name in metrics[0]:
        vals = [item[name] for item in metrics]
        if name == 'pdb_id':
            continue
        print(f'{name}: {sum(vals) / len(vals)}')
        lowest_i = min([i for i in range(len(vals))], key=lambda i: vals[i])
        highest_i = max([i for i in range(len(vals))], key=lambda i: vals[i])
        print(f'\tlowest: {vals[lowest_i]}, pdb: {pdbs[lowest_i]}')
        print(f'\thighest: {vals[highest_i]}, pdb: {pdbs[highest_i]}')
        # calculate correlation
        corr = np.corrcoef(pmets, vals)[0][1]
        print(f'\tpearson correlation with development metric: {corr}')
        log_metrics[name] = sum(vals) / len(vals)
    return log_metrics


def run_cal_metrics_antibody(test_summary_path=None, num_workers=8):
    '''
    '''
    args = argparse.Namespace(test_set=test_summary_path, num_workers=num_workers)
    metrics = main(args)
    return metrics

def parse():
    parser = argparse.ArgumentParser(description='calculate metrics')
    # parser.add_argument('--test_set', type=str, required=True, help='Path to test set')
    # parser.add_argument('--test_set', type=str, default='/data/wuhl/bfn4pep/logs/bfn_antibody_clothia[dev-a9cb598][05-15-02-21-44]_clothia/checkpoints/results/summary.json', help='Path to test set')
    parser.add_argument('--test_set', type=str, default='/data/wuhl/bfn4pep/logs/bfn_antibody[dev-4b14445][05-11-01-25-09]_no_seq_sc_mask_mixsc/checkpoints/results/summary.json', help='Path to test set')

    parser.add_argument('--num_workers', type=int, default=1, help='Number of workers to use')

    return parser.parse_args()


if __name__ == '__main__':
    main(parse())
