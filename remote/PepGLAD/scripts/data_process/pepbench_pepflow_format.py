#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse

from data.mmap_dataset import create_mmap
from data.format import VOCAB
from data.converter.pdb_to_list_blocks import pdb_to_list_blocks
from data.converter.blocks_interface import blocks_interface, blocks_cb_interface
from remote.PepGLAD.data.converter.list_blocks_to_pdb import list_blocks_to_pdb
from tqdm import tqdm
from utils.logger import print_log


def parse():
    parser = argparse.ArgumentParser(description='Process protein-peptide complexes')
    parser.add_argument('--index', type=str, default=None, help='Index file of the dataset')
    parser.add_argument('--out_dir', type=str, required=True, help='Output Directory')
    parser.add_argument('--pocket_th', type=float, default=10.0,
                        help='Threshold for determining binding site')
    return parser.parse_args()
# python -m scripts.data_process.pepbench_probayes_format --index /data/wuhl/bfn4pep/probayes_data/train_valid/all.txt --out_dir /data/wuhl/bfn4pep/probayes_data/pepbench_probayes_processed
# python -m scripts.data_process.pepbench_probayes_format --index /data/wuhl/bfn4pep/remote/PepGLAD/datasets/LNR/test.txt --out_dir /data/wuhl/bfn4pep/probayes_data/LNR_probayes_processed
def process_iterator(items, pdb_path, pocket_th):
    # for cnt, pdb_id in enumerate(items):
    for index, (pdb_id, _)  in enumerate(tqdm(items.items(),desc='Processing PDBs')):
        summary = items[pdb_id]
        rec_chain, lig_chain = summary['rec_chain'], summary['pep_chain']
        non_standard = 0
        rec_blocks, lig_blocks = pdb_to_list_blocks(summary['pdb_path'], selected_chains=[rec_chain, lig_chain])
        _, (_, pep_if_idx) = blocks_interface(rec_blocks, lig_blocks, 6.0) # 6A for atomic interaction
        if len(pep_if_idx) == 0:
            continue
        try:
            _, (pocket_idx, _) = blocks_cb_interface(rec_blocks, lig_blocks, pocket_th)  # 10A for pocket size based on CB
        except KeyError:
            print_log(f'{pdb_id} missing backbone atoms')
            continue # missing both CB and backbone atoms
        rec_num_units = sum([len(block) for block in rec_blocks])
        lig_num_units = sum([len(block) for block in lig_blocks])

        data = ([block.to_tuple() for block in rec_blocks], [block.to_tuple() for block in lig_blocks])
        rec_seq = ''.join([VOCAB.abrv_to_symbol(block.abrv) for block in rec_blocks])
        lig_seq = ''.join([VOCAB.abrv_to_symbol(block.abrv) for block in lig_blocks])
        
        # if '?' in [rec_seq[i] for i in pocket_idx] or '?' in lig_seq:
        if '?' in lig_seq:
            non_standard = 1  # has non-standard amino acids
        pdb_id_path = os.path.join(pdb_path, pdb_id)
        if not os.path.exists(pdb_id_path):
            os.makedirs(pdb_id_path)
        # save receptor
        list_blocks_to_pdb(
            [rec_blocks],
            [rec_chain],
            os.path.join(pdb_id_path, 'receptor.pdb')
        )
        # save pocket
        list_blocks_to_pdb(
            [[rec_blocks[i] for i in pocket_idx]],
            [rec_chain],
            os.path.join(pdb_id_path, 'pocket.pdb')
        )
        # save ligand
        list_blocks_to_pdb(
            [lig_blocks],
            [lig_chain],
            os.path.join(pdb_id_path, 'peptide.pdb')
        )

        # yield pdb_id, data, [
        #     len(rec_blocks), len(lig_blocks), rec_num_units, lig_num_units,
        #     rec_chain, lig_chain, rec_seq, lig_seq, non_standard,
        #     ','.join([str(idx) for idx in pocket_idx]),
        #     ], cnt


def main(args):

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # 1. get index file
    with open(args.index, 'r') as fin:
        lines = fin.readlines()
    indexes = {}
    root_dir = os.path.dirname(args.index)
    for line in lines:
        line = line.strip().split('\t')
        pdb_id = line[0]
        indexes[pdb_id] = {
            'rec_chain': line[1],
            'pep_chain': line[2],
            'pdb_path': os.path.join(root_dir, 'pdbs', pdb_id + '.pdb')
        }

    # 3. process pdb files into our format (mmap)
    # create_mmap(
    process_iterator(indexes, os.path.join(args.out_dir, 'pdbs'), args.pocket_th),len(indexes)
        # )
    
    print_log('Finished!')


if __name__ == '__main__':
    main(parse())