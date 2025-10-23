#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse

import numpy as np

from utils.logger import print_log


def parse():
    parser = argparse.ArgumentParser(description='Split peptide data')
    parser.add_argument('--train_index', type=str, required=True, help='Path for training index')
    parser.add_argument('--valid_index', type=str, required=True, help='Path for validation index')
    parser.add_argument('--test_index', type=str, default=None, help='Path for test index')
    parser.add_argument('--processed_dir', type=str, required=True, help='processed directory')
    return parser.parse_args()


def read_index(mmap_dir):
    items = {}
    index = os.path.join(mmap_dir, 'index.txt')
    with open(index, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            values = line.strip().split('\t')
            items[values[0]] = line
    return items


def transform(items, path, out):
    ids = {}
    with open(path, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            ids[line.split('\t')[0]] = 1
    miss_key_cnt = 0
    with open(out, 'w') as fout:
        # for _id in ids: fout.write(items[_id])
        for _id in ids:
            if _id in items:  # 检查键是否存在于 items 中
                fout.write(items[_id])
            else:
                miss_key_cnt += 1
                print(f"Warning: Key '{_id}' not found in items. Skipping.")

    print('miss ',miss_key_cnt, ' files')
def main(args):

    # load index file
    items = read_index(args.processed_dir)

    # load training/validation/(test)
    transform(items, args.train_index, os.path.join(args.processed_dir, 'train_index.txt'))
    transform(items, args.valid_index, os.path.join(args.processed_dir, 'valid_index.txt'))
    if args.test_index is not None:
        transform(items, args.test_index, os.path.join(args.processed_dir, 'test_index.txt'))

    print_log('Done')


if __name__ == '__main__':
    np.random.seed(12)
    main(parse())