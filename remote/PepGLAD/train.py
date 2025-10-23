#!/usr/bin/python
# -*- coding:utf-8 -*-
import os
import argparse

import yaml
import torch

from utils.logger import print_log
from utils.random_seed import setup_seed, SEED
from utils.config_utils import overwrite_values
# from utils import register as R
from remote.PepGLAD.utils import register as R
# from utils.register import construct
########### Import your packages below ##########
import wandb
from trainer import create_trainer
import models
from data import create_dataset, create_dataloader
from utils.nn_utils import count_parameters


def parse():
    parser = argparse.ArgumentParser(description='training')

    # device
    parser.add_argument('--gpus', type=int, nargs='+', required=True, help='gpu to use, -1 for cpu')
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Local rank. Necessary for using the torch.distributed.launch utility.")
    parser.add_argument('--expname', type=str, required=True, help='expname')
    # config
    parser.add_argument('--config', type=str, required=True, help='Path to the yaml configure')
    parser.add_argument('--seed', type=int, default=SEED, help='Random seed')

    return parser.parse_known_args()


def load_ckpt(model, ckpt):
    trained_model = torch.load(ckpt, map_location='cpu')
    model.load_state_dict(trained_model.state_dict())
    return model


def main(args, opt_args):
    
    # load config
    config = yaml.safe_load(open(args.config, 'r'))
    config = overwrite_values(config, opt_args)

    ########## define your model #########
    model = R.construct(config['model'])
    if 'load_ckpt' in config:
        model = load_ckpt(model, config['load_ckpt'])

    ########### load your train / valid set ###########
    train_set, valid_set, _ = create_dataset(config['dataset'])
    print('len(train set):', len(train_set))
    print('len(valid set):', len(valid_set))
    ########## define your trainer/trainconfig #########
    if len(args.gpus) > 1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', world_size=len(args.gpus))
    else:
        args.local_rank = -1

    if args.local_rank <= 0:
        print_log(f'Number of parameters: {count_parameters(model) / 1e6} M')
    
    train_loader = create_dataloader(train_set, config['dataloader'], len(args.gpus))
    valid_loader = create_dataloader(valid_set, config['dataloader'], validation=True)

    wandb.init(project="PepGLAD", sync_tensorboard=True,name=args.expname, config=config)

    trainer = create_trainer(config, model, train_loader, valid_loader)
    trainer.train(args.gpus, args.local_rank)


if __name__ == '__main__':
    args, opt_args = parse()
    print_log(f'Overwritting args: {opt_args}')
    setup_seed(args.seed)
    main(args, opt_args)
