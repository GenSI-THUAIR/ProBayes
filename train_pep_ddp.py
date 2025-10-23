import wandb

import os
os.environ['WANDB_BASE_URL'] = 'https://api.bandw.top'
import shutil
import argparse
import torch
import torch.cuda.amp as amp
import torch.distributed as distrib
import time
import datetime
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, random_split
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm
import yaml
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from probayes.utils.vc import get_version, has_changes
from probayes.utils.misc import BlackHole, inf_iterator, load_config, seed_all, get_logger, get_new_log_dir, current_milli_time, set_kwargs
from probayes.utils.data import PaddingCollate
from probayes.utils.train import ScalarMetricAccumulator, count_parameters, get_optimizer, get_scheduler, log_losses, recursive_to, sum_weighted_losses

from probayes.eval.get_ckpt_all_metrics import get_all_metrics
from probayes.dataset.pep_dataset import PepDataset
from probayes.core.flow_model import FlowModel
from probayes.core.flow_model_debug import FlowModel as FlowModel_debug
from train_pep import get_model
import easydict
from ema_pytorch import EMA
from torch.utils.data import Subset
class EasyDictDumper(yaml.Dumper):
    def represent_easydict(self, data):
        return self.represent_mapping('tag:yaml.org,2002:map', data.items())

# 注册自定义表示
yaml.add_representer(easydict.EasyDict, EasyDictDumper.represent_easydict)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./configs/learn_angle.yaml')
    parser.add_argument('--logdir', type=str, default="./logs")
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--local-rank', type=int, help='Local rank. Necessary for using the torch.distributed.launch utility.')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--name', type=str, default='probayes')
    parser.add_argument('--kwargs', nargs='*')

    args = parser.parse_args()

    local_rank=int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)

    # Version control
    branch, version = get_version()
    version_short = '%s-%s' % (branch, version[:7])

    # Load configs
    config, config_name = load_config(args.config)
    # dict_config = yaml.safe_load(open(args.config,'r'))
    seed_all(config.train.seed + local_rank * 100)
    config = set_kwargs(args.kwargs, config)
    if local_rank == 0:
        writer = BlackHole()
        # Logging
        if args.debug:
            logger = get_logger('train', None, local_rank)
            run = wandb.init(project=args.name, config=config, name='%s[%s]' % (config_name, args.tag),mode='offline')
        else:
            run = wandb.init(project=args.name, config=config, name='%s[%s]' % (config_name, args.tag))
        
        # get the log_dir
        if args.resume:
            log_dir = os.path.dirname(os.path.dirname(args.resume))
        else:
            # get the current time
            utc_time = time.gmtime()
            cur_time = time.strftime("%m-%d-%H-%M-%S", time.localtime(time.mktime(utc_time) + 8 * 3600))
            log_dir = get_new_log_dir(args.logdir, prefix='%s[%s][%s]' % (config_name, version_short,cur_time), tag=args.tag)
        
        # write the commit branch and version
        with open(os.path.join(log_dir, 'commit.txt'), 'w') as f:
            f.write(branch + '\n')
            f.write(version + '\n')
        
        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        if not args.debug: logger = get_logger('train', log_dir)
        # save the new config file
        with open(os.path.join(log_dir, os.path.basename(args.config)), 'w') as file:
            yaml.dump(config, file, Dumper=EasyDictDumper)
        # if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
        #     shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    
    if local_rank == 0: logger.info(args)
    if local_rank == 0: logger.info(config)

    # Set up DDP
    if local_rank == 0: logger.info('Initializing DDP...')
    distrib.init_process_group(backend="nccl", timeout=datetime.timedelta(hours=5))

    # Data
    if local_rank == 0: logger.info('Loading datasets...')
    train_dataset = PepDataset(structure_dir = config.dataset.train.structure_dir, dataset_dir = config.dataset.train.dataset_dir,
                                            name = config.dataset.train.name, transform=None, reset=config.dataset.train.reset)

    # TODO: use testset as valset
    if args.debug == True:
        val_size = 2
    else:
        val_size = 50
        
    val_dataset = PepDataset(structure_dir = config.dataset.val.structure_dir, dataset_dir = config.dataset.val.dataset_dir,
                                            name = config.dataset.val.name, transform=None, reset=config.dataset.val.reset)
    val_dataset = Subset(val_dataset,range(val_size))
    
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=config.train.batch_size, shuffle=False, collate_fn=PaddingCollate(), sampler=val_sampler, num_workers=args.num_workers)
    
    # TODO: to be deleted
    test_dataset = PepDataset(structure_dir = config.dataset.test.structure_dir, 
                              dataset_dir = config.dataset.test.dataset_dir,
                                name = config.dataset.test.name, 
                                transform=None, 
                                reset=config.dataset.test.reset)
    gen_size = 1000 if 'gen_size' not in config.train else config.train.gen_size
    test_dataset = Subset(test_dataset,range(min(gen_size,len(test_dataset))))
    test_sampler = DistributedSampler(test_dataset, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=min(config.train.batch_size,len(test_dataset)), shuffle=False, collate_fn=PaddingCollate(), sampler=test_sampler, num_workers=args.num_workers)
    
    train_batch_size = config.train.batch_size if local_rank!=0 else int(config.train.batch_size*0.7)
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=train_batch_size, collate_fn=PaddingCollate(), sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    
    train_iterator = inf_iterator(train_loader)
    val_iterator = inf_iterator(val_loader)
    
    # TODO: to be delted
    test_iterator = inf_iterator(test_loader)
    
    
    if local_rank == 0: logger.info('Train %d | Val %d | Test %d' % (len(train_dataset), len(val_dataset),len(test_dataset)))

    # Model
    if local_rank == 0: logger.info('Building model...')
    if args.ckpt is None:
        model = DDP(get_model(config).to(local_rank), device_ids=[local_rank])
    else:
        model = get_model(config).to(local_rank)
        model = EMA(
            model,
            beta = config.train.ema.decay,              # exponential moving average factor
            update_after_step = config.train.ema.update_after_step,    # only after this number of .update() calls will it start updating
            update_every = config.train.ema.update_every,          # how often to actually update, to save on compute (updates every 10th .update() call)
            forward_method_names=['sample']
        )
        weights = torch.load(args.ckpt, map_location=f'cuda:{local_rank}')
        model.load_state_dict(weights['ema']) #TODO: check EMA
        model = DDP(model, device_ids=[local_rank])
        logger.info('Load model from %s' % args.ckpt)
        
    # model = DDP(get_model(config).to(local_rank), device_ids=[local_rank],find_unused_parameters=True)
    
    if local_rank==0 and config.train.ema.enable:
        ema = EMA(
            model.module,
            beta = config.train.ema.decay,              # exponential moving average factor
            update_after_step = config.train.ema.update_after_step,    # only after this number of .update() calls will it start updating
            update_every = config.train.ema.update_every,          # how often to actually update, to save on compute (updates every 10th .update() call)
            forward_method_names=['sample']
        )
    
    # wandb.watch(model,log='all',log_freq=1)
    if local_rank == 0: logger.info('Number of parameters: %d' % count_parameters(model))

    # Optimizer & Scheduler
    optimizer = get_optimizer(config.train.optimizer, model)
    scheduler = get_scheduler(config.train.scheduler, optimizer)
    optimizer.zero_grad()
    it_first = 0

    # Resume
    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=f'cuda:{local_rank}')
        it_first = ckpt['iteration']  # + 1
        model.load_state_dict(ckpt['model'])
        logger.info('Resuming optimizer states...')
        optimizer.load_state_dict(ckpt['optimizer'])
        logger.info('Resuming scheduler states...')
        scheduler.load_state_dict(ckpt['scheduler'])
        # debug
        # torch.autograd.set_detect_anomaly(True)

    def train(it):
        time_start = current_milli_time()
        model.train()

        # Prepare data
        batch = recursive_to(next(train_iterator), local_rank)

        # # inspect
        # if local_rank == 0:
        #     torch.autograd.set_detect_anomaly(True)

        # Forward pass
        loss_dict = model(batch) # get loss and metrics
        loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
        time_forward_end = current_milli_time()

        loss.backward()

        # rescue for nan grad
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    param.grad[torch.isnan(param.grad)] = 0
        
        orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

        # Backward
        # if it % config.train.accum_grad ==0:
        optimizer.step()
        if local_rank==0 and config.train.ema.enable: ema.update()
        optimizer.zero_grad()
        time_backward_end = current_milli_time()

        # Logging
        if local_rank == 0:
            scalar_dict = {}
            # scalar_dict.update(metric_dict['scalar'])
            scalar_dict.update({
                'grad': orig_grad_norm,
                'lr': optimizer.param_groups[0]['lr'],
                'time_forward': (time_forward_end - time_start) / 1000,
                'time_backward': (time_backward_end - time_forward_end) / 1000,
            })
            log_losses(loss, loss_dict, scalar_dict, it=it, tag='train', logger=logger)

    def validate(it):
        scalar_accum = ScalarMetricAccumulator()
        with torch.no_grad():
            model.eval()
            val_model = ema if config.train.ema.enable else model.module
            # metrics = get_metrics(val_model, val_dataset, device=local_rank)
            # TODO: to be modifed
            metrics = get_all_metrics(ckpt_dir=ckpt_dir, model=val_model, dataset=test_dataset,
                                      n_steps=200, sample_bb=config.model.sample_bb, 
                                      sample_ang=config.model.sample_sc, 
                                      sample_seq=config.model.sample_seq, 
                                      device=local_rank, 
                                      n_samples=int(config.dataset.n_gen_samples),
                                      dataset_pdb_dir=config.dataset.test.structure_dir, 
                                      sample_mode='end_back',
                                      sc_pack=config.model.sc_pack
                                      ) 
            wandb.log(metrics, step=it)
        return metrics
    
    def val_loss(model, val_iterator, it, logger, writer, mode='val'):
        scalar_accum = ScalarMetricAccumulator()
        with torch.no_grad():
            model.eval()
            batch = recursive_to(next(val_iterator), local_rank)
            # for i, batch in enumerate(tqdm(val_iterator, desc='Validate', dynamic_ncols=True)):
            #     # Prepare data
            # batch = recursive_to(batch, local_rank)
            # Forward pass
            loss_dict = model(batch)
            loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
            scalar_accum.add(name='loss', value=loss, batchsize=len(batch['aa']), mode='mean')
            for k, v in loss_dict.items():
                scalar_accum.add(name=k, value=v, batchsize=len(batch['aa']), mode='mean')
        avg_loss = scalar_accum.get_average('loss')
        summary = scalar_accum.log(it, mode, logger=logger, writer=writer)
        return val_iterator
    

    def average_metric(metrics):
        '''consider all metrics to evaluate the ckpt, get a score where larger is better'''
        return metrics['DockQ(mean)'] + metrics['succ_rate'] - metrics['RMSD(CA)(median)']
   
    best_score = -1e4 # larger is better
    best_path = None
    try:
        for it in range(it_first, config.train.max_iters + 1):
            train(it)

            if (it+1) % 10 == 0:
                if local_rank == 0:
                    val_iterator = val_loss(model, val_iterator, it, logger, writer)       
            
                # TODO: to be deleted
                # if local_rank == 1:
                    test_iterator = val_loss(model, test_iterator, it, logger, writer, mode='test')                 
            
            if (args.debug == True or (it+1) % config.train.val_freq == 0) and local_rank == 0 and config.train.run_gen:
                if config.train.run_gen:
                    val_metrics = validate(it)
                    if val_metrics is not None:
                        score_str = '-'.join([key+'-'+f'{val:.3f}' 
                            for key,val in val_metrics.items() 
                            if key in ['DockQ(mean)','aligned_CaRMSD','succ_rate']])
                        ckpt_path = os.path.join(ckpt_dir, f'step-{it}-{score_str}.pt')
                        torch.save({
                            'config': config,
                            'model': model.state_dict(),
                            'ema': ema.state_dict() if config.train.ema.enable else None,
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'iteration': it,
                            # 'avg_val_loss': avg_val_loss,
                        }, ckpt_path)
                    # if val_metrics is not None and average_metric(val_metrics) > best_score:
                    #     best_score = average_metric(val_metrics)
                    # score_str = '-'.join([key+'-'+f'{val:.3f}' 
                    #                       for key,val in val_metrics.items() 
                    #                       if key in ['DockQ(mean)','aligned_CaRMSD','succ_rate']])
                    #     ckpt_path = os.path.join(ckpt_dir, f'step-{it}-{score_str}.pt')
                    #     # if best_path is not None:
                    #     #     os.remove(best_path) # only save the best one
                    #     best_path = ckpt_path
                    #     torch.save({
                    #         'config': config,
                    #         'model': model.state_dict(),
                    #         'ema': ema.state_dict() if config.train.ema.enable else None,
                    #         'optimizer': optimizer.state_dict(),
                    #         'scheduler': scheduler.state_dict(),
                    #         'iteration': it,
                    #         # 'avg_val_loss': avg_val_loss,
                    #     }, ckpt_path)
                else:
                    raise NotImplementedError('No validation function implemented')
                    ckpt_path = os.path.join(ckpt_dir, f'step-{it}.pt')

    except KeyboardInterrupt:
        logger.info('Terminating...')
        distrib.destroy_process_group()
