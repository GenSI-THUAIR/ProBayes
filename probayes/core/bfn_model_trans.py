import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import math
from tqdm.auto import tqdm
import functools
from torch.utils.data import DataLoader
from torch import tensor
import os
import argparse

import pandas as pd

from probayes.core.edge import EdgeEmbedder
from probayes.core.node import NodeEmbedder
from probayes.modules.common.layers import sample_from, clampped_one_hot
from probayes.modules.protein.constants import AA, BBHeavyAtom, max_num_heavyatoms, aaidx2rgidx_mat
from probayes.modules.common.geometry import construct_3d_basis
from probayes.utils.data import PaddingCollate

from probayes.modules.so3.dist import centered_gaussian,uniform_so3
from probayes.modules.common.geometry import batch_align, align

from tqdm import tqdm
from typing import Callable
import wandb

from probayes.data import so3_utils
from probayes.data import all_atom

from probayes.utils.misc import load_config
from probayes.utils.train import recursive_to
from easydict import EasyDict

from probayes.core.utils import process_dic
from probayes.core.torsion import get_torsion_angle, torsions_mask
import probayes.core.torus as torus
from probayes.modules.net.ga_trans import GAEncoder_Trans
import gc

from copy import deepcopy
from probayes.utils.data import PaddingCollate
collate_fn = PaddingCollate(eight=False)
from probayes.utils.train import recursive_to
from probayes.modules.bfn.bfn_base import bfnBase
resolution_to_num_atoms = {
    'backbone+CB': 5,
    'full': max_num_heavyatoms
}

class BFNModel_Trans(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self._model_cfg = cfg.encoder
        self._interpolant_cfg = cfg.interpolant

        self.node_embedder = NodeEmbedder(cfg.encoder.node_embed_size,max_num_heavyatoms)
        self.edge_embedder = EdgeEmbedder(cfg.encoder.edge_embed_size,max_num_heavyatoms)
        self.ga_encoder:GAEncoder_Trans = GAEncoder_Trans(cfg.encoder.ipa, cfg.bfn)

        self.sample_structure = self._interpolant_cfg.sample_structure
        self.sample_sequence = self._interpolant_cfg.sample_sequence

        self.K = self._interpolant_cfg.seqs.num_classes
        self.k = self._interpolant_cfg.seqs.simplex_value
        
        self.BFN:bfnBase = bfnBase()
        self.cfg = cfg
        if self.cfg.bfn.cond_acc:
            self.circular_var_bayesian_flow:Callable[...,tensor] = self.BFN.circular_var_bayesian_flow_sim
        else:
            self.circular_var_bayesian_flow:Callable[...,tensor] = self.BFN.circular_var_bayesian_flow
        # self._interpolant_cfg.t_normalization_clip = 0.995 #TODO: check this
        self.sigma1_trans = torch.tensor(self.cfg.bfn.sigma1_trans)
        self.beta1_seq = torch.tensor(self.cfg.bfn.beta1_seq)
        self.beta1_ang = torch.tensor(self.cfg.bfn.beta1_ang)        
        self.n_steps = self.cfg.bfn.n_steps
        self.use_dtime_loss = self.cfg.bfn.dtime_loss
        self.use_quat = self.cfg.bfn.use_quat
        
        self.epsilon = torch.tensor(1e-7)
    
    def encode(self, batch):
        rotmats_1 =  construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],batch['pos_heavyatom'][:, :, BBHeavyAtom.C],batch['pos_heavyatom'][:, :, BBHeavyAtom.N] )
        trans_1_list = [batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],batch['pos_heavyatom'][:, :, BBHeavyAtom.C],batch['pos_heavyatom'][:, :, BBHeavyAtom.N]]
        trans_1 = torch.stack(trans_1_list,dim=-2)
        seqs_1 = batch['aa']
        # ignore psi
        # batch['torsion_angle'] = batch['torsion_angle'][:,:,1:]
        # batch['torsion_angle_mask'] = batch['torsion_angle_mask'][:,:,1:]
        angles_1 = batch['torsion_angle']

        context_mask = torch.logical_and(batch['mask_heavyatom'][:, :, BBHeavyAtom.CA], ~batch['generate_mask'])
        structure_mask = context_mask if self.sample_structure else None
        sequence_mask = context_mask if self.sample_sequence else None
        node_embed = self.node_embedder(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'], 
                                        batch['mask_heavyatom'], structure_mask=structure_mask, sequence_mask=sequence_mask)
        edge_embed = self.edge_embedder(batch['aa'], batch['res_nb'], batch['chain_nb'], batch['pos_heavyatom'], 
                                        batch['mask_heavyatom'], structure_mask=structure_mask, sequence_mask=sequence_mask)
        
        return rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed
    
    def zero_center_part(self,pos,gen_mask,res_mask):
        """
        move pos by center of gen_mask
        pos: (B,N,3)
        gen_mask, res_mask: (B,N)
        """
        center = torch.sum(pos * gen_mask[...,None], dim=1) / (torch.sum(gen_mask,dim=-1,keepdim=True) + 1e-8) # (B,N,3)*(B,N,1)->(B,3)/(B,1)->(B,3)
        center = center.unsqueeze(1) # (B,1,3)
        # center = 0. it seems not center didnt influence the result, but its good for training stabilty
        pos = pos - center
        pos = pos * res_mask[...,None]
        return pos,center
    
    def seq_to_simplex(self,seqs):
        return clampped_one_hot(seqs, self.K).float() * self.k * 2 - self.k # (B,L,K)
    
    def forward(self, batch):

        num_batch, num_res = batch['aa'].shape
        DEVICE = batch['aa'].device
        gen_mask,res_mask,angle_mask = batch['generate_mask'].long(),batch['res_mask'].long(),batch['torsion_angle_mask'].long()
        #encode
        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = self.encode(batch) # no generate mask
        
        trans_1 = trans_1 / self.cfg.norm.std # norm
        # aaidx2rgidx_mat = torch.tensor(aaidx2rgidx_mat).to(DEVICE)
        
        # prepare for denoise
        # trans_1_c,_ = self.zero_center_part(trans_1,gen_mask,res_mask)
        trans_1_c = trans_1 # already centered when constructing dataset
        seqs_1_simplex = self.seq_to_simplex(seqs_1)
        seqs_1_prob = F.softmax(seqs_1_simplex,dim=-1)
        seqs_1_onehot = clampped_one_hot(seqs_1, self.K).float()

        with torch.no_grad():
            t = torch.randint(0, self.n_steps,size=(num_batch,1), 
                            device=batch['aa'].device)/self.n_steps 
            t_index = t * self.n_steps + 1
            if self.sample_structure:
                # corrupt trans
                mu_trans_t, _ = self.BFN.continuous_var_bayesian_flow(
                    t=t[...,None,None], sigma1=self.sigma1_trans, x=trans_1_c
                ) # (B,L,dim_space=3,n_atoms=3)
                mu_trans_t_c = torch.where(batch['generate_mask'][...,None,None],mu_trans_t,trans_1_c)
                mu_trans_t_calpha = mu_trans_t_c[:,:,BBHeavyAtom.CA]
                
                rotmats_t =  construct_3d_basis(mu_trans_t[:, :, BBHeavyAtom.CA],mu_trans_t[:, :, BBHeavyAtom.C],mu_trans_t[:, :, BBHeavyAtom.N])
                
                # corrup angles
                mu_angles_t, log_acc_t = self.circular_var_bayesian_flow(
                    x=angles_1,t_index=t_index, N=self.n_steps, beta1=self.beta1_ang,epsilon=self.epsilon)
                mu_angles_t = torch.where(batch['generate_mask'][...,None],mu_angles_t,angles_1)
                log_acc_t = torch.where(batch['generate_mask'][...,None],log_acc_t,torch.ones_like(log_acc_t))
            else:
                raise NotImplementedError
            
            if self.sample_sequence:
                # corrupt seqs
                theta_seqs_t = self.BFN.discrete_var_bayesian_flow(t=t[...,None], beta1=self.beta1_seq, x=seqs_1_onehot, K=self.K)
                theta_seqs_t = torch.where(batch['generate_mask'][..., None],theta_seqs_t,seqs_1_onehot)
            else:
                raise NotImplementedError

        # denoise
        pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1_prob = self.ga_encoder(
            t, rotmats_t, mu_trans_t_calpha, mu_angles_t, theta_seqs_t, node_embed, edge_embed, gen_mask, res_mask, log_acc_t)
        pred_seqs_1 = sample_from(pred_seqs_1_prob)
        pred_seqs_1 = torch.where(batch['generate_mask'],pred_seqs_1,torch.clamp(seqs_1,0,19))
        # pred_trans_1,_ = self.zero_center_part(pred_trans_1,gen_mask,res_mask)
        # TODO: do we need this?
        pred_trans_1 = pred_trans_1 # implicitly enforce zero center in gen_mask, in this way, we dont need to move receptor when sampling

        # bb aux loss
        gt_bb_atoms = all_atom.to_atom37(trans_1_c[:, :, BBHeavyAtom.CA], rotmats_1)[:, :, :3] 
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1, pred_rotmats_1)[:, :, :3]
        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * gen_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        bb_atom_loss = torch.mean(bb_atom_loss)

        # seq loss
        mask = torch.gt(gen_mask,0)
        selected_i = t_index.unsqueeze(-1).repeat(1,num_res,1)[mask,:]
        seqs_loss = self.BFN.dtime4discrete_loss_prob(
            i=selected_i,
            N=self.n_steps,
            beta1=self.beta1_seq,
            one_hot_x=seqs_1_onehot[mask,:],
            p_0=pred_seqs_1_prob[mask,:],
            K=self.K
        )
        # TODO: should we use the frame based positinal loss or original positional loss as the ground truth?
        # translation loss
        trans_loss = self.BFN.dtime4continuous_loss(
            i=selected_i, N=self.n_steps,
            sigma1=self.sigma1_trans,
            x_pred=pred_bb_atoms[mask,:],
            x=gt_bb_atoms[mask,:]
        )
        # use the original coordinate
        # trans_loss = self.BFN.dtime4continuous_loss(
        #     i=selected_i, N=self.n_steps,
        #     sigma1=self.sigma1_trans,
        #     x_pred=pred_bb_atoms[mask,:],
        #     x=trans_1_c[mask,:]
        # )

        # TODO: we should not use angle mask, as you dont know aa type when generating
        # angle loss
        angle_mask_loss = torsions_mask.to(batch['aa'].device)
        angle_mask_loss = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
        angle_mask_loss = torch.logical_and(batch['generate_mask'][...,None].bool(),angle_mask_loss)
        angle_mask = torch.gt(angle_mask_loss,0)
        angle_selected_i = t_index.unsqueeze(-1).repeat(1,num_res,5)[angle_mask]
        angle_loss = self.BFN.dtime4circular_loss(
            x=angles_1[angle_mask],x_pred=pred_angles_1[angle_mask], i=angle_selected_i,N=self.n_steps,
            alpha_i=self.BFN.alpha_wrt_index(angle_selected_i,N=self.n_steps,beta1=self.beta1_ang),
            mse_loss=False,
            device=DEVICE
        )

        # torsion_loss = torch.tensor(0.)

        return {
            "trans_loss": trans_loss,
            # 'rot_loss': rot_loss,
            'bb_atom_loss': bb_atom_loss,
            'seqs_loss': seqs_loss,
            'angle_loss': angle_loss,
            # 'torsion_loss': torsion_loss,
        }
    
    @torch.no_grad()
    def sample(self, batch, num_steps = 100, sample_bb=True, sample_ang=True, sample_seq=True, sample_mode='end_back'):
        DEVICE = batch['aa'].device
        num_batch, num_res = batch['aa'].shape
        gen_mask,res_mask = batch['generate_mask'],batch['res_mask']
        angle_mask_loss = torsions_mask.to(batch['aa'].device)
        
        # encode background information
        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = self.encode(batch) 
        
        trans_1 = trans_1 / self.cfg.norm.std # norm
        # trans_1_c,center = self.zero_center_part(trans_1,gen_mask,res_mask)
        trans_1_c = trans_1
        trans_1_calpha = trans_1_c[:,:,BBHeavyAtom.CA]
        
        # ground truth sequence as one-hot
        seqs_1_onehot = clampped_one_hot(seqs_1, self.K).float()

        # initial noise
        
        # backbone init
        if sample_bb:
            # rotation init
            rotmats_0 = uniform_so3(num_batch,num_res,device=batch['aa'].device)
            rotmats_0 = torch.where(batch['generate_mask'][...,None,None],rotmats_0,rotmats_1)
            # translation init, bfn prior for continuous variable
            mu_trans_0 = torch.zeros((num_batch,num_res,3,3), device=batch['aa'].device)
            mu_trans_0_c = torch.where(batch['generate_mask'][...,None,None],mu_trans_0,trans_1_c)            
        else:
            rotmats_0 = rotmats_1.detach().clone()
            mu_trans_0_c = trans_1_c.detach().clone()
        
        # sidechain init
        if sample_ang:
            mu_angles_0 = torus.tor_random_uniform(angles_1.shape, device=batch['aa'].device, dtype=angles_1.dtype) # (B,L,5)
            mu_angles_0 = torch.where(batch['generate_mask'][...,None],mu_angles_0,angles_1)
            log_angle_acc_t = self.BFN.norm_logbeta(
                            torch.log(torch.tensor((self.epsilon))) * torch.ones_like(mu_angles_0),
                            beta1=self.beta1_ang,epsilon=self.epsilon)
        else:
            mu_angles_0 = angles_1.detach().clone()
            
        # sequence init
        if sample_seq:
            # sequence parameter init
            theta_seqs_0 = torch.ones((num_batch, num_res, self.K)).to(DEVICE) / self.K  # [N, N_AA, K] discrete prior
            theta_seqs_1 = clampped_one_hot(seqs_1, self.K).float()
            theta_seqs_0 = torch.where(batch['generate_mask'][..., None], theta_seqs_0, theta_seqs_1)
        else:
            theta_seqs_0 = seqs_1.detach().clone()

        # Set-up time
        ts = torch.range(1, num_steps-1) / num_steps
        ts = [0] + ts.tolist()
        
        t_1 = ts[0]
        clean_traj = []
        rotmats_t_1, mu_trans_t_1_c, mu_angles_t_1, theta_seqs_t_1 = rotmats_0, mu_trans_0_c, mu_angles_0, theta_seqs_0
        mu_trans_t_1_calpha = mu_trans_t_1_c[:,:,BBHeavyAtom.CA]
        # denoise loop
        for t_2 in ts[1:]:
            # set up time
            t = torch.ones((num_batch, 1), device=batch['aa'].device) * t_1
            t_index = t * self.n_steps + 1
            
            # input to the neural network
            pred_rotmats_1, pred_trans_1_calpha, pred_angles_1, pred_seqs_1_prob = self.ga_encoder(
                t, rotmats_t_1, mu_trans_t_1_calpha, mu_angles_t_1, 
                theta_seqs_t_1, node_embed, edge_embed, 
                batch['generate_mask'].long(), batch['res_mask'].long(),
                log_angle_acc_t)
            
            # set those not in gen_mask as ground truth
            pred_rotmats_1 = torch.where(batch['generate_mask'][...,None,None],pred_rotmats_1,rotmats_1)
            pred_trans_1_calpha = torch.where(batch['generate_mask'][...,None],pred_trans_1_calpha,trans_1_calpha) # move receptor also
            pred_trans_1 = all_atom.to_atom37(pred_trans_1_calpha, pred_rotmats_1)[:, :, :3]

            # angles
            pred_angles_1 = torch.where(batch['generate_mask'][...,None],pred_angles_1,angles_1)
            # seqs
            pred_seqs_1 = sample_from(pred_seqs_1_prob)
            pred_seqs_1 = torch.where(batch['generate_mask'],pred_seqs_1,seqs_1)
            pred_seqs_1_simplex = self.seq_to_simplex(pred_seqs_1)
            
            # TODO: should we use this? aa type is not accurate now
            # seq-angle
            torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
            pred_angles_1 = torch.where(torsion_mask.bool(),pred_angles_1,torch.zeros_like(pred_angles_1))
            
            if not sample_bb:
                pred_trans_1 = trans_1_c.detach().clone()
                pred_rotmats_1 = rotmats_1.detach().clone()
            if not sample_ang:
                pred_angles_1 = angles_1.detach().clone()
            if not sample_seq:
                pred_seqs_1 = seqs_1.detach().clone()
            clean_traj.append({'rotmats':pred_rotmats_1.cpu(),'trans':pred_trans_1.cpu(),'angles':pred_angles_1.cpu(),'seqs':pred_seqs_1.cpu(),'seqs_simplex':pred_seqs_1_simplex.cpu(),
                                    'rotmats_1':rotmats_1.cpu(),'trans_1':trans_1_c.cpu(),'angles_1':angles_1.cpu(),'seqs_1':seqs_1.cpu()})
            
            # update each modality, also only for gen mask region
            mu_trans_t_2, _ = self.BFN.continuous_var_bayesian_flow(t=t_2, sigma1=self.sigma1_trans, x=pred_trans_1)
            mu_trans_t_2_c = torch.where(batch['generate_mask'][...,None,None],mu_trans_t_2,trans_1_c) # move receptor also
            
            # update angles
            angles_t_2, log_angle_acc_t = self.circular_var_bayesian_flow(
                    x=pred_angles_1,t_index=t_index+1,beta1=self.beta1_ang,N=self.n_steps,epsilon=self.epsilon)
            angles_t_2 = torch.where(batch['generate_mask'][...,None],angles_t_2,angles_1)
            # update seqs
            theta_seqs_t_2 = self.BFN.discrete_var_bayesian_flow(
                t=t_2, beta1=self.beta1_seq,
                x=pred_seqs_1_prob, K=self.K
            )
            theta_seqs_t_2 = torch.where(batch['generate_mask'][...,None],theta_seqs_t_2, seqs_1_onehot)
            # TODO: do we need this ?
            seqs_t_2 = sample_from(theta_seqs_t_2)
            seqs_t_2 = torch.where(batch['generate_mask'],seqs_t_2,seqs_1)
            # seq-angle
            torsion_mask = angle_mask_loss[seqs_t_2.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
            angles_t_2 = torch.where(torsion_mask.bool(),angles_t_2,torch.zeros_like(angles_t_2))
            log_angle_acc_t = torch.where(torsion_mask.bool(),log_angle_acc_t,torch.ones_like(log_angle_acc_t))
            # if not sample_bb:
            #     trans_t_2_c = trans_1_c.detach().clone()
            #     rotmats_t_2 = rotmats_1.detach().clone()
            # if not sample_ang:
            #     angles_t_2 = angles_1.detach().clone()
            # if not sample_seq:
            #     seqs_t_2 = seqs_1.detach().clone()
            mu_trans_t_1_c, angles_t_1, theta_seqs_t_1 = mu_trans_t_2_c, angles_t_2, theta_seqs_t_2
            mu_trans_t_1_calpha = mu_trans_t_1_c[:,:,BBHeavyAtom.CA]
            rotmats_t_1 = construct_3d_basis(mu_trans_t_1_c[:, :, BBHeavyAtom.CA],mu_trans_t_1_c[:, :, BBHeavyAtom.C],mu_trans_t_1_c[:, :, BBHeavyAtom.N])
            
            t_1 = t_2
            
        # final step
        t_1 = ts[-1]
        t = torch.ones((num_batch, 1), device=batch['aa'].device) * t_1
        pred_rotmats_1, pred_trans_1_calpha, pred_angles_1, pred_seqs_1_prob \
            = self.ga_encoder(t, rotmats_t_1, mu_trans_t_1_calpha, angles_t_1, theta_seqs_t_1, node_embed, edge_embed, batch['generate_mask'].long(), batch['res_mask'].long(), log_angle_acc_t)
        pred_rotmats_1 = torch.where(batch['generate_mask'][...,None,None],pred_rotmats_1,rotmats_1)
        # move center
        # pred_trans_1,center = self.zero_center_part(pred_trans_1,gen_mask,res_mask)
        pred_trans_1_calpha = torch.where(batch['generate_mask'][...,None],pred_trans_1_calpha,trans_1_calpha) # move receptor also
        # angles
        pred_angles_1 = torch.where(batch['generate_mask'][...,None],pred_angles_1,angles_1)
        # seqs
        pred_seqs_1 = sample_from(pred_seqs_1_prob)
        pred_seqs_1 = torch.where(batch['generate_mask'],pred_seqs_1,seqs_1)
        # pred_seqs_1_simplex = self.seq_to_simplex(pred_seqs_1)
        # seq-angle
        torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
        pred_angles_1 = torch.where(torsion_mask.bool(),pred_angles_1,torch.zeros_like(pred_angles_1))
        
        if not sample_bb:
            pred_trans_1 = trans_1_c.detach().clone()
            # _,center = self.zero_center_part(trans_1,gen_mask,res_mask)
            pred_rotmats_1 = rotmats_1.detach().clone()
        if not sample_ang:
            pred_angles_1 = angles_1.detach().clone()
        if not sample_seq:
            raise NotImplementedError
            
        pred_trans_1_calpha = pred_trans_1_calpha * self.cfg.norm.std # de norm
            
        clean_traj.append({'rotmats':pred_rotmats_1.cpu(),
                           'trans':pred_trans_1_calpha.cpu(),
                           'angles':pred_angles_1.cpu(),
                           'seqs':pred_seqs_1.cpu(),
                           'seqs_simplex':pred_seqs_1.cpu(),
                           'rotmats_1':rotmats_1.cpu(),
                           'trans_1':trans_1_calpha.cpu(),
                           'angles_1':angles_1.cpu(),
                           'seqs_1':seqs_1.cpu()})
        
        return clean_traj

