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
from probayes.modules.protein.constants import AA, BBHeavyAtom, max_num_heavyatoms
from probayes.modules.common.geometry import construct_3d_basis
from probayes.utils.data import PaddingCollate

from probayes.modules.so3.dist import centered_gaussian,uniform_so3
from probayes.modules.common.geometry import batch_align, align

from tqdm import tqdm
from typing import Callable
import wandb

from probayes.data import so3_utils
from probayes.data import all_atom

from probayes.utils import rotation_conversions as rc
from probayes.data.so3_utils import geodesic_dist
from probayes.utils.misc import load_config
from probayes.utils.train import recursive_to
from easydict import EasyDict

from probayes.core.utils import process_dic
from probayes.core.torsion import get_torsion_angle, torsions_mask
import probayes.core.torus as torus
from probayes.modules.net.ga_bfn_axis_angle_new import GAEncoder_BFN
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

class BFNModel_AxisAngle_new(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self._model_cfg = cfg.encoder
        self._interpolant_cfg = cfg.interpolant

        self.node_embedder = NodeEmbedder(cfg.encoder.node_embed_size,max_num_heavyatoms)
        self.edge_embedder = EdgeEmbedder(cfg.encoder.edge_embed_size,max_num_heavyatoms)
        self.ga_encoder:GAEncoder_BFN = GAEncoder_BFN(cfg.encoder.ipa, cfg.bfn)

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
        self.beta1_rot_ang = torch.tensor(self.cfg.bfn.beta1_rot_ang)
        self.beta1_rot_axis = torch.tensor(self.cfg.bfn.beta1_rot_axis)
        self.n_steps = self.cfg.bfn.n_steps
        self.use_dtime_loss = self.cfg.bfn.dtime_loss
        self.epsilon = torch.tensor(1e-7)
        
        self.pred_rotmat = self.cfg.bfn.pred_rotmat
    
    def encode(self, batch):
        rotmats_1 =  construct_3d_basis(batch['pos_heavyatom'][:, :, BBHeavyAtom.CA],batch['pos_heavyatom'][:, :, BBHeavyAtom.C],batch['pos_heavyatom'][:, :, BBHeavyAtom.N] )
        trans_1 = batch['pos_heavyatom'][:, :, BBHeavyAtom.CA]
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

    def zero_center_part_fix_protein(self,pos,gen_mask,res_mask,mask_heavyatoms):
        """
        move pos by center of gen_mask
        pos: (B,N,3)
        gen_mask, res_mask: (B,N)
        """
        
        protein_mask = (~gen_mask.bool()) * res_mask
        center = torch.sum(pos[:, :, BBHeavyAtom.CA] * protein_mask[...,None], dim=1) / (torch.sum(protein_mask,dim=-1,keepdim=True) + 1e-8) 
        # center = (pos * mask_heavyatoms[...,None]).sum((1,2)) / (mask_heavyatoms.sum((1,2))[...,None]+1e-8)
        center = center.unsqueeze(1).unsqueeze(1) # (B,1,1,3)
        # center = 0. it seems not center didnt influence the result, but its good for training stabilty
        pos = pos - center
        pos = pos * res_mask[...,None,None]
        return pos
    
    def seq_to_simplex(self,seqs):
        return clampped_one_hot(seqs, self.K).float() * self.k * 2 - self.k # (B,L,K)
    
    def rotmat_to_axis_angle(self, rotmats):
        rotvecs_1 = rc.matrix_to_axis_angle(rotmats, fast=True)
        rot_angle_1 = torch.norm(rotvecs_1, dim=-1, keepdim=True)
        rot_axis_1 = rotvecs_1 / rot_angle_1
        return rot_axis_1, rot_angle_1 % (2 * torch.pi)
    
    def forward(self, batch):
        num_batch, num_res = batch['aa'].shape
        DEVICE = batch['aa'].device
        gen_mask,res_mask,angle_mask = batch['generate_mask'].long(),batch['res_mask'].long(),batch['torsion_angle_mask'].long()
        #encode
        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = self.encode(batch) # no generate mask
        trans_1 = trans_1 / self.cfg.norm.std # norm
        
        # prepare for denoise
        seqs_1_simplex = self.seq_to_simplex(seqs_1)
        seqs_1_prob = F.softmax(seqs_1_simplex,dim=-1)
        seqs_1_onehot = clampped_one_hot(seqs_1, self.K).float()

        # rotvecs_1 = rc.matrix_to_axis_angle(rotmats_1, fast=True)
        # rot_angle_1 = torch.norm(rotvecs_1, dim=-1, keepdim=True)
        # rot_axis_1 = rotvecs_1 / rot_angle_1
        rot_axis_1, rot_angle_1 = self.rotmat_to_axis_angle(rotmats_1)
        rot_axis_1 = self.recover_gt(res_mask, rot_axis_1, torch.zeros_like(rot_axis_1))
        rot_angle_1 = self.recover_gt(res_mask, rot_angle_1, torch.zeros_like(rot_angle_1))
        
        rot_loss, rotang_loss, rotaxis_loss = torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)
        
        with torch.no_grad():
            t = torch.randint(0, self.n_steps,size=(num_batch,1), 
                            device=batch['aa'].device)/self.n_steps 
            t_index = t * self.n_steps + 1
            
            if self.sample_structure:
                # corrupt trans
                mu_trans_t, _ = self.BFN.continuous_var_bayesian_flow(
                    t=t[...,None], sigma1=self.sigma1_trans, x=trans_1
                )
                mu_trans_t = self.recover_gt(gen_mask, mu_trans_t, trans_1)
                # mu_trans_t = torch.where(batch['generate_mask'][...,None],mu_trans_t,trans_1)
                
                # corrupt rotmats
                # rotmats_0 = uniform_so3(num_batch,num_res,device=batch['aa'].device)
                # rotmats_t = so3_utils.geodesic_t(t[..., None], rotmats_1, rotmats_0)
                # rotmats_t = torch.where(batch['generate_mask'][...,None,None],rotmats_t,rotmats_1)
                # rotmats_t = torch.where(batch['generate_mask'][...,None,None],rotmats_t,rotmats_1)
                
                m_rotaxis_t, acc_rotaxis_t = self.BFN.sphere_var_bayesian_flow_sim(
                    x=rot_axis_1, t_index=t_index, beta1 = self.beta1_rot_axis, N=self.n_steps, epsilon=self.epsilon
                )
                m_rotaxis_t = self.recover_gt(gen_mask, m_rotaxis_t, rot_axis_1)
                acc_rotaxis_t = self.recover_acc(gen_mask, acc_rotaxis_t)
                m_rotang_t, acc_rotang_t = self.circular_var_bayesian_flow(
                    x=rot_angle_1, t_index=t_index,  N=self.n_steps, beta1 = self.beta1_rot_ang,epsilon=self.epsilon,
                )
                m_rotang_t = self.recover_gt(gen_mask, m_rotang_t, rot_angle_1)
                acc_rotang_t = self.recover_acc(gen_mask, acc_rotang_t)
                
                # corrup angles
                m_angles_t, log_acc_t = self.circular_var_bayesian_flow(
                    x=angles_1,t_index=t_index, N=self.n_steps, beta1=self.beta1_ang,epsilon=self.epsilon)
                m_angles_t = self.recover_gt(gen_mask, m_angles_t, angles_1)
                log_acc_t = self.recover_acc(gen_mask, log_acc_t)
                # m_angles_t = torch.where(batch['generate_mask'][...,None],m_angles_t,angles_1)
                # log_acc_t = torch.where(batch['generate_mask'][...,None],log_acc_t,torch.ones_like(log_acc_t))
            else:
                trans_t_c = trans_1.detach().clone()
                rotmats_t = rotmats_1.detach().clone()
                angles_t = angles_1.detach().clone()
            
            if self.sample_sequence:
                # corrupt seqs
                theta_seqs_t = self.BFN.discrete_var_bayesian_flow(t=t[...,None], beta1=self.beta1_seq, x=seqs_1_onehot, K=self.K)
                theta_seqs_t = self.recover_gt(gen_mask, theta_seqs_t, seqs_1_onehot)
                # theta_seqs_t = torch.where(batch['generate_mask'][..., None],theta_seqs_t,seqs_1_onehot)
            else:
                seqs_t = seqs_1.detach().clone()
                seqs_t_simplex = seqs_1_simplex.detach().clone()
                seqs_t_prob = seqs_1_prob.detach().clone()

        # denoise
        if self.pred_rotmat:
            pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1_prob = self.ga_encoder(
            t, m_rotaxis_t, acc_rotaxis_t, m_rotang_t, acc_rotang_t, mu_trans_t, m_angles_t, theta_seqs_t, node_embed, edge_embed, gen_mask, res_mask, log_acc_t)       
        else:
            pred_rotaxis_1, pred_rotang_1, pred_trans_1, pred_angles_1, pred_seqs_1_prob = self.ga_encoder(
            t, m_rotaxis_t, acc_rotaxis_t, m_rotang_t, acc_rotang_t, mu_trans_t, m_angles_t, theta_seqs_t, node_embed, edge_embed, gen_mask, res_mask, log_acc_t)
        pred_seqs_1 = sample_from(pred_seqs_1_prob)
        pred_seqs_1 = torch.where(batch['generate_mask'],pred_seqs_1,torch.clamp(seqs_1,0,19))
        # TODO: do we need this?
        pred_trans_1_c = pred_trans_1 # implicitly enforce zero center in gen_mask, in this way, we dont need to move receptor when sampling

        mask = torch.gt(gen_mask,0)
        selected_i = t_index.unsqueeze(-1).repeat(1,num_res,1)[mask,:]
        
        norm_scale = 1 / (1 - torch.min(t[...,None], torch.tensor(self._interpolant_cfg.t_normalization_clip))) # yim etal.trick, 1/1-t
        # rots vf loss
        # gt_rot_vf = so3_utils.calc_rot_vf(rotmats_t, rotmats_1)
        # pred_rot_vf = so3_utils.calc_rot_vf(rotmats_t, pred_rotmats_1)
        # rot_loss = torch.sum(((gt_rot_vf - pred_rot_vf) * norm_scale)**2*gen_mask[...,None],dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        # rot_loss = torch.mean(rot_loss)
        if self.pred_rotmat:
            pred_rotmats_1 = self.recover_gt(res_mask, pred_rotmats_1, rotmats_1)
            # pred_rotaxis_1, pred_rotang_1 = self.rotmat_to_axis_angle(pred_rotmats_1)
            # rot loss
            rot_loss = torch.mean(geodesic_dist(pred_rotmats_1[mask,:], rotmats_1[mask,:]))
        else:
            pred_rotaxis_1 = self.recover_gt(res_mask, pred_rotaxis_1, rot_axis_1)
            pred_rotang_1 = self.recover_gt(res_mask, pred_rotang_1, rot_angle_1)
            pred_rotmats_1 = rc.axis_angle_to_matrix(pred_rotaxis_1*pred_rotang_1, fast=True)
            
            # pred_rotvec_1 = (pred_rotaxis_1 * pred_rotang_1)[mask,:]
            # gt_rotvec_1 = rc.matrix_to_axis_angle(rotmats_1, fast=True)[mask,:]
            # killing form so3 inner product
            # rot_loss = 2 * torch.pi - (pred_rotvec_1 * gt_rotvec_1).mean()
            # rot loss
            rotang_loss = self.BFN.dtime4circular_loss(
                i = selected_i, N = self.n_steps, x_pred=pred_rotang_1[mask,:], x=rot_angle_1[mask,:],
                alpha_i = self.BFN.alpha_wrt_index(selected_i,N=self.n_steps,beta1=self.beta1_rot_ang),
            )
            rotaxis_loss = self.BFN.dtime4sphere_loss(
                x=pred_rotaxis_1[mask,:], x_pred=rot_axis_1[mask,:],
                alpha=self.BFN.sphere_alpha_wrt_index(selected_i,N=self.n_steps,beta1=self.beta1_rot_axis, p=3),
                p=3
            )

        # bb aux loss
        gt_bb_atoms = all_atom.to_atom37(trans_1, rotmats_1)[:, :, :3] 
        pred_bb_atoms = all_atom.to_atom37(pred_trans_1_c, pred_rotmats_1)[:, :, :3]

        bb_atom_loss = torch.sum(
            (gt_bb_atoms - pred_bb_atoms) ** 2 * gen_mask[..., None, None],
            dim=(-1, -2, -3)
        ) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        bb_atom_loss = torch.mean(bb_atom_loss)

        # seqs vf loss
        seqs_loss = self.BFN.dtime4discrete_loss_prob(
            i=selected_i,
            N=self.n_steps,
            beta1=self.beta1_seq,
            one_hot_x=seqs_1_onehot[mask,:],
            p_0=pred_seqs_1_prob[mask,:],
            K=self.K
        )
        trans_loss = self.BFN.dtime4continuous_loss(
            i=selected_i, N=self.n_steps,
            sigma1=self.sigma1_trans,
            x_pred=pred_trans_1_c[mask,:],
            x=trans_1[mask,:]
        )
        # we should not use angle mask, as you dont know aa type when generating
        # angle vf loss
        angle_mask_loss = torsions_mask.to(batch['aa'].device)
        angle_mask_loss = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
        # angle_mask_loss = torch.cat([angle_mask_loss,angle_mask_loss],dim=-1) # (B,L,10)
        angle_mask_loss = torch.logical_and(batch['generate_mask'][...,None].bool(),angle_mask_loss)
        angle_mask = torch.gt(angle_mask_loss,0)
        angle_selected_i = t_index.unsqueeze(-1).repeat(1,num_res,5)[angle_mask]
        angle_loss = self.BFN.dtime4circular_loss(
            x=angles_1[angle_mask],x_pred=pred_angles_1[angle_mask], i=angle_selected_i,N=self.n_steps,
            alpha_i=self.BFN.alpha_wrt_index(angle_selected_i,N=self.n_steps,beta1=self.beta1_ang),
            mse_loss=False,
            device=DEVICE
        )
        # angle_loss = torch.mean(angle_loss)
        torsion_loss = torch.tensor(0.)
        # angle aux loss
        # angles_1_vec = torch.cat([torch.sin(angles_1),torch.cos(angles_1)],dim=-1)
        # pred_angles_1_vec = torch.cat([torch.sin(pred_angles_1),torch.cos(pred_angles_1)],dim=-1)
        # # torsion_loss = torch.sum((pred_angles_1_vec - angles_1_vec)**2*gen_mask[...,None],dim=(-1,-2)) / (torch.sum(gen_mask,dim=-1) + 1e-8) # (B,)
        # aux_angle_mask_loss = torch.cat([angle_mask_loss,angle_mask_loss],dim=-1) # (B,L,10)
        # torsion_loss = torch.sum((pred_angles_1_vec - angles_1_vec)**2*aux_angle_mask_loss,dim=(-1,-2)) / (torch.sum(aux_angle_mask_loss,dim=(-1,-2)) + 1e-8) # (B,)
        # torsion_loss = torch.mean(torsion_loss)
        return {
            "trans_loss": trans_loss,
            'rot_loss': rot_loss,
            'rot_axis_loss': rotaxis_loss,
            'rot_angle_loss': rotang_loss,
            'bb_atom_loss': bb_atom_loss,
            'seqs_loss': seqs_loss,
            'angle_loss': angle_loss,
            'torsion_loss': torsion_loss,
        }
    
    
    def recover_gt(self, gen_mask, gen_state, gt_state):
        '''
        gen_mask: (num_batch, num_res)
        '''
        assert gen_state.shape == gt_state.shape and gen_mask.shape[:2] == gen_state.shape[:2]
        mask = gen_mask[[...]+[None]*(gen_state.dim()-gen_mask.dim())]
        mask = torch.gt(mask,0)
        return torch.where(mask, gen_state, gt_state)
    
    def recover_acc(self, gen_mask, acc):
        assert gen_mask.shape[:2] == acc.shape[:2]
        mask = gen_mask[[...]+[None]*(acc.dim()-gen_mask.dim())]
        mask = torch.gt(mask,0)
        return torch.where(mask, acc, torch.ones_like(acc))
    
    @torch.no_grad()
    def sample(self, batch, num_steps = 100, sample_bb=True, sample_ang=True, sample_seq=True):
        DEVICE = batch['aa'].device
        num_batch, num_res = batch['aa'].shape
        gen_mask,res_mask = batch['generate_mask'],batch['res_mask']
        angle_mask_loss = torsions_mask.to(batch['aa'].device)

        #encode
        rotmats_1, trans_1, angles_1, seqs_1, node_embed, edge_embed = self.encode(batch) 
        rotaxis_1, rotang_1 = self.rotmat_to_axis_angle(rotmats_1)
        
        trans_1 = trans_1 / self.cfg.norm.std # norm
        
        seqs_1_onehot = clampped_one_hot(seqs_1, self.K).float()

        #initial noise
        if sample_bb:
            rotmats_t = uniform_so3(num_batch,num_res,device=batch['aa'].device)
            rotaxis_t, rotang_t = self.rotmat_to_axis_angle(rotmats_t)
            rotaxis_t = self.recover_gt(res_mask, rotaxis_t, rotaxis_1)
            rotang_t = self.recover_gt(res_mask, rotang_t, rotang_1)
            acc_rot_axis = self.recover_acc(gen_mask, torch.zeros_like(rotaxis_t[...,:1]))
            acc_rot_ang = self.recover_acc(gen_mask, torch.zeros_like(rotang_t[...,:1]))
            # rotmats_t = torch.where(batch['generate_mask'][...,None,None],rotmats_t,rotmats_1)
            # bfn translation prior
            mu_trans_t = torch.zeros((num_batch,num_res,3), device=batch['aa'].device)
            mu_trans_t = self.recover_gt(gen_mask, mu_trans_t, trans_1)            
        else:
            rotmats_t = rotmats_1.detach().clone()
            trans_0_c = trans_1.detach().clone()
            
        if sample_ang:
            # angle noise
            m_angles_t = torus.tor_random_uniform(angles_1.shape, device=batch['aa'].device, dtype=angles_1.dtype) # (B,L,5)
            # m_angles_t = torch.where(batch['generate_mask'][...,None],m_angles_t,angles_1)
            m_angles_t = self.recover_gt(gen_mask, m_angles_t, angles_1)
            log_c_t = self.BFN.norm_logbeta(
                            torch.log(torch.tensor((self.epsilon))) * torch.ones_like(m_angles_t),
                            beta1=self.beta1_ang,epsilon=self.epsilon)
            log_c_t = self.recover_acc(gen_mask, log_c_t)
        else:
            angles_0 = angles_1.detach().clone()
        # seq
        if sample_seq:
            # sequence parameter init
            theta_seqs_t = torch.ones((num_batch, num_res, self.K)).to(DEVICE) / self.K  # [N, N_AA, K] discrete prior
            theta_seqs_1 = clampped_one_hot(seqs_1, self.K).float()
            # theta_seqs_t = torch.where(batch['generate_mask'][..., None], theta_seqs_t, theta_seqs_1)
            theta_seqs_t = self.recover_gt(gen_mask, theta_seqs_t, theta_seqs_1)
        else:
            raise NotImplementedError

        # Set-up time
        ts = torch.range(1, num_steps-1) / num_steps
        ts = [0] + ts.tolist()
        
        t_1 = ts[0]
        clean_traj = []

        # denoise loop
        for idx in torch.range(1, num_steps+1):
            t_1 = idx / num_steps
            t_2 = (idx + 1) / num_steps
            t = torch.ones((num_batch, 1), device=batch['aa'].device) * t_1
            t_index = t * self.n_steps + 1
            # rots
            if self.pred_rotmat:
                pred_rotmats_1, pred_trans_1, pred_angles_1, pred_seqs_1_prob = self.ga_encoder(
                t, rotaxis_t, acc_rot_axis, rotang_t, acc_rot_ang, mu_trans_t, m_angles_t, 
                theta_seqs_t, node_embed, edge_embed, 
                batch['generate_mask'].long(), batch['res_mask'].long(),log_c_t)
                pred_rotaxis_1, pred_rotang_1 = self.rotmat_to_axis_angle(pred_rotmats_1)
            else:
                pred_rotaxis_1, pred_rotang_1, pred_trans_1, pred_angles_1, pred_seqs_1_prob = self.ga_encoder(
                t, rotaxis_t, acc_rot_axis, rotang_t, acc_rot_ang, mu_trans_t, m_angles_t, 
                theta_seqs_t, node_embed, edge_embed, 
                batch['generate_mask'].long(), batch['res_mask'].long(),log_c_t)
            
            pred_rotaxis_1 = self.recover_gt(res_mask, pred_rotaxis_1, rotaxis_1)
            pred_rotang_1 = self.recover_gt(res_mask, pred_rotang_1, rotang_1)
            if not self.pred_rotmat:
                pred_rotmats_1 = rc.axis_angle_to_matrix(pred_rotaxis_1*pred_rotang_1)
            # pred_rotaxis_1, pred_rotang_1 = self.rotmat_to_axis_angle(pred_rotmats_1)

            # pred_rotmats_1 = torch.where(batch['generate_mask'][...,None,None],pred_rotmats_1,rotmats_1)
            # trans, move center
            # pred_trans_1 = torch.where(batch['generate_mask'][...,None],pred_trans_1,trans_1) # move receptor also
            pred_trans_1 = self.recover_gt(gen_mask, pred_trans_1, trans_1)
            # angles
            # pred_angles_1 = torch.where(batch['generate_mask'][...,None],pred_angles_1,angles_1)
            pred_angles_1 = self.recover_gt(gen_mask, pred_angles_1, angles_1)
            # seqs
            pred_seqs_1 = sample_from(pred_seqs_1_prob)
            # pred_seqs_1 = torch.where(batch['generate_mask'],pred_seqs_1,seqs_1)
            pred_seqs_1 = self.recover_gt(gen_mask, pred_seqs_1, seqs_1)
            # seq-angle
            torsion_mask = angle_mask_loss[pred_seqs_1.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
            pred_angles_1 = torch.where(torsion_mask.bool(),pred_angles_1,torch.zeros_like(pred_angles_1))
            
            if not sample_bb or not sample_ang or not sample_seq:
                raise NotImplementedError
            
            clean_traj.append({'rotmats':pred_rotmats_1.cpu(),
                               'trans':(pred_trans_1* self.cfg.norm.std).cpu(),
                               'angles':pred_angles_1.cpu(),
                               'seqs':pred_seqs_1.cpu(),
                                'rotmats_1':rotmats_1.cpu(),
                                'trans_1':trans_1.cpu(),
                                'angles_1':angles_1.cpu(),
                                'seqs_1':seqs_1.cpu()})
            
            if idx == num_steps:
                return clean_traj
            
            # reverse step, also only for gen mask region
            d_t = (t_2-t_1) * torch.ones((num_batch, 1), device=batch['aa'].device)
            # Euler step
            mu_trans_t, _ = self.BFN.continuous_var_bayesian_flow(t=t_2, sigma1=self.sigma1_trans, x=pred_trans_1)
            mu_trans_t = self.recover_gt(gen_mask, mu_trans_t, trans_1)
            # mu_trans_t = torch.where(batch['generate_mask'][...,None],mu_trans_t,trans_1) # move receptor also

            rotaxis_t, acc_rot_axis = self.BFN.sphere_var_bayesian_flow_sim(
                x=pred_rotaxis_1, t_index=t_index+1, beta1=self.beta1_rot_axis, N=self.n_steps
            )
            rotaxis_t = self.recover_gt(gen_mask, rotaxis_t, rotaxis_1)
            acc_rot_axis = self.recover_acc(gen_mask, acc_rot_axis)
            rotang_t, acc_rot_ang = self.circular_var_bayesian_flow(
                x=pred_rotang_1, t_index=t_index+1, beta1=self.beta1_rot_ang, N=self.n_steps, epsilon=self.epsilon
            )
            rotang_t = self.recover_gt(gen_mask, rotang_t, rotang_1)
            acc_rot_ang = self.recover_acc(gen_mask, acc_rot_ang)
            # rotmats_t = so3_utils.geodesic_t(d_t[...,None] * 10, pred_rotmats_1, rotmats_t)
            # rotmats_t = self.recover_gt(gen_mask, rotmats_t, rotmats_1)
            
            # angles
            m_angles_t, log_c_t = self.circular_var_bayesian_flow(
                    x=pred_angles_1,t_index=t_index+1,beta1=self.beta1_ang,N=self.n_steps,epsilon=self.epsilon)
            m_angles_t = self.recover_gt(gen_mask, m_angles_t, angles_1)
            # seqs
            theta_seqs_t = self.BFN.discrete_var_bayesian_flow(
                t=t_2, beta1=self.beta1_seq,
                x=pred_seqs_1_prob, K=self.K
            )
            # theta_seqs_t = torch.where(batch['generate_mask'][...,None],theta_seqs_t, seqs_1_onehot)
            theta_seqs_t = self.recover_gt(gen_mask, theta_seqs_t, seqs_1_onehot)
            
            # TODO: do we need this ?
            seqs_t_2 = sample_from(theta_seqs_t)
            seqs_t_2 = self.recover_gt(gen_mask, seqs_t_2, seqs_1)
            
            # seq-angle
            torsion_mask = angle_mask_loss[seqs_t_2.reshape(-1)].reshape(num_batch,num_res,-1) # (B,L,5)
            m_angles_t = torch.where(torsion_mask.bool(),m_angles_t,torch.zeros_like(m_angles_t))
            log_c_t = torch.where(torsion_mask.bool(),log_c_t,torch.ones_like(log_c_t))
            
            if not sample_bb or not sample_ang or not sample_seq:
                raise NotImplementedError

        return clean_traj


