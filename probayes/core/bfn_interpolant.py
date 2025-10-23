import torch
import copy
import math
import functools as fn
import torch.nn.functional as F
import probayes.utils.rotation_conversions as rc
from collections import defaultdict
from multiflow.data import so3_utils, all_atom
from multiflow.data import utils as du
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from torch import autograd
from torch.distributions.categorical import Categorical
from torch.distributions.binomial import Binomial
from probayes.modules.bfn.bfn_base import bfnBase
from tqdm import trange

def _centered_gaussian(num_batch, num_res, device):
    noise = torch.randn(num_batch, num_res, 3, device=device)
    return noise - torch.mean(noise, dim=-2, keepdims=True)


def _uniform_so3(num_batch, num_res, device):
    return torch.tensor(
        Rotation.random(num_batch*num_res).as_matrix(),
        device=device,
        dtype=torch.float32,
    ).reshape(num_batch, num_res, 3, 3)


def _trans_diffuse_mask(trans_t, trans_1, diffuse_mask):
    return trans_t * diffuse_mask[..., None] + trans_1 * (1 - diffuse_mask[..., None])


def _recover_acc( gen_mask, acc):
    assert gen_mask.shape[:2] == acc.shape[:2]
    mask = gen_mask[[...]+[None]*(acc.dim()-gen_mask.dim())]
    return torch.where(mask.bool(), acc, torch.ones_like(acc))

def _recover_gt(gen_mask, gen_state, gt_state):
    assert gen_state.shape == gt_state.shape and gen_mask.shape[:2] == gen_state.shape[:2]
    mask = gen_mask[[...]+[None]*(gen_state.dim()-gen_mask.dim())]
    return torch.where(mask.bool(), gen_state, gt_state)




class BFN_Interpolant:

    def __init__(self, cfg):
        self._cfg = cfg
        self._rots_cfg = cfg.rots
        self._trans_cfg = cfg.trans
        self._aatypes_cfg = cfg.aatypes
        self._sample_cfg = cfg.sampling
        self.BFN = bfnBase()
        self.n_steps = cfg.n_steps
        self.num_tokens = 21 if self._aatypes_cfg.interpolant_type == "masking" else 20

    def set_device(self, device):
        self._device = device

    def sample_t(self, num_batch):
        t_index = torch.randint(1, self.n_steps+1,size=(num_batch,), device=self._device)
        t = (t_index - 1) / self.n_steps
        return t

    def _corrupt_trans(self, trans_1, t, res_mask, diffuse_mask):
        trans_t, gamma_t = self.BFN.continuous_var_bayesian_flow(t[..., None], sigma1=self._trans_cfg.sigma1, x=trans_1)
        # recover not generated positions as ground truth
        trans_t = _trans_diffuse_mask(trans_t, trans_1, diffuse_mask)
        return trans_t * res_mask[..., None]
    
    def _augment_quat(self, quat):
        rand_v = (torch.rand_like(quat[...,:1]) > 0.5).double()
        return quat * ((-1) ** rand_v)

    def _corrupt_quat(self, quat_1, t, res_mask, diffuse_mask):
        beta1 = torch.tensor(self._rots_cfg.beta1)
        t_index = (t * self.n_steps).long() + 1
        # quat_1 = self._augment_quat(quat_1)
        m_quat_t, acc_quat_t = self.BFN.sphere_var_bayesian_flow_sim(
            x=quat_1, t_index=t_index, beta1=beta1, N=self.n_steps, cache_sampling=True
        )       
        acc_quat_t = _recover_acc(diffuse_mask, acc_quat_t)
        m_quat_t = _recover_gt(gen_mask=diffuse_mask, gen_state=m_quat_t, gt_state=quat_1)
        return m_quat_t, acc_quat_t

    def _corrupt_aatypes(self, aatypes_1, t, res_mask, diffuse_mask):
        num_batch, num_res = res_mask.shape
        assert aatypes_1.shape == (num_batch, num_res)
        assert t.shape == (num_batch, 1)
        assert res_mask.shape == (num_batch, num_res)
        assert diffuse_mask.shape == (num_batch, num_res)
        aatypes_1_onehot = F.one_hot(aatypes_1, num_classes=self.num_tokens)
        beta1 = torch.tensor(self._aatypes_cfg.beta1)
        theta_seqs_t = self.BFN.discrete_var_bayesian_flow(t=t[...,None], beta1=beta1, x=aatypes_1_onehot, K=self.num_tokens)
        theta_seqs_t = _recover_gt(diffuse_mask, theta_seqs_t, aatypes_1_onehot)        
        
        return theta_seqs_t

    def corrupt_batch(self, batch):
        noisy_batch = copy.deepcopy(batch)

        # [B, N, 3]
        trans_1 = batch['trans_1']  # Angstrom

        # [B, N, 3, 3]
        rotmats_1 = batch['rotmats_1']

        # [B, N]
        aatypes_1 = batch['aatypes_1']

        # [B, N]
        res_mask = batch['res_mask']
        diffuse_mask = batch['diffuse_mask']
        num_batch, num_res = diffuse_mask.shape

        # [B, 1]
        if self._cfg.codesign_separate_t:
            u = torch.rand((num_batch,), device=self._device)
            forward_fold_mask = (u < self._cfg.codesign_forward_fold_prop).float()
            inverse_fold_mask = (u < self._cfg.codesign_inverse_fold_prop + self._cfg.codesign_forward_fold_prop).float() * \
                (u >= self._cfg.codesign_forward_fold_prop).float()

            normal_structure_t = self.sample_t(num_batch)
            # inverse_fold_structure_t = torch.ones((num_batch,), device=self._device)
            inverse_fold_structure_t = torch.ones((num_batch,), device=self._device) * ((self.n_steps-1) / self.n_steps)
            normal_cat_t = self.sample_t(num_batch)
            forward_fold_cat_t = torch.ones((num_batch,), device=self._device)

            # If we are forward folding, then cat_t should be 1
            # If we are inverse folding or codesign then cat_t should be uniform
            cat_t = forward_fold_mask * forward_fold_cat_t + (1 - forward_fold_mask) * normal_cat_t

            # If we are inverse folding, then structure_t should be 1
            # If we are forward folding or codesign then structure_t should be uniform
            structure_t = inverse_fold_mask * inverse_fold_structure_t + (1 - inverse_fold_mask) * normal_structure_t

            so3_t = structure_t[:, None]
            r3_t = structure_t[:, None]
            cat_t = cat_t[:, None]
        else:
            t = self.sample_t(num_batch)[:, None]
            so3_t = t
            r3_t = t
            cat_t = t
        noisy_batch['so3_t'] = so3_t
        noisy_batch['r3_t'] = r3_t
        noisy_batch['cat_t'] = cat_t
    
        # Apply corruptions
        # normalize trans_1
        trans_1 = (trans_1 - self._trans_cfg.norm_mean) / self._trans_cfg.norm_std
        noisy_batch['trans_1'] = trans_1
        # corrupt trans
        if self._trans_cfg.corrupt:
            trans_t = self._corrupt_trans(
                trans_1, r3_t, res_mask, diffuse_mask)
        else:
            trans_t = trans_1
        if torch.any(torch.isnan(trans_t)):
            raise ValueError('NaN in trans_t during corruption')
        noisy_batch['trans_t'] = trans_t

        # corrupt rotmats as quaternion
        quats_1 = rc.matrix_to_quaternion(rotmats_1)
        quats_1 = self._augment_quat(quats_1).float()
        if self._rots_cfg.corrupt:
            quats_t, acc_quats_t = self._corrupt_quat(quats_1, so3_t, res_mask, diffuse_mask)
        else:
            quats_t = quats_1
        if torch.any(torch.isnan(quats_t)):
            raise ValueError('NaN in quats_t during corruption')
        noisy_batch['quats_t'] = quats_t
        noisy_batch['acc_quats_t'] = acc_quats_t
        noisy_batch['quats_1'] = quats_1
        # corrupt aatypes
        if self._aatypes_cfg.corrupt:
            aatypes_t = self._corrupt_aatypes(aatypes_1, cat_t, res_mask, diffuse_mask)
        else:
            aatypes_t = aatypes_1
        noisy_batch['aatypes_t'] = aatypes_t
        noisy_batch['trans_sc'] = torch.zeros_like(trans_1)
        noisy_batch['aatypes_sc'] = torch.zeros_like(
            aatypes_1)[..., None].repeat(1, 1, self.num_tokens)
        return noisy_batch

    def _trans_update_step(self, curr_t, next_t, trans_1, trans_t, acc_trans_t):
        next_mu_trans_t, _ = self.BFN.continuous_var_bayesian_flow(
                        t=next_t, sigma1=self._trans_cfg.sigma1, x=trans_1)
        next_acc_trans_t = self._trans_cfg.sigma1 ** (-2 * next_t) - 1
        return next_mu_trans_t, next_acc_trans_t

    def _rots_update_step(self, curr_t_index, next_t_index, quat_1, quat_t, acc_quat_t):
        next_quat_t, next_acc_quat_t = self.BFN.sphere_var_bayesian_flow_sim(
            x=quat_1, t_index=next_t_index, beta1=self._rots_cfg.beta1, N=self.n_steps, cache_sampling=True
        )        
        return next_quat_t, next_acc_quat_t

    def _aatypes_update_step(self, curr_t_index, next_t_index, pred_aatypes_prob, aatypes_t):
        batch_size, num_res, S = pred_aatypes_prob.shape
        assert aatypes_t.shape[:2] == (batch_size, num_res)
        next_theta_seqs_t = self.BFN.discrete_var_bayesian_flow(
            t=next_t_index, beta1=self._aatypes_cfg.beta1,
            x=pred_aatypes_prob, K=self.num_tokens
        )        
        return next_theta_seqs_t
        
    def sample(
            self,
            num_batch,
            num_res,
            model,
            num_timesteps=None,
            trans_0=None,
            rotmats_0=None,
            aatypes_0=None,
            trans_1=None,
            rotmats_1=None,
            aatypes_1=None,
            diffuse_mask=None,
            chain_idx=None,
            res_idx=None,
            t_nn=None,
            forward_folding=False,
            inverse_folding=False,
            separate_t=False,
        ):

        res_mask = torch.ones(num_batch, num_res, device=self._device)
        # Set-up initial prior samples
        if trans_0 is None:
            # trans_0 = _centered_gaussian(
            #     num_batch, num_res, self._device) * du.NM_TO_ANG_SCALE
            trans_t = torch.zeros(num_batch, num_res, 3, device=self._device)
        
        if rotmats_0 is None:
            rotmats_t = _uniform_so3(num_batch, num_res, self._device)
        quats_t = rc.matrix_to_quaternion(rotmats_t)
        acc_quats_t = torch.zeros_like(quats_t[...,:1])
        acc_trans_t = torch.zeros_like(trans_t[...,:1])
        
        # for co-design setting
        if trans_1 is None:
            trans_1 = torch.zeros(num_batch, num_res, 3, device=self._device)
        if rotmats_1 is None:
            rotmats_1 = torch.eye(3, device=self._device)[None, None].repeat(num_batch, num_res, 1, 1)
        if aatypes_1 is None:
            aatypes_1 = torch.zeros((num_batch, num_res), device=self._device).long()
        
        # init aatypes
        if aatypes_0 is None:
            aatypes_t = torch.ones((num_batch, num_res, self.num_tokens), device=self._device).long()
            aatypes_t = aatypes_t / self.num_tokens
        
        if res_idx is None:
            res_idx = torch.arange(num_res,device=self._device,
                dtype=torch.float32)[None].repeat(num_batch, 1)

        if chain_idx is None:
            chain_idx = res_mask

        if diffuse_mask is None:
            diffuse_mask = res_mask

        trans_sc = torch.zeros(num_batch, num_res, 3, device=self._device)
        aatypes_sc = torch.zeros(
            num_batch, num_res, self.num_tokens, device=self._device)
        batch = {
            'res_mask': res_mask,
            'diffuse_mask': diffuse_mask,
            'chain_idx': chain_idx,
            'res_idx': res_idx,
            'trans_sc': trans_sc,
            'aatypes_sc': aatypes_sc,
        }

        if trans_1 is None:
            trans_t = torch.zeros(num_batch, num_res, 3, device=self._device)
        if rotmats_1 is None:
            rotmats_1 = torch.eye(3, device=self._device)[None, None].repeat(num_batch, num_res, 1, 1)
        
        quat_1 = self._augment_quat(rc.matrix_to_quaternion(rotmats_1))
        
        if aatypes_1 is None:
            aatypes_1 = torch.zeros((num_batch, num_res, self.num_tokens), device=self._device).long()
        else:
            aatypes_1 = torch.nn.functional.one_hot(aatypes_1,num_classes=self.num_tokens).float()
        assert aatypes_1.shape == (num_batch, num_res, self.num_tokens)

        if forward_folding:
            assert aatypes_1 is not None
            assert self._aatypes_cfg.noise == 0
        if forward_folding and separate_t:
            aatypes_0 = aatypes_1
        
        if inverse_folding:
            assert trans_1 is not None
            assert rotmats_1 is not None
        
        if inverse_folding and separate_t:
            trans_t = trans_1
            rotmats_t = rotmats_1
            quats_t = quat_1

        logs_traj = defaultdict(list)
        # Set-up time
        num_steps = self._sample_cfg.num_timesteps

        frames_to_atom37 = lambda x,y: all_atom.atom37_from_trans_rot(x, y, res_mask).detach().cpu()

        prot_traj = [(frames_to_atom37(trans_t, rotmats_t), aatypes_t.detach().cpu())] 
        clean_traj = []

        for idx in trange(1, num_steps+1, desc='sampling', leave=False, dynamic_ncols=True):
            # assign model inputs
            if self._trans_cfg.corrupt:
                batch['trans_t'] = trans_t
            else:
                if trans_1 is None: raise ValueError('Must provide trans_1 if not corrupting.')
                batch['trans_t'] = trans_1

            if self._rots_cfg.corrupt:
                batch['quats_t'] = quats_t
                batch['acc_quats_t'] = acc_quats_t
                batch['rotmats_t'] = rotmats_t
            else:
                if rotmats_1 is None: raise ValueError('Must provide rotmats_1 if not corrupting.')
                batch['rotmats_t'] = rotmats_1
                batch['quats_t'] = quat_1

            if self._aatypes_cfg.corrupt:
                batch['aatypes_t'] = aatypes_t
            else:
                if aatypes_1 is None: raise ValueError('Must provide aatype if not corrupting.')
                batch['aatypes_t'] = aatypes_1
            
            t = torch.ones((num_batch, 1), device=self._device) * (idx - 1) / num_steps
            t_index = t * num_steps + 1
            
            if t_nn is not None:
                batch['r3_t'], batch['so3_t'], batch['cat_t'] = torch.split(t_nn(t), -1)
            else:
                batch['so3_t'] = t
                batch['r3_t'] = t
                batch['cat_t'] = t
            if forward_folding and separate_t:
                batch['cat_t'] = (1 - self._cfg.min_t) * torch.ones_like(batch['cat_t'])
            if inverse_folding and separate_t:
                batch['r3_t'] = (1 - self._cfg.min_t) * torch.ones_like(batch['r3_t'])
                batch['so3_t'] = (1 - self._cfg.min_t) * torch.ones_like(batch['so3_t'])

            with torch.no_grad():
                model_out = model(batch)

            # Process model output.
            pred_trans = model_out['pred_trans']
            pred_trans = _recover_gt(diffuse_mask, pred_trans, trans_1)
            
            pred_quats = model_out['pred_quats']
            pred_quats = _recover_gt(diffuse_mask, pred_quats, quat_1)
            
            pred_rotmats = rc.quaternion_to_matrix(pred_quats)
            pred_rotmats = _recover_gt(diffuse_mask, pred_rotmats, rotmats_1)
            
            pred_aatypes = model_out['pred_aatypes']
            pred_aatypes = _recover_gt(diffuse_mask, pred_aatypes, aatypes_1)
            clean_traj.append((frames_to_atom37(pred_trans * self._trans_cfg.norm_std + self._trans_cfg.norm_mean, pred_rotmats),
                               pred_aatypes.detach().cpu()))
            

            if inverse_folding:
                pred_trans = trans_1
                pred_rotmats = rotmats_1


            if self._cfg.self_condition:
                batch['trans_sc'] = _trans_diffuse_mask(
                    pred_trans, trans_1, diffuse_mask)
                if forward_folding:
                    batch['aatypes_sc'] = aatypes_1
                else:
                    batch['aatypes_sc'] = _trans_diffuse_mask(pred_aatypes, aatypes_1, diffuse_mask)

            # Take reverse step            
            trans_t, acc_trans_t = self._trans_update_step(
                t, t+1/num_steps, pred_trans, trans_t, acc_trans_t)
            trans_t = _recover_gt(diffuse_mask, trans_t, trans_1)

            # if idx == 40:
            #     print(quats_t)
            #     print(acc_quats_t)
            quats_t, acc_quats_t = self._rots_update_step(
                t_index, t_index+1, pred_quats, quats_t, acc_quats_t)
            quats_t = _recover_gt(diffuse_mask, quats_t, quat_1)
            rotmats_t = rc.quaternion_to_matrix(quats_t)
            acc_quats_t = _recover_acc(diffuse_mask, acc_quats_t)

            aatypes_t = self._aatypes_update_step(t_index, t_index+1, pred_aatypes, aatypes_t)
            aatypes_t = _recover_gt(diffuse_mask, aatypes_t, aatypes_1)

            prot_traj.append((frames_to_atom37(trans_t * self._trans_cfg.norm_std + self._trans_cfg.norm_mean, rotmats_t), aatypes_t.cpu().detach()))

            if idx == num_steps:
                return prot_traj, clean_traj

