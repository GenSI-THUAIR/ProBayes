
import torch
import numpy as np
import scipy.special
from numbers import Number
from typing import Tuple
import os
import math
import torch
from torch.distributions.kl import register_kl
from probayes.modules.bfn.von_mises_fisher_utils import VonMisesFisher


class VonMisesFisherCache:
    '''
    A cached sampler for the von Mises-Fisher distribution.
    '''
    def __init__(self, alpha_schedule, beta1, p, cache_size, cache_dir, cache_refresh_rate, overwrite_cache=False):
        # self.dtype = loc.dtype
        self.device = alpha_schedule.device
        self.dtype = alpha_schedule.dtype
        self.n_steps = alpha_schedule.size(0)
        self.cache_size = cache_size # number of samples to cache, larger one means more accuracy
        self.p = p
        self.alpha_schedule = alpha_schedule
        self.beta1 = beta1
        self.cache_dir = cache_dir
        self.__e1 = (torch.Tensor([1.0] + [0] * (p - 1))).to(self.device)
        
        self._setup_sample_w_cache(overwrite=overwrite_cache)

    def sample(self, loc, t_index):
        shape = loc.shape[:-1]
        w = self._sample_w_cache(shape, t_index)
        
        # v = (
        #     torch.distributions.Normal(0, 1)
        #     # .sample(shape + torch.Size(loc.shape))
        #     .sample(torch.Size(loc.shape))
        #     .to(self.device)
        #     .transpose(0, -1)[1:]
        # ).transpose(0, -1)
        v = (
            torch.distributions.Normal(torch.tensor(0.,device=loc.device), torch.tensor(1.,device=loc.device))
            # .sample(shape + torch.Size(loc.shape))
            .sample(torch.Size(loc.shape))
            .to(self.device)
        )[...,1:]  
        v = v / v.norm(dim=-1, keepdim=True)

        w_ = torch.sqrt(torch.clamp(1 - (w ** 2), 1e-10))
        x = torch.cat((w, w_ * v), -1)
        z = self.__householder_rotation(x, loc)
        return z
    
    def __householder_rotation(self, x, loc):
        u = self.__e1 - loc
        u = u / (u.norm(dim=-1, keepdim=True) + 1e-5)
        z = x - 2 * (x * u).sum(-1, keepdim=True) * u
        return z
    
    def _sample_w_cache(self, shape, t_index):
        '''
        Sample w from the cache.
        '''
        idx = torch.randint(low=0, high=self.cache_size, size=shape, device=self.device)
        return self.cache[idx, t_index-1, :]
    
    def _get_cache_fname(self, ):
        return f'{self.cache_dir}/vmf_cache_size{int(self.cache_size)}_step{self.n_steps}_beta{int(self.beta1)}_dim{self.p}.pt'
    
    def _setup_sample_w_cache(self, overwrite=False):
        if os.path.exists(self._get_cache_fname()) and (not overwrite):
            self.cache = torch.load(self._get_cache_fname(),map_location=self.device)
        else:
            loc = torch.zeros((self.cache_size, self.n_steps, self.p),device=self.device)
            scale = self.alpha_schedule.unsqueeze(0).unsqueeze(-1).repeat(self.cache_size, 1, 1)
            torch_vmf = VonMisesFisher(loc,scale)
            # samples = torch_vmf.sample()
            samples = torch_vmf._sample_w_rej(torch.Size())
            self.cache = samples
            
            if not os.path.exists(self.cache_dir):
                os.makedirs(self.cache_dir)
            torch.save(samples, self._get_cache_fname())
        

if __name__ == '__main__':
    import torch

    alpha_schedule = torch.load('cache_files/sphere_schedules/linear_entropy_alphas_s200_beta1000_dim4.pt', map_location='cuda:1')
    
    vmf_cache = VonMisesFisherCache(alpha_schedule=alpha_schedule, beta1=1000,
                                    p=4, cache_size=10000, 
                                    cache_dir='cache_files/vmf_cache', 
                                    cache_refresh_rate=1000,
                                    overwrite_cache=False)
    x = torch.randn(100, 4, device=alpha_schedule.device)
    x = x / x.norm(dim=-1, keepdim=True)
    t_index = torch.randint(low=0, high=200, size=x.shape[:-1],device=x.device) 
    sampled = vmf_cache.sample(loc=x, t_index=t_index)
    print(sampled)