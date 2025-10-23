import scipy
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import vonmises_fisher, probplot
from von_mises_fisher_utils import VonMisesFisher
from von_mises_fisher_cache import VonMisesFisherCache
from von_mises_fisher_test import compare_quantile
from bfn_base import bfnBase

if __name__ == '__main__':
    
    n_samples = 100000
    n_steps = 200
    beta1 = torch.tensor(1e3)
    device = 'cuda:1'

    mean_direction = torch.randn(1, 2).to(device).unsqueeze(0).repeat(n_samples, 1, 1)
    mean_direction = mean_direction / mean_direction.norm(dim=-1,keepdim=True)

    BFN = bfnBase()

    # t_index = torch.range(1, n_steps).unsqueeze(-1).to(device)
    t_index = 100 * torch.ones_like(mean_direction[...,0])

    m_t_cached, acc_t = BFN.sphere_var_bayesian_flow_sim(x=mean_direction,
                                                t_index=t_index,
                                                beta1=beta1, 
                                                N=n_steps, 
                                                cache_sampling=True, cache_size=100000)
    print('cached done')
    m_t_not_cached, acc_t = BFN.sphere_var_bayesian_flow_sim(x=mean_direction,
                                                t_index=t_index,
                                                beta1=beta1, 
                                                N=n_steps, 
                                                cache_sampling=False)
    print('not cached done')
    compare_quantile(m_t_cached.cpu().numpy(), m_t_not_cached.cpu().numpy())



