import os
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
# print(os.getcwd())

import torch
from tqdm import tqdm
from torch.special import i0e, i1e
from scipy.optimize import root_scalar
from scipy.stats import circvar,circstd
from von_mises_fisher_utils import VonMisesFisher
from scipy.stats import vonmises_fisher, uniform_direction

class SphereAccuracySchedule(torch.nn.Module):
    def __init__(self, n_steps, beta1, n_dim, device='cuda:0'):
        super(SphereAccuracySchedule, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.beta1 = torch.tensor(beta1,device=device)
        self.n_dim = n_dim
        
    
    def vmf_entropy(self, kappa):
        kappa = max(float(kappa),1e-10)
        return vonmises_fisher.entropy(mu=np.array([np.sqrt(1/self.n_dim)]*self.n_dim),kappa=kappa)
    
    def linear_entropy(self, step):
        assert (step <= self.n_steps).all() and (step >= 0).all()
        t = step / self.n_steps
        entropy1 = self.vmf_entropy(float(self.beta1.cpu()))
        entropy0 = self.vmf_entropy(torch.tensor(0.))
        slope = entropy1 - entropy0
        return entropy0 + slope * t
    
    def get_beta(self, step:torch.tensor):
        assert (step <= self.n_steps).all() and (step >= 0).all()
        t = step / self.n_steps
        return self.beta_schedule(t)              
    
    def get_alpha(self, step:torch.tensor, schedule='add'):
        assert (step <= self.n_steps).all() and (step >= 1).all()
        if schedule == 'add':
            step_prev = step - 1
            alpha = self.get_beta(step) - self.get_beta(step_prev)
        else:
            raise NotImplementedError
        return alpha
        
    
    @torch.no_grad()
    def analyze_schedule(self, schedule=None, n_samples=100000):
        steps = torch.range(1,self.n_steps,1, device=self.device).long().to(self.device)
        if schedule == None:
            schedule = self.get_alpha(steps, schedule='add')
        assert schedule.shape == (self.n_steps,)
        schedule = schedule.to(self.device)
        # x = torch.tensor(uniform_direction.rvs(dim=3,size=(self.n_steps, n_samples)),device=self.device)
        x = torch.tensor([np.sqrt(1/self.n_dim)]*self.n_dim,device=self.device).unsqueeze(0).unsqueeze(0).repeat(self.n_steps,n_samples,1)
        # prior_mean = torch.tensor([np.sqrt(0.5),np.sqrt(0.5),0],device=self.device).unsqueeze(0).repeat(n_samples,1)
        sender_alpha = schedule.unsqueeze(-1).repeat(1,n_samples).unsqueeze(-1).to(self.device)
        torch_vmf = VonMisesFisher(loc=x, scale=sender_alpha)
        y = torch_vmf.sample()
        poster = (y * sender_alpha).cumsum(dim=0)
        poster_acc = torch.linalg.norm(poster,dim=-1).mean(dim=-1).cpu()
        poster_mean = poster / poster_acc.cuda()[..., None, None]        
        final_acc = poster_acc[-1]
        # entropy = vonmises_fisher.entropy(None,poster_acc)
        entropy = [vonmises_fisher.entropy(mu=uniform_direction.rvs(dim=self.n_dim), kappa=float(poster_acc[i])) for i in range(self.n_steps)]
        
        poster_stds = poster_mean.std(dim=1).cpu().unbind(-1)
        plt.figure(dpi=300)
        for i in range(self.n_dim):
            plt.plot(poster_stds[i],label=f'dim{i}')
        plt.legend()
        plt.show()
        plt.savefig(f'cache_files/sphere_schedules/poster_stds_alphas_s{self.n_steps}_beta{int(self.beta1)}_dim{self.n_dim}.png')
        
        
        # 绘制entropy图
        plt.figure(dpi=300)
        plt.plot(entropy,label='simulated entropy')
        linear_entropy = self.linear_entropy(steps)
        plt.plot(linear_entropy.cpu(),label='linear entropy')
        plt.legend()
        plt.show()
        plt.savefig(f'cache_files/sphere_schedules/linear_entropy_alphas_s{self.n_steps}_beta{int(self.beta1)}_dim{self.n_dim}.png')
        return final_acc, entropy
    
    def entropy_equation(self, tar_entropy):
        return lambda kappa: self.vmf_entropy(kappa) - tar_entropy
    
    def find_beta(self):
        steps = torch.range(1,self.n_steps,1, device=self.device).long()
        linear_entropies = self.linear_entropy(steps).unsqueeze(-1).cpu()
        betas = []
        for i in tqdm(range(len(linear_entropies)-1)):
            tar_entropy = linear_entropies[i]
            root = root_scalar(self.entropy_equation(tar_entropy),bracket=[0,self.beta1])
            if root.converged:
                betas.append(root.root)
            else:
                assert False, 'root not converged!'
        betas.append(float(self.beta1.cpu()))
        return torch.tensor(betas)

    def alpha_equation_accurate(self, prior_alphas, prior_beta, tar_beta, n_samples):
        prior_alpha = torch.stack(prior_alphas).to(self.device)
        def func(alpha):
            # prior_mean = torch.tensor(uniform_direction.rvs(dim=3,size=(n_samples)),device=self.device)
            gt_x = torch.tensor([np.sqrt(1/self.n_dim)]*self.n_dim,device=self.device).unsqueeze(0).repeat(n_samples,1)
            gt_x_prev = torch.stack([gt_x]*prior_alpha.shape[0],dim=0) # (n_prev_steps, p, n_samples)
            prev_samples_scale = prior_alpha.unsqueeze(-1).repeat(1,n_samples)
            prev_samples_vmf = VonMisesFisher(loc=gt_x_prev, scale=prev_samples_scale.unsqueeze(-1))
            prev_samples = prev_samples_vmf.sample()
            sender_alpha = alpha * torch.ones(size=prev_samples_scale.shape,device=self.device).unsqueeze(-1)
            torch_vmf = VonMisesFisher(loc=gt_x, scale=sender_alpha)
            y = torch_vmf.sample()
            poster = y * alpha + (prev_samples* prior_alpha.unsqueeze(-1).unsqueeze(-1)).sum(dim=0) 
            poster_acc = torch.linalg.norm(poster,dim=-1).mean().cpu()
            return poster_acc - tar_beta
        return func

    def alpha_equation_approx(self, prior_alphas, prior_beta, tar_beta, n_samples):
        def func(alpha):
            # prior_mean = torch.tensor(uniform_direction.rvs(dim=3,size=(n_samples)),device=self.device)
            prior_mean = torch.tensor([np.sqrt(1/self.n_dim)]*self.n_dim,device=self.device).unsqueeze(0).repeat(n_samples,1)
            sender_alpha = alpha * torch.ones(size=prior_mean.shape[:-1],device=self.device).unsqueeze(-1)
            torch_vmf = VonMisesFisher(loc=prior_mean, scale=sender_alpha)
            y = torch_vmf.sample()
            poster = y * alpha + prior_mean * prior_beta
            poster_acc = torch.linalg.norm(poster,dim=-1).mean().cpu()
            return poster_acc - tar_beta
        return func

    @torch.no_grad()
    def find_linear(self, n_samples=500000):
        res_betas = self.find_beta()
        sender_alpha = [] # search sender alpha
        sender_alpha.append(res_betas[0])
        for i in tqdm(range(1,self.n_steps)):
            prior_beta = res_betas[i-1] #上一步达到的beta
            target_beta = res_betas[i] #目标beta
            if i < self.n_steps // 3:
                root_alpha = root_scalar(self.alpha_equation_accurate(sender_alpha, prior_beta, target_beta, n_samples=n_samples),
                                        bracket=[target_beta-prior_beta, self.beta1],xtol=1e-4)
            else:
                root_alpha = root_scalar(self.alpha_equation_approx(sender_alpha, prior_beta, target_beta, n_samples=n_samples),
                    bracket=[target_beta-prior_beta, self.beta1],xtol=1e-4)
            assert root_alpha.converged, 'alpha root not converged!'
            sender_alpha.append(torch.tensor(root_alpha.root))
        return torch.stack(sender_alpha)          
    
    
    
if __name__ == '__main__':
    n_steps = 200
    beta1 = 1e2
    n_dim = 4
    n_samples = 50000
    DEVICE = 'cuda:1'
    fname = f'./cache_files/sphere_schedules/linear_entropy_alphas_s{n_steps}_beta{int(beta1)}_dim{n_dim}.pt'
    find_linear = True

    acc_schedule = SphereAccuracySchedule(n_steps=n_steps,beta1=beta1,n_dim=n_dim,)
    t = torch.range(0, n_steps-1, 1) / n_steps
    if find_linear:
        sender_alphas = acc_schedule.find_linear(n_samples=n_samples)
        torch.save(sender_alphas, fname)
    else:
        if os.path.exists(fname):
            sender_alphas = torch.load(fname)
        else:
            raise FileNotFoundError
    final_acc, linear_entropy = acc_schedule.analyze_schedule(sender_alphas, n_samples=n_samples)
    print('final accuracy:', final_acc,'designed beta:', beta1)
    
    
    
    
