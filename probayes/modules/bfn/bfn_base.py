import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.special import i0e, i1e
from tqdm import tqdm
from torch.distributions import Normal
import numpy as np
from absl import logging
from torch.distributions.von_mises import VonMises
import torch.distributions as D

from probayes.modules.bfn.von_mises_fisher_utils import VonMisesFisher, ive
from probayes.modules.bfn.von_mises_fisher_cache import VonMisesFisherCache

def corrupt_t_pred(self, mu, t, gamma):
    # if t < self.t_min:
    #   return torch.zeros_like(mu)
    # else:
    # eps_pred = self.model()
    t = torch.clamp(t, min=self.t_min)
    # t = torch.ones((mu.size(0),1)).cuda() * t
    eps_pred = self.model(mu, t)
    x_pred = mu / gamma - torch.sqrt((1 - gamma) / gamma) * eps_pred
    return x_pred


class bfnBase(nn.Module):
    # this is a general method which could be used for implement vector field in CNF or
    def __init__(self, *args, **kwargs):
        super(bfnBase, self).__init__(*args, **kwargs)
        self.sphere_dim = 3
        self.epsilon = torch.tensor(1e-7)
    
    def zero_center_of_mass(self, x_pos, segment_ids):
        size = x_pos.size()
        assert len(size) == 2  # TODO check this
        seg_means = scatter_mean(x_pos, segment_ids, dim=0)
        mean_for_each_segment = seg_means.index_select(0, segment_ids)
        x = x_pos - mean_for_each_segment
        return x

    def get_k_params(self, bins):
        """
        function to get the k parameters for the discretised variable
        """
        # k = torch.ones_like(mu)
        # ones_ = torch.ones((mu.size()[1:])).cuda()
        # ones_ = ones_.unsqueeze(0)
        list_c = []
        list_l = []
        list_r = []
        for k in range(1, int(bins + 1)):
            # k = torch.cat([k,torch.ones_like(mu)*(i+1)],dim=1
            k_c = (2 * k - 1) / bins - 1
            k_l = k_c - 1 / bins
            k_r = k_c + 1 / bins
            list_c.append(k_c)
            list_l.append(k_l)
            list_r.append(k_r)
        # k_c = torch.cat(list_c,dim=0)
        # k_l = torch.cat(list_l,dim=0)
        # k_r = torch.cat(list_r,dim=0)

        return list_c, list_l, list_r

    def discretised_cdf(self, mu, sigma, x):
        """
        cdf function for the discretised variable
        """
        # in this case we use the discretised cdf for the discretised output function
        mu = mu.unsqueeze(1)
        sigma = sigma.unsqueeze(1)  # B,1,D

        f_ = 0.5 * (1 + torch.erf((x - mu) / ((sigma) * np.sqrt(2))))
        flag_upper = torch.ge(x, 1)
        flag_lower = torch.le(x, -1)
        f_ = torch.where(flag_upper, torch.ones_like(f_), f_)
        f_ = torch.where(flag_lower, torch.zeros_like(f_), f_)
        return f_

    def continuous_var_bayesian_flow(self, t, sigma1, x):
        """
        t: [B, ...], ... is the shape of x
        x: [B, ...]
        sigma1: float
        """
        assert torch.tensor(t <= 1.01).all()
        gamma = 1 - torch.pow(sigma1, 2 * t)  # [B]
        eps = torch.randn_like(x)  # [B, D]
        mu = gamma * x + eps * torch.sqrt(gamma * (1 - gamma))
        return mu, gamma


    def discrete_var_bayesian_flow(self, t, beta1, x, K, ret_eps=False):
        """
        x: [N, K]
        """
        beta = beta1 * (t**2)  # (B,)
        one_hot_x = x  # (N, K)
        mean = beta * (K * one_hot_x - 1)
        std = (beta * K).sqrt()
        eps = torch.randn_like(mean)
        y = mean + std * eps
        theta = F.softmax(y, dim=-1)
        if not ret_eps:
            return theta
        else:
            return theta, eps

    def ctime4continuous_loss(self, t, sigma1, x_pred, x):
        loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))

    def dtime4continuous_loss(self, i, N, sigma1, x_pred, x, segment_ids=None, wn=False, return_raw=False):
        sigma1 = torch.tensor(sigma1)
        if wn:
            steps = torch.arange(1,N+1).to(self.device)
            all_weights = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * steps / N))
            weight_norm = all_weights.mean().detach().clone()
        else:
            weight_norm = 1.
        if segment_ids is not None:
            weight = N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N))
            loss = scatter_mean(weight.view(-1)*((x_pred - x)**2).mean(-1),segment_ids,dim=0)
        else:
            loss =  N * (1 - torch.pow(sigma1, 2 / N)) / (2 * torch.pow(sigma1, 2 * i / N)) * (x_pred - x).view(x.shape[0], -1).abs().pow(2)

        if return_raw:
            return loss / weight_norm
        else:
            return loss.mean() / weight_norm

    def ctime4discrete_loss(self, t, beta1, one_hot_x, p_0, K):
        e_x = one_hot_x  # [N, K]
        e_hat = p_0  # (N, K)
        L_infinity = K * beta1 * t.view(-1) * ((e_x - e_hat) ** 2).sum(dim=-1)
        return L_infinity.mean()

    def ctime4discreteised_loss(self, t, sigma1, x_pred, x):
        loss = (x_pred - x).view(x.shape[0], -1).abs().pow(2).sum(dim=1)
        return -torch.log(sigma1) * loss * torch.pow(sigma1, -2 * t.view(-1))

    # def dtime4discrete_loss(self, i, N, beta1, one_hot_x, p_0, K, segment_ids=None, mult_constant=True):
    #     alpha = beta1 * (2 * i - 1) / (N**2)  # [N]
    #     e_x = one_hot_x  # [N, T, K]
    #     e_hat = p_0  # (N, T, K)
    #     if mult_constant:
    #         L_n = N * (K * alpha * (((e_x - e_hat) ** 2).sum(dim=-1)))
    #     else:
    #         L_n = (K * alpha * (((e_x - e_hat) ** 2).sum(dim=-1)))
    #     # L_n = N * (K * alpha * (((e_x - e_hat) ** 2).sum(dim=2).sum(dim=1))).mean()
    #     return L_n

    # def dtime4discrete_loss(self, i, N, beta1, one_hot_x, p_0, K, segment_ids=None,mult_constant=True):
    #     if not mult_constant:
    #         N = 1
    #     # i in {1,n}
    #     # Algorithm 7 in BFN
    #     D = one_hot_x.size()[0]
    #     e_x = one_hot_x  # [D, K]
    #     e_hat = p_0  # (D, K)
    #     assert e_x.size() == e_hat.size()
    #     alpha = beta1 * (2 * i - 1) / N**2  # [D]
    #     mean_ = alpha * (K * e_x - 1)  # [D, K]
    #     std_ = torch.sqrt(alpha * K)  # [D,1] TODO check shape
    #     eps = torch.randn_like(mean_)  # [D,K,]
    #     y_ = mean_ + std_ * eps  # [D, K]
    #     matrix_ek = torch.eye(K, K).to(e_x.device).unsqueeze(0).repeat(D,1,1)  # [D, K, K]
    #     mean_matrix = alpha.unsqueeze(-1) * (K * matrix_ek - 1)  # [K, K]
    #     std_matrix = torch.sqrt(alpha * K).unsqueeze(-1)  #
    #     LOG2PI = torch.log(torch.tensor(2 * np.pi))
    #     _log_gaussians = (  # [D, K]
    #         (-0.5 * LOG2PI - torch.log(std_matrix))
    #         - (y_.unsqueeze(1) - mean_matrix) ** 2 / (2 * std_matrix**2)
    #     ).sum(-1)
    #     _inner_log_likelihood = torch.log(torch.sum(e_hat * torch.exp(_log_gaussians), dim=-1))  # (D,)
    #     if segment_ids is not None:
    #         L_N = -scatter_mean(_inner_log_likelihood, segment_ids, dim=0)
    #     else:
    #         L_N = -_inner_log_likelihood.sum(dim=-1)  # [D]
    #     return N * L_N

    def dtime4discrete_loss_prob(
        self, i, N, beta1, one_hot_x, p_0, K, n_samples=200, segment_ids=None, time_scheduler ="quad", beta_init=None
    ):
        # this is based on the official implementation of BFN.
        # import pdb
        # pdb.set_trace()
        target_x = one_hot_x  # [D, K]
        e_hat = p_0  # (D,  K)

        if time_scheduler == "quad":
            alpha = beta1 * (2 * i - 1) / (N**2)  # [N]
        elif time_scheduler == "linear":
            alpha = beta1 / N
        elif time_scheduler == "hybrid":
            assert beta_init is not None
            alpha = (beta1 - beta_init) * (2 * i - 1) / (N**2) + beta_init / N     
        else:
            raise NotImplementedError   
        
        alpha = alpha.view(-1, 1) # [D, 1]

        classes = torch.arange(K, device=target_x.device).long().unsqueeze(0)  # [ 1, K]
        e_x = F.one_hot(classes.long(), K) #[1,K, K]
        # print(e_x.shape)
        receiver_components = D.Independent(
            D.Normal(
                alpha.unsqueeze(-1) * ((K * e_x) - 1), # [D K, K]
                (K * alpha.unsqueeze(-1)) ** 0.5, # [D, 1, 1]
            ),
            1,
        )  # [D,T, K, K]
        receiver_mix_distribution = D.Categorical(probs=e_hat)  # [D, K]
        receiver_dist = D.MixtureSameFamily(
            receiver_mix_distribution, receiver_components
        )  # [D, K]
        # pdb.set_trace()
        # print(receiver_dist.event_shape)

        sender_dist = D.Independent( D.Normal(
            alpha* ((K * target_x) - 1), ((K * alpha) ** 0.5)
        ),1)  # [D, K]

        y = sender_dist.sample(torch.Size([n_samples])) 

        # print(sender_dist.log_prob(y).size())
        # import pdb
        # pdb.set_trace()

        loss = N * (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0).mean(
            -1, keepdims=True
        )

        # loss = (
        #         (sender_dist.log_prob(y) - receiver_dist.log_prob(y))
        #         .mean(0)
        #         .flatten(start_dim=1)
        #         .mean(1, keepdims=True)
        #     )
        # #
        return loss.mean()




    def dtime4discrete_loss(
        self, i, N, beta1, one_hot_x, p_0, K, n_samples=200, segment_ids=None, return_raw=False
    ):
        if K == 1:
            return torch.zeros_like(one_hot_x)
            n_samples = 5
        # this is based on the official implementation of BFN.
        target_x = one_hot_x  # [D, K]
        e_hat = p_0  # (D,  K)
        alpha = beta1 * (2 * i - 1) / (N**2)  # [D]
        alpha = alpha.view(-1, 1) # [D, 1]
        classes = torch.arange(K, device=target_x.device).long().unsqueeze(0)  # [ 1, K]
        e_x = F.one_hot(classes.long(), K) #[1,K, K]
        # print(e_x.shape)
        receiver_components = D.Independent(
            D.Normal(
                alpha.unsqueeze(-1) * ((K * e_x) - 1), # [D K, K]
                (K * alpha.unsqueeze(-1)) ** 0.5, # [D, 1, 1]
            ),
            1,
        )  # [D,T, K, K]
        receiver_mix_distribution = D.Categorical(probs=e_hat)  # [D, K]
        receiver_dist = D.MixtureSameFamily(
            receiver_mix_distribution, receiver_components
        )  # [D, K]
        sender_dist = D.Independent( D.Normal(
            alpha* ((K * target_x) - 1), ((K * alpha) ** 0.5)
        ),1)  # [D, K]
        y = sender_dist.sample(torch.Size([n_samples])) 
        # if segment_ids != None:
        #     loss = scatter_mean(N * (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0), segment_ids, dim=0)
        # else:
        if return_raw:
            loss = N * (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0)
            return loss
        else:
            loss = N * (sender_dist.log_prob(y) - receiver_dist.log_prob(y)).mean(0).mean(
            -1, keepdims=True
            )
            return loss.mean()

    def dtime4circular_loss(self, i, N,  x_pred, x, beta1, do_weight_norm=False, mse_loss=False):
        freqs = torch.arange(-10,11).to(x.device).unsqueeze(0).unsqueeze(0)*np.pi*2
        tar_coord = x.unsqueeze(-1) + freqs
        coord_diff = (tar_coord - x_pred.unsqueeze(-1)).square().min(dim=-1).values
        
        alpha_i = self.alpha_wrt_index(i,N=N,beta1=beta1)
        
        weight = (i1e(alpha_i) / i0e(alpha_i)) * alpha_i   
        if mse_loss:
            loss = N * weight * coord_diff
        else:
            loss = N * weight * (1 - torch.cos(x_pred - x))
            
        if not do_weight_norm:
            weight_norm = 1.
        else:
            steps = torch.arange(1,N+1).to(x.device)
            alpha_steps = self.alpha_wrt_index(steps, N, beta1=beta1)
            weights = N * (i1e(alpha_steps) / i0e(alpha_steps)) * alpha_steps
            weight_norm = weights.mean().detach().clone()
        
        return loss.mean() / weight_norm

    
    def alpha_wrt_index(self, t_index, N, beta1):
        assert (t_index >= 1).all() and (t_index <= N).all()
        fname = f'./cache_files/schedules/linear_entropy_alphas_s{int(N)}_{int(beta1)}.pt'
        acc_schedule = torch.load(fname,map_location=t_index.device)
        return acc_schedule[t_index.long()-1]

    def circular_norm_logbeta(self, logbeta, beta1, epsilon=1e-7):
        '''
        Normalize logbeta to [0,1]
        '''
        epsilon = torch.tensor(epsilon)
        return (logbeta - torch.log(epsilon))/(torch.log(beta1) - torch.log(epsilon))

    @torch.no_grad()
    def circular_var_bayesian_flow_sim(self, x, t_index, beta1, N, epsilon=1e-7):
        '''
        the returned variable is in [0, 2 * pi)
        :param x: the input circular variable, shape (BxN, D)
        :param t: time , shape (BxN,)
        :param beta_t: the concentration of the von Mises distribution, shape (BxN, D)
        return: mu: the mean of the posterior, shape (BxN, D)
                kappa: the concentration of the posterior, shape (BxN, D)
        '''
        n_dim = x.shape[-1]
        x = x.view(x.shape[0],-1)
        t_index = t_index.long()
        n_samples = 1
        device = x.device
        # assert (t_index >= 1).all() and 
        alpha_index = self.alpha_wrt_index(torch.arange(1, N+1).to(device).unsqueeze(-1).long(), N, beta1).squeeze(-1)
        alpha = alpha_index.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,x.shape[0],x.shape[1],n_samples)
        # y = self.vm_helper.sample_cache(loc=x)
        torch_vm = VonMises(loc=x.unsqueeze(0).unsqueeze(-1).repeat(alpha.shape[0],1,1,n_samples).double(),concentration=alpha.double().clip(float(epsilon)))
        # y = self.vm_helper.sample(loc=x.unsqueeze(0).unsqueeze(-1).repeat(alpha.shape[0],1,1,n_samples), concentration=alpha, n_samples=1)
        y = torch_vm.sample()
        # sample prior
        prior_mu = (2*torch.pi*torch.rand((1,x.shape[0],x.shape[1]))).to(device)
        prior_cos, prior_sin = prior_mu.cos(), prior_mu.sin()
        # sample posterior
        poster_cos_cum, poster_sin_cum = (alpha * y.cos()).cumsum(dim=0).mean(-1), (alpha * y.sin()).cumsum(dim=0).mean(-1)
        poster_cos_cum = torch.cat([prior_cos, poster_cos_cum], dim=0)
        poster_sin_cum = torch.cat([prior_sin, poster_sin_cum], dim=0)
        # concat prior and posterior
        poster_cos = torch.gather(poster_cos_cum, dim=0, 
                                  index=(t_index-1).unsqueeze(0).repeat(1,1,x.shape[-1])).squeeze(0)
        poster_sin = torch.gather(poster_sin_cum, dim=0, 
                                  index=(t_index-1).unsqueeze(0).repeat(1,1,x.shape[-1])).squeeze(0)
        poster_mu = (torch.atan2(poster_sin, poster_cos)) % (np.pi * 2)
        poster_kappa = (torch.sqrt(poster_cos**2 + poster_sin**2)+epsilon).log().float()
        # assign kappa = 0 for every t_index = 1
        poster_kappa[t_index.squeeze()==1] = torch.log(torch.tensor((epsilon))).to(device).float()
        # normalize log beta
        poster_kappa = self.circular_norm_logbeta(poster_kappa, beta1=beta1, epsilon=epsilon)
        return poster_mu.view(x.shape[0],-1,n_dim).detach().float(), poster_kappa.view(x.shape[0],-1,n_dim).detach().float()

    def circular_var_bayesian_flow(self, x, t_index, N, beta1, epsilon):
        '''
        Compute p(θ|x, t) = E_{vM(y|x,β(t))}δ(θ-h(θ, y, β(t)))
        the returned variable is in [-pi, pi)
        :param x: the input circular variable, shape (BxN, 3 or 1)
        :param t: time , shape (BxN,)
        :param beta_t: the concentration of the von Mises distribution, shape (BxN, 3 or 1)
        return: mu: the mean of the posterior, shape (BxN, 3 or 1)
                kappa: the concentration of the posterior, shape (BxN, 3 or 1)
        '''
        DEVICE = x.device
        t_index = t_index.long().squeeze()
        alpha_index = self.alpha_wrt_index(torch.arange(1, N+1).to(DEVICE).long(), N, beta1)
        beta = torch.tensor([0.] + alpha_index.cumsum(dim=0).cpu().numpy().tolist()).to(DEVICE)
        beta_t = beta[t_index].unsqueeze(-1).unsqueeze(-1).repeat(1,x.shape[1],x.shape[2])
        torch_vm = VonMises(loc=x.double(),concentration=beta_t.double().clip(float(epsilon)))
        y = torch_vm.sample().float() % (np.pi * 2)
        return y.detach().clone(), torch.zeros_like(y)
    
    def sphere_norm_log_conc(self, logbeta, beta1, epsilon=1e-7):
        '''
        Normalize logbeta to [0,1]
        '''
        epsilon = torch.tensor(epsilon) if isinstance(epsilon, float) else epsilon
        beta1 = torch.tensor(beta1) if isinstance(beta1, float) else beta1
        return (logbeta - torch.log(epsilon))/(torch.log(beta1) - torch.log(epsilon))

    def dtime4sphere_loss(self, x, x_pred, t_index, beta1, N,  p, do_weight_norm=False, return_raw=False):
        """
        x: [N, D]
        x_pred: [N, D]
        t_index: [N, 1]
        loss = alpha * (ive(v+1,alpha) / ive(v,alpha)) * (1 - x_pred^T x)
        """
        # sanity check
        diff = 1 - torch.einsum("bi,bi->b", x, x_pred.float())
        v = p / 2 - 1
        alpha = self.sphere_alpha_wrt_index(t_index, N, beta1, p).squeeze()
        coef = N * alpha * (ive(v+1,alpha) / ive(v,alpha))
        loss = coef * diff
        
        if do_weight_norm:
            steps = torch.arange(1,N+1).to(x.device)
            alpha_steps = self.sphere_alpha_wrt_index(steps, N, beta1, p)
            weights = N * alpha_steps * (ive(v+1,alpha_steps) / ive(v,alpha_steps))
            weight_norm = weights.mean().detach().clone()
            loss = loss / weight_norm        
        
        if return_raw:
            return loss
        else:
            return loss.mean()

    def sphere_var_bayesian_update(self, loc_prev, conc_prev, pred_x, t_index, n_steps, beta1, p):
        '''
        loc_prev: [n_batch, dim]
        conc_prev: [n_batch, 1]
        '''
        alpha_i = self.sphere_alpha_wrt_index(t_index, N=n_steps, beta1=beta1, p=p) # (B, 1, 1)
        alpha_i = alpha_i.repeat(1, loc_prev.shape[1], 1) # (B, D, 1)
        y_i = VonMisesFisher(loc=pred_x,scale=alpha_i).sample()
        theta = loc_prev * conc_prev + alpha_i * y_i
        conc = torch.linalg.norm(theta,dim=-1,keepdim=True)    
        loc = theta / conc
        return loc, conc
    
    def circular_var_bayesian_update(self, loc_prev, conc_prev, alpha, pred_x):
        '''
        Compute (m_out, c_out) = h(m, c , y, α)
        according to 
        m_out = arctan((α sin(y) + c sin(m))/( α cos(y) + c cos(m))
        c_out = sqrt(α^2 + c^2 + 2αc cos(y-m))
        :param m: the previous mean, shape (D,)
        :param c: the previous concentration, shape (D,)
        return: m_out, c_out, shape (D,)
        '''
        torch_vm = VonMises(loc=pred_x, concentration=alpha)
        y = torch_vm.sample() % (np.pi * 2)
        m_out = torch.atan2(alpha * torch.sin(y) + conc_prev * torch.sin(loc_prev)+1e-6, 
                            alpha * torch.cos(y) + conc_prev * torch.cos(loc_prev)+1e-6)
        c_out = torch.sqrt(alpha**2 + conc_prev**2 + 2 * alpha * conc_prev * torch.cos(y - loc_prev))
        return m_out % (np.pi * 2), c_out

    def sphere_var_bayesian_flow_sim(self, x, t_index, beta1, N, epsilon=1e-7, dtype=torch.float32, 
                                     cache_sampling=False, cache_size=100000,cache_dir='./cache_files/vmf_cache'):
        '''
        x: (n_batch, seq_len, p)
        t_index: (n_batch, 1)
        return
        mu: (n_batch, seq_len, p)
        normed_log_conc: (n_batch, seq_len, 1)
        '''
        # shape check
        assert len(x.shape) == 3 and t_index.shape[0] == x.shape[0] and t_index.shape[-1]==1, 'shape incorrect'
        # prepare
        x = x.to(dtype)
        n_batch, seq_len, p = x.shape
        t_index = t_index.long().unsqueeze(-2)
        device = x.device
        alpha_schedule = self.sphere_alpha_wrt_index(torch.arange(1, N+1).to(device).long(), N, beta1, p) # [n_steps]
        # sample the whole trajectory
        alpha = alpha_schedule.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).repeat(1,n_batch, seq_len,1) 
        sample_x = x.unsqueeze(0).repeat(alpha.shape[0],1,1,1).double() # (n_steps,n_batch, seq_len, dim)
        
        if cache_sampling:
            cache_vmf = VonMisesFisherCache(alpha_schedule=alpha_schedule, 
                                            beta1=beta1, p=p, cache_size=cache_size, 
                                            cache_dir=cache_dir, 
                                            cache_refresh_rate=0.1,
                                            overwrite_cache=False)
            sample_t_index = torch.arange(1, N+1).to(device).long()
            sample_t_index = sample_t_index.view(-1,1,1).expand_as(sample_x[...,0])
            y = cache_vmf.sample(loc=sample_x, t_index=sample_t_index)
        else:        
            torch_vmf = VonMisesFisher(loc=sample_x, scale=alpha.double().clip(float(epsilon)))
            y = torch_vmf.sample()
        poster_cum = (y * alpha).cumsum(dim=0)
        prior = VonMisesFisher(loc=x,scale=epsilon*torch.ones((n_batch,seq_len,1),device=x.device)).sample()
        posters = torch.cat([prior.unsqueeze(0),poster_cum],dim=0) #[n_steps+1,n_batch,dim]
        # select
        selected_index = (t_index-1).unsqueeze(0).repeat(1,1,seq_len,p) # (1, n_batch, seq_len, p)
        selected_poster = torch.gather(posters, dim=0, index=selected_index).squeeze(0)
        # post process
        kappa = torch.functional.norm(selected_poster,dim=-1,p=2)
        mu = torch.nn.functional.normalize(selected_poster, p=2, dim=-1)
        log_kappa = (kappa+epsilon).log().float().unsqueeze(-1)
        t_index_select = t_index.repeat(1,seq_len,1)
        log_kappa[t_index_select==1] = torch.log(torch.tensor((epsilon))).to(device).float()
        normed_log_conc = self.sphere_norm_log_conc(log_kappa,beta1=beta1,epsilon=epsilon)
        return mu, normed_log_conc

    def sphere_alpha_wrt_index(self, t_index, N, beta1, p):
        assert (t_index >= 1).all() and (t_index <= N).all()
        fname = f'./cache_files/sphere_schedules/linear_entropy_alphas_s{int(N)}_beta{int(beta1)}_dim{p}.pt'
        acc_schedule = torch.load(fname,map_location=t_index.device)
        return acc_schedule[t_index.long()-1]
    
    def sphere_denorm_conc(self, normed_log_conc, beta1, epsilon=1e-7):
        '''
        Denormalize logbeta 
        '''
        beta1 = torch.tensor(beta1)
        epsilon = torch.tensor(epsilon)
        log_conc = normed_log_conc * (torch.log(beta1) - torch.log(epsilon)) + torch.log(epsilon)
        return log_conc.exp()
    
    def circular_denorm_conc(self, normed_log_conc, beta1, epsilon=1e-7):
        '''
        Denormalize logbeta 
        '''
        beta1 = torch.tensor(beta1)
        epsilon = torch.tensor(epsilon)
        log_conc = normed_log_conc * (torch.log(beta1) - torch.log(epsilon)) + torch.log(epsilon)
        return log_conc.exp()

    def interdependency_modeling(self):
        raise NotImplementedError

    def forward(self):
        raise NotImplementedError

    def loss_one_step(self):
        raise NotImplementedError

    def sample(self):
        raise NotImplementedError

