import scipy
import numpy as np
import matplotlib.pyplot as plt
import torch
from scipy.stats import vonmises_fisher, probplot
from von_mises_fisher_utils import VonMisesFisher




def compare_quantile(samples, ref_samples):
    # quantile torch_samples
    quantiles = np.linspace(0, 1, 10000)
    gen_quantiles = np.quantile(samples, quantiles)
    ref_quantiles = np.quantile(ref_samples, quantiles)

    plt.figure()
    plt.scatter(gen_quantiles, ref_quantiles, s=3)

    x_min = min(gen_quantiles.min(), ref_quantiles.min())
    x_max = max(gen_quantiles.max(), ref_quantiles.max())

    plt.plot([x_min,x_max],[x_min,x_max],'r')
    plt.xlabel('torch')
    plt.ylabel('scipy')
    plt.title('Q-Q plot')
    plt.show()  
    plt.savefig('qqplot.png')
    
if __name__ == '__main__':
    n_samples = 1000000

    mean_direction = torch.randn(4)
    mean_direction = mean_direction / mean_direction.norm()
    print(mean_direction)

    kappa = torch.tensor([3.])
    # 生成 n_samples 个样本
    samples = vonmises_fisher.rvs(mean_direction, float(kappa), size=n_samples)
    torch_vmf = VonMisesFisher(torch.tensor(mean_direction), torch.tensor(kappa))

    torch_samples = torch_vmf.sample(n_samples)
    print(torch_samples)
    print(samples)
    # Quantile-Quantile 图
    torch_dim1 = torch_samples[:,0].cpu().numpy()
    scipy_dim1 = samples[:,0]

    compare_quantile(torch_dim1, scipy_dim1)
