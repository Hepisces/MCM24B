import numpy as np
from scipy.stats import beta
from scipy.integrate import quad
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['axes.unicode_minus'] = False 
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def p_exp(p, sigma):
    """
    p:正品率
    sigma = 0.05  # bayes修正置信度
    """
    p = 1 - p
    sample = int(np.round((1.645*np.sqrt(p*(1-p))/(0.02))**2,0))   # 样本量

    p_min = 0  # 最小概率
    p_max = np.round(p + 1.645 * np.sqrt(p * (1 - p) / sample), 3)  # 允许的最大概率

    # beta先验超参数
    alpha_prior = np.round((1 - p) * ((p * (1 - p)) / sigma**2 - 1), 2)
    beta_prior = np.round(p * ((p * (1 - p)) / sigma**2 - 1), 2)

    # 抽样
    k = np.random.binomial(sample, p)

    # 后验修正
    alpha_post = alpha_prior + k
    beta_post = beta_prior + sample - k

    # 期望
    expectation = truncated_beta_expectation(alpha_post, beta_post, p_min, p_max)

    return alpha_post, beta_post, p_min, p_max, round(expectation, 2),alpha_prior,beta_prior


def truncated_beta_expectation(alpha_post, beta_post, p_min, p_max):
    """
    截断后验 Beta 分布的期望
    """
    normalization_constant, _ = quad(
        lambda p: beta.pdf(p, alpha_post, beta_post), p_min, p_max
    )
    expectation_numerator, _ = quad(
        lambda p: p * beta.pdf(p, alpha_post, beta_post), p_min, p_max
    )
    return expectation_numerator / (normalization_constant)

def truncated_beta_pdf(alpha, beta_val, p_min, p_max):
    """
    返回截断 Beta 分布的概率密度函数
    """
    # 归一化常数
    normalization_constant, _ = quad(
        lambda p: beta.pdf(p, alpha, beta_val), p_min, p_max
    )
    
    # 定义截断后的PDF
    def pdf(p):
        if p_min <= p <= p_max:
            return beta.pdf(p, alpha, beta_val) / normalization_constant
        else:
            return 0
    
    return pdf

def p_correct(p,sigma):
    """
    从截断后的 Beta 分布中
    """
    alpha_post,beta_post,p_min,p_max,_,_,_ = p_exp(p,sigma)
    while True:
        sample = beta.rvs(alpha_post, beta_post)
        if p_min <= sample <= p_max:
            break
    return round(sample,2)

def sample_from_truncated_beta(alpha_post, beta_post, p_min, p_max, num_samples):
        """
        从截断后的 Beta 分布中进行抽样
        """
        samples = []
        while len(samples) < num_samples:
            sample = beta.rvs(alpha_post, beta_post)
            if p_min <= sample <= p_max:
                samples.append(sample)
        return np.array(samples)

def plot(p, sigma):
    alpha_post, beta_post, p_min, p_max, _, alpha_prior, beta_prior = p_exp(p, sigma)
    pdf_truncated = truncated_beta_pdf(alpha_post, beta_post, p_min, p_max)

    # 在 [0, 1] 范围内画出原始和截断的 Beta 分布
    x = np.linspace(0, 1, 1000)
    y_original = beta.pdf(x, alpha_post, beta_post)
    y_truncated = np.array([pdf_truncated(p) for p in x])

    # 创建子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # 第一张图：原始和截断的 Beta 分布
    # ax1.plot(x, y_original, label='原始的 Beta 分布', linestyle='--')
    ax1.plot(x, y_truncated, label='截断后的 Beta 分布', color='red')
    ax1.fill_between(x, 0, y_truncated, where=((x >= p_min) & (x <= p_max)), color='red', alpha=0.3)
    ax1.axvline(p_min, color='green', linestyle=':', label=f'左截断点 p_min={p_min}')
    ax1.axvline(p_max, color='green', linestyle=':', label=f'右截断点 p_max={p_max}')
    ax1.set_title('截断后的 Beta 分布')
    ax1.set_xlabel('p')
    ax1.set_xlim(p_min, p_max+0.1)
    ax1.legend()
    ax1.grid(True)

    # 从截断后的 Beta 分布中抽样
    num_samples = 1000
    samples = sample_from_truncated_beta(alpha_post, beta_post, p_min, p_max, num_samples)

    # 第二张图：抽样的直方图
    ax2.hist(samples, bins=30, density=True, alpha=0.6, color='blue', edgecolor='black', label='截断 Beta 分布样本')
    ax2.plot(x, y_truncated, label='截断后的 Beta 分布', color='red')
    ax2.set_title('截断 Beta 分布的样本')
    ax2.set_xlabel('p')
    ax2.set_ylabel('密度')
    ax2.set_xlim(p_min-0.01, p_max+0.01)
    ax2.legend()
    ax2.grid(True)

    # 显示图像
    plt.tight_layout()
    plt.show()

# if __name__ == "__main__":
#     print(p_correct(0.9, 0.05))
#     #plot(0.8, 0.05)
