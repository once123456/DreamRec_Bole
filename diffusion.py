import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

class Diffusion(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        # 基础参数（从配置读取）
        self.timesteps = config['timesteps']
        self.beta_start = config['beta_start']
        self.beta_end = config['beta_end']
        self.beta_sche = config['beta_sche']
        self.w = config['w']
        self.diffuser_type = config['diffuser_type']
        self.device = device
        
        # 新增：读取配置中的损失类型和权重
        self.loss_type = config['loss_type']  # 从配置读取损失类型
        self.loss_weight = config['loss_weight']  # 可选权重

        # 1. 生成beta序列
        self.register_buffer('betas', self.get_beta_schedule())
        alphas = 1.0 - self.betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

        # 2. 预计算所有扩散系数（不变）
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))
        
        # 反向扩散系数（不变）
        self.register_buffer(
            'posterior_mean_coef1', 
            self.betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        )
        self.register_buffer(
            'posterior_mean_coef2', 
            (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        )
        self.register_buffer(
            'posterior_variance', 
            self.betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        )

        # 3. 初始化diffuser（不变）
        if self.diffuser_type == 'mlp1':
            self.diffuser = nn.Sequential(
                nn.Linear(config['hidden_size']*3, config['hidden_size'])
            ).to(device)
        elif self.diffuser_type == 'mlp2':
            self.diffuser = nn.Sequential(
                nn.Linear(config['hidden_size']*3, config['hidden_size']*2),
                nn.GELU(),
                nn.Linear(config['hidden_size']*2, config['hidden_size'])
            ).to(device)
    
    def get_beta_schedule(self):
        """生成beta序列（不变）"""
        if self.beta_sche == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, steps=self.timesteps, device=self.device)
        elif self.beta_sche == 'exp':
            x = torch.linspace(1, 2 * self.timesteps + 1, self.timesteps, device=self.device)
            betas = 1 - torch.exp(- 0.1 / self.timesteps - x * 0.5 * (10 - 0.1) / (self.timesteps * self.timesteps))
            return betas
        elif self.beta_sche == 'cosine':
            steps = torch.arange(self.timesteps + 1, device=self.device)
            alphas_cumprod = torch.cos(((steps / self.timesteps) + 0.008) / (1 + 0.008) * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return betas.clamp(0.0001, 0.9999)
        elif self.beta_sche == 'sqrt':
            def alpha_bar(t):
                return 1 - np.sqrt(t + 0.0001)
            betas = []
            for i in range(self.timesteps):
                t1 = i / self.timesteps
                t2 = (i + 1) / self.timesteps
                betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
            return torch.tensor(betas, device=self.device).float()
        else:
            raise ValueError(f"不支持的 beta 调度：{self.beta_sche}")

    def q_sample(self, x_start, t, noise=None):
        """前向加噪（不变）"""
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, denoise_model, x_start, h, t, noise=None):
        """修改：从配置读取损失类型，不再传参"""
        if noise is None:
            noise = torch.randn_like(x_start)
        
        # 1. 前向加噪
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        # 2. 去噪模型预测
        predicted_x = denoise_model(x_noisy, h, t)
        
        # 3. 根据配置选择损失函数
        if self.loss_type == 'l1':
            loss = F.l1_loss(x_start, predicted_x)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(x_start, predicted_x)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(x_start, predicted_x)
        else:
            raise ValueError(f"不支持的损失类型：{self.loss_type}，可选 l1/l2/huber")
        
        # 4. 应用损失权重
        loss = loss * self.loss_weight
        
        return loss, predicted_x

    @torch.no_grad()
    def p_sample(self, model_forward, model_forward_uncon, x, h, t, t_index):
        """单步去噪（不变）"""
        x_start = (1 + self.w) * model_forward(x, h, t) - self.w * model_forward_uncon(x, t)
        model_mean = (
            extract(self.posterior_mean_coef1, t, x.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x.shape) * x
        )
        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise 

    @torch.no_grad()
    def sample(self, model_forward, model_forward_uncon, h):
        """完整去噪（不变）"""
        x = torch.randn_like(h, device=self.device)
        for n in reversed(range(0, self.timesteps)):
            t = torch.full((h.shape[0], ), n, device=self.device, dtype=torch.long)
            x = self.p_sample(model_forward, model_forward_uncon, x, h, t, n)
        return x