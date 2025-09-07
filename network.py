import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import normalize


class DiffusionClustering(nn.Module):
    def __init__(self, input_dim, output_dim, T=1000):
        super().__init__()
        self.T = T
        # 增加数值稳定性：使用 LayerNorm 和更小的初始化
        self.denoise_net = nn.Sequential(
            nn.Linear(output_dim + input_dim, 512),
            nn.LayerNorm(512),  # 添加层标准化
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        
        # 更安全的噪声调度
        self.register_buffer('alphas', 1 - torch.sqrt(torch.arange(1, T+1, dtype=torch.float32)/T + 1e-4))
        
        # 创建加速采样索引
        self.sampling_steps = torch.linspace(T-1, 0, 45, dtype=torch.long).tolist()
        
        # 网络参数初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)

    def p_sample(self, zt, condition, t):
        """DDIM反向去噪（数值稳定版本）"""
        inputs = torch.cat([zt, condition], dim=1)
        z0_pred = self.denoise_net(inputs)
        
        if t > 0:
            idx = self.sampling_steps.index(t)
            prev_t = self.sampling_steps[idx-1] if idx > 0 else 0
            
            # 数值稳定性处理
            alpha_ratio = torch.clamp(self.alphas[prev_t] / self.alphas[t], min=1e-5, max=1.0)
            sigma_sq = torch.clamp((1 - self.alphas[prev_t]) / (1 - self.alphas[t]), min=0.0, max=0.99)
            
            # 添加数值保护
            coeff = torch.clamp(1 - alpha_ratio - sigma_sq, min=0.0)
            sqrt_coeff = torch.sqrt(coeff)
            
            zt = (torch.sqrt(alpha_ratio) * zt + sqrt_coeff * z0_pred)
        return zt

    def forward(self, condition, B=5):
        batch_size = condition.shape[0]
        output_dim = self.denoise_net[-1].out_features
        
        z0_samples = []
        for _ in range(B):
            # 缩小噪声范围
            zt = torch.randn(batch_size, output_dim).to(condition.device) * 0.1
            
            for t in self.sampling_steps:
                zt = self.p_sample(zt, condition, t)
                
            z0_samples.append(zt)
        
        return torch.mean(torch.stack(z0_samples), dim=0)

class FeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(FeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Encoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 2000),
            nn.ReLU(),
            nn.Linear(2000, feature_dim),
        )

    def forward(self, x):
        return self.encoder(x)

class Decoder(nn.Module):
    def __init__(self, input_dim, feature_dim):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(feature_dim, 2000),
            nn.ReLU(),
            nn.Linear(2000, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, input_dim)
        )
    def forward(self, x):
        return self.decoder(x)

class GCFAggMVC(nn.Module):
    def __init__(self, view, input_size, low_feature_dim, high_feature_dim, device):
        super(GCFAggMVC, self).__init__()
        self.encoders = []
        self.decoders = []
        for v in range(view):
            self.encoders.append(Encoder(input_size[v], low_feature_dim).to(device))
            self.decoders.append(Decoder(input_size[v], low_feature_dim).to(device))
        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)
        self.Specific_view = nn.Sequential(
            nn.Linear(low_feature_dim, high_feature_dim),
        )
        
        self.view = view

        
        # 扩散模型维度一致性修复
        self.diffusion = DiffusionClustering(
            input_dim=low_feature_dim * view,  # 条件输入维度
            output_dim=high_feature_dim  # 输出嵌入维度
        ).to(device)
        

        
    def forward(self, xs):
        xrs = []
        zs = []
        hs = []
        for v in range(self.view):
            x = xs[v]
            z = self.encoders[v](x)
            h = normalize(self.Specific_view(z), dim=1)
            xr = self.decoders[v](z)
            hs.append(h)
            zs.append(z)
            xrs.append(xr)
        return xrs, zs, hs

    def GCFAgg(self, xs):
        zs = []
        Alist = []
        for v in range(self.view):
            x = xs[v]
            A = self.computeA(F.normalize(x), mode='knn')
            Alist.append(A)
            z = self.encoders[v](x)
            zs.append(z)
            
        

        commonz = torch.cat(zs, dim=1)
        
        # 扩散模型生成聚类嵌入向量z0（论文Section 4）
        z0 = self.diffusion(commonz, B=10)  # [batch, high_feature_dim]
        
       
        
        
        
        return z0, torch.mean(torch.stack(Alist), dim=0)

    def computeA(self, x, mode):
        if mode == 'cos':
            a = F.normalize(x, p=2, dim=1)
            b = F.normalize(x.T, p=2, dim=0)
            A = torch.mm(a, b)
            A = (A + 1) / 2
        if mode == 'kernel':
            x = torch.nn.functional.normalize(x, p=1.0, dim=1)
            a = x.unsqueeze(1)
            A = torch.exp(-torch.sum(((a - x.unsqueeze(0)) ** 2) * 1000, dim=2))
        if mode == 'knn':
            dis2 = (-2 * x.mm(x.t())) + torch.sum(torch.square(x), axis=1, keepdim=True) + torch.sum(
                torch.square(x.t()), axis=0, keepdim=True)
            A = torch.zeros(dis2.shape).cuda()
            A[(torch.arange(len(dis2)).unsqueeze(1), torch.topk(dis2, 10, largest=False).indices)] = 1
            A = A.detach()
        if mode == 'sigmod':
            A = 1/(1+torch.exp(-torch.mm(x, x.T)))
        return A
