import torch
import torch.nn as nn
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from typing import Optional, List, Tuple
import numpy as np
from tqdm import tqdm
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:x.size(0)]

class TimeEmbedding(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        assert dim % 2 == 0, "嵌入维度需要是偶数"
        half_dim = dim // 2
        
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer('emb', emb)
        
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, t):
        t = t.float() / self.emb.numel()
        emb = t.unsqueeze(-1) * self.emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.proj(emb)

class TransformerDiffusionBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = nn.GELU()
        
    def forward(
        self,
        x: torch.Tensor,
        combined_cond: torch.Tensor,
    ) -> torch.Tensor:
        # Self attention
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2)[0])
        
        # Cross attention with combined condition
        x2 = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(x2, combined_cond, combined_cond)[0])
        
        # Feed forward
        x2 = self.norm3(x)
        x = x + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(x2)))))
        
        return x

class TransformerDiffusion(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        # 条件投影
        self.cond_proj = nn.Sequential(
            nn.Linear(1, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model)
        )
        
        self.time_embedding = TimeEmbedding(d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.layers = nn.ModuleList([
            TransformerDiffusionBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(
        self,
        x: torch.Tensor,  # [B, seq_len, 1] - 归一化的ID值
        timesteps: torch.Tensor,  # [B]
        condition: torch.Tensor,  # [B, seq_len, 1] - 归一化的条件ID值
    ) -> torch.Tensor:
        # 输入投影
        x = self.input_proj(x)  # [B, seq_len, d_model]
        
        # 条件投影
        cond = self.cond_proj(condition)  # [B, seq_len, d_model]
        
        # 时间嵌入
        t_emb = self.time_embedding(timesteps)  # [B, d_model]
        t_emb = t_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # [B, seq_len, d_model]
        
        # 将时间信息和条件信息相加
        combined_cond = t_emb + cond  # [B, seq_len, d_model]
        
        # 位置编码
        x = x.transpose(0, 1)  # [seq_len, B, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [B, seq_len, d_model]
        
        # Transformer层
        for layer in self.layers:
            x = layer(x, combined_cond)
        
        # 输出投影
        x = self.output_proj(x)  # [B, seq_len, 1]
        return x

class NormalizedTrajectoryTransformerDiffusion(pl.LightningModule):
    def __init__(
        self,
        num_ids: int = 3953,
        embedding_dim: int = 512,
        num_layers: int = 6,
        learning_rate: float = 1e-4,
        num_timesteps: int = 300
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_ids = num_ids
        
        self.model = TransformerDiffusion(
            d_model=embedding_dim,
            nhead=8,
            num_layers=num_layers,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1
        )
        
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
        self.learning_rate = learning_rate
        self.num_timesteps = num_timesteps
        
    def normalize_ids(self, ids):
        # 将ID归一化到[0,10]范围
        # 最小值是0，最大值是num_ids + 1
        min_val = 0
        max_val = self.num_ids + 1
        normalized = (ids.float() - min_val) / (max_val - min_val) * 10
        return normalized
        
    def denormalize_ids(self, normalized_ids):
        # 将归一化的值（0-10）转换回原始ID范围
        min_val = 0
        max_val = self.num_ids + 1
        denormalized = normalized_ids * (max_val - min_val) / 10 + min_val
        rounded_ids = torch.round(denormalized)
        return rounded_ids.long().clamp(0, self.num_ids + 1)
    
    def forward(self, noisy_x, timesteps, condition_ids, mask=None):
        # 将条件ID归一化并扩展维度
        normalized_condition = self.normalize_ids(condition_ids).unsqueeze(-1)
        noise_pred = self.model(noisy_x, timesteps, normalized_condition)
        return noise_pred
    
    def training_step(self, batch, batch_idx):
        original_ids, condition_ids, mask = batch
        
        # 归一化ID
        x0 = self.normalize_ids(original_ids).unsqueeze(-1)  # [B, seq_len, 1]
        
        # 添加噪声
        noise = torch.randn_like(x0)
        timesteps = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=x0.device).long()
        timesteps[0] = 1
        noisy_x = self.noise_scheduler.add_noise(x0, noise, timesteps)
        
        # 预测噪声
        noise_pred = self(noisy_x, timesteps, condition_ids, mask)
        
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        self.log("train_loss", loss.item(), prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
    def sample(self, condition_ids, mask=None):
        device = next(self.parameters()).device
        batch_size = condition_ids.shape[0]
        seq_len = condition_ids.shape[1]
        
        # 初始化噪声
        x = torch.randn((batch_size, seq_len, 1), device=device)
        
        # 逐步去噪
        for t in tqdm(self.noise_scheduler.timesteps):
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            with torch.no_grad():
                noise_pred = self(x, timesteps, condition_ids, mask)
            scheduler_output = self.noise_scheduler.step(noise_pred, t, x)
            x = scheduler_output.prev_sample
        
        # 将最终结果转换回ID
        return self.denormalize_ids(x.squeeze(-1))

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置超参数
    num_timesteps = 500
    batch_size = 64
    seq_len = 32
    num_ids = 250
    
    # 创建样本数据
    original_ids = torch.randint(1, num_ids + 1, (batch_size, seq_len)).to(device)
    condition_ids = original_ids.clone()  # 在实际应用中，这里应该是你的条件ID
    
    # 创建模型
    model = NormalizedTrajectoryTransformerDiffusion(
        num_ids=num_ids,
        embedding_dim=256,
        num_layers=6,
        learning_rate=1e-4,
        num_timesteps=num_timesteps
    ).to(device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 训练循环
    print("\n开始训练:")
    num_epochs = 1000
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # 准备批次数据
        batch = (original_ids, condition_ids, None)
        
        # 计算损失
        loss = model.training_step(batch, epoch)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
            
            # 生成样本
            model.eval()
            with torch.no_grad():
                generated_ids = model.sample(condition_ids)
                
                # 计算准确率
                correct = (generated_ids == original_ids).float().mean()
                print(f"准确率: {correct.item():.4f}")       
                # 打印样本对比
                if (epoch + 1) % 200 == 0:
                    print("\n样本对比 (前3个序列):")
                    print("原始ID:")
                    print(original_ids[:3].cpu().numpy())
                    print("生成ID:")
                    print(generated_ids[:3].cpu().numpy())

if __name__ == "__main__":
    main() 