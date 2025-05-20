import torch
import torch.nn as nn
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from typing import Optional, List, Tuple
import numpy as np
from tqdm import tqdm
import math

"""
模型：UNet1D
数据：
    原始数据：ID序列
    条件数据：ID序列
方案：
    id嵌入和扩散模型一起训练。
    对嵌入后的id做加噪、去噪处理

"""

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        # 条件嵌入投影
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )
        
    def forward(self, x: torch.Tensor, fused_condition: torch.Tensor) -> torch.Tensor:
        condition_emb = self.condition_mlp(fused_condition)
        condition_emb = condition_emb.unsqueeze(-1).expand(-1, -1, x.shape[-1])
        
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = h + condition_emb
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h

class ConditionFusion(nn.Module):
    def __init__(self, time_dim=256, cond_dim=64):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, cond_dim)
        self.conv_compress = nn.Conv1d(cond_dim, cond_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, time_emb, cond_emb):
        time_feat = self.time_proj(time_emb)
        time_feat = time_feat.unsqueeze(1).expand(-1, cond_emb.shape[1], -1)
        fused = time_feat + cond_emb
        
        fused = fused.transpose(1, 2)
        fused = self.conv_compress(fused)
        fused = self.pool(fused)
        fused = fused.squeeze(-1)
        
        return fused

class DownBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
        super().__init__()
        self.down = nn.Conv1d(in_channels, out_channels, 4, stride=2, padding=1)
        self.conv = ConvBlock1D(out_channels, out_channels, condition_dim, stride=1)
        
    def forward(self, x: torch.Tensor, fused_condition: torch.Tensor) -> torch.Tensor:
        x = self.down(x)
        x = self.conv(x, fused_condition)
        return x

class UpBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, 4, stride=2, padding=1)
        self.concat_conv = nn.Conv1d(out_channels * 2, out_channels, 1)
        self.conv = ConvBlock1D(out_channels, out_channels, condition_dim, stride=1)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor, fused_condition: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.concat_conv(x)
        x = self.conv(x, fused_condition)
        return x

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        assert dim % 2 == 0, "嵌入维度需要是偶数"
        half_dim = dim // 2
        
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer('emb', emb)

        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, t):
        t = t.float() / self.emb.numel()
        emb = t.unsqueeze(-1) * self.emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.proj(emb)

class IDEmbedding(nn.Module):
    def __init__(self, num_ids: int, embedding_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(num_ids + 2, embedding_dim)
        
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, ids):
        emb = self.embedding(ids)
        lstm_out, _ = self.bilstm(emb)
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        out = lstm_out + attn_out
        out = self.proj(out)
        return out

class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int = 256,
        condition_dim: int = 64,
        hidden_dims: List[int] = [128, 256, 512]
    ):
        super().__init__()
        
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.fusion = ConditionFusion(time_emb_dim, condition_dim)
        self.init_conv = nn.Conv1d(in_channels, hidden_dims[0], 3, padding=1)
        
        self.down_blocks = nn.ModuleList([
            DownBlock1D(hidden_dims[i], hidden_dims[i+1], condition_dim)
            for i in range(len(hidden_dims)-1)
        ])
        
        self.mid_conv1 = ConvBlock1D(hidden_dims[-1], hidden_dims[-1], condition_dim)
        self.mid_conv2 = ConvBlock1D(hidden_dims[-1], hidden_dims[-1], condition_dim)
        
        self.up_blocks = nn.ModuleList([
            UpBlock1D(hidden_dims[i], hidden_dims[i-1], condition_dim)
            for i in range(len(hidden_dims)-1, 0, -1)
        ])
        
        self.final_conv = nn.Conv1d(hidden_dims[0], out_channels, 1)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        t = self.time_embedding(t)
        condition = condition.transpose(1, 2)
        fused_condition = self.fusion(t, condition)
        
        x = self.init_conv(x)
        
        skip_connections = []
        for down_block in self.down_blocks:
            skip_connections.append(x)
            x = down_block(x, fused_condition)
        
        x = self.mid_conv1(x, fused_condition)
        x = self.mid_conv2(x, fused_condition)
        
        for up_block in self.up_blocks:
            skip = skip_connections.pop()
            x = up_block(x, skip, fused_condition)
        
        return self.final_conv(x)

class EmbeddingDiffusionModel(pl.LightningModule):
    def __init__(
        self,
        num_ids: int = 3953,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        learning_rate: float = 1e-4,
        num_timesteps: int = 100
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # ID嵌入层
        self.id_embedding = IDEmbedding(num_ids, embedding_dim, hidden_dim)
        
        # UNet模型 - 注意输入输出通道数现在是embedding_dim
        self.model = UNet1D(
            in_channels=embedding_dim,  # 输入是嵌入向量
            out_channels=embedding_dim,  # 输出也是嵌入向量
            time_emb_dim=256,
            condition_dim=embedding_dim,
            hidden_dims=[hidden_dim, hidden_dim * 2, hidden_dim * 4]
        )
        
        # 用于将嵌入映射回ID空间的投影层
        self.proj_to_logits = nn.Linear(embedding_dim, num_ids + 2)
        
        # 扩散调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
        self.learning_rate = learning_rate
        self.num_timesteps = num_timesteps
        
    def forward(self, noisy_emb, timesteps, masked_ids, mask):
        """
        前向传播过程
        
        Args:
            noisy_emb: 带噪声的嵌入 [B, 32, embedding_dim]
            timesteps: 时间步 [B]
            masked_ids: 带缺失值的ID序列 [B, 32]
            mask: 掩码 [B, 32]
        
        Returns:
            预测的噪声 [B, embedding_dim, 32]
        """
        # 获取条件嵌入
        condition_embedding = self.id_embedding(masked_ids)
        
        # 调整维度顺序
        noisy_emb = noisy_emb.transpose(1, 2)  # [B, embedding_dim, 32]
        condition_embedding = condition_embedding.transpose(1, 2)  # [B, embedding_dim, 32]
        
        # 预测噪声
        noise_pred = self.model(noisy_emb, condition_embedding, timesteps)
        return noise_pred
    
    def training_step(self, batch, batch_idx):
        original_ids, masked_ids, mask = batch
        
        # 获取原始ID的嵌入表示
        x0_emb = self.id_embedding(original_ids)  # [B, 32, embedding_dim]
        
        # 添加噪声
        noise = torch.randn_like(x0_emb)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (x0_emb.shape[0],), device=x0_emb.device).long()
        noisy_emb = self.noise_scheduler.add_noise(x0_emb, noise, timesteps)
        
        # 预测噪声
        noise_pred = self(noisy_emb, timesteps, masked_ids, mask)
        
        # 计算损失
        loss = nn.functional.mse_loss(noise_pred, noise.transpose(1, 2))
        
        self.log("train_loss", loss.item(), prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
    def sample(self, masked_ids, mask):
        """在嵌入空间中采样生成轨迹"""
        device = next(self.parameters()).device
        batch_size = masked_ids.shape[0]
        seq_len = masked_ids.shape[1]
        
        # 初始化噪声（在嵌入空间）
        x = torch.randn(
            (batch_size, seq_len, self.hparams.embedding_dim),
            device=device
        )
        
        # 逐步去噪
        for t in tqdm(self.noise_scheduler.timesteps):
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            
            # 预测噪声
            noise_pred = self(x, timesteps, masked_ids, mask)
            
            # 去噪步骤
            scheduler_output = self.noise_scheduler.step(noise_pred.transpose(1, 2), t, x)
            x = scheduler_output.prev_sample
            
            # 如果是最后一个时间步，将嵌入映射回ID空间
            if t == self.noise_scheduler.timesteps[-1]:
                # 将嵌入投影到ID空间
                logits = self.proj_to_logits(x)
                # 应用softmax得到概率分布
                probs = torch.softmax(logits, dim=-1)
                return probs

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置超参数
    num_timesteps = 500
    batch_size = 64
    seq_len = 32
    num_ids = 250
    embedding_dim = 128
    
    # 创建训练数据
    original_ids = torch.randint(1, num_ids + 1, (batch_size, seq_len)).to(device)
    masked_ids = original_ids.clone()
    mask = torch.ones_like(original_ids, dtype=torch.bool)
    
    # 创建模型
    model = EmbeddingDiffusionModel(
        num_ids=num_ids,
        embedding_dim=embedding_dim,
        hidden_dim=256,
        num_layers=6,
        learning_rate=1e-5,
        num_timesteps=num_timesteps
    ).to(device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    
    # 训练循环
    print("\n开始训练:")
    num_epochs = 1000
    
    # 用于记录loss和准确率的列表
    losses = []
    top1_accuracies = []
    top5_accuracies = []
    top10_accuracies = []
    
    def evaluate_model():
        model.eval()
        with torch.no_grad():
            final_predictions = model.sample(masked_ids, mask)
            
            topk_values = [1, 5, 10]
            topk_correct = torch.zeros(len(topk_values), device=final_predictions.device)
            
            _, topk_indices = torch.topk(final_predictions, k=max(topk_values), dim=2)
            
            for i, k in enumerate(topk_values):
                correct = torch.any(topk_indices[:, :, :k] == original_ids.unsqueeze(-1), dim=-1)
                topk_correct[i] = correct.float().mean()
            
            print(f"\n当前测试结果:")
            for i, k in enumerate(topk_values):
                print(f"Top-{k} 准确率: {topk_correct[i].item():.4f}")
            
            return topk_correct.cpu().numpy()
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        
        # 获取原始ID的嵌入表示
        x0_emb = model.id_embedding(original_ids)
        
        # 添加噪声
        timesteps = torch.randint(0, model.num_timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(x0_emb)
        noisy_emb = model.noise_scheduler.add_noise(x0_emb, noise, timesteps)
        
        # 前向传播
        noise_pred = model(noisy_emb, timesteps, masked_ids, mask)
        
        # 计算损失
        loss = nn.functional.mse_loss(noise_pred, noise.transpose(1, 2))
        losses.append(loss.item())
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
        
        if (epoch + 1) % 100 == 0:  # 每100轮进行一次评估
            print(f"\n=== 第 {epoch + 1} 轮测试 ===")
            accuracies = evaluate_model()
            top1_accuracies.append(accuracies[0])
            top5_accuracies.append(accuracies[1])
            top10_accuracies.append(accuracies[2])
    
    # 绘制loss和准确率曲线
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('训练损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    x = list(range(100, num_epochs + 1, 100))  # 每100轮一个点
    plt.plot(x, top1_accuracies, label='Top-1')
    plt.plot(x, top5_accuracies, label='Top-5')
    plt.plot(x, top10_accuracies, label='Top-10')
    plt.title('准确率曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('准确率')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # 最终测试
    print("\n=== 最终测试结果 ===")
    model.eval()
    with torch.no_grad():
        final_predictions = model.sample(masked_ids, mask)
        
        print("\n样本对比 (前5个序列的前10个时间步):")
        print("原始ID:")
        print(original_ids[:5, :10].cpu().numpy())
        print("\n生成ID (Top-1预测):")
        _, topk_indices = torch.topk(final_predictions, k=1, dim=2)
        print(topk_indices[:5, :10, 0].cpu().numpy())
        
        print("\n详细预测示例 (第一个序列的前3个时间步):")
        for step in range(3):
            print(f"\n时间步 {step}:")
            probs = final_predictions[0, step]
            top_probs, top_ids = torch.topk(probs, k=10)
            for j in range(10):
                print(f"ID {top_ids[j].item():4d}: {top_probs[j].item():.4f}", end="  ")
                if j == 0:
                    print("(Top-1)", end="")
                elif j == 4:
                    print("(Top-5)", end="")
                elif j == 9:
                    print("(Top-10)", end="")
            print()

if __name__ == "__main__":
    main() 