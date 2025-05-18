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
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        return x + self.pe[:x.size(0)]

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim=128):
        super().__init__()
        assert dim % 2 == 0, "嵌入维度需要是偶数"
        half_dim = dim // 2
        
        # 预计算指数衰减系数 log(10000)/ (half_dim -1)
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer('emb', emb)  # (half_dim,)

        # 可选的后续处理层
        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, t):
        """
        输入参数:
            t : 时间步张量, 形状为 (batch_size,)
        
        输出:
            嵌入向量, 形状为 (batch_size, dim)
        """
        # 归一化到[0,1]
        t = t.float() / self.emb.numel()  # 建议使用1000步时不用此归一化
        
        # 计算位置编码
        emb = t.unsqueeze(-1) * self.emb.unsqueeze(0)  # (batch_size, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (batch_size, dim)
        
        # 可选：通过MLP增强表达能力
        return self.proj(emb)

class TimeEmbedding(nn.Module):
    def __init__(self, dim=256):
        super().__init__()
        assert dim % 2 == 0, "嵌入维度需要是偶数"
        half_dim = dim // 2
        
        # 预计算指数衰减系数
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        self.register_buffer('emb', emb)
        
        # MLP层
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
    
    def forward(self, t):
        """
        输入参数:
            t : 时间步张量, 形状为 (batch_size,)
        输出:
            嵌入向量, 形状为 (batch_size, dim)
        """
        # 计算位置编码
        t = t.float() / self.emb.numel()
        emb = t.unsqueeze(-1) * self.emb.unsqueeze(0)  # (batch_size, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)  # (batch_size, dim)
        
        # 通过MLP增强表达能力
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
        time_emb: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        # Self attention
        x2 = self.norm1(x)
        x = x + self.dropout1(self.self_attn(x2, x2, x2)[0])
        
        # Cross attention with condition
        x2 = self.norm2(x)
        x = x + self.dropout2(self.cross_attn(x2, condition, condition)[0])
        
        # Feed forward
        x2 = self.norm3(x)
        x = x + self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(x2)))))
        
        return x

class TransformerDiffusion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        # 输入投影
        self.input_proj = nn.Linear(num_classes, d_model)
        
        # 时间嵌入
        self.time_embedding = TimeEmbedding(d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerDiffusionBlock(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])
        
        # 输出投影
        self.output_proj = nn.Linear(d_model, num_classes)
        
    def forward(
        self,
        x: torch.Tensor,  # [B, seq_len, num_classes]
        timesteps: torch.Tensor,  # [B]
        condition: torch.Tensor,  # [B, seq_len, d_model]
    ) -> torch.Tensor:
        # 输入投影
        x = self.input_proj(x)  # [B, seq_len, d_model]
        
        # 时间嵌入
        t_emb = self.time_embedding(timesteps)  # [B, d_model]
        
        # 扩展时间嵌入到序列长度
        t_emb = t_emb.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B, seq_len, d_model]
        
        # 添加时间信息
        x = x + t_emb
        
        # 添加位置编码
        x = x.transpose(0, 1)  # [seq_len, B, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [B, seq_len, d_model]
        
        # Transformer层
        for layer in self.layers:
            x = layer(x, t_emb, condition)
        
        # 输出投影
        x = self.output_proj(x)  # [B, seq_len, num_classes]
        
        return x

class IDEmbedding(nn.Module):
    def __init__(self, num_ids: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_ids + 2, embedding_dim)  # +2 for padding and missing tokens
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        
    def forward(self, ids):
        # ids: [batch_size, seq_len]
        x = self.embedding(ids)  # [batch_size, seq_len, embedding_dim]
        
        # 添加位置编码
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        # Transformer编码
        x = self.transformer(x)
        
        return x

class TrajectoryTransformerDiffusion(pl.LightningModule):
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
        
        # 条件嵌入层
        self.id_embedding = IDEmbedding(num_ids, embedding_dim)
        
        # Transformer扩散模型
        self.model = TransformerDiffusion(
            num_classes=num_ids + 2,  # one-hot编码维度
            d_model=embedding_dim,
            nhead=8,
            num_layers=num_layers,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1
        )
        
        # 扩散调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
        self.learning_rate = learning_rate
        self.num_timesteps = num_timesteps
        
    def forward(self, noisy_x, timesteps, masked_ids, mask):
        # 获取条件嵌入
        condition_embedding = self.id_embedding(masked_ids)  # [B, seq_len, embedding_dim]
        
        # 预测噪声
        noise_pred = self.model(noisy_x, timesteps, condition_embedding)
        return noise_pred
    
    def training_step(self, batch, batch_idx):
        original_ids, x0, masked_ids, mask = batch
        
        # 添加噪声
        noise = torch.randn_like(x0)
        timesteps = torch.randint(0, self.num_timesteps, (x0.shape[0],), device=x0.device).long()
        noisy_x = self.noise_scheduler.add_noise(x0, noise, timesteps)
        
        # 预测噪声
        noise_pred = self(noisy_x, timesteps, masked_ids, mask)
        
        # 计算损失
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        self.log("train_loss", loss.item(), prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
    def sample(self, masked_ids, mask):
        device = next(self.parameters()).device
        batch_size = masked_ids.shape[0]
        seq_len = masked_ids.shape[1]
        
        # 初始化噪声
        x = torch.randn(
            (batch_size, seq_len, self.hparams.num_ids + 2),
            device=device
        )
        
        # 逐步去噪
        for t in tqdm(self.noise_scheduler.timesteps):
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            
            # 预测噪声
            noise_pred = self(x, timesteps, masked_ids, mask)
            
            # 去噪步骤
            scheduler_output = self.noise_scheduler.step(noise_pred, t, x)
            x = scheduler_output.prev_sample
            
            # 如果是最后一个时间步，对预测结果进行处理
            if t == self.noise_scheduler.timesteps[-1]:
                x = torch.nn.functional.softmax(x, dim=2)
        
        return x

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置超参数
    num_timesteps = 500
    batch_size = 64
    seq_len = 32
    num_ids = 250
    
    # 创建样本数据
    original_ids = torch.randint(1, num_ids + 1, (batch_size, seq_len)).to(device)
    x0 = torch.zeros(batch_size, seq_len, num_ids + 2).to(device)
    for i in range(batch_size):
        for j in range(seq_len):
            x0[i, j, original_ids[i, j]] = 1
    
    # 创建模型
    model = TrajectoryTransformerDiffusion(
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
    
    # 用于记录loss和准确率的列表
    losses = []
    top1_accuracies = []
    top5_accuracies = []
    top10_accuracies = []
    
    def evaluate_model():
        model.eval()
        with torch.no_grad():
            final_predictions = model.sample(original_ids, None)
            
            # 计算top-k准确率
            topk_values = [1, 5, 10]
            topk_correct = torch.zeros(len(topk_values), device=final_predictions.device)
            
            # 获取top-k预测
            _, topk_indices = torch.topk(final_predictions, k=max(topk_values), dim=2)
            
            # 计算每个k的准确率
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
        
        # 添加噪声
        timesteps = torch.randint(0, model.num_timesteps, (batch_size,), device=device).long()
        noise = torch.randn_like(x0)
        noisy_x = model.noise_scheduler.add_noise(x0, noise, timesteps)
        
        # 前向传播
        noise_pred = model(noisy_x, timesteps, original_ids, None)
        
        # 计算损失
        loss = nn.functional.mse_loss(noise_pred, noise)
        
        # 记录loss
        losses.append(loss.item())
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 打印训练信息
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
            print(f"\n=== 第 {epoch + 1} 轮测试 ===")
            accuracies = evaluate_model()
            top1_accuracies.append(accuracies[0])
            top5_accuracies.append(accuracies[1])
            top10_accuracies.append(accuracies[2])
    
    # 绘制loss和准确率曲线
    import matplotlib.pyplot as plt
    
    # 绘制loss曲线
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('Training Loss Curve')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    x = list(range(50, num_epochs + 1, 50))
    plt.plot(x, top1_accuracies, label='Top-1')
    plt.plot(x, top5_accuracies, label='Top-5')
    plt.plot(x, top10_accuracies, label='Top-10')
    plt.title('Accuracy Curves')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

    # 最终测试
    print("\n=== 最终测试结果 ===")
    model.eval()
    with torch.no_grad():
        final_predictions = model.sample(original_ids, None)
        
        # 打印一些样本进行对比
        print("\n样本对比 (前5个序列的前10个时间步):")
        print("原始ID:")
        print(original_ids[:5, :10].cpu().numpy())
        print("\n生成ID (Top-1预测):")
        _, topk_indices = torch.topk(final_predictions, k=1, dim=2)
        print(topk_indices[:5, :10, 0].cpu().numpy())
        
        # 打印详细的top-k预测示例
        print("\n详细预测示例 (第一个序列的前3个时间步):")
        for step in range(3):
            print(f"\n时间步 {step}:")
            probs = torch.softmax(final_predictions[0, step], dim=0)
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