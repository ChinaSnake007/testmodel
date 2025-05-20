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
    用one-hot向量表示id序列，对one-hot向量做加噪、去噪处理
"""


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        # 条件嵌入投影，将条件嵌入投影到对应维度
        self.condition_mlp = nn.Sequential(
            nn.Linear(condition_dim, out_channels),
            nn.SiLU(),
            nn.Linear(out_channels, out_channels)
        )
        
    def forward(self, x: torch.Tensor, fused_condition: torch.Tensor) -> torch.Tensor:
        # 条件嵌入投影
        condition_emb = self.condition_mlp(fused_condition)  # [B, out_channels]
        # 重复到与x相同的序列长度
        condition_emb = condition_emb.unsqueeze(-1).expand(-1, -1, x.shape[-1])  # [B, out_channels, seq_len]
        
        # 第一个卷积块
        h = self.conv1(x)  # [B, out_channels, seq_len]
        h = self.norm1(h)
        h = self.act(h)
        h = h + condition_emb  # 维度完全匹配
        
        # 第二个卷积块
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h

class ConditionFusion(nn.Module):
    def __init__(self, time_dim=256, cond_dim=64):
        super().__init__()
        self.time_proj = nn.Linear(time_dim, cond_dim)
        # 使用1D卷积压缩序列维度
        self.conv_compress = nn.Conv1d(cond_dim, cond_dim, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, time_emb, cond_emb):
        # time_emb: [B, time_dim]
        # cond_emb: [B, seq_len, cond_dim]
        
        # 投影时间嵌入
        time_feat = self.time_proj(time_emb)  # [B, cond_dim]
        
        # 调整维度并相加
        time_feat = time_feat.unsqueeze(1).expand(-1, cond_emb.shape[1], -1)  # [B, seq_len, cond_dim]
        fused = time_feat + cond_emb  # [B, seq_len, cond_dim]
        
        # 转换维度用于卷积
        fused = fused.transpose(1, 2)  # [B, cond_dim, seq_len]
        fused = self.conv_compress(fused)  # [B, cond_dim, seq_len]
        
        # 使用自适应平均池化压缩序列维度
        fused = self.pool(fused)  # [B, cond_dim, 1]
        fused = fused.squeeze(-1)  # [B, cond_dim]
        
        return fused

class DownBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
        super().__init__()
        # 下采样层
        self.down = nn.Conv1d(in_channels, out_channels, 4, stride=2, padding=1)
        self.conv = ConvBlock1D(out_channels, out_channels, condition_dim, stride=1)
        
    def forward(self, x: torch.Tensor, fused_condition: torch.Tensor) -> torch.Tensor:
        # 下采样特征
        x = self.down(x)  # [B, out_channels, seq_len/2]
        x = self.conv(x, fused_condition)  # fused_condition保持[B, embedding_dim]不变
        return x

class UpBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, condition_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, 4, stride=2, padding=1)
        # 添加处理拼接特征的卷积层
        self.concat_conv = nn.Conv1d(out_channels * 2, out_channels, 1)
        self.conv = ConvBlock1D(out_channels, out_channels, condition_dim, stride=1)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor, fused_condition: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # 使用拼接操作替代加法
        x = torch.cat([x, skip], dim=1)  # 在通道维度上拼接
        x = self.concat_conv(x)  # 使用1x1卷积调整通道数
        x = self.conv(x, fused_condition)  # fused_condition保持[B, embedding_dim]不变
        return x

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
        
        # 时间嵌入
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        
        # 时空融合模块
        self.fusion = ConditionFusion(time_emb_dim, condition_dim)
        
        # 初始卷积
        self.init_conv = nn.Conv1d(in_channels, hidden_dims[0], 3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList([
            DownBlock1D(hidden_dims[i], hidden_dims[i+1], condition_dim)
            for i in range(len(hidden_dims)-1)
        ])
        
        # 中间块
        self.mid_conv1 = ConvBlock1D(hidden_dims[-1], hidden_dims[-1], condition_dim)
        self.mid_conv2 = ConvBlock1D(hidden_dims[-1], hidden_dims[-1], condition_dim)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList([
            UpBlock1D(hidden_dims[i], hidden_dims[i-1], condition_dim)
            for i in range(len(hidden_dims)-1, 0, -1)
        ])
        
        # 输出卷积
        self.final_conv = nn.Conv1d(hidden_dims[0], out_channels, 1)
        
    def forward(self, x: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 时间嵌入
        t = self.time_embedding(t)  # [B, time_emb_dim]
        
        # 调整条件嵌入维度
        condition = condition.transpose(1, 2)  # [B, seq_len, embedding_dim]
        
        # 时空融合
        fused_condition = self.fusion(t, condition)  # [B, embedding_dim]
        
        # 初始特征
        x = self.init_conv(x)  # [B, hidden_dims[0], seq_len]
        
        # 下采样路径
        skip_connections = []
        for down_block in self.down_blocks:
            skip_connections.append(x)
            x = down_block(x, fused_condition)
        
        # 中间块
        x = self.mid_conv1(x, fused_condition)
        x = self.mid_conv2(x, fused_condition)
        
        # 上采样路径
        for up_block in self.up_blocks:
            skip = skip_connections.pop()
            x = up_block(x, skip, fused_condition)
        
        # 输出
        return self.final_conv(x)

class IDEmbedding(nn.Module):
    def __init__(self, num_ids: int, embedding_dim: int, hidden_dim: int = 128):
        """
        初始化ID嵌入模块
        
        Args:
            num_ids: ID的总数量
            embedding_dim: ID嵌入向量的维度
            hidden_dim: LSTM隐藏层的维度,默认为128
        """
        super().__init__()
        self.embedding = nn.Embedding(num_ids + 2, embedding_dim)  # +2 for padding and missing tokens
        
        # BiLSTM层
        self.bilstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1
        )
        
        # 自注意力层
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # BiLSTM的输出维度是hidden_dim*2
            num_heads=4,
            dropout=0.1,
            batch_first=True
        )
        
        # 输出投影层
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.Dropout(0.1)
        )
        
    def forward(self, ids):
        # ids: [batch_size, seq_len]
        # 获取词嵌入
        emb = self.embedding(ids)  # [batch_size, seq_len, embedding_dim]
        
        # BiLSTM处理
        lstm_out, _ = self.bilstm(emb)  # [batch_size, seq_len, hidden_dim*2]
        
        # 自注意力处理
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # [batch_size, seq_len, hidden_dim*2]
        
        # 残差连接
        out = lstm_out + attn_out
        
        # 投影到所需维度
        out = self.proj(out)  # [batch_size, seq_len, embedding_dim]
        
        return out

class TrajectoryDiffusionModel(pl.LightningModule):
    def __init__(
        self,
        num_ids: int = 3953,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        learning_rate: float = 1e-4,
        num_timesteps: int = 100  # 添加时间步数参数
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 条件嵌入层
        self.id_embedding = IDEmbedding(num_ids, embedding_dim, hidden_dim)
        
        # UNet模型
        self.model = UNet1D(
            in_channels=num_ids + 2,  # one-hot编码维度
            out_channels=num_ids + 2,
            time_emb_dim=256,
            condition_dim=embedding_dim,  # 条件嵌入维度
            hidden_dims=[hidden_dim, hidden_dim * 2, hidden_dim * 4]  # 保持三层结构
        )
        
        # 扩散调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,  # 使用传入的时间步数
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
        self.learning_rate = learning_rate
        self.num_timesteps = num_timesteps  # 保存时间步数
        
    def forward(self, noisy_x, timesteps, masked_ids, mask):
        """
        前向传播过程
        
        Args:
            noisy_x: 带噪声的输入 [B, 32, 3954]
            timesteps: 时间步 [B]
            masked_ids: 带缺失值的ID序列 [B, 32]
            mask: 掩码 [B, 32]
        
        Returns:
            预测的噪声 [B, 3954, 32]
        """
        # 获取条件嵌入
        condition_embedding = self.id_embedding(masked_ids)  # [B, 32, embedding_dim]
        
        # 调整维度顺序
        noisy_x = noisy_x.transpose(1, 2)  # [B, 3954, 32]
        condition_embedding = condition_embedding.transpose(1, 2)  # [B, embedding_dim, 32]
        
        # 预测噪声
        noise_pred = self.model(noisy_x, condition_embedding, timesteps)
        return noise_pred
    
    def training_step(self, batch, batch_idx):
        original_ids, x0, masked_ids, mask = batch  # x0已经是one-hot编码 [B, 32, 3954]
        
        # 添加噪声
        noise = torch.randn_like(x0)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (x0.shape[0],), device=x0.device).long()
        noisy_x = self.noise_scheduler.add_noise(x0, noise, timesteps)
        
        # 预测噪声
        noise_pred = self(noisy_x, timesteps, masked_ids, mask)
        
        # 计算所有位置的损失
        loss = nn.functional.mse_loss(noise_pred, noise.transpose(1, 2))
        
        # 使用float()确保损失值是标量
        self.log("train_loss", loss.item(), prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
    def sample(self, masked_ids, mask):
        """采样生成轨迹"""
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
            # 将时间步转换为张量并移动到正确的设备
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            
            # 预测噪声
            noise_pred = self(x, timesteps, masked_ids, mask)
            
            # 去噪步骤
            scheduler_output = self.noise_scheduler.step(noise_pred.transpose(1, 2), t, x)
            x = scheduler_output.prev_sample
            
            # 如果是最后一个时间步，对预测结果进行处理
            if t == self.noise_scheduler.timesteps[-1]:
                # 对最终预测进行softmax处理
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
    
    
    # 创建单个样本数据
    original_ids = torch.randint(1, num_ids + 1, (batch_size, seq_len)).to(device)
    
    # 创建one-hot编码
    x0 = torch.zeros(batch_size, seq_len, num_ids + 2).to(device)
    for i in range(batch_size):
        for j in range(seq_len):
            x0[i, j, original_ids[i, j]] = 1
    
    # 创建模型
    model = TrajectoryDiffusionModel(
        num_ids=num_ids,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=6,
        learning_rate=1e-5,
        num_timesteps=num_timesteps  # 传入时间步数
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
            # 使用训练数据进行测试
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
        loss = nn.functional.mse_loss(noise_pred, noise.transpose(1, 2))
        
        # 记录loss
        losses.append(loss.item())
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # 打印训练信息
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
        
        # 每1000次迭代进行一次测试
        if (epoch + 1) % 1000 == 0:
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
    x = list(range(1000, num_epochs + 1, 1000))
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