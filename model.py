import torch
import torch.nn as nn
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from typing import Optional, List, Tuple
import numpy as np

class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.act = nn.SiLU()
        
        # 时间嵌入投影
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 时间嵌入
        time_emb = self.time_mlp(t)
        time_emb = time_emb.unsqueeze(-1)  # [B, C, 1]
        
        # 第一个卷积块
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        h = h + time_emb
        
        # 第二个卷积块
        h = self.conv2(h)
        h = self.norm2(h)
        h = self.act(h)
        
        return h

class DownBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.conv = ConvBlock1D(in_channels, out_channels, time_emb_dim)
        self.down = nn.Conv1d(out_channels, out_channels, 4, stride=2, padding=1)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.conv(x, t)
        return self.down(x), x

class UpBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_channels, out_channels, 4, stride=2, padding=1)
        self.conv = ConvBlock1D(out_channels, out_channels, time_emb_dim)
        
    def forward(self, x: torch.Tensor, skip: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self.conv(x, t)
        return x

class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int = 256,
        hidden_dims: List[int] = [128, 256, 512]
    ):
        super().__init__()
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # 初始卷积
        self.init_conv = nn.Conv1d(in_channels, hidden_dims[0], 3, padding=1)
        
        # 下采样路径
        self.down_blocks = nn.ModuleList([
            DownBlock1D(hidden_dims[i], hidden_dims[i+1], time_emb_dim)
            for i in range(len(hidden_dims)-1)
        ])
        
        # 中间块
        self.mid_conv1 = ConvBlock1D(hidden_dims[-1], hidden_dims[-1], time_emb_dim)
        self.mid_conv2 = ConvBlock1D(hidden_dims[-1], hidden_dims[-1], time_emb_dim)
        
        # 上采样路径
        self.up_blocks = nn.ModuleList([
            UpBlock1D(hidden_dims[i], hidden_dims[i-1], time_emb_dim)
            for i in range(len(hidden_dims)-1, 0, -1)
        ])
        
        # 输出卷积
        self.final_conv = nn.Conv1d(hidden_dims[0], out_channels, 1)
        
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # 时间嵌入
        t = self.time_mlp(t.unsqueeze(-1))
        
        # 初始特征
        x = self.init_conv(x)
        
        # 下采样路径
        skip_connections = []
        for down_block in self.down_blocks:
            x, skip = down_block(x, t)
            skip_connections.append(skip)
        
        # 中间块
        x = self.mid_conv1(x, t)
        x = self.mid_conv2(x, t)
        
        # 上采样路径
        for up_block in self.up_blocks:
            skip = skip_connections.pop()
            x = up_block(x, skip, t)
        
        # 输出
        return self.final_conv(x)

class IDEmbedding(nn.Module):
    def __init__(self, num_ids: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_ids + 2, embedding_dim)  # +2 for padding and missing tokens
        
    def forward(self, x):
        # x: [batch_size, seq_len, num_ids+2]
        # 获取one-hot编码中值为1的位置索引
        ids = torch.argmax(x, dim=-1)  # [batch_size, seq_len]
        return self.embedding(ids)  # [batch_size, seq_len, embedding_dim]

class TrajectoryDiffusionModel(pl.LightningModule):
    def __init__(
        self,
        num_ids: int = 3953,
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 4,
        learning_rate: float = 1e-4
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # 条件嵌入层
        self.id_embedding = IDEmbedding(num_ids, embedding_dim)
        
        # 计算输入通道数
        input_channels = num_ids + 2 + embedding_dim  # one-hot编码 + 嵌入维度
        
        # UNet模型
        self.model = UNet1D(
            in_channels=input_channels,
            out_channels=num_ids + 2,
            time_emb_dim=256,
            hidden_dims=[hidden_dim, hidden_dim * 2, hidden_dim * 4]  # 保持三层结构
        )
        
        # 扩散调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
        self.learning_rate = learning_rate
        
    def forward(self, x, timesteps, condition_mask):
        # 调整输入维度顺序 [batch_size, seq_len, channels] -> [batch_size, channels, seq_len]
        x = x.transpose(1, 2)
        
        # 获取条件嵌入
        condition_embedding = self.id_embedding(x.transpose(1, 2))  # 转回原始顺序进行嵌入
        
        # 调整条件嵌入维度顺序
        condition_embedding = condition_embedding.transpose(1, 2)
        
        # 将条件嵌入与输入拼接
        model_input = torch.cat([x, condition_embedding], dim=1)
        
        # 预测噪声
        noise_pred = self.model(model_input, timesteps)
        return noise_pred
    
    def training_step(self, batch, batch_idx):
        original_ids, x, masked_ids, mask = batch
        
        # 添加噪声
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (x.shape[0],), device=x.device).long()  # 使用long类型
        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)
        
        # 预测噪声
        noise_pred = self(noisy_x, timesteps.float(), mask)  # 在forward时转换为float类型
        
        # 计算所有位置的损失
        loss = nn.functional.mse_loss(noise_pred, noise.transpose(1, 2))
        
        # 使用float()确保损失值是标量
        self.log("train_loss", loss.item(), prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
    def sample(self, condition_mask, num_inference_steps=50):
        # 初始化为随机噪声
        x = torch.randn(1, self.hparams.num_ids + 2, 32, device=self.device)  # 注意维度顺序
        
        # 逐步去噪
        for t in self.noise_scheduler.timesteps:
            timesteps = torch.tensor([t], device=self.device, dtype=torch.float32)  # 确保是浮点类型
            noise_pred = self(x.transpose(1, 2), timesteps, condition_mask)  # 转回原始顺序
            x = self.noise_scheduler.step(noise_pred, t, x).prev_sample
            
        return x.transpose(1, 2)  # 转回原始顺序

def main():
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 创建示例数据
    batch_size = 4
    seq_len = 32
    num_ids = 3953
    
    # 生成随机的ID序列
    original_ids = torch.randint(1, num_ids + 1, (batch_size, seq_len))
    
    # 创建one-hot编码
    x = torch.zeros(batch_size, seq_len, num_ids + 2)
    for i in range(batch_size):
        for j in range(seq_len):
            x[i, j, original_ids[i, j]] = 1
    
    # 创建掩码（随机将一些位置设为缺失）
    mask = torch.ones(batch_size, seq_len)
    missing_positions = torch.rand(batch_size, seq_len) < 0.3  # 30%的位置设为缺失
    mask[missing_positions] = 0
    
    # 创建模型
    model = TrajectoryDiffusionModel(
        num_ids=num_ids,
        embedding_dim=64,
        hidden_dim=128,
        num_layers=4,
        learning_rate=1e-4
    )
    
    # 打印模型结构
    print("模型结构:")
    print(model)
    
    # 测试前向传播
    print("\n测试前向传播:")
    timesteps = torch.randint(0, 1000, (batch_size,)).float()
    output = model(x, timesteps, mask)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    
    # 测试采样
    print("\n测试采样:")
    sampled = model.sample(mask)
    print(f"采样结果形状: {sampled.shape}")
    
    # 计算损失
    print("\n测试训练步骤:")
    loss = model.training_step((original_ids, x, x, mask), 0)
    print(f"训练损失: {loss.item():.4f}")

if __name__ == "__main__":
    main() 