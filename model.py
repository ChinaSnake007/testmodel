import torch
import torch.nn as nn
import pytorch_lightning as pl
from diffusers import DDPMScheduler, UNet1DModel
from typing import Optional

class IDEmbedding(nn.Module):
    def __init__(self, num_ids: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_ids + 2, embedding_dim)  # +2 for padding and missing tokens
        
    def forward(self, x):
        # x: [batch_size, seq_len]
        return self.embedding(x)  # [batch_size, seq_len, embedding_dim]

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
        
        # UNet模型
        self.model = UNet1DModel(
            sample_size=30,  # 最大序列长度
            in_channels=num_ids + 2,  # one-hot编码维度 (+2 for padding and missing tokens)
            out_channels=num_ids + 2,
            layers_per_block=2,
            block_out_channels=(hidden_dim, hidden_dim * 2, hidden_dim * 4),
            down_block_types=("DownBlock1D", "DownBlock1D", "DownBlock1D"),
            up_block_types=("UpBlock1D", "UpBlock1D", "UpBlock1D"),
            mid_block_type="UNetMidBlock1D",
            norm_num_groups=8,
        )
        
        # 扩散调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
        
        self.learning_rate = learning_rate
        
    def forward(self, x, timesteps, condition_mask):
        # 获取条件嵌入
        condition_embedding = self.id_embedding(x)
        
        # 将条件嵌入与输入拼接
        model_input = torch.cat([x, condition_embedding], dim=-1)
        
        # 预测噪声
        noise_pred = self.model(model_input, timesteps)
        return noise_pred
    
    def training_step(self, batch, batch_idx):
        original_ids, x, masked_ids, mask = batch
        
        # 添加噪声
        noise = torch.randn_like(x)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (x.shape[0],), device=x.device)
        noisy_x = self.noise_scheduler.add_noise(x, noise, timesteps)
        
        # 预测噪声
        noise_pred = self(noisy_x, timesteps, mask)
        
        # 计算损失（只计算有效位置的损失）
        loss = nn.functional.mse_loss(noise_pred * mask.unsqueeze(-1), noise * mask.unsqueeze(-1))
        
        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
    def sample(self, condition_mask, num_inference_steps=50):
        # 初始化为随机噪声
        x = torch.randn(1, 30, self.hparams.num_ids + 2, device=self.device)
        
        # 逐步去噪
        for t in self.noise_scheduler.timesteps:
            timesteps = torch.full((1,), t, device=self.device)
            noise_pred = self(x, timesteps, condition_mask)
            x = self.noise_scheduler.step(noise_pred, t, x).prev_sample
            
        return x 