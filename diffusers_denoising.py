import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from diffusers import DDPMScheduler, UNet2DModel
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from pathlib import Path
from pytorch_lightning.callbacks import Callback
"""
数据：一张图片
模型：diffusers 的 UNet2DModel
生成：无条件生成

"""

class DiffusionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # 使用 diffusers 的 UNet2DModel，添加注意力机制
        self.model = UNet2DModel(
            sample_size=128,           # 图片大小
            in_channels=3,            # 输入通道数
            out_channels=3,           # 输出通道数
            layers_per_block=2,       # 每个块的层数
            block_out_channels=(128, 256, 512, 512),  # 每个块的输出通道数
            down_block_types=(
                "DownBlock2D",
                "DownBlock2D",
                "AttnDownBlock2D",    # 添加注意力机制
                "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D",
                "AttnUpBlock2D",      # 添加注意力机制
                "UpBlock2D",
                "UpBlock2D"
            ),
        )
        
        # 初始化噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon"
        )

    def forward(self, x, t):
        return self.model(x, t).sample

    def training_step(self, batch, batch_idx):
        clean_images = batch
        
        # 采样随机时间步
        batch_size = clean_images.shape[0]
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (batch_size,), device=self.device)
        
        # 添加噪声
        noise = torch.randn_like(clean_images)
        noisy_images = self.noise_scheduler.add_noise(clean_images, noise, timesteps)
        
        # 预测噪声
        noise_pred = self(noisy_images, timesteps)
        loss = F.mse_loss(noise_pred, noise)
        
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer

    def sample(self, batch_size=1, device="cuda"):
        # 从随机噪声开始采样
        image = torch.randn(batch_size, 3, 128, 128).to(device)
        
        # 逐步去噪
        for t in reversed(range(self.noise_scheduler.num_train_timesteps)):
            timesteps = torch.tensor([t], device=device).long()
            timesteps = timesteps.repeat(batch_size)
            
            # 预测噪声
            with torch.no_grad():
                noise_pred = self(image, timesteps)
            
            # 使用调度器进行去噪步骤
            image = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=image
            ).prev_sample
        
        # 将图像转换到[0,1]范围
        image = (image.clamp(-1, 1) + 1) / 2
        return image

class MultiImageDataset(Dataset):
    def __init__(self, image_dir, image_size=32):
        self.image_size = image_size
        self.image_paths = list(Path(image_dir).glob('*.jpg')) + list(Path(image_dir).glob('*.png'))
        self.images = []
        
        # 预加载所有图片
        for img_path in self.image_paths:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.image_size, self.image_size))
            img = torch.from_numpy(img).float() / 255.0
            img = img.permute(2, 0, 1)
            self.images.append(img)
            
    def __len__(self):
        return len(self.images) * 100

    def __getitem__(self, idx):
        return self.images[idx % len(self.images)]

class GenerateImageCallback(Callback):
    def __init__(self):
        super().__init__()
        
    def on_train_epoch_end(self, trainer, pl_module):
        current_epoch = trainer.current_epoch + 1
        # 每100个epoch进行一次采样
        if current_epoch % 10 == 0:
            # 创建特定epoch的输出目录
            output_dir = f"diffusers_generated/epoch_{current_epoch}"
            os.makedirs(output_dir, exist_ok=True)
            
            # 设置设备
            device = pl_module.device
            pl_module.eval()
            
            # 生成100张图片
            num_images = 25
            batch_size = 25
            num_batches = num_images // batch_size
            
            for batch_idx in range(num_batches):
                with torch.no_grad():
                    samples = pl_module.sample(batch_size=batch_size, device=device)
                samples = samples.cpu().permute(0, 2, 3, 1).numpy()
                
                # 保存这批图片
                for i, sample in enumerate(samples):
                    img_idx = batch_idx * batch_size + i
                    sample = (sample * 255).astype(np.uint8)
                    sample = cv2.cvtColor(sample, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(os.path.join(output_dir, f'generated_{img_idx:03d}.jpg'), sample)
            
            # 显示一些生成的样本
            plt.figure(figsize=(15, 3))
            for i in range(5):
                plt.subplot(1, 5, i + 1)
                generated_img = cv2.imread(os.path.join(output_dir, f'generated_{i:03d}.jpg'))
                generated_img = cv2.cvtColor(generated_img, cv2.COLOR_BGR2RGB)
                plt.imshow(generated_img)
                plt.axis('off')
                plt.title(f'Epoch {current_epoch}\nSample {i+1}')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'samples_summary.png'))
            plt.close()
            
            pl_module.train()

def main():
    # 创建主输出目录
    os.makedirs("diffusers_generated", exist_ok=True)
    
    # 创建模型和数据集
    model = DiffusionModel()
    dataset = MultiImageDataset('test_picture', image_size=32)
    
    # 显示原始训练图片
    plt.figure(figsize=(15, 3))
    for i in range(5):
        if i < len(dataset.images):
            img = dataset.images[i].permute(1, 2, 0).numpy()
            plt.subplot(1, 5, i + 1)
            plt.imshow(img)
            plt.axis('off')
            plt.title(f'Original Image {i+1}')
    plt.tight_layout()
    plt.savefig('diffusers_generated/original_images.png')
    plt.close()
    
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # 创建训练器，添加回调
    trainer = pl.Trainer(
        max_epochs=500,
        accelerator='auto',
        devices=1,
        logger=False,
        enable_checkpointing=False,
        log_every_n_steps=1,
        callbacks=[GenerateImageCallback()]
    )
    
    # 开始训练
    trainer.fit(model, dataloader)

if __name__ == "__main__":
    main() 