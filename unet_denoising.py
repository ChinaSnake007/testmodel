import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from diffusers import DDPMScheduler
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
from pathlib import Path
from pytorch_lightning.callbacks import Callback
"""
模型：Unet结构，自定义DoubleConv
输入：
    1. 原始图片
    2. 时间步长
输出：
    1. 预测噪声
"""
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(1, out_channels),
            nn.SiLU(),
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # Encoder
        self.inc = DoubleConv(3, 128)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 512))
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, 256),
            nn.SiLU(),
            nn.Linear(256, 512),
        )
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2)
        self.up_conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.up_conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.up_conv3 = DoubleConv(256, 128)
        self.outc = nn.Conv2d(128, 3, kernel_size=1)
        
        # 初始化噪声调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=1000,
            beta_schedule="linear",
            prediction_type="epsilon"
        )

    def forward(self, x, t):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Time embedding
        t = t.unsqueeze(-1).float()
        t = self.time_mlp(t)
        t = t.unsqueeze(-1).unsqueeze(-1)
        x4 = x4 + t
        
        # Decoder
        x = self.up1(x4)
        x = self.up_conv1(torch.cat([x, x3], dim=1))
        x = self.up2(x)
        x = self.up_conv2(torch.cat([x, x2], dim=1))
        x = self.up3(x)
        x = self.up_conv3(torch.cat([x, x1], dim=1))
        x = self.outc(x)
        return x

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
        image = torch.randn(batch_size, 3, 64, 64).to(device)
        
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
    def __init__(self, image_dir, image_size=64):
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
        if current_epoch % 100 == 0:
            # 创建特定epoch的输出目录
            output_dir = f"general_picture/epoch_{current_epoch}"
            os.makedirs(output_dir, exist_ok=True)
            
            # 设置设备
            device = pl_module.device
            pl_module.eval()
            
            # 生成100张图片
            num_images = 100
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
    os.makedirs("general_picture", exist_ok=True)
    
    # 创建模型和数据集
    model = UNet()
    dataset = MultiImageDataset('test_picture', image_size=64)
    
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
    plt.savefig('general_picture/original_images.png')
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