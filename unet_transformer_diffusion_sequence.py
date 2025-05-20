import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from diffusers import DDPMScheduler

"""
模型：Unet结构+残差块（提取特征用一维卷积）和注意力
将条件信息，以序列的方式和x相加
输入：
    1. 原始ID除以num_ids+1，再乘以10，扩大范围
    2. 条件ID除以num_ids+1
    3. 时间步长

输出：
    1. 预测噪声
    2. 预测ID
"""

def get_timestep_embedding(timesteps, embedding_dim):
    """时间步编码"""
    assert len(timesteps.shape) == 1
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1, 0, 0))
    return emb

def nonlinearity(x):
    return x * torch.sigmoid(x)  # swish激活函数

def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)

class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=2, padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (1, 1)
            x = F.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = F.avg_pool1d(x, kernel_size=2, stride=2)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False, dropout=0.1, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h

class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, w = q.shape
        q = q.permute(0, 2, 1)  # b,w,c
        w_ = torch.bmm(q, k)  # b,w,w
        w_ = w_ * (int(c)**(-0.5))
        w_ = F.softmax(w_, dim=2)

        w_ = w_.permute(0, 2, 1)  # b,w,w
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, w)

        h_ = self.proj_out(h_)

        return x + h_

class IDConditionModel(nn.Module):
    def __init__(self, num_ids, embedding_dim, use_transformer=True):
        super().__init__()
        self.id_embedding = nn.Embedding(num_ids + 2, embedding_dim)  # +2 for padding
        self.use_transformer = use_transformer
        
        if use_transformer:
            self.pos_encoder = nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=8,
                dim_feedforward=embedding_dim * 4,
                batch_first=True
            )
            self.transformer = nn.TransformerEncoder(self.pos_encoder, num_layers=3)
        else:
            # 使用简单的线性层替代Transformer
            self.linear_layers = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim * 2),
                nn.SiLU(),
                nn.Linear(embedding_dim * 2, embedding_dim)
            )
            
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, ids):
        x = self.id_embedding(ids)
        
        if self.use_transformer:
            x = self.transformer(x)
        else:
            x = self.linear_layers(x)
            
        x = self.output_proj(x)
        x = x.permute(0, 2, 1)  # [B, embed_dim, seq_len]
        return x

class UNetTransformerDiffusion(pl.LightningModule):
    def __init__(
        self,
        num_ids=3953,
        seq_length=32,
        ch=64,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        dropout=0.1,
        learning_rate=1e-4,
        num_timesteps=1000
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.num_ids = num_ids
        self.seq_length = seq_length
        self.learning_rate = learning_rate
        
        # 基础参数设置
        self.ch = ch
        self.temb_ch = self.ch * 4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        in_ch_mult = (1,) + tuple(ch_mult)
        
        # 时间嵌入
        self.temb = nn.ModuleDict({
            'dense0': nn.Linear(self.ch, self.temb_ch),
            'dense1': nn.Linear(self.temb_ch, self.temb_ch),
        })
        
        # 条件编码
        self.cond_encoder = IDConditionModel(num_ids, self.ch * 4, use_transformer=True)
        
        # 条件嵌入的通道数调整层
        self.cond_channel_adjust = nn.ModuleList([
            nn.Conv1d(self.ch * 4, ch * ch_mult[i], kernel_size=1)
            for i in range(self.num_resolutions)
        ])
        
        # 输入投影
        self.conv_in = nn.Conv1d(1, self.ch, kernel_size=3, stride=1, padding=1)
        
        # 下采样部分
        self.down = nn.ModuleList()
        curr_res = seq_length
        block_in = None
        
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(
                    in_channels=block_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch,
                    dropout=dropout
                ))
                block_in = block_out
                attn.append(AttnBlock(block_in))
            
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, True)
                curr_res = curr_res // 2
            self.down.append(down)
        
        # 中间层
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in, block_in, temb_channels=self.temb_ch)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(block_in, block_in, temb_channels=self.temb_ch)
        
        # 上采样部分
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            skip_in = ch * ch_mult[i_level]
            
            for i_block in range(self.num_res_blocks + 1):
                if i_block == self.num_res_blocks:
                    skip_in = ch * in_ch_mult[i_level]
                block.append(ResnetBlock(
                    in_channels=block_in + skip_in,
                    out_channels=block_out,
                    temb_channels=self.temb_ch
                ))
                block_in = block_out
                attn.append(AttnBlock(block_in))
            
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, True)
                curr_res = curr_res * 2
            self.up.insert(0, up)
        
        # 输出层
        self.norm_out = Normalize(block_in)
        self.conv_out = nn.Conv1d(block_in, 1, kernel_size=3, stride=1, padding=1)
        
        # 扩散调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,
            beta_schedule="linear",
            prediction_type="epsilon"
        )
    
    def forward(self, x, t, cond):
        # 时间编码
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense0(temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense1(temb)
        
        # 条件编码（只生成一次，后续动态调整）
        cond_emb = self.cond_encoder(cond)  # [B, embed_dim, seq_len]
        
        # 下采样路径
        hs = [self.conv_in(x)]
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                # 动态调整条件嵌入的通道数和特征长度
                # 1. 使用cond_channel_adjust调整通道数,使其与当前层特征图通道数匹配
                cond_emb_level = self.cond_channel_adjust[i_level](cond_emb)
                # 2. 使用线性插值调整特征长度,使其与当前层特征图长度匹配
                cond_emb_level = F.interpolate(cond_emb_level, size=h.shape[2], mode='linear')
                h = h + cond_emb_level
                h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        
        # 中间层
        h = hs[-1]
        cond_emb_mid = self.cond_channel_adjust[-1](cond_emb)
        cond_emb_mid = F.interpolate(cond_emb_mid, size=h.shape[2], mode='linear')
        h = self.mid.block_1(h, temb)
        h = h + cond_emb_mid
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        h = h + cond_emb_mid
        
        # 上采样路径
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                # 动态调整条件嵌入
                cond_emb_level = self.cond_channel_adjust[i_level](cond_emb)
                cond_emb_level = F.interpolate(cond_emb_level, size=h.shape[2], mode='linear')
                h = h + cond_emb_level
                h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        # 输出层
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        
        return h
    
    def normalize_ids(self, ids):
        # 将ID归一化到[0,1]范围
        return ids.float() / (self.num_ids + 1)
    
    def denormalize_ids(self, normalized_ids):
        # 从[0,1]范围恢复到原始ID
        scaled_ids = normalized_ids * (self.num_ids + 1)
        # 对缩放后的ID值进行四舍五入,将连续值转换为离散的整数ID
        rounded_ids = torch.round(scaled_ids)
        # 将四舍五入后的ID值转换为长整型,并限制在合法范围内(0到num_ids+1)
        return rounded_ids.long().clamp(0, self.num_ids + 1)
    
    def training_step(self, batch, batch_idx):
        original_ids, masked_ids = batch
        
        # 归一化ID
        x0 = self.normalize_ids(original_ids).unsqueeze(1)  # [B, 1, seq_len]
        
        # 添加噪声
        noise = torch.randn_like(x0)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps,
                                (x0.shape[0],), device=x0.device).long()
        noisy_x = self.noise_scheduler.add_noise(x0, noise, timesteps)
        
        # 预测噪声
        noise_pred = self(noisy_x, timesteps, masked_ids)
        
        # 计算损失
        loss = F.mse_loss(noise_pred, noise)
        
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
    def sample(self, masked_ids):
        device = masked_ids.device
        batch_size = masked_ids.shape[0]
        
        # 初始化噪声
        x = torch.randn((batch_size, 1, self.seq_length), device=device)
        
        # 逐步去噪
        for t in self.noise_scheduler.timesteps:
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            with torch.no_grad():
                noise_pred = self(x, timesteps, masked_ids)
                
            # 去噪步骤
            scheduler_output = self.noise_scheduler.step(noise_pred, t, x)
            x = scheduler_output.prev_sample
        
        # 将结果转换回ID
        return self.denormalize_ids(x.squeeze(1))

def main():
    # 模型参数
    model_config = {
        'num_ids': 20,
        'seq_length': 16,
        'ch': 64,
        'ch_mult': (1, 2, 4, 8),
        'num_res_blocks': 2,
        'dropout': 0.1,
        'learning_rate': 1e-2,
        'num_timesteps': 500
    }
    
    # 训练参数
    train_config = {
        'batch_size': 1,
        'num_epochs': 1000
    }
    
    # 创建模型
    model = UNetTransformerDiffusion(**model_config)
    
    # 使用示例数据进行测试
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # 创建示例数据
    batch_size = train_config['batch_size']
    seq_length = model_config['seq_length']
    original_ids = torch.randint(1, model_config['num_ids'] + 1, (batch_size, seq_length)).to(device)
    masked_ids = original_ids.clone()  # 在实际应用中，这里应该是条件ID
    
    # 训练循环
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config['learning_rate'])
    
    print("\n开始训练:")
    for epoch in range(train_config['num_epochs']):
        model.train()
        optimizer.zero_grad()
        
        # 准备批次数据
        batch = (original_ids, masked_ids)
        
        # 计算损失
        loss = model.training_step(batch, epoch)
        
        # 反向传播
        loss.backward()
        optimizer.step()
        
        # 记录最高准确率
        if not hasattr(model, 'best_accuracy'):
            model.best_accuracy = 0.0
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{train_config['num_epochs']}], Loss: {loss.item():.6f}")
            
            # 生成样本
            model.eval()
            with torch.no_grad():
                generated_ids = model.sample(masked_ids)
                
                # 计算准确率
                correct_count = (generated_ids == original_ids).sum().item()
                total_ids = original_ids.numel()
                accuracy = correct_count / total_ids
                
                # 更新并打印最高准确率
                if accuracy > model.best_accuracy:
                    model.best_accuracy = accuracy
                print(f"当前准确率: {accuracy:.4f}, 最高准确率: {model.best_accuracy:.4f}")
                
                # 打印样本对比
                if (epoch + 1) % 20 == 0:
                    print("\n样本对比 (前3个序列):")
                    print("原始ID:")
                    print(original_ids[:3].cpu().numpy())
                    print("生成ID:")
                    print(generated_ids[:3].cpu().numpy())

if __name__ == "__main__":
    main()