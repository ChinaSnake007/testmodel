import torch
import torch.nn as nn
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from typing import Optional, List, Tuple
import numpy as np
from tqdm import tqdm
import math
import random
import torch.nn.functional as F

"""
模型：两阶段UNet结构
    第一阶段：PretrainIDEmbedding，做id的嵌入学习
    第二阶段：UNetDiffusion，使用UNet结构进行去噪
输入：
    1. 原始ID嵌入
    2. 条件ID嵌入
    3. 时间步长
输出：
    1. 预测ID
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

class PretrainIDEmbedding(pl.LightningModule):
    def __init__(self, num_ids: int, embedding_dim: int, temperature: float = 0.1):
        super().__init__()
        self.save_hyperparameters()
        self.temperature = temperature
        
        # ID嵌入层
        self.embedding = nn.Embedding(num_ids + 2, embedding_dim)
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
        
        # 序列级别的投影头
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # 添加解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embedding_dim,
            nhead=8,
            dim_feedforward=embedding_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=3)
        
        # 解码器输出层
        self.decoder_output = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, num_ids + 2)
        )
        
    def forward(self, ids):
        # ids: [batch_size, seq_len]
        x = self.embedding(ids)  # [batch_size, seq_len, embedding_dim]
        
        # 对整个词表的嵌入进行最大最小值归一化
        with torch.no_grad():
            # 获取完整词表的嵌入
            all_embeddings = self.embedding.weight  # [num_ids+2, embedding_dim]
            # 计算每个维度的最大最小值
            emb_min = all_embeddings.min(dim=0)[0]  # [embedding_dim]
            emb_max = all_embeddings.max(dim=0)[0]  # [embedding_dim]
            # 避免除零
            emb_range = (emb_max - emb_min).clamp(min=1e-5)
        
        # 对当前batch的嵌入进行归一化
        x = (x - emb_min) / emb_range
        
        # 添加位置编码
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        # Transformer编码
        memory = self.transformer(x)  # [batch_size, seq_len, embedding_dim]
        
        return memory  # [batch_size, seq_len, embedding_dim]
    
    def decode(self, memory):
        """解码器前向传播
        Args:
            memory: [batch_size, seq_len, embedding_dim] 编码器的输出
        Returns:
            [batch_size, seq_len, num_ids+2] 解码后的logits
        """
        # 创建目标序列的掩码
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(memory.size(1)).to(memory.device)
        
        # 使用编码器的输出作为解码器的输入
        decoded = self.decoder(
            tgt=memory,
            memory=memory,
            tgt_mask=tgt_mask
        )
        
        # 通过输出层得到logits
        logits = self.decoder_output(decoded)
        
        return logits
    
    def get_sequence_representation(self, x):
        """获取序列级别的表示
        Args:
            x: [batch_size, seq_len, embedding_dim] 的张量
        Returns:
            [batch_size, embedding_dim] 的序列表示
        """
        # 使用平均池化获取序列表示
        x = x.mean(dim=1)  # [batch_size, embedding_dim]
        
        # 通过投影头
        x = self.projection(x)  # [batch_size, embedding_dim]
        
        # 归一化
        x = nn.functional.normalize(x, dim=-1)
        
        return x
    
    def training_step(self, batch, batch_idx):
        # 从batch中解包数据
        original_ids, x, masked_ids, mask = batch
        
        # 通过数据增强创建正样本对
        augmented_ids = self._augment_trajectory(original_ids)
        
        # 获取序列表示
        x1 = self(original_ids)  # [batch_size, seq_len, embedding_dim]
        x2 = self(augmented_ids)  # [batch_size, seq_len, embedding_dim]
        
        # 获取序列级别的表示用于对比学习
        z1 = self.get_sequence_representation(x1)  # [batch_size, embedding_dim]
        z2 = self.get_sequence_representation(x2)  # [batch_size, embedding_dim]
        
        # 计算对比损失
        contrastive_loss = self._contrastive_loss(z1, z2)
        
        # 计算重建损失
        logits = self.decode(x1)  # [batch_size, seq_len, num_ids+2]
        reconstruction_loss = nn.functional.cross_entropy(
            logits.view(-1, self.hparams.num_ids + 2),
            original_ids.view(-1)
        )
        
        # 总损失
        total_loss = contrastive_loss + reconstruction_loss
        
        # 打印损失信息
        if batch_idx % 10 == 0:  # 每10个batch打印一次
            print(f"Batch {batch_idx}, 对比损失: {contrastive_loss.item():.4f}, 重建损失: {reconstruction_loss.item():.4f}, 总损失: {total_loss.item():.4f}")
        
        return total_loss
    
    def _augment_trajectory(self, ids):
        """
        对轨迹进行数据增强，可以包括：
        1. 随机裁剪
        2. 随机掩码
        3. 随机替换
        4. 时序扰动
        """
        augmented = ids.clone()
        batch_size, seq_len = ids.shape
        
        # 随机掩码
        mask_prob = 0.15
        mask = torch.rand(batch_size, seq_len, device=ids.device) < mask_prob
        augmented[mask] = random.randint(1, self.hparams.num_ids)
        
        # 随机时序扰动
        if random.random() < 0.5:
            # 随机交换相邻位置
            for b in range(batch_size):
                for i in range(0, seq_len-1, 2):
                    if random.random() < 0.3:
                        augmented[b, i], augmented[b, i+1] = augmented[b, i+1], augmented[b, i].clone()
        
        return augmented
    
    def _contrastive_loss(self, z1, z2):
        """
        计算对比损失
        使用InfoNCE损失函数
        """
        batch_size = z1.shape[0]
        
        # 计算相似度矩阵
        sim = torch.mm(z1, z2.t()) / self.temperature
        
        # 正样本对应的标签
        labels = torch.arange(batch_size, device=z1.device)
        
        # 计算对比损失
        loss = nn.functional.cross_entropy(sim, labels) + nn.functional.cross_entropy(sim.t(), labels)
        return loss / 2
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=0.01)
        return optimizer

class UNetDiffusion(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        ch: int = 64,
        ch_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        dropout: float = 0.1,
        num_timesteps: int = 1000
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
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
        
        # 输入投影
        self.conv_in = nn.Conv1d(embedding_dim, self.ch, kernel_size=3, stride=1, padding=1)
        
        # 下采样部分
        self.down = nn.ModuleList()
        curr_res = 32  # 假设序列长度为32
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
        self.conv_out = nn.Conv1d(block_in, embedding_dim, kernel_size=3, stride=1, padding=1)
        
        # 扩散调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,
            beta_schedule="squaredcos_cap_v2",
            prediction_type="epsilon"
        )
    
    def forward(self, x, t, condition):
        # 时间编码
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense0(temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense1(temb)
        
        # 输入投影
        x = x.transpose(1, 2)  # [B, embed_dim, seq_len]
        x = self.conv_in(x)
        
        # 下采样路径
        hs = [x]
        
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        
        # 中间层
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)
        
        # 上采样路径
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1), temb)
                h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)
        
        # 输出层
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        
        return h.transpose(1, 2)  # [B, seq_len, embed_dim]

class TwoStageUNetDiffusion(pl.LightningModule):
    def __init__(
        self,
        num_ids: int = 3953,
        embedding_dim: int = 512,
        learning_rate: float = 1e-4,
        num_timesteps: int = 1000,
        pretrained_embedding: Optional[PretrainIDEmbedding] = None
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['pretrained_embedding'])
        
        # 使用预训练的嵌入模型或创建新的
        self.id_embedding = pretrained_embedding if pretrained_embedding is not None else PretrainIDEmbedding(num_ids, embedding_dim)
        
        # 如果使用预训练模型，冻结其参数
        if pretrained_embedding is not None:
            for param in self.id_embedding.parameters():
                param.requires_grad = False
        
        # UNet扩散模型
        self.model = UNetDiffusion(
            embedding_dim=embedding_dim,
            ch=64,
            ch_mult=(1, 2, 4, 8),
            num_res_blocks=2,
            dropout=0.1,
            num_timesteps=num_timesteps
        )
        
        self.learning_rate = learning_rate
        self.num_timesteps = num_timesteps
    
    def forward(self, noisy_emb, timesteps, masked_ids, mask):
        # 获取条件嵌入
        with torch.no_grad():
            condition_embedding = self.id_embedding(masked_ids)
        
        # 预测噪声
        noise_pred = self.model(noisy_emb, timesteps, condition_embedding)
        return noise_pred
    
    def training_step(self, batch, batch_idx):
        original_ids, x, masked_ids, mask = batch

        # 用预训练嵌入模型对original_ids和masked_ids做嵌入
        with torch.no_grad():
            original_emb = self.id_embedding(original_ids)      # [B, seq_len, embedding_dim]

        # 对embedding加噪声
        noise = torch.randn_like(original_emb)
        timesteps = torch.randint(0, self.num_timesteps, (original_emb.shape[0],), device=original_emb.device)
        noisy_emb = self.model.noise_scheduler.add_noise(original_emb, noise, timesteps)

        # 预测噪声
        noise_pred = self(noisy_emb, timesteps, masked_ids, mask)

        # 计算损失
        loss = F.mse_loss(noise_pred, noise)

        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
    
    def sample(self, masked_ids, mask):
        device = next(self.parameters()).device
        batch_size = masked_ids.shape[0]
        seq_len = masked_ids.shape[1]
        
        masked_ids = masked_ids.to(device)
        mask = mask.to(device)
        
        x = torch.randn(
            (batch_size, seq_len, self.hparams.embedding_dim),
            device=device
        )
        
        for t in tqdm(self.model.noise_scheduler.timesteps):
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            noise_pred = self(x, timesteps, masked_ids, mask)
            scheduler_output = self.model.noise_scheduler.step(noise_pred, t, x)
            x = scheduler_output.prev_sample
            
            if t == self.model.noise_scheduler.timesteps[-1]:
                logits = self.id_embedding.decode(x)
                probs = torch.softmax(logits, dim=-1)
                return probs, x  # 返回解码概率和还原的x0嵌入

    def test_step(self, batch, batch_idx):
        """测试步骤
        计算准确率
        """
        original_ids, x, masked_ids, mask = batch
        
        # 确保所有数据都在正确的设备上
        device = next(self.parameters()).device
        original_ids = original_ids.to(device)
        x = x.to(device)
        masked_ids = masked_ids.to(device)
        mask = mask.to(device)
        
        # 生成预测
        with torch.no_grad():
            probs, x0_pred_emb = self.sample(masked_ids, mask)
        
        # 计算准确率
        _, top1_indices = torch.topk(probs, k=1, dim=-1)  # [batch_size, seq_len, 1]
        correct = (top1_indices.squeeze(-1) == original_ids)  # [batch_size, seq_len]
        
        # 只考虑掩码位置
        valid_positions = (mask == 1)
        accuracy = (correct & valid_positions).float().sum() / valid_positions.float().sum()
        
        # 计算x0嵌入与真实嵌入的MAE
        real_emb = self.id_embedding(original_ids)
        mae = torch.abs(x0_pred_emb - real_emb).mean().item()
        
        # 打印当前batch的结果
        print(f"测试准确率: {accuracy.item():.4f}")
        print(f"嵌入MAE: {mae:.6f}")
        
        return accuracy.item(), mae

def pretrain_embedding_model(
    num_ids: int,
    embedding_dim: int,
    num_epochs: int,
    batch_size: int,
    device: torch.device,
    original_ids: torch.Tensor  # 直接接收原始ID数据
):
    print("开始预训练嵌入模型...")
    
    # 创建预训练模型
    model = PretrainIDEmbedding(
        num_ids=num_ids,
        embedding_dim=embedding_dim,
        temperature=0.1
    ).to(device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 计算每个epoch的批次数
    num_samples = original_ids.shape[0]
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    # 用于提前停止的变量
    low_loss_count = 0
    
    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        # 打乱数据
        indices = torch.randperm(num_samples)
        
        for i in range(0, num_samples, batch_size):
            # 获取当前批次的数据
            batch_indices = indices[i:i + batch_size]
            batch_ids = original_ids[batch_indices].to(device)
            
            # 创建one-hot编码
            x = torch.zeros(batch_ids.shape[0], batch_ids.shape[1], num_ids + 2, device=device)
            x.scatter_(2, batch_ids.unsqueeze(-1), 1)
            
            # 创建掩码和带缺失值的序列
            mask = torch.ones_like(batch_ids, dtype=torch.bool, device=device)
            masked_ids = batch_ids.clone()
            
            # 前向传播和损失计算
            optimizer.zero_grad()
            batch_idx = i // batch_size  # 计算当前batch的索引
            loss = model.training_step((batch_ids, x, masked_ids, mask), batch_idx)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}")
        
        # 检查是否满足提前停止条件
        if avg_loss < 0.2:
            low_loss_count += 1
            if low_loss_count >= 10:
                print(f"损失连续{low_loss_count}代小于0.2，提前停止训练!")
                break
        else:
            low_loss_count = 0
    
    print("预训练完成！")
    return model

def main():
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置超参数
    num_timesteps = 500
    batch_size = 64
    seq_len = 32
    num_ids = 25
    embedding_dim = 64  # 确保这个维度在预训练和扩散模型中保持一致
    
    # 创建训练数据
    original_ids = torch.randint(1, num_ids + 1, (batch_size, seq_len)).to(device)
    x = F.one_hot(original_ids, num_classes=num_ids + 2).float()
    masked_ids = original_ids.clone()
    mask = torch.ones_like(original_ids, dtype=torch.bool)
    
    # 第一阶段：预训练嵌入模型
    print("\n开始预训练嵌入模型...")
    pretrained_model = pretrain_embedding_model(
        num_ids=num_ids,
        embedding_dim=embedding_dim,  # 使用相同的embedding_dim
        num_epochs=2000,  # 预训练轮数
        batch_size=batch_size,
        device=device,
        original_ids=original_ids  # 直接接收原始ID数据
    )
    
    # 第二阶段：创建并训练扩散模型
    print("\n开始训练扩散模型...")
    model = TwoStageUNetDiffusion(
        num_ids=num_ids,
        embedding_dim=embedding_dim,  # 使用相同的embedding_dim
        learning_rate=1e-4,
        num_timesteps=num_timesteps,
        pretrained_embedding=pretrained_model  # 传入预训练模型
    ).to(device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 训练循环
    num_epochs = 5000
    losses = []
    accuracies = []
    maes = []
    
    def evaluate_model():
        model.eval()
        with torch.no_grad():
            probs, x0_pred_emb = model.sample(masked_ids, mask)
            # 只计算Top-1准确率
            _, top1_indices = torch.topk(probs, k=1, dim=2)
            correct = (top1_indices.squeeze(-1) == original_ids)
            accuracy = correct.float().mean()
            # 计算真实嵌入
            real_emb = model.id_embedding(original_ids)
            mae = torch.abs(x0_pred_emb - real_emb).mean().item()
            print(f"\n当前测试结果:")
            print(f"准确率: {accuracy.item():.4f}")
            print(f"嵌入MAE: {mae:.6f}")
            return accuracy.item(), mae
    
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        loss = model.training_step((original_ids, x, masked_ids, mask), None)
        losses.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
        if (epoch + 1) % 100 == 0:
            print(f"\n=== 第 {epoch + 1} 轮测试 ===")
            accuracy, mae = evaluate_model()
            accuracies.append(accuracy)
            maes.append(mae)
    
    # 绘制loss、准确率和MAE曲线
    import matplotlib.pyplot as plt
    import matplotlib
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.plot(losses)
    plt.title('训练损失曲线', fontsize=12)
    plt.xlabel('迭代次数', fontsize=10)
    plt.ylabel('损失值', fontsize=10)
    plt.grid(True)
    plt.subplot(1, 3, 2)
    x_axis = list(range(100, num_epochs + 1, 100))
    plt.plot(x_axis, accuracies, label='准确率')
    plt.title('准确率曲线', fontsize=12)
    plt.xlabel('迭代次数', fontsize=10)
    plt.ylabel('准确率', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.subplot(1, 3, 3)
    plt.plot(x_axis, maes, label='嵌入MAE', color='orange')
    plt.title('嵌入MAE曲线', fontsize=12)
    plt.xlabel('迭代次数', fontsize=10)
    plt.ylabel('MAE', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.close()

    # 最终测试
    print("\n=== 最终测试结果 ===")
    model.eval()
    with torch.no_grad():
        final_predictions = model.sample(masked_ids, mask)
        
        print("\n样本对比 (前5个序列的前10个时间步):")
        print("原始ID:")
        print(original_ids[:5, :10].cpu().numpy())
        print("\n生成ID (预测):")
        _, top1_indices = torch.topk(final_predictions, k=1, dim=2)
        print(top1_indices[:5, :10, 0].cpu().numpy())
        
        print("\n详细预测示例 (第一个序列的前3个时间步):")
        for step in range(3):
            print(f"\n时间步 {step}:")
            probs = final_predictions[0, step]
            top_probs, top_ids = torch.topk(probs, k=1)
            print(f"预测ID: {top_ids[0].item():4d}, 概率: {top_probs[0].item():.4f}")

if __name__ == "__main__":
    main() 