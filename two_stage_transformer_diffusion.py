import torch
import torch.nn as nn
import pytorch_lightning as pl
from diffusers import DDPMScheduler
from typing import Optional, List, Tuple
import numpy as np
from tqdm import tqdm
import math
import random

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
        x = self.transformer(x)  # [batch_size, seq_len, embedding_dim]
        
        return x  # [batch_size, seq_len, embedding_dim]
    
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
        loss = self._contrastive_loss(z1, z2)
        
        self.log("pretrain_loss", loss)
        return loss
    
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

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1, attention_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm_cond = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, condition):
        # Cross-attention with residual
        x = x + self.cross_attn(self.norm1(x), self.norm_cond(condition), self.norm_cond(condition))[0]
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerDiffusion(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        cross_attention_layers: int = 3,  # 交叉注意力层数
    ):
        super().__init__()
        
        self.time_embedding = TimeEmbedding(embedding_dim)
        self.pos_encoder = PositionalEncoding(embedding_dim)
        
        # 时间嵌入投影
        self.time_proj = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
        # 自注意力层
        self.self_attention_layers = nn.ModuleList([
            TransformerBlock(
                dim=embedding_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout
            ) for _ in range(num_layers)
        ])
        
        # 交叉注意力层
        self.cross_attention_layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=embedding_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                attention_dropout=attention_dropout
            ) for _ in range(cross_attention_layers)
        ])
        
        # 输出投影
        self.final_norm = nn.LayerNorm(embedding_dim)
        self.noise_predictor = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )
        
    def forward(
        self,
        x: torch.Tensor,  # [B, seq_len, embedding_dim]
        timesteps: torch.Tensor,  # [B]
        condition: torch.Tensor,  # [B, seq_len, embedding_dim]
    ) -> torch.Tensor:
        # 时间嵌入
        t_emb = self.time_embedding(timesteps)  # [B, embedding_dim]
        t_emb = self.time_proj(t_emb)  # [B, embedding_dim]
        t_emb = t_emb.unsqueeze(1)  # [B, 1, embedding_dim]
        t_emb = t_emb.expand(-1, x.shape[1], -1)  # [B, seq_len, embedding_dim]
        
        # 添加时间信息和位置编码
        x = x + t_emb
        x = x.transpose(0, 1)
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)
        
        # 自注意力处理
        for layer in self.self_attention_layers:
            x = layer(x)
        
        # 交叉注意力处理
        for layer in self.cross_attention_layers:
            x = layer(x, condition)
        
        # 最终输出
        x = self.final_norm(x)
        noise_pred = self.noise_predictor(x)
        
        return noise_pred

class TwoStageTrajectoryDiffusion(pl.LightningModule):
    def __init__(
        self,
        num_ids: int = 3953,
        embedding_dim: int = 512,
        num_layers: int = 6,
        learning_rate: float = 1e-4,
        num_timesteps: int = 300,
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
        
        # Transformer扩散模型
        self.model = TransformerDiffusion(
            embedding_dim=embedding_dim,
            num_heads=8,
            num_layers=num_layers,
            mlp_ratio=4.0,
            dropout=0.1,
            attention_dropout=0.1,
            cross_attention_layers=3
        )
        
        # 用于将嵌入映射回ID空间的投影层
        self.proj_to_logits = nn.Linear(embedding_dim, num_ids + 2)
        
        # 扩散调度器
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_timesteps,
            beta_schedule="squaredcos_cap_v2",  # 使用余弦调度
            beta_start=0.0001,  # 起始beta值
            beta_end=0.02,  # 终点beta值
            prediction_type="epsilon"
        )
        
        self.learning_rate = learning_rate
        self.num_timesteps = num_timesteps
    
    def setup(self, stage=None):
        """在训练开始时将调度器的张量移动到正确的设备上"""
        device = next(self.parameters()).device
        # 将调度器的所有张量移动到设备上
        for key in self.noise_scheduler.betas.__dict__.keys():
            if torch.is_tensor(self.noise_scheduler.betas.__dict__[key]):
                self.noise_scheduler.betas.__dict__[key] = self.noise_scheduler.betas.__dict__[key].to(device)
        for key in self.noise_scheduler.alphas.__dict__.keys():
            if torch.is_tensor(self.noise_scheduler.alphas.__dict__[key]):
                self.noise_scheduler.alphas.__dict__[key] = self.noise_scheduler.alphas.__dict__[key].to(device)
    
    def forward(self, noisy_emb, timesteps, masked_ids, mask):
        # 获取条件嵌入
        with torch.no_grad():
            condition_embedding = self.id_embedding(masked_ids)
        
        # 预测噪声
        noise_pred = self.model(noisy_emb, timesteps, condition_embedding)
        return noise_pred
    
    def training_step(self, batch, batch_idx):
        original_ids, x, masked_ids, mask = batch  # 修改这里以匹配数据加载器的输出
        
        # 获取原始ID的嵌入表示
        with torch.no_grad():
            x0_emb = self.id_embedding(original_ids)  # [B, seq_len, embedding_dim]
        
        # 添加噪声
        noise = torch.randn_like(x0_emb)
        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (x0_emb.shape[0],), device=x0_emb.device).long()
        noisy_emb = self.noise_scheduler.add_noise(x0_emb, noise, timesteps)
        
        # 预测噪声
        noise_pred = self(noisy_emb, timesteps, masked_ids, mask)
        
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
        
        # 确保输入在正确的设备上
        masked_ids = masked_ids.to(device)
        mask = mask.to(device)
        
        # 初始化噪声
        x = torch.randn(
            (batch_size, seq_len, self.hparams.embedding_dim),
            device=device
        )
        
        # 将调度器的张量移到正确的设备上
        self.noise_scheduler.betas = self.noise_scheduler.betas.to(device)
        self.noise_scheduler.alphas = self.noise_scheduler.alphas.to(device)
        self.noise_scheduler.alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(device)
        
        # 逐步去噪
        for t in tqdm(self.noise_scheduler.timesteps):
            timesteps = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            
            # 预测噪声
            noise_pred = self(x, timesteps, masked_ids, mask)
            
            # 去噪步骤
            scheduler_output = self.noise_scheduler.step(noise_pred, t, x)
            x = scheduler_output.prev_sample
            
            # 如果是最后一个时间步，将嵌入映射回ID空间
            if t == self.noise_scheduler.timesteps[-1]:
                logits = self.proj_to_logits(x)
                probs = torch.softmax(logits, dim=-1)
                return probs

    def test_step(self, batch, batch_idx):
        """测试步骤
        计算Top-1, Top-5, Top-10准确率
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
            probs = self.sample(masked_ids, mask)
        
        # 计算不同的Top-k准确率
        topk_values = [1, 5, 10]
        accuracies = {}
        
        for k in topk_values:
            # 获取top-k预测
            _, topk_preds = torch.topk(probs, k=k, dim=-1)  # [batch_size, seq_len, k]
            
            # 检查原始ID是否在top-k预测中
            correct = torch.any(topk_preds == original_ids.unsqueeze(-1), dim=-1)  # [batch_size, seq_len]
            
            # 只考虑掩码位置
            valid_positions = (mask == 1)
            accuracy = (correct & valid_positions).float().sum() / valid_positions.float().sum()
            
            # 记录结果
            accuracies[f'test_top{k}_acc'] = accuracy
            
            # 打印当前batch的结果
            self.log(f'test_top{k}_acc', accuracy, prog_bar=True)
        
        return accuracies

def pretrain_embedding_model(
    num_ids: int,
    embedding_dim: int,
    num_epochs: int,
    batch_size: int,
    device: torch.device
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
    
    # 创建训练数据（这里使用随机数据作为示例）
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 100  # 每个epoch的批次数
        
        for _ in range(num_batches):
            # 生成随机轨迹序列
            original_ids = torch.randint(1, num_ids + 1, (batch_size, 32), device=device)
            # 创建one-hot编码
            x = torch.zeros(batch_size, 32, num_ids + 2, device=device)
            x.scatter_(2, original_ids.unsqueeze(-1), 1)
            # 创建掩码和带缺失值的序列
            mask = torch.ones_like(original_ids, dtype=torch.bool, device=device)
            masked_ids = original_ids.clone()
            
            # 前向传播和损失计算
            optimizer.zero_grad()
            loss = model.training_step((original_ids, x, masked_ids, mask), None)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.6f}")
    
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
    num_ids = 250
    embedding_dim = 256  # 确保这个维度在预训练和扩散模型中保持一致
    
    # 创建训练数据
    original_ids = torch.randint(1, num_ids + 1, (batch_size, seq_len)).to(device)
    masked_ids = original_ids.clone()
    mask = torch.ones_like(original_ids, dtype=torch.bool)
    
    # 第一阶段：预训练嵌入模型
    print("\n开始预训练嵌入模型...")
    pretrained_model = pretrain_embedding_model(
        num_ids=num_ids,
        embedding_dim=embedding_dim,  # 使用相同的embedding_dim
        num_epochs=20,  # 预训练轮数
        batch_size=batch_size,
        device=device
    )
    
    # 第二阶段：创建并训练扩散模型
    print("\n开始训练扩散模型...")
    model = TwoStageTrajectoryDiffusion(
        num_ids=num_ids,
        embedding_dim=embedding_dim,  # 使用相同的embedding_dim
        num_layers=6,
        learning_rate=1e-4,
        num_timesteps=num_timesteps,
        pretrained_embedding=pretrained_model  # 传入预训练模型
    ).to(device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 训练循环
    num_epochs = 1000
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
        
        # 前向传播
        loss = model.training_step((original_ids, masked_ids, mask), None)
        losses.append(loss.item())
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if (epoch + 1) % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}")
        
        if (epoch + 1) % 100 == 0:
            print(f"\n=== 第 {epoch + 1} 轮测试 ===")
            accuracies = evaluate_model()
            top1_accuracies.append(accuracies[0])
            top5_accuracies.append(accuracies[1])
            top10_accuracies.append(accuracies[2])
    
    # 绘制loss和准确率曲线
    import matplotlib.pyplot as plt
    import matplotlib
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
    
    plt.figure(figsize=(15, 5))
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.title('训练损失曲线', fontsize=12)
    plt.xlabel('迭代次数', fontsize=10)
    plt.ylabel('损失值', fontsize=10)
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    x = list(range(100, num_epochs + 1, 100))
    plt.plot(x, top1_accuracies, label='Top-1准确率')
    plt.plot(x, top5_accuracies, label='Top-5准确率')
    plt.plot(x, top10_accuracies, label='Top-10准确率')
    plt.title('准确率曲线', fontsize=12)
    plt.xlabel('迭代次数', fontsize=10)
    plt.ylabel('准确率', fontsize=10)
    plt.legend(fontsize=10)
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

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