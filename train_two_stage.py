import pytorch_lightning as pl
from data_utils import create_dataloaders
from two_stage_transformer_diffusion import TwoStageTrajectoryDiffusion, pretrain_embedding_model
import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_topk_accuracy(pred_probs, original_ids, mask, k=1):
    """计算Top-K准确率
    Args:
        pred_probs: [batch_size, seq_len, num_ids] 预测概率
        original_ids: [batch_size, seq_len] 原始ID
        mask: [batch_size, seq_len] 掩码
        k: int, top-k的k值
    """
    # 获取top-k预测
    _, topk_preds = torch.topk(pred_probs, k=k, dim=-1)  # [batch_size, seq_len, k]
    
    # 检查原始ID是否在top-k预测中
    correct = torch.any(topk_preds == original_ids.unsqueeze(-1), dim=-1)  # [batch_size, seq_len]
    
    # 只考虑掩码位置
    valid_positions = (mask == 1)
    accuracy = (correct & valid_positions).float().sum() / valid_positions.float().sum()
    
    return accuracy.item()

def plot_training_curves(losses, top1_accs, top5_accs, save_path='training_curves.png'):
    """绘制训练曲线"""
    plt.figure(figsize=(15, 5))
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 绘制损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(losses, label='训练损失')
    plt.title('训练损失曲线')
    plt.xlabel('迭代次数')
    plt.ylabel('损失值')
    plt.grid(True)
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(top1_accs, label='Top-1准确率')
    plt.plot(top5_accs, label='Top-5准确率')
    plt.title('准确率曲线')
    plt.xlabel('评估次数')
    plt.ylabel('准确率')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # 设置设备和优化选项
    torch.set_float32_matmul_precision('medium')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path="trajectories_sample.csv",
        batch_size=64,  # 较小的batch size以适应两阶段模型
        max_length=32,
        num_ids=3953,
        miss_ratio=0.3
    )
    
    # 获取原始ID数据
    original_ids = []
    for batch in train_loader:
        original_ids.append(batch[0])  # 获取每个批次的原始ID
    original_ids = torch.cat(original_ids, dim=0)  # 将所有批次的数据连接起来
    
    # 模型参数
    model_config = {
        "num_ids": 3953,
        "embedding_dim": 256,
        "num_layers": 6,
        "learning_rate": 1e-4,
        "num_timesteps": 500
    }
    
    # 第一阶段：预训练嵌入模型
    print("\n=== 第一阶段：预训练ID嵌入模型 ===")
    pretrained_model = pretrain_embedding_model(
        num_ids=model_config["num_ids"],
        embedding_dim=model_config["embedding_dim"],
        num_epochs=50,
        batch_size=64,
        device=device,
        original_ids=original_ids  # 直接传入原始ID数据
    )
    
    # 第二阶段：训练扩散模型
    print("\n=== 第二阶段：训练扩散模型 ===")
    
    # 创建扩散模型
    model = TwoStageTrajectoryDiffusion(
        num_ids=model_config["num_ids"],
        embedding_dim=model_config["embedding_dim"],
        num_layers=model_config["num_layers"],
        learning_rate=model_config["learning_rate"],
        num_timesteps=model_config["num_timesteps"],
        pretrained_embedding=pretrained_model
    ).to(device)
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=1.0,
        enable_checkpointing=False,  # 禁用检查点保存
        logger=False,  # 禁用日志记录
        enable_progress_bar=True  # 保留进度条显示
    )
    
    # 开始训练
    trainer.fit(model, train_loader, val_loader)
    
    # 测试模型
    print("\n=== 模型测试 ===")
    trainer.test(model, test_loader)
    
        
        
if __name__ == "__main__":
    main() 