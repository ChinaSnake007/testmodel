import pytorch_lightning as pl
from data_utils import create_dataloaders
from model import TrajectoryDiffusionModel
import torch
import os
import numpy as np
from tqdm import tqdm

def calculate_accuracy(pred_ids, original_ids, mask):
    """计算预测准确率"""
    # 只考虑非padding且非缺失的位置
    valid_positions = (mask == 1)
    correct = (pred_ids == original_ids) & valid_positions
    accuracy = correct.sum().float() / valid_positions.sum().float()
    return accuracy.item()

def main():
    # 设置Tensor Core优化
    torch.set_float32_matmul_precision('medium')
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path="trajectories.csv",
        batch_size=512,
        max_length=32,
        num_ids=3953,
        miss_ratio=0.3
    )
    
    # 创建模型
    model = TrajectoryDiffusionModel(
        num_ids=3953,
        embedding_dim=128,
        hidden_dim=256,
        num_layers=6,
        learning_rate=1e-5
    )
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=1.0,
        log_every_n_steps=50,
        logger=False,  # 暂时禁用logger
        enable_checkpointing=False  # 禁用检查点保存
    )
    
    # 开始训练
    trainer.fit(model, train_loader)
    
if __name__ == "__main__":
    main()
