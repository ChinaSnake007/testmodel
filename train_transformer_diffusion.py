import pytorch_lightning as pl
from data_utils import create_dataloaders
from transformer_diffusion import TrajectoryTransformerDiffusion
import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_topk_accuracy(pred_probs, original_ids, mask=None, k=1):
    """计算Top-K准确率
    Args:
        pred_probs: [batch_size, seq_len, num_ids] 预测概率
        original_ids: [batch_size, seq_len] 原始ID
        mask: [batch_size, seq_len] 掩码，如果为None则考虑所有位置
        k: int, top-k的k值
    """
    # 获取top-k预测
    _, topk_preds = torch.topk(pred_probs, k=k, dim=-1)  # [batch_size, seq_len, k]
    
    # 检查原始ID是否在top-k预测中
    correct = torch.any(topk_preds == original_ids.unsqueeze(-1), dim=-1)  # [batch_size, seq_len]
    # 计算所有位置的准确率
    all_accuracy = correct.float().mean()
    
    # 计算掩码位置的准确率
    if mask is not None:
        valid_positions = (mask == 1)
        mask_accuracy = (correct & valid_positions).float().sum() / valid_positions.float().sum()
        return all_accuracy.item(), mask_accuracy.item()
    
    return all_accuracy.item(), None
    
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

def evaluate_model(model, val_loader, device):
    """评估模型性能"""
    model.eval()
    total_top1_acc = 0
    total_top5_acc = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            original_ids, x0, masked_ids, mask = [b.to(device) for b in batch]
            
            # 生成预测
            predictions = model.sample(masked_ids, mask)
            
            # 计算准确率
            top1_acc = calculate_topk_accuracy(predictions, original_ids, mask, k=1)
            top5_acc = calculate_topk_accuracy(predictions, original_ids, mask, k=5)
            
            total_top1_acc += top1_acc
            total_top5_acc += top5_acc
            num_batches += 1
    
    avg_top1_acc = total_top1_acc / num_batches
    avg_top5_acc = total_top5_acc / num_batches
    
    return avg_top1_acc, avg_top5_acc

def main():
    # 设置设备和优化选项
    torch.set_float32_matmul_precision('medium')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path="trajectories_sample.csv",
        batch_size=64,
        max_length=32,
        num_ids=3953,
        miss_ratio=0.3
    )
    
    # 模型参数
    model_config = {
        "num_ids": 3953,
        "embedding_dim": 256,
        "num_layers": 6,
        "learning_rate": 1e-4,
        "num_timesteps": 500
    }
    
    # 创建模型
    model = TrajectoryTransformerDiffusion(
        num_ids=model_config["num_ids"],
        embedding_dim=model_config["embedding_dim"],
        num_layers=model_config["num_layers"],
        learning_rate=model_config["learning_rate"],
        num_timesteps=model_config["num_timesteps"]
    ).to(device)
    
    # 创建训练器
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        gradient_clip_val=1.0,
        enable_checkpointing=False,  # 禁用检查点保存
        logger=False,  # 禁用日志记录
        enable_progress_bar=True
    )
    
    # 记录训练过程
    losses = []
    top1_accs = []
    top5_accs = []
    
    # 开始训练
    print("\n=== 开始训练 ===")
    trainer.fit(model, train_loader, val_loader)
    
    # 每个epoch结束后评估模型
    for epoch in range(trainer.max_epochs):
        # 训练一个epoch
        for batch in train_loader:
            original_ids, x0, masked_ids, mask = [b.to(device) for b in batch]
            loss = model.training_step((original_ids, x0, masked_ids, mask), 0)
            losses.append(loss.item())
        
        # 评估模型
        top1_acc, top5_acc = evaluate_model(model, val_loader, device)
        top1_accs.append(top1_acc)
        top5_accs.append(top5_acc)
        
        # 打印进度
        print(f"Epoch {epoch+1}/{trainer.max_epochs}")
        print(f"Top-1 Accuracy: {top1_acc:.4f}")
        print(f"Top-5 Accuracy: {top5_acc:.4f}")
        
        # 绘制训练曲线
        plot_training_curves(losses, top1_accs, top5_accs)
    
    # 测试模型
    print("\n=== 模型测试 ===")
    trainer.test(model, test_loader)
    
    # 保存模型
    torch.save(model.state_dict(), 'transformer_diffusion_model.pth')
    print("模型已保存到 transformer_diffusion_model.pth")

if __name__ == "__main__":
    main() 