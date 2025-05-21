import pytorch_lightning as pl
from data_utils import create_dataloaders
from unet_transformer_diffusion_maxid_state import UNetTransformerDiffusion
import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def calculate_accuracy(pred_ids, original_ids, mask=None):
    """计算准确率
    Args:
        pred_ids: [batch_size, seq_len] 预测的ID
        original_ids: [batch_size, seq_len] 原始ID
        mask: [batch_size, seq_len] 掩码，如果为None则考虑所有位置
    """
    # 计算正确预测的位置
    correct = (pred_ids == original_ids)  # [batch_size, seq_len]
    
    # 计算所有位置的准确率
    all_accuracy = correct.float().mean()
    
    # 计算掩码位置的准确率
    if mask is not None:
        valid_positions = (mask == 1)
        mask_accuracy = (correct & valid_positions).float().sum() / valid_positions.float().sum()
        return all_accuracy.item(), mask_accuracy.item()
    
    return all_accuracy.item(), None

def plot_training_curves(losses, accuracies, masked_accuracies=None, save_path='training_curves.png'):
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
    plt.plot(accuracies, label='总体准确率')
    if masked_accuracies is not None:
        plt.plot(masked_accuracies, label='掩码位置准确率')
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
    total_accuracy = 0
    total_masked_accuracy = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="验证进度")
        for batch in pbar:
            original_ids, masked_ids = [b.to(device) for b in batch]
            
            # 生成预测
            predictions = model.sample(masked_ids)
            
            # 计算准确率
            accuracy, masked_accuracy = calculate_accuracy(predictions, original_ids)
            
            total_accuracy += accuracy
            if masked_accuracy is not None:
                total_masked_accuracy += masked_accuracy
            num_batches += 1
            
            # 更新进度条显示当前batch的准确率
            pbar.set_postfix({
                'batch_acc': f'{accuracy:.4f}',
                'batch_mask_acc': f'{masked_accuracy:.4f}' if masked_accuracy is not None else 'N/A'
            })
    
    avg_accuracy = total_accuracy / num_batches
    avg_masked_accuracy = total_masked_accuracy / num_batches if masked_accuracy is not None else None
    
    return avg_accuracy, avg_masked_accuracy

def main():
    # 设置设备和优化选项
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path='test_data/sequences_id50_ratio0.4_len20_count300000.csv',
        batch_size=128,
        max_length=20,
        num_ids=50,
        miss_ratio=0.3,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
    )
    
    # 模型参数
    model_config = {
        "num_ids": 50,  # 根据你的数据集修改
        "seq_length": 20,
        "ch": 64,
        "ch_mult": (1, 2, 4, 8),
        "num_res_blocks": 2,
        "dropout": 0.1,
        "learning_rate": 1e-4,
        "num_timesteps": 500
    }
    
    # 创建模型
    model = UNetTransformerDiffusion(**model_config).to(device)
    
    # 创建优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=model_config["learning_rate"])
    
    # 记录训练过程
    losses = []
    accuracies = []
    masked_accuracies = []
    
    # 开始训练
    print("\n=== 开始训练 ===")
    
    # 每个epoch的训练循环
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        epoch_losses = []
        
        # 训练一个epoch
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            original_ids, masked_ids = [b.to(device) for b in batch]

            # 计算损失
            loss = model.training_step((original_ids, masked_ids), 0)
            epoch_losses.append(loss.item())
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 优化器步进
            optimizer.step()
        
        # 记录当前epoch的平均损失
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        # 打印训练损失
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(f"Average Loss: {avg_loss:.6f}")
        
        if (epoch + 1) % 2 == 0:
            accuracy, masked_accuracy = evaluate_model(model, val_loader, device)
            accuracies.append(accuracy)
            if masked_accuracy is not None:
                masked_accuracies.append(masked_accuracy)
            
            print(f"Accuracy: {accuracy:.4f}")
            if masked_accuracy is not None:
                print(f"Masked Accuracy: {masked_accuracy:.4f}")
            
            # 绘制训练曲线
            plot_training_curves(
                losses, 
                accuracies,
                masked_accuracies if masked_accuracies else None,
                save_path=f'training_curves_epoch_{epoch+1}.png'
            )
    
    # 测试模型
    print("\n=== 模型测试 ===")
    test_accuracy, test_masked_accuracy = evaluate_model(model, test_loader, device)
    print(f"测试集准确率: {test_accuracy:.4f}")
    if test_masked_accuracy is not None:
        print(f"测试集掩码位置准确率: {test_masked_accuracy:.4f}")

if __name__ == "__main__":
    main() 