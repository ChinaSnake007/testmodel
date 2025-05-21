import pytorch_lightning as pl
from data_utils import create_dataloaders
from unet_transformer_diffusion_maxid_state import UNetTransformerDiffusion
import torch
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
# 设置matplotlib中文字体
import matplotlib.pyplot as plt

# 设置matplotlib字体和负号显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用系统自带的黑体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号
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
            original_ids, one_hot, masked_ids, mask = [b.to(device) for b in batch]
            
            # 生成预测
            predictions = model.sample(masked_ids)
            
            # 计算准确率
            accuracy, masked_accuracy = calculate_accuracy(predictions, original_ids, mask)
            
            total_accuracy += accuracy
            if masked_accuracy is not None:
                total_masked_accuracy += masked_accuracy
            num_batches += 1
            
            # 更新进度条显示当前batch的准确率
            pbar.set_postfix({
                '整体准确率': f'{accuracy:.4f}',
                '掩码准确率': f'{masked_accuracy:.4f}' if masked_accuracy is not None else 'N/A'
            })
    
    avg_accuracy = total_accuracy / num_batches
    avg_masked_accuracy = total_masked_accuracy / num_batches if masked_accuracy is not None else None
    
    return avg_accuracy, avg_masked_accuracy

def main():
    # 设置设备和优化选项
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='指定使用的GPU ID')
    parser.add_argument('--csv_path', type=str, default='test_data/sequences_id50_ratio0.4_len20_count300000.csv', help='数据文件路径')
    parser.add_argument('--batch_size', type=int, default=128, help='批次大小')
    parser.add_argument('--max_length', type=int, default=20, help='序列最大长度')
    parser.add_argument('--num_ids', type=int, default=50, help='ID数量')
    parser.add_argument('--miss_ratio', type=float, default=0.3, help='缺失率')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例')
    parser.add_argument('--num_epochs', type=int, default=10, help='训练迭代次数')
    parser.add_argument('--ch', type=int, default=64, help='UNet基础通道数')
    parser.add_argument('--num_res_blocks', type=int, default=2, help='残差块数量')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='学习率')
    parser.add_argument('--num_timesteps', type=int, default=500, help='扩散步数')
    args = parser.parse_args()
    
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.gpu)  # 设置当前使用的GPU
    print(f"使用设备: {device}")
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path=args.csv_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_ids=args.num_ids,
        miss_ratio=args.miss_ratio,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )
    
    
    # 模型参数    
    model_config = {
        'num_ids': args.num_ids,
        'seq_length': args.max_length,
        'ch': args.ch,
        'ch_mult': (1, 2, 4, 8),
        'num_res_blocks': args.num_res_blocks,
        'dropout': args.dropout,
        'learning_rate': args.learning_rate,
        'num_timesteps': args.num_timesteps
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
    for epoch in range(args.num_epochs):
        model.train()
        epoch_losses = []
        
        # 训练一个epoch
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for batch in pbar:
            original_ids, one_hot, masked_ids, mask = [b.to(device) for b in batch]

            # 计算损失
            loss = model.training_step((original_ids, masked_ids), 0)
            epoch_losses.append(loss.item())
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 优化器步进
            optimizer.step()
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        # 记录当前epoch的平均损失
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        
        # 打印训练损失
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print(f"Average Loss: {avg_loss:.6f}")
        
        if (epoch + 1) % 5 == 0:
            accuracy, masked_accuracy = evaluate_model(model, val_loader, device)
            accuracies.append(accuracy)
            if masked_accuracy is not None:
                masked_accuracies.append(masked_accuracy)
            
            print(f"整体准确率: {accuracy:.4f}")
            if masked_accuracy is not None:
                print(f"掩码位置准确率: {masked_accuracy:.4f}")
    
    # 测试模型
    print("\n=== 模型测试 ===")
    test_accuracy, test_masked_accuracy = evaluate_model(model, test_loader, device)
    print(f"测试集整体准确率: {test_accuracy:.4f}")
    if test_masked_accuracy is not None:
        print(f"测试集掩码位置准确率: {test_masked_accuracy:.4f}")
    
    # 绘制总的训练曲线
    filename = f'training_curves_id{args.num_ids}_len{args.max_length}_miss{args.miss_ratio}_epochs{args.num_epochs}_batch{args.batch_size}_lr{args.learning_rate}.png'
    plot_training_curves(
        losses, 
        accuracies,
        masked_accuracies if masked_accuracies else None,
        save_path=filename
    )
    print(f"\n训练曲线已保存至: {filename}")

if __name__ == "__main__":
    main() 