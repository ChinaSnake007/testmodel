import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from typing import List, Tuple, Optional
from tqdm import tqdm

class TrajectoryDataset(Dataset):
    def __init__(self, csv_path: str, max_length: int = 32, num_ids: int = 3953, miss_ratio: float = 0.3):
        self.max_length = max_length
        self.num_ids = num_ids
        self.miss_ratio = miss_ratio
        self.pad_token = num_ids + 1  # 使用num_ids+1作为padding标记
        
        # 只读取原始数据，不进行预处理
        print("正在加载数据...")
        df = pd.read_csv(csv_path)
        self.trajectories = df['trajectory'].apply(eval).tolist()
        print(f"数据加载完成，共{len(self.trajectories)}条轨迹")
        
    def __len__(self):
        return len(self.trajectories)
    
    def __getitem__(self, idx):
        traj = self.trajectories[idx]
        traj_len = min(len(traj), self.max_length)
        
        # 创建原始id序列（带padding）
        original_ids = torch.full((self.max_length,), self.pad_token, dtype=torch.long)
        original_ids[:traj_len] = torch.tensor(traj[:traj_len])
        
        # 创建one-hot编码
        x = torch.zeros(self.max_length, self.num_ids + 2)  # +2 for padding token and missing token
        x[torch.arange(traj_len), original_ids[:traj_len]] = 1  # 使用向量化操作
        
        # 创建mask（用于条件生成）
        mask = torch.zeros(self.max_length)
        mask[:traj_len] = 1  # 先将有效位置标记为1
        
        # 创建带缺失值的id序列
        masked_ids = original_ids.clone()
        # 只对非padding位置进行缺失处理
        valid_positions = torch.where(mask == 1)[0]
        # 随机选择位置进行缺失
        miss_positions = valid_positions[torch.rand(len(valid_positions)) < self.miss_ratio]
        masked_ids[miss_positions] = 0  # 将选中的位置设置为0
        # 将缺失位置在mask中也标记为0
        mask[miss_positions] = 0
        
        return original_ids, x, masked_ids, mask

def create_dataloaders(
    csv_path: str,
    batch_size: int = 32,
    max_length: int = 32,
    num_ids: int = 3953,
    miss_ratio: float = 0.3,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    创建训练集、验证集和测试集的数据加载器
    
    Args:
        csv_path: CSV文件路径
        batch_size: 批次大小
        max_length: 最大序列长度
        num_ids: ID总数
        miss_ratio: 缺失率
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例
        seed: 随机种子
    
    Returns:
        训练集、验证集和测试集的数据加载器
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 创建数据集
    dataset = TrajectoryDataset(csv_path, max_length, num_ids, miss_ratio)
    
    # 计算各集合的大小
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size
    
    # 分割数据集
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    print(f"\n数据集划分完成:")
    print(f"训练集: {len(train_dataset)}条")
    print(f"验证集: {len(val_dataset)}条")
    print(f"测试集: {len(test_dataset)}条")
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,  # 使用pin_memory加速数据传输
        prefetch_factor=2,  # 预取因子
        persistent_workers=True,  # 保持worker进程存活
        drop_last=True  # 丢弃不完整的最后一个batch
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        drop_last=True
    )
    
    return train_loader, val_loader, test_loader

def main():
    # 创建数据加载器
    print("初始化数据加载器...")
    train_loader, val_loader, test_loader = create_dataloaders(
        csv_path='trajectories_sample.csv',
        batch_size=4,
        max_length=32,
        num_ids=3953,
        miss_ratio=0.3
    )
    
    # 获取一个批次的数据
    print("\n开始处理数据批次...")
    for loader, name in [(train_loader, "训练集"), (val_loader, "验证集"), (test_loader, "测试集")]:
        print(f"\n{name}数据示例:")
        # 移除tqdm，直接使用for循环
        for original_ids, x, masked_ids, mask in loader:
            print(f"\n批次数据形状:")
            print(f"原始ID序列形状: {original_ids.shape}")
            print(f"One-hot编码形状: {x.shape}")
            print(f"带缺失值的ID序列形状: {masked_ids.shape}")
            print(f"掩码形状: {mask.shape}")
            
            # 打印第一个样本的详细信息
            print("\n第一个样本的详细信息:")
            print("原始ID序列:")
            print(original_ids[0].numpy())
            print("\n带缺失值的ID序列:")
            print(masked_ids[0].numpy())
            print("\n掩码 (1=有效位置, 0=缺失或padding):")
            print(mask[0].numpy())
            break

if __name__ == "__main__":
    main() 