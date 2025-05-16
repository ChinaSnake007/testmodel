import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
from tqdm import tqdm

class TrajectoryDataset(Dataset):
    def __init__(self, csv_path: str, max_length: int = 32, num_ids: int = 3953, miss_ratio: float = 0.3):
        self.max_length = max_length
        self.num_ids = num_ids
        self.miss_ratio = miss_ratio
        self.pad_token = num_ids + 1  # 使用num_ids+1作为padding标记
        
        # 读取CSV文件
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
        for i in range(traj_len):
            x[i, original_ids[i]] = 1
        
        # 创建mask（用于条件生成）
        # 1表示有效位置（非padding且非缺失），0表示padding或缺失位置
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

def create_dataloader(csv_path: str, batch_size: int = 32, max_length: int = 32, num_ids: int = 3953, miss_ratio: float = 0.3):
    dataset = TrajectoryDataset(csv_path, max_length, num_ids, miss_ratio)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

def main():
    # 创建数据加载器
    print("初始化数据加载器...")
    dataloader = create_dataloader(
        csv_path='trajectories_sample.csv',
        batch_size=4,
        max_length=32,
        num_ids=3953,
        miss_ratio=0.3
    )
    
    # 获取一个批次的数据
    print("\n开始处理数据批次...")
    for original_ids, x, masked_ids, mask in tqdm(dataloader, desc="处理数据批次"):
        print("\n批次数据示例:")
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
        
        # 打印one-hot编码中的非零位置
        print("\nOne-hot编码中的非零位置:")
        for i in range(x[0].shape[0]):
            if mask[0, i] == 1:  # 只打印有效位置
                id_idx = torch.argmax(x[0, i]).item()
                print(f"位置 {i}: ID = {id_idx}")
        break

if __name__ == "__main__":
    main() 