import pandas as pd
from pathlib import Path

def process_trajectories():
    # 读取grid_id映射字典
    mapping_df = pd.read_csv('grid_id_mapping.csv')
    grid_id_dict = dict(zip(mapping_df['grid_id'], mapping_df['mapped_id']))
    
    # 获取CSV文件列表
    csv_dir = Path('simplified_trajectory')
    csv_files = list(csv_dir.glob('*.csv'))[:3]
    
    # 用于存储所有轨迹
    all_trajectories = []
    
    # 处理每个CSV文件
    for csv_file in csv_files:
        print(f"Processing {csv_file.name}...")
        df = pd.read_csv(csv_file)
        
        # 按userID分组，组内按arrive_time排序
        grouped = df.groupby('userID')
        
        # 处理每个用户的轨迹
        for _, group in grouped:
            # 按时间排序
            sorted_group = group.sort_values('arrive_time')
            
            # 将grid_id转换为映射后的id
            trajectory = [grid_id_dict[int(grid_id)] for grid_id in sorted_group['grid_id']]
            
            # 只保留长度在8-32之间的轨迹
            if 8 <= len(trajectory) <= 32:
                all_trajectories.append({
                    'trajectory': trajectory
                })
    
    # 将结果保存为CSV，包含轨迹序列和列名
    result_df = pd.DataFrame(all_trajectories)
    result_df.to_csv('trajectories_sample.csv', index=False)
    print(f"Processed {len(all_trajectories)} valid trajectories from {len(csv_files)} files")
    print("Results saved to trajectories_sample.csv")

if __name__ == "__main__":
    process_trajectories() 