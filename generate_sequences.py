import numpy as np
import random
import pandas as pd
import os
from tqdm import tqdm

def generate_adjacency_matrix(id_count, ratio):
    """
    生成随机邻接矩阵，对角线元素为0（表示不能跳回自身）
    :param id_count: ID的数量
    :param ratio: 1在矩阵中的占比
    :return: 邻接矩阵
    """
    # 创建全0矩阵
    matrix = np.zeros((id_count, id_count), dtype=int)
    
    # 计算非对角线元素的数量
    non_diagonal_size = id_count * id_count - id_count
    # 计算需要设置为1的元素数量
    ones_count = int(non_diagonal_size * ratio)
    
    # 使用numpy的随机函数直接生成非对角线位置的索引
    indices = np.random.choice(id_count * id_count, ones_count, replace=False)
    rows = indices // id_count
    cols = indices % id_count
    
    # 过滤掉对角线元素
    mask = rows != cols
    matrix[rows[mask], cols[mask]] = 1
    
    return matrix

def generate_sequence(adj_matrix, seq_len):
    """
    基于邻接矩阵生成满足条件的序列
    :param adj_matrix: 邻接矩阵
    :param seq_len: 序列长度
    :return: 生成的序列
    """
    id_count = len(adj_matrix)
    sequence = []
    current_id = random.randint(1, id_count)
    sequence.append(current_id)
    
    # 预计算每个ID可以连接的下一个ID列表
    next_ids = {i+1: np.where(adj_matrix[i] == 1)[0] + 1 for i in range(id_count)}
    
    for _ in range(seq_len - 1):
        possible_next = next_ids[current_id]
        if len(possible_next) == 0:
            current_id = random.randint(1, id_count)
        else:
            current_id = np.random.choice(possible_next)
        sequence.append(current_id)
    
    return sequence

def generate_sequences(id_count, ratio, seq_len, count):
    """
    生成指定数量的序列
    :param id_count: ID的数量
    :param ratio: 邻接矩阵中1的占比
    :param seq_len: 序列长度
    :param count: 要生成的序列数量
    :return: 邻接矩阵和生成的序列列表
    """
    adj_matrix = generate_adjacency_matrix(id_count, ratio)
    sequences = [generate_sequence(adj_matrix, seq_len) for _ in tqdm(range(count), desc="生成序列")]
    return adj_matrix, sequences

if __name__ == "__main__":
    # 从命令行解析参数
    import argparse
    parser = argparse.ArgumentParser(description='生成序列数据')
    parser.add_argument('--id_count', type=int, default=50, help='ID数量')
    parser.add_argument('--ratio', type=float, default=0.2, help='邻接矩阵中1的占比')
    parser.add_argument('--seq_len', type=int, default=20, help='序列长度')
    parser.add_argument('--count', type=int, default=300000, help='生成序列的数量')
    args = parser.parse_args()

    # 获取参数
    id_count = args.id_count
    ratio = args.ratio
    seq_len = args.seq_len
    count = args.count
    
    adj_matrix, sequences = generate_sequences(id_count, ratio, seq_len, count)
    
    
    # 创建DataFrame并保存到CSV
    df = pd.DataFrame(sequences)
    
    # 创建文件名，包含参数信息
    filename = f"test_data/sequences_id{id_count}_ratio{ratio}_len{seq_len}_count{count}.csv"
    
    # 将序列转换为字符串格式
    df = pd.DataFrame({'trajectory': [str(seq) for seq in sequences]})
    # 保存到CSV文件
    df.to_csv(filename, index=False)
    print(f"\n序列已保存到文件：{filename}")