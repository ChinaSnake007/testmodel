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
    
    # 获取所有非对角线位置的索引
    non_diagonal_indices = []
    for i in range(id_count):
        for j in range(id_count):
            if i != j:  # 排除对角线元素
                non_diagonal_indices.append((i, j))
    
    # 随机选择位置设置为1
    selected_indices = random.sample(non_diagonal_indices, ones_count)
    for i, j in selected_indices:
        matrix[i][j] = 1
    
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
    
    for _ in range(seq_len - 1):
        # 获取当前ID可以连接的所有可能的下一个ID
        possible_next = [i+1 for i in range(id_count) if adj_matrix[current_id-1][i] == 1]
        if not possible_next:
            # 如果没有可用的下一个ID，随机选择一个
            current_id = random.randint(1, id_count)
        else:
            current_id = random.choice(possible_next)
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
    # 示例使用
    id_count = 50  # ID数量
    ratio = 0.4   # 邻接矩阵中1的占比
    seq_len = 20  # 序列长度
    count = 300000     # 生成序列的数量
    
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