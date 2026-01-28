"""
    自注意力机制没有位置信息 需要显式地进行位置编码
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

d_model = 512
max_seq_len = 100

class PositionalEncoding(nn.Module):
    def __init__(self,d_model:int,max_seq_len: int = 5000,dropout:float=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        #初始化位置编码矩阵
        pe = torch.zeros(max_seq_len,d_model)
        
        #位置索引
        position = torch.arange(0,max_seq_len).unsqueeze(1).float()
        # unsqueeze 把[100]变成[100,1] 在第1维插入了新的维度
        
        # 计算频率项
        # torch.arange 指定起始值、结束值和步长 生成一个等差数列的张量 相当于2i 
        # exp(指数) 相当于: exp(-2i * log(10000) / d_model) = 10000^(-2i/d_model)
        div_term = torch.exp(torch.arange(0,d_model,2).float() * -np.log(10000)/d_model)
        
        
        # sin处理所有行的偶数列
        # cos处理所有行的奇数列
        pe[:,0::2] = torch.sin(position * div_term)
        pe[:,1::2] = torch.cos(position * div_term)
        
        # 添加batch维度并注册为buffer（不参与训练）
        pe = pe.unsqueeze(0) # [1,max_seq_len,d_model]
        self.register_buffer('pe',pe)
    
    def forward(self,x: torch.Tensor) -> torch.Tensor:
        #x [batch,seq_len,d_model]
        #切片获取与输入序列长度匹配的位置编码
        seq_len = x.size(1) #获取第1维的大小,即seq_len
        x = x+self.pe[:,:seq_len,:]
        return self.dropout(x)
    
    def visualize(self, max_positions=50, max_dims=64):
        """可视化位置编码的热图"""
        pos_enc = self.pe.squeeze(0)[:max_positions, :max_dims]
        
        plt.figure(figsize=(12, 6))
        plt.imshow(pos_enc.T, aspect='auto', cmap='RdBu')
        plt.colorbar()
        plt.xlabel('Position')
        plt.ylabel('Dimension')
        plt.title('Positional Encoding Heatmap')
        plt.show()
        
        
        
def test_positional_encoding():
    """测试位置编码功能"""
    # 创建实例
    pos_encoder = PositionalEncoding(d_model=512, max_seq_len=100)
    
    # 创建模拟输入 (类似随机词嵌入)
    batch_size = 4
    seq_len = 20
    x = torch.randn(batch_size, seq_len, 512)  # [batch, seq_len, d_model]
    
    print("输入形状:", x.shape)
    print("位置编码形状:", pos_encoder.pe.shape)
    
    # 应用位置编码
    output = pos_encoder(x)
    print("输出形状:", output.shape)
    
    # 验证位置编码的特性
    print("\n=== 位置编码特性验证 ===")
    
    # 1. 不同位置的编码应该不同
    pos1 = pos_encoder.pe[0, 0, :10]  # 第一个位置的前10维
    pos2 = pos_encoder.pe[0, 1, :10]  # 第二个位置的前10维
    print(f"位置0 vs 位置1 (前10维):")
    print(f"位置0: {pos1.numpy().round(3)}")
    print(f"位置1: {pos2.numpy().round(3)}")
    print(f"是否不同: {not torch.allclose(pos1, pos2)}")
    
    # 2. 相对位置关系 - 可以通过点积验证
    print("\n=== 相对位置关系 ===")
    positions = [0, 1, 2, 5, 10]
    print("位置之间的余弦相似度:")
    for i in positions:
        for j in positions:
            vec_i = pos_encoder.pe[0, i]
            vec_j = pos_encoder.pe[0, j]
            similarity = torch.cosine_similarity(vec_i.unsqueeze(0), 
                                                vec_j.unsqueeze(0)).item()
            print(f"pos{i}-pos{j}: {similarity:.3f}", end=" | ")
        print()
    
    return pos_encoder, output

# 运行测试
pos_encoder, output = test_positional_encoding()

def test_visualization():
    """可视化测试位置编码"""
    # 初始化位置编码
    pos_encoder = PositionalEncoding(d_model=512, dropout=0.1)
    
    # 创建测试输入
    batch_size = 2
    seq_len = 30
    x = torch.randn(batch_size, seq_len, 512)
    
    print("测试位置编码层:")
    print(f"输入形状: {x.shape}")
    
    # 前向传播
    output = pos_encoder(x)
    print(f"输出形状: {output.shape}")
    print(f"是否保留梯度: {output.requires_grad}")
    
    # 可视化
    pos_encoder.visualize()

# test_visualization()
        
        