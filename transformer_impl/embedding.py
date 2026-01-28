"""
嵌入层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import matplotlib.pyplot as plt
try:
    # 当作为包执行（推荐）
    from .positional_encoding import PositionalEncoding
except Exception:
    # 当直接用 `python embedding.py` 运行时（临时兼容）
    from positional_encoding import PositionalEncoding
try :
    from .token_embedding import TokenEmbedding
except Exception:
    from token_embedding import TokenEmbedding

class TransformerEmbedding(nn.Module):
    """
    完整的Transformer嵌入层 = 词嵌入 + 位置编码
    """
    def __init__(self, vocab_size: int, d_model: int, max_seq_len: int = 5000, 
                 dropout: float = 0.1, padding_idx: int = 0):
        super().__init__()
        
        # 词嵌入层
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        
        # 位置编码层
        self.positional_encoding = PositionalEncoding(
            d_model=d_model, 
            max_seq_len=max_seq_len, 
            dropout=dropout
        )
        
        # 可选的层归一化（有些实现会加）
        self.layer_norm = nn.LayerNorm(d_model)
        
        # 额外的投影层（可选，用于调整维度）
        self.projection = None  # 可以设为线性层如果需要
        
        self.dropout_rate = dropout
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, seq_len]
        返回:
            嵌入向量 [batch_size, seq_len, d_model]
        """
        # 1. 词嵌入
        token_embeddings = self.token_embedding(x)
        
        # 2. 添加位置编码（在PositionalEncoding类中已经包含加法）
        embeddings = self.positional_encoding(token_embeddings)
        
        # 3. 层归一化（可选）
        embeddings = self.layer_norm(embeddings)
        
        return embeddings
    
    def get_token_embeddings(self):
        """获取词嵌入权重"""
        return self.token_embedding.get_embedding_weights()
    
    def visualize_embeddings(self, num_words=20, figsize=(12, 8)):
        """可视化部分词嵌入（使用PCA降维到2D）"""
        import numpy as np
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        
        # 获取嵌入权重
        embeddings = self.get_token_embeddings().detach().cpu().numpy()
        
        # 取前num_words个词（假设0是padding，从1开始）
        sample_embeddings = embeddings[1:num_words+1]
        
        # 使用PCA降维到2D
        pca = PCA(n_components=2)
        reduced = pca.fit_transform(sample_embeddings)
        
        # 绘制
        plt.figure(figsize=figsize)
        plt.scatter(reduced[:, 0], reduced[:, 1])
        
        # 添加标签
        for i in range(num_words):
            plt.annotate(f"word_{i+1}", (reduced[i, 0], reduced[i, 1]))
        
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.title(f"Word Embeddings Visualization (First {num_words} words)")
        plt.grid(True, alpha=0.3)
        plt.show()
        
        # 打印解释方差比
        print(f"PCA解释方差比: {pca.explained_variance_ratio_}")
        print(f"总解释方差: {sum(pca.explained_variance_ratio_):.2%}")
        
        

def test_embedding_layers():
    """测试词嵌入层和完整嵌入层"""
    
    # 设置参数
    vocab_size = 10000  # 假设词汇表有10000个词
    d_model = 512
    batch_size = 4
    seq_len = 20
    
    print("=== 测试基础词嵌入层 ===")
    
    # 测试TokenEmbedding
    token_embedding = TokenEmbedding(vocab_size, d_model)
    
    # 创建模拟输入（词汇索引，范围在[0, vocab_size-1]）
    # 假设0是padding，所以从1开始
    input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
    
    print(f"输入形状: {input_ids.shape}")
    print(f"输入示例（第一行前5个）: {input_ids[0, :5].tolist()}")
    
    # 前向传播
    token_embeddings = token_embedding(input_ids)
    print(f"词嵌入输出形状: {token_embeddings.shape}")
    print(f"词嵌入示例（第一个词）: {token_embeddings[0, 0, :10]}")  # 显示前10维
    
    # 验证缩放
    print(f"\n=== 验证缩放 ===")
    # 嵌入权重
    weights = token_embedding.get_embedding_weights()
    print(f"嵌入权重形状: {weights.shape}")
    
    # 检查是否应用了缩放
    unnormalized = weights[input_ids[0, 0]]  # 未缩放的嵌入
    normalized = token_embeddings[0, 0]  # 缩放后的嵌入
    scaling_factor = math.sqrt(d_model)
    
    print(f"缩放因子: {scaling_factor:.2f}")
    print(f"未缩放向量范数: {torch.norm(unnormalized):.2f}")
    print(f"缩放后向量范数: {torch.norm(normalized):.2f}")
    print(f"缩放因子×未缩放范数: {scaling_factor * torch.norm(unnormalized):.2f}")
    
    print("\n=== 测试完整Transformer嵌入层 ===")
    
    # 测试完整嵌入层
    transformer_embedding = TransformerEmbedding(
        vocab_size=vocab_size,
        d_model=d_model,
        max_seq_len=100,
        dropout=0.1
    )
    
    # 前向传播
    embeddings = transformer_embedding(input_ids)
    print(f"完整嵌入层输出形状: {embeddings.shape}")
    
    # 检查位置编码的影响
    print(f"\n=== 检查位置编码效果 ===")
    
    # 创建一个简单的测试：相同词在不同位置
    test_input = torch.tensor([[1, 1, 1, 1],  # 4个相同的词
                               [2, 2, 2, 2]])
    
    # 获取嵌入
    test_embeddings = transformer_embedding(test_input)
    
    print(f"测试输入形状: {test_input.shape}")
    print(f"位置1和位置2的嵌入是否相同？")
    print(f"  batch0: {torch.allclose(test_embeddings[0, 0], test_embeddings[0, 1], rtol=1e-3)}")
    print(f"  batch1: {torch.allclose(test_embeddings[1, 0], test_embeddings[1, 1], rtol=1e-3)}")
    
    # 查看差异
    diff = torch.abs(test_embeddings[0, 0] - test_embeddings[0, 1])
    print(f"  平均差异: {diff.mean():.6f}")
    print(f"  最大差异: {diff.max():.6f}")
    
    return token_embedding, transformer_embedding

# 运行测试
token_embedding, transformer_embedding = test_embedding_layers()