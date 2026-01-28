"""
词嵌入层
1) 将离散的词汇索引转换为连续的向量（像Word2Vec/GloVe）
2) 与位置编码结合，为Transformer提供输入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TokenEmbedding(nn.Module):
    """
    vocab_size词表大小 d_model嵌入维度
    """
    def __init__(self,vocab_size:int,d_model:int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # 创建嵌入矩阵 [vocab_size,d_model]
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=0  # 通常0用于填充（可选）
        )
        
        # 初始化权重 重要！
        self.__init_weights()
    
    def __init_weights(self):
        """
        初始化嵌入权重
        论文中没有明确指定，但通常使用正态分布或Xavier初始化
        """
        # 方法1: Xavier均匀分布初始化（推荐）
        nn.init.xavier_uniform_(self.embedding.weight)
        
        # 方法2: 正态分布初始化（也可使用）
        # nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model**-0.5)
        
        # 如果有padding_idx，将其权重设为0
        if self.embedding.padding_idx is not None:
            nn.init.constant_(self.embedding.weight[self.embedding.padding_idx], 0)
            
    def forward(self,x: torch.Tensor) ->torch.Tensor:
        
        # 执行嵌入查找
        embeddings = self.embedding(x)
        
        # 缩放嵌入：乘以 sqrt(d_model) （论文中的做法）
        # 这有助于在添加位置编码前控制嵌入的幅度
        embeddings = embeddings * math.sqrt(self.d_model)
        
        return embeddings
    
    def get_embedding_weights(self):
        """获取嵌入权重矩阵"""
        return self.embedding.weight
    
    def set_embedding_weights(self, pretrained_embeddings: torch.Tensor, freeze: bool = True):
        """
        加载预训练嵌入权重
        
        参数:
            pretrained_embeddings: 预训练嵌入矩阵
            freeze: 是否冻结嵌入层（不更新权重）
        """
        if pretrained_embeddings.shape != (self.vocab_size, self.d_model):
            raise ValueError(f"形状不匹配: 期望{(self.vocab_size, self.d_model)}, "
                           f"得到{pretrained_embeddings.shape}")
        
        self.embedding.weight.data = pretrained_embeddings
        if freeze:
            self.embedding.weight.requires_grad = False
        
        
def test_token_embedding():
    """测试词嵌入层"""
    
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
    
    return token_embedding

# 运行测试
token_embedding = test_token_embedding()