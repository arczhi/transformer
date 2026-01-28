import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
多头注意力的并行体现在两处：

1) 形状扩展：_split_heads 把 [batch, seq_len, d_model] reshape 成 [batch, num_heads, seq_len, head_dim]。这一额外的 num_heads 维度告诉 PyTorch，后续运算要在每个 head 的子空间上同时进行。
2) 统一算子：scaled_dot_product_attention(q, k, v, mask) 里 torch.matmul、F.softmax 等操作都接受带有 num_heads 的张量。PyTorch 会把这个维度当成批次维度的一部分，在底层一次性对所有 head 做矩阵乘、softmax、加权求和；不需要手动循环逐头计算。
因此，每次前向传播时，所有注意力头共享同一套张量操作，利用 GPU/向量化实现真正的"并行"。

支持模式：
- 自注意力（Self-Attention）：Q=K=V，用于同一序列内部注意
- 掩码自注意力（Masked Self-Attention）：Q=K=V + 因果掩码，用于自回归解码
- 交叉注意力（Cross-Attention）：Q来自一个序列，K/V来自另一个，用于Encoder-Decoder
"""

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    缩放点积注意力（支持多头，输入形状需包含 num_heads 维度）。
    参数:
        query: [batch_size, num_heads, seq_len_q, head_dim]
        key:   [batch_size, num_heads, seq_len_k, head_dim]
        value: [batch_size, num_heads, seq_len_v, head_dim]
        mask:  [batch_size, 1, seq_len_q, seq_len_k] 或相同形状（可选）
              1 表示允许关注，0 表示屏蔽（不允许关注）
    返回:
        output: [batch_size, num_heads, seq_len_q, head_dim]
        attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
    """
    # 1) 相关性评分 - 查询矩阵和键矩阵进行矩阵乘法
    #维度
    d_k = query.size(-1) 
    #key矩阵的最后两个维度（行列）进行倒置，以便进行矩阵乘法计算
    scores = torch.matmul(query,key.transpose(-2,-1))
    # 2）缩放 - 使用d_k的平方根，防止值过大，让softmax更稳定
    scores = scores / math.sqrt(d_k)
    # 3) 掩码 - 掩掉不需要关注的位置，例如未来的token或padding，避免对处理中及之前的产生影响
    if mask is not None:
        # 掩码为0的地方 替换成负无穷 -inf
        scores = scores.masked_fill(mask==0,float('-inf'))
    # 4) softmax进行归一化处理 确保各项之和为1 处理最后一维（对最后一维进行归一化处理）
    # 注意力权重整体就是“一批样本 × 查询位置 × 被关注位置”，形成 [batch_size, num_heads,seq_len_q, seq_len_k]
    attention_weights = F.softmax(scores,dim=-1)
    # 5) 乘以值矩阵 - 注意力权重加权求和
    # attention_weights [batch_size, num_heads, seq_len_q, seq_len_k]
    # value [batch_size, num_heads, seq_len_v, head_dim]
    # `softmax` 之后得到的 `attention_weights[b,h,i, :]`：就是第 i 个位置对整句所有位置的注意力分布，所有元素加起来 = 1。
    # - 再乘 `V`，就是拿这些权重，对所有 token 的 `value` 做加权求和。
    output = torch.matmul(attention_weights,value)
    return output,attention_weights


class MultiHeadAttention(nn.Module):
    """
    多头注意力模块，支持自注意力、掩码自注意力和交叉注意力。
    """
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        参数:
            d_model: 模型维度（必须能被 num_heads 整除）
            num_heads: 注意力头数
            dropout: dropout 比率
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # 线性投影层
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def _split_heads(self, x):
        """把 [batch, seq_len, d_model] reshape 成 [batch, num_heads, seq_len, head_dim]。"""
        batch_size, seq_len, _ = x.shape
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x):
        """把 [batch, num_heads, seq_len, head_dim] 合回 [batch, seq_len, d_model]。"""
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch_size, seq_len, self.d_model)

    def forward(self, query, key=None, value=None, mask=None):
        """
        多头注意力前向传播。支持三种模式：
        
        模式1 - 自注意力（Self-Attention）：
            forward(x) 或 forward(x, x, x)
            Q=K=V，用于同一序列内部注意
            
        模式2 - 掩码自注意力（Masked Self-Attention）：
            forward(x, mask=causal_mask) 或 forward(x, x, x, mask=causal_mask)
            Q=K=V + 因果掩码，用于自回归解码（decoder）
            
        模式3 - 交叉注意力（Cross-Attention）：
            forward(query, key, value) 或 forward(query, key, value, mask)
            Q来自一个序列，K/V来自另一个，用于Encoder-Decoder
            Q来自解码器的序列 K/V来自编码器的序列
        
        参数:
            query: [batch_size, seq_len_q, d_model]
            key: [batch_size, seq_len_k, d_model]（可选，默认=query）
            value: [batch_size, seq_len_v, d_model]（可选，默认=query）
            mask: [batch_size, seq_len_q, seq_len_k] 或 [batch_size, 1, seq_len_q, seq_len_k]
                  1 表示允许关注，0 表示屏蔽
        
        返回:
            output: [batch_size, seq_len_q, d_model]
            attention_weights: [batch_size, num_heads, seq_len_q, seq_len_k]
        """
        # 如果 key/value 未提供，则为自注意力
        if key is None:
            key = query
        if value is None:
            value = query
        
        # 保存输入用于残差连接（只对 query 做残差）
        residual = query
        
        # 线性投影并拆分成多个头
        q = self._split_heads(self.W_q(query))  # [batch, num_heads, seq_len_q, head_dim]
        k = self._split_heads(self.W_k(key))    # [batch, num_heads, seq_len_k, head_dim]
        v = self._split_heads(self.W_v(value))  # [batch, num_heads, seq_len_v, head_dim]

        # 处理掩码：如果是 3D 掩码，扩展为 4D 以匹配多头维度
        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # [batch, 1, seq_len_q, seq_len_k]

        # 缩放点积注意力
        attention_output, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        # 合并多个头
        attention_output = self._merge_heads(attention_output)  # [batch, seq_len_q, d_model]

        # 输出投影
        out = self.W_o(attention_output)
        
        # Dropout + 残差连接 + 层归一化
        out = self.dropout(out)
        out = self.layer_norm(residual + out)
        
        return out, attention_weights
    
    def attention_visualization(self, attention_weights, seq_len=10):
        """
        可视化注意力权重
        
        参数:
            attention_weights: 注意力权重张量
                - 4D: [batch, num_heads, seq_len, seq_len] 
                - 3D: [batch, seq_len, seq_len] (已平均或单头)
            seq_len: 要可视化的序列长度
        """
        import matplotlib.pyplot as plt
        
        # 检查维度并提取可视化数据
        if attention_weights.dim() == 4:
            # 4D: [batch, num_heads, seq_len, seq_len]
            # 取第一个batch，第一个头
            attn_head = attention_weights[0, 0, :seq_len, :seq_len].detach().cpu().numpy()
        elif attention_weights.dim() == 3:
            # 3D: [batch, seq_len, seq_len]
            # 取第一个batch
            attn_head = attention_weights[0, :seq_len, :seq_len].detach().cpu().numpy()
        else:
            raise ValueError(f"期望 3D 或 4D 张量，但得到 {attention_weights.dim()}D")
        
        plt.figure(figsize=(8, 6))
        plt.imshow(attn_head, cmap='viridis')
        plt.colorbar()
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        plt.title("Attention Weights (Head 0)")
        plt.show()


def demo_self_attention():
    """演示1：自注意力（Self-Attention）"""
    print("=" * 50)
    print("演示1：自注意力（Self-Attention）")
    print("=" * 50)
    
    batch_size, seq_len, d_model, num_heads = 2, 4, 8, 2
    x = torch.randn(batch_size, seq_len, d_model)
    attention_layer = MultiHeadAttention(d_model, num_heads)
    
    # 自注意力：只传 query，内部自动设置 key=value=query
    output, attention_weights = attention_layer(x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"输出=输入维度: {output.shape == x.shape}")
    print()


def demo_masked_self_attention():
    """演示2：掩码自注意力（Masked Self-Attention）"""
    print("=" * 50)
    print("演示2：掩码自注意力（Masked Self-Attention - 因果掩码）")
    print("=" * 50)
    
    batch_size, seq_len, d_model, num_heads = 2, 4, 8, 2
    x = torch.randn(batch_size, seq_len, d_model)
    attention_layer = MultiHeadAttention(d_model, num_heads)
    
    # 创建因果掩码（下三角矩阵）：每个位置只能看到当前及之前的位置
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))  # [seq_len, seq_len]
    
    # 掩码自注意力
    output, attention_weights = attention_layer(x, mask=causal_mask)
    
    print(f"输入形状: {x.shape}")
    print(f"因果掩码:\n{causal_mask}")
    print(f"输出形状: {output.shape}")
    print(f"注意力权重形状: {attention_weights.shape}")
    
    # 验证因果性：位置 i 对位置 j>i 的注意力应该为 0
    print(f"\n验证因果性（第一个样本第一头）:")
    weights_sample = attention_weights[0, 0]  # [seq_len, seq_len]
    print(f"位置0对位置1的注意力: {weights_sample[0, 1].item():.6f}")
    print(f"位置1对位置0的注意力: {weights_sample[1, 0].item():.6f}")
    print(f"位置1对位置2的注意力: {weights_sample[1, 2].item():.6f}")
    print()


def demo_cross_attention():
    """演示3：交叉注意力（Cross-Attention）"""
    print("=" * 50)
    print("演示3：交叉注意力（Cross-Attention）")
    print("=" * 50)
    
    batch_size, d_model, num_heads = 2, 8, 2
    seq_len_q, seq_len_k = 4, 6  # Query 和 Key-Value 的序列长度可以不同
    
    query = torch.randn(batch_size, seq_len_q, d_model)
    key = torch.randn(batch_size, seq_len_k, d_model)
    value = torch.randn(batch_size, seq_len_k, d_model)
    
    attention_layer = MultiHeadAttention(d_model, num_heads)
    
    # 交叉注意力：Q 和 K/V 来自不同序列
    output, attention_weights = attention_layer(query, key, value)
    
    print(f"Query 形状: {query.shape}")
    print(f"Key 形状: {key.shape}")
    print(f"Value 形状: {value.shape}")
    print(f"输出形状: {output.shape} (应该与 Query 长度相同)")
    print(f"注意力权重形状: {attention_weights.shape}")
    print(f"输出序列长度 = Query 序列长度: {output.shape[1] == query.shape[1]}")
    print()


if __name__ == "__main__":
    demo_self_attention()
    demo_masked_self_attention()
    demo_cross_attention()
