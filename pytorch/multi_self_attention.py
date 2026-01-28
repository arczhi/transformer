import torch
import torch.nn as nn
import torch.nn.functional as F
import math

"""
多头注意力的并行体现在两处：

1) 形状扩展：_split_heads 把 [batch, seq_len, d_model] reshape 成 [batch, num_heads, seq_len, head_dim]。这一额外的 num_heads 维度告诉 PyTorch，后续运算要在每个 head 的子空间上同时进行。
2) 统一算子：scaled_dot_product_attention(q, k, v, mask) 里 torch.matmul、F.softmax 等操作都接受带有 num_heads 的张量。PyTorch 会把这个维度当成批次维度的一部分，在底层一次性对所有 head 做矩阵乘、softmax、加权求和；不需要手动循环逐头计算。
因此，每次前向传播时，所有注意力头共享同一套张量操作，利用 GPU/向量化实现真正的“并行”。
"""

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    缩放点积注意力（支持多头，输入形状需包含 num_heads 维度）。
    参数:
        query: [batch_size, num_heads, seq_len, head_dim]
        key:   [batch_size, num_heads, seq_len, head_dim]
        value: [batch_size, num_heads, seq_len, head_dim]
        mask:  [batch_size, 1, seq_len, seq_len] 或相同形状（可选）
    """
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attention_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, value)
    return output, attention_weights


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        """
        与单头不同：引入 num_heads，并将总维度均匀分配到多个 head。
        """
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model 必须能被 num_heads 整除")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads # 整除 向下取整

        # 关键差异：使用 d_model->d_model 的线性层，之后再 reshape 成多个 head。
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

    def forward(self, x, mask=None):
        """多头流程：线性投影 -> 切分多头 -> 缩放点积 -> 拼接 -> 输出映射。"""
        residual = x

        q = self._split_heads(self.W_q(x))
        k = self._split_heads(self.W_k(x))
        v = self._split_heads(self.W_v(x))

        if mask is not None:
            if mask.dim() == 3:
                mask = mask.unsqueeze(1)  # 关键差异：扩展 mask 以匹配多头维度。

        attention_output, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        attention_output = self._merge_heads(attention_output)

        out = self.W_o(attention_output)
        out = self.dropout(out)
        out = self.layer_norm(residual + out)
        return out, attention_weights


def demo(batch=2, seq_len=4, d_model=8, num_heads=2):
    x = torch.randn(batch, seq_len, d_model)
    mask = torch.ones(batch, seq_len, seq_len)
    attention_layer = MultiHeadSelfAttention(d_model, num_heads)
    output, attention_weights = attention_layer(x, mask)

    print(f"输入形状 input: {x.shape}")
    print(f"注意力输出形状 output: {output.shape}")
    print(f"注意力权重形状 attention_weights: {attention_weights.shape}")
    print(f"注意力权重 attention_weights: {attention_weights}")


if __name__ == "__main__":
    demo()
