下面直接给出**单头自注意力（Single-Head Self-Attention）**的 PyTorch 实现，并**一步步、从代码出发**讲清楚每一行在干什么。你已经懂原理了，这里更多是把「公式 → 代码」打通。

---

## 一、整体目标

实现一个这样的模块：

```text
输入: x ∈ ℝ^{batch_size × seq_len × d_model}
输出: y ∈ ℝ^{batch_size × seq_len × d_model}
同时返回注意力权重 matrix: attn_weights ∈ ℝ^{batch_size × seq_len × seq_len}
```

包含：
- 线性层生成 Q, K, V
- 缩放点积注意力 Scaled Dot-Product Attention
- 输出线性层
- Dropout（可选）
- 残差连接 + LayerNorm（对齐 Transformer Block 的习惯写法）

---

## 二、先实现「缩放点积注意力」

这是自注意力的数学核心：

\[
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
\]

对应代码如下：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    缩放点积注意力（单头版）

    参数:
        query: [batch_size, seq_len, d_k]
        key:   [batch_size, seq_len, d_k]
        value: [batch_size, seq_len, d_v]
        mask:  [seq_len, seq_len] 或 [batch_size, seq_len, seq_len]，元素为0或1（可选）
              1 表示允许关注，0 表示屏蔽（不允许关注）

    返回:
        output: [batch_size, seq_len, d_v]
        attention_weights: [batch_size, seq_len, seq_len]
    """
    # 1. 计算注意力分数 scores = QK^T
    #   query: [b, L, d_k]
    #   key:   [b, L, d_k] -> [b, d_k, L] (transpose 后)
    #   scores: [b, L, L]，每一行表示一个 token 对其他所有 token 的相关性
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1))  # Q × K^T

    # 2. 缩放：除以 sqrt(d_k)，让数值别太大，softmax 更稳定
    scores = scores / math.sqrt(d_k)

    # 3. 掩码（mask）
    if mask is not None:
        # mask == 0 的地方填充为 -inf，这样 softmax 之后概率会变成 0
        # 注意：mask 的形状可以是 [L, L]，会自动广播到 [b, L, L]
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 4. softmax 归一化，得到注意力权重
    #    在最后一维（即每行）上做 softmax，保证每一行的和为 1
    attention_weights = F.softmax(scores, dim=-1)  # [b, L, L]

    # 5. 用注意力权重加权求和 Value
    #    [b, L, L] × [b, L, d_v] -> [b, L, d_v]
    output = torch.matmul(attention_weights, value)

    return output, attention_weights
```

### 这一段怎么理解？

- `scores[b, i, j]`：表示**第 b 个样本中，第 i 个位置对第 j 个位置的“相关性分数”**。
- `softmax` 之后得到的 `attention_weights[b, i, :]`：就是第 i 个位置对整句所有位置的注意力分布，所有元素加起来 = 1。
- 再乘 `V`，就是拿这些权重，对所有 token 的 `value` 做加权求和。

---

## 三、实现单头自注意力模块

现在我们用上面的函数，包装成一个「可直接丢进模型里」的模块。

```python
class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, dropout=0.1):
        """
        单头自注意力模块

        参数:
            d_model: 输入/输出特征维度（例如 512）
            d_k:     Q、K 的特征维度
            d_v:     V 的特征维度
        """
        super(SingleHeadAttention, self).__init__()

        # 1. 三个线性层，把输入 x 投影成 Q、K、V
        self.W_q = nn.Linear(d_model, d_k)
        self.W_k = nn.Linear(d_model, d_k)
        self.W_v = nn.Linear(d_model, d_v)

        # 2. 输出线性层，把注意力输出再映射回 d_model 维度
        self.W_o = nn.Linear(d_v, d_model)

        # 3. Dropout & LayerNorm
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        前向传播

        参数:
            x: [batch_size, seq_len, d_model]
            mask: [seq_len, seq_len] 或 [batch_size, seq_len, seq_len]，0/1 掩码（可选）

        返回:
            out: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, seq_len, seq_len]
        """
        # 保存残差分支用
        residual = x

        # ========= 1. 生成 Q、K、V =========
        # 每个位置的 token 都会被线性层变换成三种“视角”
        Q = self.W_q(x)  # [b, L, d_k]
        K = self.W_k(x)  # [b, L, d_k]
        V = self.W_v(x)  # [b, L, d_v]

        # ========= 2. 计算缩放点积注意力 =========
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)
        # attn_output: [b, L, d_v]

        # ========= 3. 投影回 d_model 维度 =========
        out = self.W_o(attn_output)  # [b, L, d_model]

        # ========= 4. Dropout + 残差连接 + LayerNorm =========
        out = self.dropout(out)
        out = self.layer_norm(residual + out)  # 残差加回去再做归一化

        return out, attn_weights
```

### 每一步的直观比喻

- `W_q, W_k, W_v`：
  - 相当于给每个词准备 3 副不同的“眼镜”，让它从三个不同的角度看整句。
- `scaled_dot_product_attention`：
  - 用 Q 和 K 算“谁该看谁”，用 softmax 变成概率，再对 V 做加权求和。
- `W_o`：
  - 把注意力结果“重混合”回原来的特征空间（d_model）。
- `residual + LayerNorm`：
  - 残差：即使注意力层学得不好，至少还能“退回原输入”；
  - LayerNorm：让整体数值更稳定，训练更顺滑。

---

## 四、跑一个小 demo 看看形状与注意力分布

### 1. 基础不带 mask 的例子

```python
batch_size = 2
seq_len = 4
d_model = 8
d_k = 8
d_v = 8

x = torch.randn(batch_size, seq_len, d_model)  # 随机输入

attention_layer = SingleHeadAttention(d_model, d_k, d_v)

out, attn_weights = attention_layer(x)

print("输入 x 形状:", x.shape)               # [2, 4, 8]
print("输出 out 形状:", out.shape)          # [2, 4, 8]
print("注意力权重形状:", attn_weights.shape)  # [2, 4, 4]
```

- `attn_weights[b]` 是一个 4×4 的矩阵：
  - 第 `i` 行表示**第 i 个 token 对句子中所有 token 的注意力分布**。

### 2. 带因果 mask 的例子（解码器常用）

**因果 mask：**位置 i 不能看到 i 之后的 token，只能看自己和前面。

```python
seq_len = 4
mask = torch.tril(torch.ones(seq_len, seq_len))  # 下三角为1，上三角为0

print("因果掩码:")
print(mask)
# tensor([[1., 0., 0., 0.],
#         [1., 1., 0., 0.],
#         [1., 1., 1., 0.],
#         [1., 1., 1., 1.]])

out, attn_weights = attention_layer(x, mask)

print("注意力权重形状:", attn_weights.shape)   # [2, 4, 4]
print("第一个样本第一行注意力:", attn_weights[0, 0])
```

- 第 0 行（第一个 token）只能看到自己（mask 中只有 (0,0) 为 1）；
- 第 3 行可以同时看到 0,1,2,3 四个位置。

### 3. 再直观一点：打印整块注意力矩阵

你可以像下面这样检查注意力是不是符合 softmax 的性质（每行和为 1）：

```python
batch_size = 1
seq_len = 5
d_model = 10
d_k = 8
d_v = 8

x = torch.randn(batch_size, seq_len, d_model)
attention_layer = SingleHeadAttention(d_model, d_k, d_v)

# 示例 mask：后两个 token 不能看前 3 个
mask = torch.ones(seq_len, seq_len)
mask[3:, :3] = 0

out, attn_weights = attention_layer(x, mask)

print("掩码矩阵:")
print(mask)
print("\n注意力权重矩阵 (batch=0):")
print(attn_weights[0])                 # [5, 5]
print("\n每行和 (应为1):")
print(attn_weights[0].sum(dim=1))      # 约等于 tensor([1,1,1,1,1])
```

你会看到：
- `mask` 为 0 的地方，对应的 `attn_weights` 为 0；
- 每一行的和都接近 1，符合 softmax 的归一化。

---

## 五、如何在自己的项目中使用？

1. **直接拷贝上面的 `scaled_dot_product_attention` 和 `SingleHeadAttention` 类**。
2. 在你的模型中初始化：

   ```python
   self.attn = SingleHeadAttention(d_model=128, d_k=64, d_v=64)
   ```

3. 在 `forward` 里调用：

   ```python
   out, attn_weights = self.attn(x, mask=mask)
   ```

4. `out` 接着可以送到后续的前馈网络（FFN）等模块；
5. `attn_weights` 可以用来**可视化模型在看哪里**（比如画热力图）。

---

如果你愿意，下一步可以在这个基础上扩展成**多头注意力（Multi-Head Attention）**：  
- 把 `d_model` 切成 `n_heads` 份；
- 对每一份做一次上面的单头注意力；
- 最后拼接起来再过一个线性层。

但概念上，其实就是把当前这个单头版本复制成多份并行而已。