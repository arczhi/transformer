# Performer-PyTorch 关键逻辑分析：K 和 V 如何聚合

## 概述

Performer 是一种注意力机制，它用线性复杂度近似标准注意力，而不是二次复杂度。它使用 FAVOR+（Fast Attention Via positive Orthogonal Random features）方法来实现这种效率。

## 核心注意力机制

### FastAttention 类

`FastAttention` 类实现了 Performer 的核心逻辑：

```python
class FastAttention(nn.Module):
    def __init__(self, dim_heads, nb_features = None, ortho_scaling = 0, causal = False, generalized_attention = False, kernel_fn = nn.ReLU(), no_projection = False):
        # ...
```

关键参数：
- `dim_heads`: 每个注意力头的维度
- `nb_features`: 投影的随机特征数量（默认为 int(dim_heads * math.log(dim_heads))）
- `ortho_scaling`: 正交随机矩阵的缩放方法
- `causal`: 是否使用因果（掩码）注意力
- `generalized_attention`: 是否使用广义注意力核
- `kernel_fn`: 广义注意力的核函数
- `no_projection`: 是否跳过随机投影

### 核函数

Performer 使用核函数来近似 softmax 注意力。两种主要方法：

#### 1. Softmax 核（标准方法）
```python
def softmax_kernel(data, *, projection_matrix, is_query, normalize_data=True, eps=1e-4, device = None):
    # 应用随机投影后进行指数变换
    # 近似 softmax 注意力
```

#### 2. 广义核（替代方法）
```python
def generalized_kernel(data, *, projection_matrix, kernel_fn = nn.ReLU(), kernel_epsilon = 0.001, normalize_data = True, device = None):
    # 使用自定义核函数而不是 softmax 近似
```

### K 和 V 如何聚合

关键聚合发生在 `linear_attention` 函数中：

```python
def linear_attention(q, k, v):
    k_cumsum = k.sum(dim = -2)  # K 在序列维度上的总和
    D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))  # 归一化因子
    context = torch.einsum('...nd,...ne->...de', k, v)  # 聚合的 KV 上下文矩阵
    out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)  # 将 Q 应用于上下文并归一化
    return out
```

让我们分解这个聚合过程：

1. **上下文矩阵创建**:
   ```python
   context = torch.einsum('...nd,...ne->...de', k, v)
   ```
   这通过乘以 K^T 和 V 创建一个上下文矩阵。结果具有形状 `(..., d, e)`，其中 `d` 是键维度，`e` 是值维度。这表示来自所有键值对的累积信息。

2. **查询应用**:
   ```python
   out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
   ```
   上下文矩阵被应用于查询，然后进行归一化。

### 详细逐步过程

1. **输入转换**:
   - 输入 Q、K、V 张量使用随机投影转换以创建正特征
   - 对于标准 softmax 近似：`q = softmax_kernel(Q, projection_matrix, is_query=True)`
   - 对于键：`k = softmax_kernel(K, projection_matrix, is_query=False)`

2. **归一化**:
   ```python
   k_cumsum = k.sum(dim = -2)  # 形状: (batch, heads, key_dim)
   D_inv = 1. / torch.einsum('...nd,...d->...n', q, k_cumsum.type_as(q))  # 形状: (batch, heads, seq_len)
   ```
   这计算归一化因子以确保注意力概率总和为 1。

3. **KV 聚合**:
   ```python
   context = torch.einsum('...nd,...ne->...de', k, v)
   ```
   这是核心聚合步骤，其中所有键值对被组合成单个上下文矩阵，形状为 `(batch, heads, key_dim, value_dim)`。

4. **输出计算**:
   ```python
   out = torch.einsum('...de,...nd,...n->...ne', context, q, D_inv)
   ```
   上下文被应用于每个查询位置，归一化，并产生最终输出。

### 因果线性注意力

对于自回归模型，有一个因果版本，尊重因果约束：

```python
def causal_linear_attention(q, k, v, eps = 1e-6):
    # 使用 CUDA 优化的因果点积
    # 维护累积和以进行高效计算
```

### 随机特征投影

随机投影使用以下方式创建：

```python
def gaussian_orthogonal_random_matrix(nb_rows, nb_columns, scaling = 0, device = None):
    # 创建具有高斯缩放的正交随机矩阵
```

这些投影在训练期间定期重绘以提高近似质量：

```python
class ProjectionUpdater(nn.Module):
    def redraw_projections(self):
        # 在指定间隔重绘投影矩阵
```

## 关键创新

1. **线性复杂度**: 不是 O(N²) 注意力计算，Performer 通过避免显式计算完整注意力矩阵来实现 O(N) 复杂度。

2. **随机特征**: 使用随机投影来近似 softmax 核，允许可以高效计算的正特征。

3. **核聚合**: 而不是显式计算注意力权重，它将 K 和 V 聚合到可以高效应用于查询的上下文矩阵中。

4. **内存效率**: 与标准注意力机制相比显著减少内存使用，特别是对于长序列。

## 总结

Performer 通过以下方式聚合 K 和 V：
1. 使用随机投影转换它们以创建正特征
2. 计算累积上下文矩阵 `K^T @ V`
3. 将此上下文应用于查询并适当归一化
4. 这避免了计算和存储完整注意力矩阵的需要，实现线性复杂度

这种方法在保持注意力表示能力的同时，显著提高了长序列的计算效率。