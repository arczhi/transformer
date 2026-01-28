Mamba 等新架构、高效注意力（如 FlashAttention）

BoPE

KV缓存

GeLU

logits

采样 贪婪 top-k top-p

temperature
在 `GreedySampling` 类中，`temperature` 参数直接影响了模型生成的随机性。以下是原因：

### 1. **`temperature` 的作用**
在代码中，`temperature` 是通过以下方式影响 logits 的：

```python
logits = logits / self.temperature
```

- **logits** 是模型输出的分数（未归一化的对数概率）。
- 除以 `temperature` 会改变 logits 的分布：
  - 当 `temperature < 1` 时，logits 的值会被放大，差距变大，概率分布会更加偏向于最高分的 token，生成的结果更确定（更“贪婪”）。
  - 当 `temperature > 1` 时，logits 的值会被缩小，差距变小，概率分布会更加平滑，生成的结果更随机。

### 2. **数学解释**
在生成过程中，logits 会通过 softmax 函数转化为概率分布：

$$
P(x_i) = \frac{e^{\text{logits}_i / T}}{\sum_j e^{\text{logits}_j / T}}
$$

- 当 $T = 1$（`temperature=1`），softmax 的输出是标准的概率分布。
- 当 $T < 1$，分母中的 $e^{\text{logits}_j / T}$ 被放大，导致概率分布更加偏向于分数较高的 token，随机性降低。
- 当 $T > 1$，分母中的 $e^{\text{logits}_j / T}$ 被缩小，导致概率分布更加平滑，分数较低的 token 也有更高的概率被选中，随机性增加。

### 3. **直观理解**
- **低温（`temperature < 1`）**：模型更倾向于选择概率最高的 token，生成的文本更确定，但可能缺乏多样性。
- **高温（`temperature > 1`）**：模型会更倾向于探索其他可能性，选择概率较低的 token，生成的文本更随机，但可能会出现不连贯或不合理的内容。

### 4. **举例**
假设 logits 是 `[2.0, 1.0, 0.1]`，我们计算 softmax 概率：

#### 当 `temperature = 1`：
$$
P(x_i) = \frac{e^{\text{logits}_i}}{\sum_j e^{\text{logits}_j}}
$$
- $P(x_1) = 0.71, P(x_2) = 0.26, P(x_3) = 0.03$

#### 当 `temperature = 0.5`（低温）：
$$
P(x_i) = \frac{e^{\text{logits}_i / 0.5}}{\sum_j e^{\text{logits}_j / 0.5}}
$$
- $P(x_1) = 0.91, P(x_2) = 0.08, P(x_3) = 0.01$
- 最高分的 token（$x_1$）的概率更高，其他 token 的概率更低。

#### 当 `temperature = 2.0`（高温）：
$$
P(x_i) = \frac{e^{\text{logits}_i / 2.0}}{\sum_j e^{\text{logits}_j / 2.0}}
$$
- $P(x_1) = 0.49, P(x_2) = 0.36, P(x_3) = 0.15$
- 概率分布更平滑，低分 token 的概率增加，随机性更高。

### 总结
`temperature` 控制了 logits 的平滑程度，从而影响了 softmax 概率分布的形状。温度越高，分布越平滑，随机性越高；温度越低，分布越尖锐，随机性越低。
