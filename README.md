# Transformer & GPT 学习笔记

> 一个以「动手实现 + 深度笔记」为主线，从零理解 Transformer、GPT 与新一代注意力机制的代码库。既是学习索引，也是随学随记的笔记仓库。

## 项目说明

从「Attention Is All You Need」原论文出发，用 PyTorch 手写 Transformer 的每一个组件（词嵌入、位置编码、多头注意力、Encoder/Decoder、前馈网络），再基于 Decoder 搭建 GPT，最后追踪 FlashAttention、Longformer、Performer、Mamba 等让 Transformer 更快、更长的前沿方案。**原则：每个组件先懂数学，再用代码还原，最后看它如何演进。**

## 目录索引

```
.
├── guide/                 # 学习指南：路线图与论文原文
├── transformer_impl/      # 核心！从零实现 Transformer 全组件 + 简化 GPT
├── pytorch/               # PyTorch 入门 + 自注意力/多头注意力小实验
├── attention/             # 前沿注意力机制论文解读与源码阅读
├── doc/                   # 公式 → 代码的打通笔记
├── gpt_demo/              # GPT 完整实现指南（含采样策略、聊天机器人）
└── test/                  # 测试脚本
```

## 目录结构与核心笔记

### 1. 学习路线图 `guide/`

- `learning_guide_deepseek.md` / `learning_guide_miro.md` — 10 周分阶段学习计划（基础 → 核心 → 实践 → 拓展）
- `llm_learning_path_web_dev.md` — Web 开发背景转 LLM 的定制路线
- `1706.03762v7.pdf` — Transformer 原论文 *Attention Is All You Need*

### 2. Transformer 从零实现 `transformer_impl/`

按「数据流」顺序阅读，每个文件对应一个组件：

| 文件 | 组件 | 一句话核心 |
|---|---|---|
| `token_embedding.py` | 词嵌入层 | 把离散 token 变成向量 |
| `positional_encoding.py` | 位置编码 | 正弦函数注入位置信息（附可视化图） |
| `multi_head_attention.py` | 多头注意力 | 多头 = 在多个子空间并行做缩放点积注意力 |
| `feed_forward_network.py` | 前馈网络 MLP | 逐位置的 512→2048→512 非线性变换 |
| `encoder.py` | Encoder 块 | 注意力/FFN + 残差连接 + LayerNorm |
| `decoder.py` | Decoder 块 | 掩码自注意力 + 交叉注意力 + FFN |
| `transfomer.py` | 完整模型 | 组装 Encoder-Decoder 并跑通前向 |
| `gpt.py` | 简化 GPT | 只用 Decoder，做文本生成 |

> 关键领悟：**多头注意力的并行性**来自 reshape 多出的 `num_heads` 维度——PyTorch 把它当 batch 维度一次性做矩阵乘和 softmax，无需循环逐头计算。

### 3. PyTorch 入门实验 `pytorch/`

- `self_attention.py` → 单头缩放点积注意力的逐步实现
- `multi_self_attention.py` → 多头版本（对照 `transformer_impl` 食用更佳）
- `MLP.py`、`forward_neural_network.py` → 手写全连接层与训练流程
- `pipeline.py`、`tokenizer.py` → 体验 HuggingFace pipeline 与 GPT 分词器

### 4. 公式与代码打通笔记 `doc/`

- `self_attention_impl.md` — 从「公式 → 代码」逐行拆解单头自注意力
- `attention_weights.md` — 为什么 `output = attention_weights @ value`

### 5. 前沿注意力机制 `attention/`

| 笔记 | 解决什么问题 | 核心思路 |
|---|---|---|
| `flash_attention.md` | 注意力显存 O(n²) 瓶颈 | 分块计算，把中间矩阵 `QK^T`、Softmax 留在 SRAM 里，避免频繁读写 HBM |
| `sliding_window_attention_summary.md` | 长文档 O(n²) | 每个位置只看窗口 w 内的邻居，复杂度降到 O(n×w) |
| `longformer_explained.md` | 超长文档 | 滑动窗口 + 全局 token，可处理上万 token |
| `performer_analysis.md` | 二次复杂度 | FAVOR+ 随机特征分解 QK^T，线性时间近似注意力 |
| `mamba_architecture_explained.md` | 长序列效率 | 状态空间模型替代注意力，线性复杂度 O(n) |
| `guide/` | 脉络串联 | 随机特征、Performer 论文精读、新型注意力机制通俗对比 |

> 主线笔记：**attention（二次，精度高）→ FlashAttention（工程提速）→ 稀疏/线性近似（理论降复杂度）→ Mamba（架构级替代）**。相关第三方源码（longformer / mamba / performer）为对照阅读素材，不作为主仓库内容维护。

### 6. GPT 从理解到实现 `gpt_demo/`

- `README.md` — GPT 与 Transformer 的 5 点差异 + 十一步完整代码：RoPE、因果掩码注意力、KV 缓存、贪婪/top-k/top-p/束搜索、对话管理
- `knowledge_points.md` — 速记卡片：KV Cache、FlashAttention、RoPE、Mamba 等高频考点
- `config.py` — GPTConfig 超参配置

## 核心知识脉络（快照笔记）

```
Transformer
 ├── 无位置信息 → 位置编码（正弦 / 可学习 / RoPE 旋转）
 ├── 注意力: Q、K、V 三矩阵 → QK^T 打分 → softmax → 加权 V
 │     └── 多头: 不同子空间并行捕捉多种关系
 ├── 深度训练技巧: 残差连接 + LayerNorm（先 norm 再 attn 的 Pre-LN）
 └── Encoder(双向) / Decoder(掩码单向 + 交叉注意力)

GPT = 只留 Decoder 的掩码自注意力 + 大规模预训练 + 自回归生成
 ├── 生成加速: KV Cache（只算最新 token，旧 KV 复用）
 ├── 采样策略: 贪心 / Top-k / Top-p / Beam Search + 温度
 └── 位置编码升级: 学习位置编码 → RoPE（外推到更长序列）

效率演进: 标准 O(n²) → 窗口/稀疏 O(n·w) → 线性化 O(n) → Mamba
```

## 环境与运行

代码基于 Python + PyTorch，仅 CPU 即可运行大部分演示：

```bash
# 手写多头注意力的前向演示
python transformer_impl/multi_head_attention.py

# 简化 GPT 文本生成演示（gpt.py 自带 main）
python transformer_impl/gpt.py
```

> 说明：`pytorch/pipeline.py`、`pytorch/tokenizer.py` 依赖 `transformers` 库，可按需安装。

## 学习建议

1. 按 `guide/` 路线走：**先通读一篇全实现笔记，再对照 `transformer_impl/` 源码逐行理解，最后回论文验证细节**。
2. 动手改超参（层数、头数、维度），观察参数量与效果变化。
3. 进阶必读顺序：FlashAttention → Longformer → Performer → Mamba，配合仓库内每篇笔记「它动了哪里、代价是什么」一起思考。
