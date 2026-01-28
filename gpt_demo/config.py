import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any

class GPTConfig:
    """GPT配置类"""
    def __init__(
        self,
        vocab_size: int = 50257,  # GPT-2的词汇表大小
        block_size: int = 1024,   # 上下文长度
        n_layer: int = 12,        # Transformer层数
        n_head: int = 12,         # 注意力头数
        n_embd: int = 768,        # 嵌入维度
        dropout: float = 0.1,
        bias: bool = True,        # 是否在线性层中使用偏置
        use_learned_pos_emb: bool = True,  # 是否使用学习的位置编码
        use_rope: bool = False,   # 是否使用RoPE (Rotary Positional Encoding)
    ):
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.dropout = dropout
        self.bias = bias
        self.use_learned_pos_emb = use_learned_pos_emb
        self.use_rope = use_rope
        
        # 验证配置
        assert n_embd % n_head == 0, "n_embd必须能被n_head整除"