# GPT (Generative Pre-trained Transformer) 实现指南

太棒了！基于已有的Transformer，我们可以构建一个类ChatGPT的聊天模型。GPT本质上是仅使用Transformer的Decoder部分，并进行一些调整。让我们一步步实现。

## 第一步：理解GPT与Transformer的区别

GPT的核心特点：
1. **仅使用Decoder**：没有Encoder部分
2. **移除交叉注意力**：只保留带掩码的自注意力
3. **使用更大的模型**：更多层、更多头、更大维度
4. **不同的位置编码**：通常使用学习的位置编码
5. **生成任务**：自回归语言建模

## 第二步：实现GPT模型架构

```python
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
```

## 第三步：实现RoPE (Rotary Positional Encoding)

```python
class RotaryPositionalEmbedding(nn.Module):
    """旋转位置编码 (RoPE) - 更现代的位置编码方法"""
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        
        # 计算频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # 预计算sin和cos
        self._build_cache(max_seq_len)
    
    def _build_cache(self, max_seq_len: int):
        """预计算位置编码缓存"""
        seq = torch.arange(max_seq_len)
        freqs = torch.einsum("i,j->ij", seq.float(), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用RoPE到Q或K
        
        参数:
            x: [batch_size, seq_len, n_head, head_dim]
        返回:
            旋转后的x
        """
        seq_len = x.size(1)
        cos = self.cos_cached[:, :, :seq_len, ...]
        sin = self.sin_cached[:, :, :seq_len, ...]
        
        # 分离偶数和奇数维度
        x1, x2 = x[..., 0::2], x[..., 1::2]
        
        # 应用旋转
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x2 * cos + x1 * sin
        
        # 重新组合
        out = torch.stack([rotated_x1, rotated_x2], dim=-1).flatten(-2, -1)
        return out
```

## 第四步：实现GPT注意力层

```python
class GPTAttention(nn.Module):
    """GPT的自注意力层（带因果掩码）"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        
        # 注意力机制
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # 正则化
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # 注册缓冲区（用于因果掩码）
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
        )
        
        # 可选：RoPE
        if config.use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim, config.block_size)
        else:
            self.rope = None
            
    def forward(self, x: torch.Tensor, use_cache: bool = False, 
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        前向传播
        
        参数:
            x: 输入张量 [batch, seq_len, n_embd]
            use_cache: 是否使用KV缓存
            layer_past: 过去的KV缓存 (key, value)
            
        返回:
            output: [batch, seq_len, n_embd]
            present: 更新的KV缓存 (key, value) 或 None
        """
        batch_size, seq_len, _ = x.shape
        
        # 计算Q, K, V
        qkv = self.c_attn(x)  # [batch, seq_len, 3 * n_embd]
        q, k, v = qkv.split(self.n_embd, dim=2)
        
        # 重塑为多头
        q = q.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_head, self.head_dim).transpose(1, 2)
        
        # 应用RoPE（如果使用）
        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        
        # KV缓存（用于自回归生成）
        if use_cache and layer_past is not None:
            past_key, past_value = layer_past
            # 连接过去的KV
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)
        
        present = (k, v) if use_cache else None
        
        # 因果自注意力
        att = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        
        # 应用因果掩码
        att = att.masked_fill(self.bias[:, :, :seq_len, :k.size(-2)] == 0, float('-inf'))
        
        # Softmax
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # 计算输出
        y = torch.matmul(att, v)
        
        # 重塑回原始形状
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, self.n_embd)
        
        # 输出投影
        y = self.resid_dropout(self.c_proj(y))
        
        return y, present
```

## 第五步：实现GPT前馈网络

```python
class GPTFeedForward(nn.Module):
    """GPT的前馈网络"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        hidden_dim = 4 * config.n_embd  # 按照GPT标准
        
        self.c_fc = nn.Linear(config.n_embd, hidden_dim, bias=config.bias)
        self.c_proj = nn.Linear(hidden_dim, config.n_embd, bias=config.bias)
        self.act = nn.GELU()  # GPT使用GELU激活函数
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```

## 第六步：实现GPT Transformer块

```python
class GPTBlock(nn.Module):
    """GPT的Transformer块"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = GPTAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = GPTFeedForward(config)
        
    def forward(self, x: torch.Tensor, use_cache: bool = False,
                layer_past: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        # 自注意力子层（带残差连接）
        attn_out, present = self.attn(self.ln_1(x), use_cache=use_cache, layer_past=layer_past)
        x = x + attn_out
        
        # 前馈网络子层（带残差连接）
        mlp_out = self.mlp(self.ln_2(x))
        x = x + mlp_out
        
        return x, present
```

## 第七步：实现完整的GPT模型

```python
class GPT(nn.Module):
    """完整的GPT模型"""
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        # Token嵌入
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        
        # 位置嵌入
        if config.use_learned_pos_emb:
            self.wpe = nn.Embedding(config.block_size, config.n_embd)
        else:
            self.wpe = None
        
        # Dropout
        self.drop = nn.Dropout(config.dropout)
        
        # Transformer块
        self.blocks = nn.ModuleList([GPTBlock(config) for _ in range(config.n_layer)])
        
        # 最终层归一化
        self.ln_f = nn.LayerNorm(config.n_embd, bias=config.bias)
        
        # 语言模型头
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # 权重绑定（共享嵌入权重）
        self.wte.weight = self.lm_head.weight
        
        # 初始化权重
        self._init_weights()
        
        # KV缓存（用于推理加速）
        self.kv_cache = None
        
    def _init_weights(self):
        """初始化权重"""
        # 应用GPT风格的初始化
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用正态分布初始化，标准差较小
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.zeros_(module.bias)
                torch.nn.init.ones_(module.weight)
    
    def create_position_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        """创建位置ID"""
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return position_ids
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None
    ) -> Tuple[torch.Tensor, Optional[List[Tuple[torch.Tensor, torch.Tensor]]]]:
        """
        前向传播
        
        参数:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (可选)
            position_ids: [batch_size, seq_len] (可选)
            use_cache: 是否使用KV缓存
            past_key_values: 过去的KV缓存列表
            
        返回:
            logits: [batch_size, seq_len, vocab_size]
            presents: KV缓存列表或None
        """
        batch_size, seq_len = input_ids.shape
        
        # 创建位置ID（如果未提供）
        if position_ids is None:
            position_ids = self.create_position_ids(input_ids)
        
        # Token嵌入
        tok_emb = self.wte(input_ids)
        
        # 位置嵌入
        if self.wpe is not None:
            pos_emb = self.wpe(position_ids)
            hidden_states = tok_emb + pos_emb
        else:
            hidden_states = tok_emb
        
        hidden_states = self.drop(hidden_states)
        
        # 注意力掩码
        if attention_mask is not None:
            # 扩展掩码维度以匹配注意力分数形状
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # 通过Transformer块
        presents = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            layer_past = past_key_values[i] if past_key_values is not None else None
            
            hidden_states, present = block(
                hidden_states,
                use_cache=use_cache,
                layer_past=layer_past
            )
            
            if use_cache:
                presents.append(present)
        
        # 最终层归一化
        hidden_states = self.ln_f(hidden_states)
        
        # 语言模型头
        logits = self.lm_head(hidden_states)
        
        return logits, presents
    
    def clear_kv_cache(self):
        """清除KV缓存"""
        self.kv_cache = None
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """为生成准备输入"""
        # 如果使用缓存，只传递最后一个token
        if past_key_values:
            input_ids = input_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": True,
        }
```

## 第八步：实现文本生成策略

```python
class TextGenerationStrategy:
    """文本生成策略基类"""
    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature
    
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        """生成下一个token"""
        raise NotImplementedError


class GreedySampling(TextGenerationStrategy):
    """贪婪采样"""
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        logits = logits / self.temperature
        return torch.argmax(logits, dim=-1)


class TopKSampling(TextGenerationStrategy):
    """Top-k采样"""
    def __init__(self, k: int = 50, temperature: float = 1.0):
        super().__init__(temperature)
        self.k = k
    
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        batch_size, vocab_size = logits.shape
        
        # 应用温度
        logits = logits / self.temperature
        
        # 获取top-k
        top_k_values, top_k_indices = torch.topk(logits, self.k, dim=-1)
        
        # 创建概率分布
        probs = F.softmax(top_k_values, dim=-1)
        
        # 采样
        sampled_indices = torch.multinomial(probs, num_samples=1)
        
        # 获取原始词汇表中的索引
        sampled_tokens = torch.gather(top_k_indices, -1, sampled_indices)
        
        return sampled_tokens.squeeze(-1)


class TopPSampling(TextGenerationStrategy):
    """Top-p（核）采样"""
    def __init__(self, p: float = 0.9, temperature: float = 1.0):
        super().__init__(temperature)
        self.p = p
    
    def __call__(self, logits: torch.Tensor) -> torch.Tensor:
        # 应用温度
        logits = logits / self.temperature
        
        # 获取概率
        probs = F.softmax(logits, dim=-1)
        
        # 排序概率
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        
        # 计算累积概率
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # 移除概率总和超过p的token
        sorted_indices_to_remove = cumulative_probs > self.p
        
        # 确保至少保留一个token
        sorted_indices_to_remove[..., 0] = False
        
        # 创建掩码
        indices_to_remove = torch.zeros_like(probs, dtype=torch.bool)
        indices_to_remove.scatter_(-1, sorted_indices, sorted_indices_to_remove)
        
        # 将不需要的token的概率设为0
        probs = probs.masked_fill(indices_to_remove, 0.0)
        
        # 重新归一化
        probs = probs / probs.sum(dim=-1, keepdim=True)
        
        # 采样
        sampled_indices = torch.multinomial(probs, num_samples=1)
        
        return sampled_indices.squeeze(-1)


class BeamSearch:
    """束搜索"""
    def __init__(self, beam_size: int = 5, length_penalty: float = 1.0):
        self.beam_size = beam_size
        self.length_penalty = length_penalty
    
    def __call__(self, model, input_ids, max_length, **kwargs):
        """执行束搜索"""
        batch_size = input_ids.shape[0]
        
        # 初始化束
        beams = [(input_ids, 0.0)]  # (序列, 分数)
        
        for step in range(max_length - input_ids.shape[1]):
            new_beams = []
            
            for seq, score in beams:
                # 获取下一个token的logits
                with torch.no_grad():
                    logits, _ = model(seq)
                    next_token_logits = logits[:, -1, :]
                
                # 获取top-k tokens
                topk_logits, topk_indices = torch.topk(
                    next_token_logits, 
                    self.beam_size, 
                    dim=-1
                )
                
                # 计算分数
                for i in range(self.beam_size):
                    token_id = topk_indices[0, i].item()
                    token_logit = topk_logits[0, i].item()
                    
                    new_seq = torch.cat([
                        seq, 
                        torch.tensor([[token_id]], device=seq.device)
                    ], dim=1)
                    
                    # 应用长度惩罚
                    length = new_seq.shape[1]
                    new_score = score + token_logit / (length ** self.length_penalty)
                    
                    new_beams.append((new_seq, new_score))
            
            # 选择分数最高的beam_size个序列
            new_beams.sort(key=lambda x: x[1], reverse=True)
            beams = new_beams[:self.beam_size]
        
        # 返回最佳序列
        best_seq, best_score = beams[0]
        return best_seq
```

## 第九步：实现对话管理器

```python
class ConversationManager:
    """管理对话上下文"""
    def __init__(
        self,
        model: GPT,
        tokenizer,
        max_context_length: int = 2048,
        system_prompt: str = "You are a helpful AI assistant."
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.max_context_length = max_context_length
        
        # 对话历史
        self.history = []
        
        # 添加系统提示
        if system_prompt:
            self.add_message("system", system_prompt)
    
    def add_message(self, role: str, content: str):
        """添加消息到历史"""
        self.history.append({"role": role, "content": content})
    
    def get_conversation_text(self) -> str:
        """将对话历史转换为文本格式"""
        # 不同的模型可能有不同的格式
        # 这里使用ChatGPT的格式
        conversation = []
        
        for message in self.history:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                conversation.append(f"System: {content}")
            elif role == "user":
                conversation.append(f"User: {content}")
            elif role == "assistant":
                conversation.append(f"Assistant: {content}")
        
        return "\n".join(conversation)
    
    def get_prompt(self) -> str:
        """获取完整的提示文本"""
        conversation_text = self.get_conversation_text()
        return f"{conversation_text}\nAssistant:"
    
    def tokenize_conversation(self, max_length: Optional[int] = None) -> torch.Tensor:
        """将对话历史tokenize"""
        if max_length is None:
            max_length = self.max_context_length
        
        prompt = self.get_prompt()
        tokens = self.tokenizer.encode(prompt)
        
        # 截断到最大长度
        if len(tokens) > max_length:
            tokens = tokens[-max_length:]
        
        return torch.tensor([tokens])
    
    def generate_response(
        self,
        user_message: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        strategy: str = "top_p",
        stop_tokens: List[str] = None
    ) -> str:
        """
        生成响应
        
        参数:
            user_message: 用户消息
            max_new_tokens: 最大生成长度
            temperature: 温度参数
            top_k: top-k参数
            top_p: top-p参数
            strategy: 采样策略（greedy, top_k, top_p）
            stop_tokens: 停止token列表
        """
        # 添加用户消息
        self.add_message("user", user_message)
        
        # 准备输入
        input_ids = self.tokenize_conversation()
        
        # 设置生成策略
        if strategy == "greedy":
            sampler = GreedySampling(temperature)
        elif strategy == "top_k":
            sampler = TopKSampling(k=top_k, temperature=temperature)
        elif strategy == "top_p":
            sampler = TopPSampling(p=top_p, temperature=temperature)
        else:
            raise ValueError(f"未知的采样策略: {strategy}")
        
        # 生成响应
        generated_tokens = []
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 前向传播
                logits, _ = self.model(input_ids)
                
                # 获取最后一个token的logits
                next_token_logits = logits[0, -1, :]
                
                # 应用采样策略
                next_token_id = sampler(next_token_logits.unsqueeze(0))
                
                # 检查停止条件
                token_str = self.tokenizer.decode([next_token_id.item()])
                
                # 检查停止token
                if stop_tokens and any(stop in token_str for stop in stop_tokens):
                    break
                
                # 添加到生成的tokens
                generated_tokens.append(next_token_id.item())
                
                # 更新输入
                input_ids = torch.cat([
                    input_ids,
                    next_token_id.unsqueeze(0).unsqueeze(0)
                ], dim=1)
                
                # 检查上下文长度
                if input_ids.shape[1] >= self.max_context_length:
                    break
        
        # 解码响应
        response = self.tokenizer.decode(generated_tokens)
        
        # 清理响应（移除可能的停止token）
        if stop_tokens:
            for stop in stop_tokens:
                response = response.split(stop)[0]
        
        # 添加到历史
        self.add_message("assistant", response)
        
        return response
    
    def clear_history(self):
        """清空对话历史"""
        self.history = []
    
    def get_token_count(self) -> int:
        """获取当前对话的token数量"""
        prompt = self.get_prompt()
        tokens = self.tokenizer.encode(prompt)
        return len(tokens)
```

## 第十步：实现简单的Tokenizer

```python
import json
from collections import Counter

class SimpleTokenizer:
    """简单的字符级tokenizer（用于演示）"""
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        
        # 特殊token
        self.pad_token = 0
        self.eos_token = 1
        self.bos_token = 2
        self.unk_token = 3
        
        # 构建基本vocab
        self._build_vocab()
    
    def _build_vocab(self):
        """构建词汇表"""
        # 添加特殊token
        special_tokens = {
            "[PAD]": self.pad_token,
            "[EOS]": self.eos_token,
            "[BOS]": self.bos_token,
            "[UNK]": self.unk_token,
        }
        
        # 添加常见字符
        vocab_items = list(special_tokens.items())
        
        # 添加ASCII字符
        for i in range(32, 127):  # 可打印ASCII字符
            char = chr(i)
            token_id = len(vocab_items)
            vocab_items.append((char, token_id))
        
        # 构建字典
        for token, token_id in vocab_items:
            self.vocab[token] = token_id
            self.inverse_vocab[token_id] = token
    
    def encode(self, text: str) -> List[int]:
        """将文本编码为token IDs"""
        tokens = []
        
        # 添加BOS token
        tokens.append(self.bos_token)
        
        # 编码每个字符
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.unk_token)
        
        # 添加EOS token
        tokens.append(self.eos_token)
        
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """将token IDs解码为文本"""
        text = ""
        
        for token_id in token_ids:
            if token_id in self.inverse_vocab:
                # 跳过特殊token（除了UNK）
                if token_id in [self.bos_token, self.eos_token, self.pad_token]:
                    continue
                elif token_id == self.unk_token:
                    text += "�"
                else:
                    text += self.inverse_vocab[token_id]
            else:
                text += "�"
        
        return text
    
    def save(self, path: str):
        """保存tokenizer"""
        data = {
            'vocab': self.vocab,
            'vocab_size': self.vocab_size,
            'special_tokens': {
                'pad': self.pad_token,
                'eos': self.eos_token,
                'bos': self.bos_token,
                'unk': self.unk_token
            }
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self, path: str):
        """加载tokenizer"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocab = data['vocab']
        self.vocab_size = data['vocab_size']
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # 恢复特殊token
        special_tokens = data['special_tokens']
        self.pad_token = special_tokens['pad']
        self.eos_token = special_tokens['eos']
        self.bos_token = special_tokens['bos']
        self.unk_token = special_tokens['unk']


class BPETokenizer:
    """BPE (Byte Pair Encoding) tokenizer的简化实现"""
    def __init__(self, vocab_size: int = 10000):
        self.vocab_size = vocab_size
        self.vocab = {}
        self.inverse_vocab = {}
        self.merges = {}
        
        # 特殊token
        self.pad_token = 0
        self.eos_token = 1
        self.bos_token = 2
        self.unk_token = 3
        
    def train(self, texts: List[str]):
        """训练BPE tokenizer"""
        # 统计字符频率
        word_counts = Counter()
        for text in texts:
            words = text.split()
            for word in words:
                word_counts[word] += 1
        
        # 初始vocab：字符
        vocab = set()
        for word in word_counts.keys():
            vocab.update(list(word))
        
        # 转换为列表并添加特殊token
        vocab = list(vocab)
        vocab = ["[PAD]", "[EOS]", "[BOS]", "[UNK]"] + vocab
        
        # BPE合并
        merges = {}
        while len(vocab) < self.vocab_size:
            # 这里简化了BPE算法
            # 实际实现需要统计相邻字符对频率
            break
        
        # 构建vocab字典
        for i, token in enumerate(vocab[:self.vocab_size]):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
```

## 第十一步：创建完整的聊天系统

```python
class ChatBot:
    """完整的聊天机器人系统"""
    def __init__(
        self,
        model_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        config: Optional[GPTConfig] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        
        # 加载或创建配置
        if config is None:
            config = GPTConfig(
                vocab_size=1000,
                block_size=512,
                n_layer=6,
                n_head=8,
                n_embd=384,
                dropout=0.1
            )
        
        # 创建或加载tokenizer
        if tokenizer_path:
            self.tokenizer = SimpleTokenizer()
            self.tokenizer.load(tokenizer_path)
        else:
            self.tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
        
        # 创建或加载模型
        self.model = GPT(config)
        
        if model_path:
            self.load_model(model_path)
        
        self.model.to(device)
        self.model.eval()
        
        # 创建对话管理器
        self.conversation = ConversationManager(
            model=self.model,
            tokenizer=self.tokenizer,
            max_context_length=config.block_size,
            system_prompt="You are a helpful AI assistant named ChatBot."
        )
    
    def load_model(self, path: str):
        """加载模型权重"""
        state_dict = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state_dict)
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save(self.model.state_dict(), path)
    
    def chat(
        self,
        message: str,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        strategy: str = "top_p"
    ) -> str:
        """
        聊天接口
        
        参数:
            message: 用户消息
            max_new_tokens: 最大回复长度
            temperature: 温度参数（越高越随机）
            top_k: top-k参数
            top_p: top-p参数
            strategy: 采样策略
        
        返回:
            助手回复
        """
        response = self.conversation.generate_response(
            message,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            strategy=strategy,
            stop_tokens=["\n", "[EOS]", "User:", "System:"]
        )
        
        return response
    
    def clear_conversation(self):
        """清空对话历史"""
        self.conversation.clear_history()
    
    def get_conversation_history(self) -> List[Dict]:
        """获取对话历史"""
        return self.conversation.history.copy()
    
    def set_system_prompt(self, prompt: str):
        """设置系统提示"""
        self.conversation.clear_history()
        self.conversation.add_message("system", prompt)
    
    def tokenize_text(self, text: str) -> List[int]:
        """将文本tokenize"""
        return self.tokenizer.encode(text)
    
    def detokenize(self, token_ids: List[int]) -> str:
        """将token IDs转换为文本"""
        return self.tokenizer.decode(token_ids)
```

## 第十二步：训练和演示

```python
def train_gpt_model():
    """训练GPT模型（简化示例）"""
    print("=== 训练GPT模型 ===")
    
    # 配置
    config = GPTConfig(
        vocab_size=1000,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=0.1
    )
    
    # 创建模型
    model = GPT(config)
    
    # 创建tokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    
    # 创建简单的训练数据（文本反转任务）
    def create_training_data(num_samples=1000, max_len=20):
        X, y = [], []
        
        for _ in range(num_samples):
            # 随机生成文本
            length = torch.randint(5, max_len, (1,)).item()
            text = ''.join([chr(torch.randint(32, 127, (1,)).item()) for _ in range(length)])
            
            # Tokenize
            tokens = tokenizer.encode(text)
            
            # 输入是文本，目标是文本的反转
            reversed_text = text[::-1]
            target_tokens = tokenizer.encode(reversed_text)
            
            # 确保长度一致
            max_seq_len = max(len(tokens), len(target_tokens))
            tokens = tokens + [tokenizer.pad_token] * (max_seq_len - len(tokens))
            target_tokens = target_tokens + [tokenizer.pad_token] * (max_seq_len - len(target_tokens))
            
            X.append(tokens)
            y.append(target_tokens)
        
        return torch.tensor(X), torch.tensor(y)
    
    # 创建数据
    X_train, y_train = create_training_data(100, 20)
    X_val, y_val = create_training_data(20, 20)
    
    print(f"训练数据: {len(X_train)} 样本")
    print(f"验证数据: {len(X_val)} 样本")
    
    # 训练配置
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # 训练循环
    epochs = 10
    batch_size = 8
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        # 随机打乱
        indices = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            
            # 前向传播
            logits, _ = model(batch_X)
            
            # 计算损失
            loss = criterion(
                logits.view(-1, config.vocab_size),
                batch_y.view(-1)
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 验证
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_X = X_val[i:i+batch_size]
                batch_y = y_val[i:i+batch_size]
                
                logits, _ = model(batch_X)
                loss = criterion(
                    logits.view(-1, config.vocab_size),
                    batch_y.view(-1)
                )
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  训练损失: {total_loss/len(X_train):.4f}")
        print(f"  验证损失: {val_loss/len(X_val):.4f}")
    
    print("训练完成!")
    
    # 测试
    print("\n=== 测试模型 ===")
    model.eval()
    
    test_text = "Hello World"
    tokens = tokenizer.encode(test_text)
    input_tensor = torch.tensor([tokens])
    
    with torch.no_grad():
        # 生成响应
        generated_tokens = []
        
        for _ in range(20):  # 最大生成长度
            logits, _ = model(input_tensor)
            next_token_logits = logits[0, -1, :]
            
            # 使用贪婪采样
            next_token = torch.argmax(next_token_logits).item()
            
            if next_token == tokenizer.eos_token:
                break
            
            generated_tokens.append(next_token)
            input_tensor = torch.cat([
                input_tensor,
                torch.tensor([[next_token]])
            ], dim=1)
        
        generated_text = tokenizer.decode(generated_tokens)
        print(f"输入: {test_text}")
        print(f"生成: {generated_text}")
        print(f"期望: {test_text[::-1]}")
    
    return model, tokenizer


def demo_chat_system():
    """演示聊天系统"""
    print("=== 聊天系统演示 ===")
    
    # 创建配置
    config = GPTConfig(
        vocab_size=1000,
        block_size=256,
        n_layer=4,
        n_head=4,
        n_embd=256,
        dropout=0.1
    )
    
    # 创建tokenizer
    tokenizer = SimpleTokenizer(vocab_size=config.vocab_size)
    
    # 创建模型
    model = GPT(config)
    
    # 创建聊天机器人
    chatbot = ChatBot(
        config=config,
        device="cpu"
    )
    
    # 设置自定义系统提示
    chatbot.set_system_prompt(
        "You are a helpful AI assistant. "
        "You provide concise and accurate responses."
    )
    
    print("\n系统提示已设置")
    print(f"词汇表大小: {config.vocab_size}")
    print(f"上下文长度: {config.block_size}")
    print(f"模型层数: {config.n_layer}")
    print(f"注意力头数: {config.n_head}")
    print(f"嵌入维度: {config.n_embd}")
    
    # 演示不同采样策略
    strategies = [
        ("贪婪采样", "greedy", 0.8, 50, 0.9),
        ("Top-k采样", "top_k", 0.8, 10, 0.9),
        ("Top-p采样", "top_p", 0.8, 50, 0.9),
    ]
    
    test_messages = [
        "Hello, how are you?",
        "What is artificial intelligence?",
        "Tell me a joke."
    ]
    
    for strategy_name, strategy_type, temperature, top_k, top_p in strategies:
        print(f"\n=== 使用 {strategy_name} ===")
        print(f"  温度: {temperature}, Top-k: {top_k}, Top-p: {top_p}")
        
        for message in test_messages[:1]:  # 只测试第一条
            print(f"\n用户: {message}")
            
            response = chatbot.chat(
                message,
                max_new_tokens=50,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                strategy=strategy_type
            )
            
            print(f"助手: {response}")
        
        # 清空对话历史以便下一个策略
        chatbot.clear_conversation()
        chatbot.set_system_prompt("You are a helpful AI assistant.")
    
    return chatbot


def compare_sampling_strategies():
    """比较不同采样策略"""
    print("=== 采样策略比较 ===")
    
    # 创建测试logits
    vocab_size = 10
    logits = torch.randn(vocab_size)
    
    print(f"\n原始logits: {logits}")
    print(f"Softmax概率: {F.softmax(logits, dim=-1)}")
    
    # 测试不同策略
    strategies = {
        "贪婪采样": GreedySampling(temperature=1.0),
        "Top-k采样 (k=3)": TopKSampling(k=3, temperature=1.0),
        "Top-p采样 (p=0.7)": TopPSampling(p=0.7, temperature=1.0),
    }
    
    # 多次采样以查看分布
    num_samples = 1000
    results = {}
    
    for name, strategy in strategies.items():
        samples = []
        for _ in range(num_samples):
            sample = strategy(logits.unsqueeze(0))
            samples.append(sample.item())
        
        # 统计频率
        from collections import Counter
        counter = Counter(samples)
        
        results[name] = {
            "samples": samples,
            "distribution": {k: v/num_samples for k, v in counter.items()}
        }
        
        print(f"\n{name}:")
        for token_id, prob in sorted(results[name]["distribution"].items()):
            print(f"  Token {token_id}: {prob:.3%}")
    
    return results


# 运行演示
if __name__ == "__main__":
    print("=" * 60)
    print("GPT 聊天模型实现")
    print("=" * 60)
    
    # 比较采样策略
    results = compare_sampling_strategies()
    
    print("\n" + "=" * 60)
    
    # 演示聊天系统
    chatbot = demo_chat_system()
    
    print("\n" + "=" * 60)
    print("实现完成！")
    print("=" * 60)
```

## 关键特性总结：

1. **GPT架构**：仅Decoder，无Encoder
2. **位置编码**：支持学习的位置编码和RoPE
3. **KV缓存**：加速自回归生成
4. **采样策略**：贪婪、top-k、top-p、束搜索
5. **对话管理**：上下文窗口管理、历史记录
6. **文本生成**：温度控制、停止token处理

## 下一步优化建议：

1. **使用预训练权重**：加载GPT-2或GPT-3的权重
2. **实现更高效的attention**：如Flash Attention
3. **添加LoRA/QLoRA**：用于高效微调
4. **实现流式输出**：逐个token生成显示
5. **添加工具调用**：让模型能调用外部工具
6. **实现多模态**：支持图像、音频输入

这个实现提供了构建类ChatGPT模型的基础框架。你可以在此基础上添加更多功能！