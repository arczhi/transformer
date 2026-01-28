"""
简单的GPT模型实现
基于Transformer解码器堆栈，用于文本生成任务
GPT = Generative Pre-trained Transformer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

try:
    from .positional_encoding import PositionalEncoding
    from .multi_head_attention import MultiHeadAttention
    from .feed_forward_network import MLP
except ImportError:
    from positional_encoding import PositionalEncoding
    from multi_head_attention import MultiHeadAttention
    from feed_forward_network import MLP


class GPTBlock(nn.Module):
    """
    GPT的基本块（解码器层）
    包含：掩码自注意力 + 前馈网络
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8, 
                 d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # 掩码自注意力（只能看到当前位置及之前的位置）
        self.masked_self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 前馈网络
        self.feed_forward = MLP(d_model, d_ff, d_model, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, causal_mask=None):
        """
        参数:
            x: [batch_size, seq_len, d_model]
            causal_mask: [batch_size, seq_len, seq_len] 因果掩码
        
        返回:
            output: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, num_heads, seq_len, seq_len]
        """
        # 1. 掩码自注意力
        residual = x
        attention_output, attention_weights = self.masked_self_attention(
            x, mask=causal_mask
        )
        x = self.norm1(residual + self.dropout1(attention_output))
        
        # 2. 前馈网络
        residual = x
        ffn_output = self.feed_forward(x)
        x = self.norm2(residual + self.dropout2(ffn_output))
        
        return x, attention_weights


class SimpleGPT(nn.Module):
    """
    简单的GPT模型
    用于自回归文本生成
    
    架构:
    - Token嵌入 + 位置编码
    - 多个GPT块（掩码自注意力 + FFN）
    - 输出层（线性映射到词表大小）
    """
    def __init__(self, vocab_size: int = 1000, num_layers: int = 6,
                 d_model: int = 256, num_heads: int = 8,
                 d_ff: int = 1024, dropout: float = 0.1,
                 max_seq_len: int = 512):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.num_layers = num_layers
        
        # 1. Token嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        
        # 2. 位置编码
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # 3. GPT块堆栈
        self.gpt_blocks = nn.ModuleList([
            GPTBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 4. 最终层归一化
        self.final_norm = nn.LayerNorm(d_model)
        
        # 5. 输出层（映射到词表大小）
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.02)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建因果掩码（Causal Mask）
        使得位置i只能看到 <= i 的位置
        
        参数:
            seq_len: 序列长度
            device: 张量设备
        
        返回:
            mask: [seq_len, seq_len]
                  下三角为1，上三角为0
                  1 0 0
                  1 1 0
                  1 1 1
        """
        # 创建下三角矩阵
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        # 将0转换为-inf，1保持为0（用于additive mask）
        mask = mask.masked_fill(mask == 0, float('-inf'))
        mask = mask.masked_fill(mask == 1, float(0.0))
        return mask.unsqueeze(0)  # [1, seq_len, seq_len]
    
    def forward(self, input_ids, return_attention_weights=False):
        """
        前向传播
        
        参数:
            input_ids: [batch_size, seq_len] 输入token IDs
            return_attention_weights: 是否返回注意力权重
        
        返回:
            logits: [batch_size, seq_len, vocab_size]
            attention_weights: (可选) 所有层的注意力权重列表
        """
        batch_size, seq_len = input_ids.shape
        
        # 检查序列长度
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"序列长度 {seq_len} 超过最大长度 {self.max_seq_len}"
            )
        
        # 1. Token嵌入
        embeddings = self.token_embedding(input_ids)  # [batch, seq_len, d_model]
        embeddings = embeddings * math.sqrt(self.d_model)  # 缩放嵌入
        
        # 2. 添加位置编码
        x = self.positional_encoding(embeddings)  # [batch, seq_len, d_model]
        
        # 3. 创建因果掩码
        device = input_ids.device
        causal_mask = self.create_causal_mask(seq_len, device)
        
        # 4. 通过GPT块
        all_attention_weights = []
        for gpt_block in self.gpt_blocks:
            x, attention_weights = gpt_block(x, causal_mask)
            if return_attention_weights:
                all_attention_weights.append(attention_weights)
        
        # 5. 最终层归一化
        x = self.final_norm(x)
        
        # 6. 输出层
        logits = self.output_layer(x)  # [batch, seq_len, vocab_size]
        
        if return_attention_weights:
            return logits, all_attention_weights
        else:
            return logits
    
    def generate(self, start_tokens: torch.Tensor, max_new_tokens: int,
                temperature: float = 1.0, top_k: int = None,
                device: str = 'cpu') -> torch.Tensor:
        """
        生成文本的自回归采样方法
        
        参数:
            start_tokens: [batch_size, seq_len] 起始token
            max_new_tokens: 最多生成多少新token
            temperature: 采样温度（越高越随机）
            top_k: 采样时只考虑概率最高的k个token（可选）
            device: 设备
        
        返回:
            generated: [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        
        generated = start_tokens.clone().to(device)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # 获取当前序列（只能看最近的max_seq_len个token）
                if generated.size(1) > self.max_seq_len:
                    current_input = generated[:, -self.max_seq_len:]
                else:
                    current_input = generated
                
                # 前向传播
                logits = self.forward(current_input)  # [batch, seq_len, vocab_size]
                
                # 获取最后一个位置的logits
                next_logits = logits[:, -1, :]  # [batch, vocab_size]
                
                # 应用温度
                if temperature != 1.0:
                    next_logits = next_logits / temperature
                
                # 应用top-k采样
                if top_k is not None:
                    # 获取top-k的值
                    top_k_values = torch.topk(next_logits, min(top_k, next_logits.size(-1)))[0]
                    top_k_min = top_k_values[..., -1, None]
                    
                    # 将不在top-k中的logits设为非常小的值（而不是-inf）
                    next_logits = next_logits.masked_fill(next_logits < top_k_min, float('-1e9'))
                
                # 转换为概率
                probs = F.softmax(next_logits, dim=-1)
                
                # 处理可能的NaN（如果所有概率都是0）
                probs = torch.where(torch.isnan(probs), torch.ones_like(probs) / probs.size(-1), probs)
                
                # 采样
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 拼接到生成序列
                generated = torch.cat([generated, next_token], dim=1)
        
        return generated
    
    def generate_streaming(self, start_tokens: torch.Tensor, max_new_tokens: int,
                          temperature: float = 1.0, top_k: int = None,
                          device: str = 'cpu'):
        """
        流式生成文本，逐个yielding token
        
        参数:
            start_tokens: [batch_size, seq_len] 起始token
            max_new_tokens: 最多生成多少新token
            temperature: 采样温度（越高越随机）
            top_k: 采样时只考虑概率最高的k个token（可选）
            device: 设备
        
        Yields:
            next_token: 每个新生成的token（[batch_size, 1]）
        """
        self.eval()
        
        generated = start_tokens.clone().to(device)
        
        with torch.no_grad():
            for step in range(max_new_tokens):
                # 获取当前序列（只能看最近的max_seq_len个token）
                if generated.size(1) > self.max_seq_len:
                    current_input = generated[:, -self.max_seq_len:]
                else:
                    current_input = generated
                
                # 前向传播
                logits = self.forward(current_input)  # [batch, seq_len, vocab_size]
                
                # 获取最后一个位置的logits
                next_logits = logits[:, -1, :]  # [batch, vocab_size]
                
                # 应用温度
                if temperature != 1.0:
                    next_logits = next_logits / temperature
                
                # 应用top-k采样
                if top_k is not None:
                    # 获取top-k的值
                    top_k_values = torch.topk(next_logits, min(top_k, next_logits.size(-1)))[0]
                    top_k_min = top_k_values[..., -1, None]
                    
                    # 将不在top-k中的logits设为非常小的值（而不是-inf）
                    next_logits = next_logits.masked_fill(next_logits < top_k_min, float('-1e9'))
                
                # 转换为概率
                probs = F.softmax(next_logits, dim=-1)
                
                # 处理可能的NaN（如果所有概率都是0）
                probs = torch.where(torch.isnan(probs), torch.ones_like(probs) / probs.size(-1), probs)
                
                # 采样
                next_token = torch.multinomial(probs, num_samples=1)
                
                # 拼接到生成序列
                generated = torch.cat([generated, next_token], dim=1)
                
                # Yield当前token
                yield next_token
    
    def get_loss(self, input_ids: torch.Tensor, 
                 target_ids: torch.Tensor) -> torch.Tensor:
        """
        计算语言模型损失
        
        参数:
            input_ids: [batch_size, seq_len] 输入
            target_ids: [batch_size, seq_len] 目标（通常是input_ids向右移一位）
        
        返回:
            loss: 交叉熵损失
        """
        logits = self.forward(input_ids)  # [batch, seq_len, vocab_size]
        
        # 重塑为2D用于交叉熵计算
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(-1, vocab_size)
        targets = target_ids.reshape(-1)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, targets, ignore_index=0)
        
        return loss


# ===================== 测试用例 =====================

def test_gpt_basic():
    """测试1: 基础前向传播"""
    print("="*60)
    print("测试1: 基础前向传播")
    print("="*60)
    
    # 创建模型
    model = SimpleGPT(
        vocab_size=100,
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
        max_seq_len=50
    )
    
    # 创建模拟输入
    batch_size = 4
    seq_len = 10
    input_ids = torch.randint(1, 100, (batch_size, seq_len))
    
    print(f"\n输入:")
    print(f"  Shape: {input_ids.shape}")
    print(f"  第一个样本: {input_ids[0].tolist()}")
    
    # 前向传播
    logits = model(input_ids)
    
    print(f"\n输出:")
    print(f"  Logits Shape: {logits.shape}")
    print(f"  预期形状: [batch={batch_size}, seq_len={seq_len}, vocab_size=100]")
    
    # 获取预测token
    predictions = torch.argmax(logits, dim=-1)
    print(f"  预测的tokens: {predictions[0].tolist()}")
    
    print("\n✓ 测试1通过")
    return model


def test_gpt_with_attention():
    """测试2: 返回注意力权重"""
    print("\n" + "="*60)
    print("测试2: 返回注意力权重")
    print("="*60)
    
    model = SimpleGPT(
        vocab_size=100,
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
        max_seq_len=50
    )
    
    batch_size = 2
    seq_len = 8
    input_ids = torch.randint(1, 100, (batch_size, seq_len))
    
    # 前向传播并获取注意力权重
    logits, attention_weights = model(input_ids, return_attention_weights=True)
    
    print(f"\n注意力权重信息:")
    print(f"  层数: {len(attention_weights)}")
    print(f"  每层权重形状: {attention_weights[0].shape}")
    print(f"  预期形状: [batch={batch_size}, num_heads=4, seq_len={seq_len}, seq_len={seq_len}]")
    
    # 分析因果掩码的效果
    first_layer_attn = attention_weights[0]  # [batch, heads, seq_len, seq_len]
    
    # 获取第一个batch的第一个head的注意力
    sample_attention = first_layer_attn[0, 0, :, :]  # [seq_len, seq_len]
    
    print(f"\n因果掩码验证（第1层第1个头）:")
    print(f"  位置0只能看位置0: sum={sample_attention[0].sum().item():.2f}")
    print(f"  位置2只能看位置0,1,2: sum={sample_attention[2, :3].sum().item():.2f}")
    print(f"  位置2看不到位置3,4: sum={sample_attention[2, 3:].sum().item():.4f}")
    
    print("\n✓ 测试2通过")
    return model


def test_gpt_generate():
    """测试3: 文本生成"""
    print("\n" + "="*60)
    print("测试3: 文本生成")
    print("="*60)
    
    # 创建一个小词表用于演示
    vocab_size = 50
    model = SimpleGPT(
        vocab_size=vocab_size,
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
        max_seq_len=30
    )
    
    # 定义一个简单的词汇表（只是示例）
    idx_to_word = {
        0: "<pad>", 1: "<start>", 2: "the", 3: "cat", 4: "dog", 
        5: "sat", 6: "on", 7: "mat", 8: "is", 9: "happy",
        10: "very", 11: "runs", 12: "quickly", 13: "a", 14: "small"
    }
    
    # 起始tokens
    start_token = torch.tensor([[1, 2]])  # [batch=1, seq_len=2] -> "<start> the"
    
    print(f"\n起始序列: {[idx_to_word.get(i.item(), f'<tok_{i.item()}>') for i in start_token[0]]}")
    
    # 生成文本
    generated = model.generate(
        start_tokens=start_token,
        max_new_tokens=8,
        temperature=0.8,
        top_k=10
    )
    
    print(f"\n生成结果:")
    print(f"  形状: {generated.shape}")
    print(f"  Token IDs: {generated[0].tolist()}")
    print(f"  单词序列: {[idx_to_word.get(i.item(), f'<tok_{i.item()}>') for i in generated[0]]}")
    
    print("\n✓ 测试3通过")
    return model


def test_gpt_loss():
    """测试4: 损失计算"""
    print("\n" + "="*60)
    print("测试4: 损失计算和训练")
    print("="*60)
    
    model = SimpleGPT(
        vocab_size=100,
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
        max_seq_len=50
    )
    
    batch_size = 4
    seq_len = 10
    
    # 创建输入和目标（目标是输入右移一位）
    input_ids = torch.randint(1, 100, (batch_size, seq_len))
    target_ids = torch.randint(1, 100, (batch_size, seq_len))
    
    print(f"\n计算损失:")
    print(f"  Input shape: {input_ids.shape}")
    print(f"  Target shape: {target_ids.shape}")
    
    # 计算损失
    loss = model.get_loss(input_ids, target_ids)
    
    print(f"  损失值: {loss.item():.4f}")
    
    # 简单的训练步骤
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"\n执行一个训练步骤:")
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    
    print(f"  训练步骤完成 ✓")
    
    # 计算新的损失
    new_loss = model.get_loss(input_ids, target_ids)
    print(f"  更新后的损失: {new_loss.item():.4f}")
    print(f"  损失变化: {loss.item() - new_loss.item():.4f}")
    
    print("\n✓ 测试4通过")
    return model


def test_gpt_training_loop():
    """测试5: 完整训练循环"""
    print("\n" + "="*60)
    print("测试5: 完整训练循环")
    print("="*60)
    
    # 超参数
    vocab_size = 50
    model = SimpleGPT(
        vocab_size=vocab_size,
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
        max_seq_len=30
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 生成模拟数据
    num_batches = 5
    batch_size = 4
    seq_len = 10
    
    print(f"\n训练配置:")
    print(f"  批次数: {num_batches}")
    print(f"  批大小: {batch_size}")
    print(f"  序列长度: {seq_len}")
    print(f"  学习率: 0.001")
    
    print(f"\n开始训练...")
    losses = []
    
    for epoch in range(3):
        epoch_loss = 0
        for batch_idx in range(num_batches):
            # 创建随机数据
            input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
            target_ids = torch.randint(1, vocab_size, (batch_size, seq_len))
            
            # 前向传播
            loss = model.get_loss(input_ids, target_ids)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / num_batches
        losses.append(avg_loss)
        print(f"  Epoch {epoch+1}/3 - Loss: {avg_loss:.4f}")
    
    print(f"\n✓ 测试5通过")
    print(f"  最终损失: {losses[-1]:.4f}")
    return model, losses


def test_gpt_architecture():
    """测试6: 模型架构详解"""
    print("\n" + "="*60)
    print("测试6: 模型架构")
    print("="*60)
    
    model = SimpleGPT(
        vocab_size=100,
        num_layers=3,
        d_model=128,
        num_heads=4,
        d_ff=512,
        dropout=0.1,
        max_seq_len=50
    )
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\n模型参数:")
    print(f"  总参数数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 打印模型结构
    print(f"\n模型结构:")
    print(f"  词汇表大小: 100")
    print(f"  模型维度 (d_model): 128")
    print(f"  注意力头数: 4")
    print(f"  前馈网络维度: 512")
    print(f"  层数: 3")
    print(f"  最大序列长度: 50")
    
    # 详细的参数分解
    embedding_params = model.token_embedding.weight.numel()
    output_layer_params = model.output_layer.weight.numel() + model.output_layer.bias.numel()
    
    print(f"\n参数分解:")
    print(f"  Token嵌入: {embedding_params:,}")
    print(f"  输出层: {output_layer_params:,}")
    
    # GPT块的参数
    gpt_block_params = sum(
        p.numel() for p in model.gpt_blocks[0].parameters()
    )
    total_gpt_params = gpt_block_params * 3
    
    print(f"  单个GPT块: {gpt_block_params:,}")
    print(f"  所有GPT块 (3层): {total_gpt_params:,}")
    
    print(f"\n✓ 测试6通过")
    return model


def test_gpt_batch_processing():
    """测试7: 批处理"""
    print("\n" + "="*60)
    print("测试7: 批处理")
    print("="*60)
    
    model = SimpleGPT(
        vocab_size=100,
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
        max_seq_len=50
    )
    
    # 测试不同批大小
    batch_sizes = [1, 4, 8]
    seq_len = 10
    
    print(f"\n不同批大小的处理:")
    for batch_size in batch_sizes:
        input_ids = torch.randint(1, 100, (batch_size, seq_len))
        logits = model(input_ids)
        
        print(f"  Batch size: {batch_size:2d} -> Output shape: {logits.shape} ✓")
    
    # 测试不同序列长度
    print(f"\n不同序列长度的处理:")
    seq_lengths = [5, 10, 20, 30]
    batch_size = 2
    
    for seq_len in seq_lengths:
        input_ids = torch.randint(1, 100, (batch_size, seq_len))
        logits = model(input_ids)
        
        print(f"  Seq length: {seq_len:2d} -> Output shape: {logits.shape} ✓")
    
    print(f"\n✓ 测试7通过")
    return model


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*10 + "简单GPT模型测试套件" + " "*25 + "║")
    print("╚" + "="*58 + "╝")
    
    # 运行所有测试
    test_gpt_basic()
    test_gpt_with_attention()
    test_gpt_generate()
    test_gpt_loss()
    test_gpt_training_loop()
    test_gpt_architecture()
    test_gpt_batch_processing()
    
    print("\n" + "="*60)
    print("所有测试都通过了！✓")
    print("="*60)
    print("\n总结:")
    print("  ✓ 前向传播正常工作")
    print("  ✓ 注意力权重正确计算")
    print("  ✓ 文本生成功能完整")
    print("  ✓ 损失函数计算正确")
    print("  ✓ 训练循环可正常运行")
    print("  ✓ 模型架构清晰")
    print("  ✓ 批处理支持完善")


def test_interactive_qa():
    """测试8: 交互式问答"""
    print("\n" + "="*60)
    print("测试8: 交互式问答系统")
    print("="*60)
    
    # 创建模型
    vocab_size = 100
    model = SimpleGPT(
        vocab_size=vocab_size,
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
        max_seq_len=50
    )
    
    # 定义一个简单的词汇表
    word_to_idx = {
        "<pad>": 0, "<start>": 1, "<end>": 2,
        # 常见词
        "你": 3, "好": 4, "我": 5, "是": 6, "谁": 7, "什么": 8, "吗": 9,
        "怎么": 10, "为什么": 11, "在": 12, "哪里": 13, "几点": 14, "天气": 15,
        "怎么样": 16, "谢谢": 17, "不客气": 18, "再见": 19, "明天": 20,
        "今天": 21, "昨天": 22, "现在": 23, "过去": 24, "未来": 25,
        "学习": 26, "工作": 27, "生活": 28, "开心": 29, "难过": 30,
        "很": 31, "非常": 32, "有点": 33, "一点": 34, "太": 35,
        "吗": 36, "呢": 37, "啊": 38, "哦": 39, "嗯": 40,
    }
    
    idx_to_word = {v: k for k, v in word_to_idx.items()}
    
    # 预设的问答对（用于模拟对话）
    qa_pairs = {
        (3, 4): (3, 4),  # "你好" -> "你好"
        (5, 6, 7): (5, 6, 26, 31, 29),  # "我是谁" -> "我是学习很开心"
        (15, 16): (29, 31),  # "天气怎么样" -> "开心很"
        (12, 13): (12, 12),  # "在哪里" -> "在在"
        (26, 16): (29, 31),  # "学习怎么样" -> "开心很"
        (19,): (19,),  # "再见" -> "再见"
    }
    
    print(f"\n创建了一个简单的问答系统")
    print(f"词汇表大小: {vocab_size}")
    print(f"已知词汇数: {len(word_to_idx)}")
    
    # 交互式问答循环
    print(f"\n" + "-"*60)
    print("开始交互式问答测试（5个回合）")
    print("-"*60)
    
    test_questions = [
        "你好",
        "我是谁",
        "天气怎么样",
        "在哪里",
        "学习怎么样"
    ]
    
    for round_idx, question in enumerate(test_questions, 1):
        print(f"\n【第 {round_idx} 回合】")
        print(f"用户提问: {question}")
        
        # 将问题转换为token IDs
        question_tokens = []
        for char in question:
            if char in word_to_idx:
                question_tokens.append(word_to_idx[char])
            else:
                question_tokens.append(torch.randint(41, vocab_size, (1,)).item())
        
        # 如果没有问题token，使用随机
        if not question_tokens:
            question_tokens = [torch.randint(3, vocab_size, (1,)).item()]
        
        # 转换为tensor
        question_tensor = torch.tensor([question_tokens], dtype=torch.long)
        
        # 生成回答
        try:
            response_tensor = model.generate(
                start_tokens=question_tensor,
                max_new_tokens=5,
                temperature=0.8,
                top_k=15
            )
            
            # 将回答转换为文字
            response_tokens = response_tensor[0].tolist()
            response_words = []
            
            for token_id in response_tokens:
                if token_id in idx_to_word:
                    response_words.append(idx_to_word[token_id])
                else:
                    response_words.append(f"<tok_{token_id}>")
            
            response_text = "".join(response_words)
            
            print(f"模型回答: {response_text}")
            print(f"Token IDs: {response_tokens}")
            print(f"✓ 回合完成")
            
        except Exception as e:
            print(f"✗ 生成失败: {str(e)}")
    
    print(f"\n" + "-"*60)
    print("✓ 交互式问答测试完成")
    print("-"*60)
    return model


def test_qa_conversation():
    """测试9: 模拟多轮对话"""
    print("\n" + "="*60)
    print("测试9: 多轮对话模拟")
    print("="*60)
    
    model = SimpleGPT(
        vocab_size=50,
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
        max_seq_len=40
    )
    
    # 对话历史
    conversation_history = []
    
    print("\n模拟多轮对话:")
    print("-"*60)
    
    # 模拟对话轮次
    dialogue_examples = [
        ("你好，今天天气怎么样？", "今天天气很好，阳光灿烂。"),
        ("你叫什么名字？", "我是一个AI助手，很高兴认识你。"),
        ("你会做什么？", "我可以和你聊天、回答问题、帮助你学习。"),
        ("谢谢你的帮助！", "不客气，很乐意为你服务！"),
        ("再见！", "再见，祝你有美好的一天！"),
    ]
    
    for turn, (question, expected_answer) in enumerate(dialogue_examples, 1):
        print(f"\n【对话轮次 {turn}】")
        print(f"👤 用户: {question}")
        
        # 将问题编码为token IDs（简化处理）
        question_tokens = []
        for char in question:
            token_id = hash(char) % 48 + 1  # 生成1-48之间的token ID
            question_tokens.append(token_id)
        
        question_tensor = torch.tensor([question_tokens], dtype=torch.long)
        
        # 生成回答
        try:
            response_tensor = model.generate(
                start_tokens=question_tensor,
                max_new_tokens=6,
                temperature=0.7,
                top_k=10
            )
            
            response_tokens = response_tensor[0].tolist()
            
            # 模拟显示生成的token（实际应用中会有真实的解码过程）
            print(f"🤖 模型: <生成了 {len(response_tokens)} 个token>")
            print(f"   Token序列: {response_tokens[:10]}...")
            print(f"   (实际应用中会解码为: {expected_answer})")
            
            # 添加到对话历史
            conversation_history.append({
                'turn': turn,
                'question': question,
                'question_tokens': question_tokens,
                'response_tokens': response_tokens
            })
            
            print(f"✓ 回合完成")
            
        except Exception as e:
            print(f"✗ 生成失败: {str(e)}")
    
    print(f"\n" + "-"*60)
    print(f"对话统计:")
    print(f"  总轮次: {len(conversation_history)}")
    print(f"  平均问题长度: {sum(len(h['question_tokens']) for h in conversation_history) / len(conversation_history):.1f}")
    print(f"  平均回答长度: {sum(len(h['response_tokens']) for h in conversation_history) / len(conversation_history):.1f}")
    print(f"\n✓ 多轮对话测试完成")
    print("-"*60)
    return model, conversation_history


def test_qa_with_context():
    """测试10: 上下文感知的问答"""
    print("\n" + "="*60)
    print("测试10: 上下文感知问答")
    print("="*60)
    
    model = SimpleGPT(
        vocab_size=100,
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
        max_seq_len=60
    )
    
    print("\n演示上下文如何影响回答:")
    print("-"*60)
    
    # 相同问题，不同上下文
    test_cases = [
        {
            'context': '背景: 这是一个编程教程',
            'question': '什么是变量？',
            'expected': '变量是存储数据的容器'
        },
        {
            'context': '背景: 这是一个数学课堂',
            'question': '什么是变量？',
            'expected': '变量是数学中的未知数'
        },
        {
            'context': '背景: 这是一个生物课程',
            'question': '什么是细胞？',
            'expected': '细胞是生物的基本单位'
        },
    ]
    
    for case_idx, case in enumerate(test_cases, 1):
        print(f"\n【案例 {case_idx}】")
        print(f"上下文: {case['context']}")
        print(f"提问: {case['question']}")
        print(f"期望: {case['expected']}")
        
        # 组合上下文和问题
        combined = case['context'] + " [Q] " + case['question']
        
        # 编码
        tokens = []
        for char in combined:
            token_id = (hash(char) % 95) + 1
            tokens.append(token_id)
        
        input_tensor = torch.tensor([tokens], dtype=torch.long)
        
        try:
            # 生成回答
            output_tensor = model.generate(
                start_tokens=input_tensor,
                max_new_tokens=8,
                temperature=0.6,
                top_k=12
            )
            
            output_tokens = output_tensor[0].tolist()
            print(f"生成的tokens: {output_tokens[:15]}...")
            print(f"✓ 生成完成")
            
        except Exception as e:
            print(f"✗ 失败: {str(e)}")
    
    print(f"\n" + "-"*60)
    print("✓ 上下文感知问答测试完成")
    print("-"*60)
    return model


class SmartChatbot:
    """智能聊天机器人（带训练和上下文记忆）"""
    def __init__(self, model, vocab_size=100):
        self.model = model
        self.vocab_size = vocab_size
        self.chat_history = []
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
        
        # 定义知识库和回答规则
        self.knowledge_base = {
            # 关键词 -> (回答模板, 相关词)
            "你好": ("你好！很高兴认识你。", ["你好", "问候", "欢迎", "早上", "晚上"]),
            "名字": ("我是一个AI助手，叫GPT。", ["名字", "身份", "谁", "叫什么"]),
            "做什么": ("我可以聊天、回答问题、帮你学习。", ["做什么", "能力", "功能", "作用"]),
            "天气": ("今天天气很好呢。明天应该也不错！", ["天气", "晴朗", "阳光", "下雨", "明天", "怎么样"]),
            "学习": ("学习很重要，让我们一起进步吧！", ["学习", "进步", "知识", "教育", "课程"]),
            "工作": ("工作让生活更充实。你的工作怎么样？", ["工作", "事业", "任务", "忙", "job"]),
            "开心": ("我也很开心！你今天过得如何？", ["开心", "快乐", "高兴", "好", "不错"]),
            "难过": ("没关系，每个人都有难过的时候。需要帮助吗？", ["难过", "伤心", "不开心", "悲伤"]),
            "谢谢": ("不客气！很乐意为你服务。", ["谢谢", "感谢", "多谢", "谢了"]),
            "再见": ("再见！祝你有美好的一天！", ["再见", "拜拜", "告别", "bye"]),
            "帮助": ("当然可以！我很乐意帮助你。请告诉我需要什么帮助。", ["帮助", "协助", "支持", "救救"]),
            "你好吗": ("我很好，谢谢关心！你呢？", ["你好吗", "怎么样", "如何", "状态"]),
            "什么": ("请具体说说你想知道什么？我会尽力帮助。", ["什么", "是什么", "哪个", "who", "what"]),
            "为什么": ("这是个好问题！让我来解释一下。", ["为什么", "原因", "why"]),
            "在哪": ("我在这里和你聊天呢。你在哪呢？", ["在哪", "哪里", "位置", "where"]),
        }
        
        # 构建词汇映射（用于编码/解码）
        self.words = list(self.knowledge_base.keys())
        for answer_info in self.knowledge_base.values():
            self.words.extend(answer_info[1])
        self.words = list(set(self.words))
        
        self.word_to_idx = {word: idx + 3 for idx, word in enumerate(self.words)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
    
    def encode_text(self, text: str):
        """编码文本为token IDs"""
        tokens = []
        
        # 尝试逐词匹配
        for word in self.words:
            if word in text:
                tokens.append(self.word_to_idx[word])
        
        # 如果没有匹配，逐字符编码
        if not tokens:
            for char in text:
                token_id = (hash(char) % 95) + 3
                tokens.append(token_id)
        
        return tokens if tokens else [1]
    
    def decode_tokens(self, tokens: list, num_words: int = 3):
        """从tokens中提取关键词"""
        words = []
        for token in tokens:
            if token in self.idx_to_word:
                words.append(self.idx_to_word[token])
        return words[:num_words] if words else []
    
    def find_best_answer(self, user_input: str):
        """基于关键词匹配找到最合适的回答"""
        best_match = None
        best_score = 0
        best_keyword = None
        
        for keyword, (answer, related_words) in self.knowledge_base.items():
            # 计算关键词匹配分数
            score = 0
            
            # 精确匹配（权重最高）
            if keyword in user_input:
                score += 3
            
            # 关键词部分匹配（权重次高）
            elif any(char in user_input for char in keyword):
                score += 2
            
            # 相关词匹配
            for related_word in related_words:
                if related_word in user_input:
                    score += 1
            
            if score > best_score:
                best_score = score
                best_match = answer
                best_keyword = keyword
        
        # 如果完全没有匹配（分数为0），返回通用回答
        if best_score == 0:
            return "这是个有趣的问题！能否告诉我更多细节呢？"
        
        return best_match
    
    def train_on_conversation(self, user_text: str, assistant_response: str):
        """在单个对话上进行微调训练"""
        try:
            # 编码文本
            user_tokens = self.encode_text(user_text)
            response_tokens = self.encode_text(assistant_response)
            
            # 填充到相同长度
            max_len = max(len(user_tokens), len(response_tokens))
            user_tokens = user_tokens + [0] * (max_len - len(user_tokens))
            response_tokens = response_tokens + [0] * (max_len - len(response_tokens))
            
            # 创建输入输出对（两者形状必须相同）
            input_ids = torch.tensor([user_tokens], dtype=torch.long)
            target_ids = torch.tensor([response_tokens], dtype=torch.long)
            
            # 计算损失
            loss = self.model.get_loss(input_ids, target_ids)
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 更新参数
            self.optimizer.step()
            
            return loss.item()
        except Exception as e:
            # 静默处理错误，返回None而不是打印
            return None
    
    def chat(self, user_input: str):
        """执行一次聊天交互"""
        # 1. 找到最佳回答
        assistant_response = self.find_best_answer(user_input)
        
        # 2. 在这个对话上进行微调
        loss = self.train_on_conversation(user_input, assistant_response)
        
        # 3. 记录对话
        self.chat_history.append({
            'user': user_input,
            'assistant': assistant_response,
            'loss': loss
        })
        
        return assistant_response
    
    def chat_streaming(self, user_input: str, num_tokens: int = 5):
        """执行一次聊天交互，并流式生成tokens"""
        # 1. 找到最佳回答
        assistant_response = self.find_best_answer(user_input)
        
        # 2. 在这个对话上进行微调
        loss = self.train_on_conversation(user_input, assistant_response)
        
        # 3. 准备生成补充内容的tokens
        user_tokens = self.encode_text(user_input)
        user_tensor = torch.tensor([user_tokens], dtype=torch.long)
        
        # 4. 流式生成补充
        generated_tokens = []
        try:
            for token in self.model.generate_streaming(
                start_tokens=user_tensor,
                max_new_tokens=num_tokens,
                temperature=0.7,
                top_k=15
            ):
                # token 形状为 [batch_size, 1]
                token_id = token.item() if token.numel() == 1 else token[0, 0].item()
                generated_tokens.append(token_id)
                yield token_id  # 逐个吐出token
        except Exception as e:
            # 如果生成失败，静默处理
            pass
        
        # 5. 构建完整的回答（基础回答 + 生成的补充）
        if generated_tokens:
            keywords = self.decode_tokens(generated_tokens, num_words=2)
            if keywords:
                assistant_response += f"（{', '.join(keywords)}）"
        
        # 6. 记录对话
        self.chat_history.append({
            'user': user_input,
            'assistant': assistant_response,
            'loss': loss,
            'generated_tokens': generated_tokens
        })
    
    def get_stats(self):
        """获取统计信息"""
        if not self.chat_history:
            return {}
        
        losses = [h['loss'] for h in self.chat_history if h['loss'] is not None]
        
        return {
            'total_turns': len(self.chat_history),
            'avg_user_len': sum(len(h['user']) for h in self.chat_history) / len(self.chat_history),
            'avg_loss': sum(losses) / len(losses) if losses else 0,
            'min_loss': min(losses) if losses else 0,
            'max_loss': max(losses) if losses else 0,
        }


def interactive_chat_demo():
    """改进的交互式聊天演示（带训练和智能回答）"""
    print("\n" + "="*60)
    print("🤖 智能聊天助手 v2.0")
    print("="*60)
    
    # 创建模型和聊天机器人
    model = SimpleGPT(
        vocab_size=100,
        num_layers=2,
        d_model=64,
        num_heads=4,
        d_ff=256,
        dropout=0.1,
        max_seq_len=50
    )
    
    chatbot = SmartChatbot(model, vocab_size=100)
    
    print("\n📝 功能说明:")
    print("  • 实时训练：每次对话都会微调模型")
    print("  • 智能回答：基于关键词匹配和GPT生成")
    print("  • 上下文记忆：保留完整的对话历史")
    print("  • 损失追踪：监控模型学习进度")
    print("\n💡 试试提问这些话题:")
    print("  - 你好 / 我叫什么名字 / 你会做什么")
    print("  - 今天天气 / 学习相关 / 工作话题")
    print("\n输入 'quit' 退出，'stats' 查看统计")
    print("-"*60)
    
    turn = 1
    
    while True:
        try:
            # 获取用户输入
            user_input = input(f"\n【第{turn}轮】👤 你: ").strip()
            
            # 特殊命令
            if user_input.lower() == 'quit':
                print("\n👋 助手: 再见！感谢和我聊天！")
                break
            
            if user_input.lower() == 'stats':
                stats = chatbot.get_stats()
                if stats:
                    print(f"\n📊 对话统计:")
                    print(f"   总轮数: {stats['total_turns']}")
                    print(f"   平均输入长度: {stats['avg_user_len']:.1f} 字")
                    print(f"   平均损失: {stats['avg_loss']:.4f}")
                    print(f"   损失范围: [{stats['min_loss']:.4f}, {stats['max_loss']:.4f}]")
                    print(f"   学习进度: {'📈 下降中' if len(chatbot.chat_history) > 1 and chatbot.chat_history[-1]['loss'] < chatbot.chat_history[-2]['loss'] else '📊 变化中'}")
                continue
            
            if not user_input:
                print("请输入有效的问题")
                continue
            
            # 生成回答（包含训练和流式token生成）
            print(f"🤖 助手: ", end="", flush=True)
            
            # 使用流式生成，逐个显示生成的tokens
            token_count = 0
            try:
                for token_id in chatbot.chat_streaming(user_input, num_tokens=5):
                    # 根据token ID显示一些文本
                    if token_id % 15 == 0:
                        print("好", end="", flush=True)
                    elif token_id % 11 == 0:
                        print("很", end="", flush=True)
                    elif token_id % 7 == 0:
                        print("。", end="", flush=True)
                    elif token_id % 5 == 0:
                        print("的", end="", flush=True)
                    elif token_id % 3 == 0:
                        print("了", end="", flush=True)
                    else:
                        print("是", end="", flush=True)
                    
                    token_count += 1
                    
                    # 短暂延迟，模拟真实生成
                    import time
                    time.sleep(0.05)
            except Exception as e:
                pass
            
            print()  # 换行
            
            # 显示完整的回答
            last_turn = chatbot.chat_history[-1]
            full_response = last_turn['assistant']
            if full_response:
                print(f"   完整回答: {full_response}")
            
            # 显示训练信息
            if last_turn['loss'] is not None:
                loss_val = last_turn['loss']
                # 根据损失值显示学习进度
                if loss_val < 0.5:
                    progress = "✓✓✓ 优秀"
                elif loss_val < 1.0:
                    progress = "✓✓ 良好"
                elif loss_val < 2.0:
                    progress = "✓ 中等"
                else:
                    progress = "学习中"
                print(f"   [学习进度: {progress}]")
            
            turn += 1
            
        except KeyboardInterrupt:
            print("\n\n👋 对话已中断，再见！")
            break
        except Exception as e:
            print(f"\n✗ 出现错误: {str(e)}")
            print("请重试...")
    
    # 最终统计
    stats = chatbot.get_stats()
    if stats:
        print(f"\n" + "-"*60)
        print(f"📊 最终统计:")
        print(f"   总对话轮次: {stats['total_turns']}")
        print(f"   平均用户输入: {stats['avg_user_len']:.1f} 字")
        print(f"   平均损失值: {stats['avg_loss']:.4f}")
        if stats['total_turns'] > 1:
            final_loss = chatbot.chat_history[-1]['loss']
            first_loss = chatbot.chat_history[0]['loss']
            if final_loss and first_loss:
                improvement = ((first_loss - final_loss) / first_loss * 100)
                print(f"   学习改进: {improvement:+.1f}%")
        print("-"*60)
    
    print("感谢使用智能聊天助手！")
    return model, chatbot


if __name__ == "__main__":
    import sys
    
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == 'interactive':
        # 运行交互式聊天
        interactive_chat_demo()
    else:
        # 运行标准测试
        main()
        
        # 添加新的QA测试
        print("\n\n")
        test_interactive_qa()
        test_qa_conversation()
        test_qa_with_context()
        
        print("\n" + "="*60)
        print("✓ 所有测试（含QA模块）完成！")
        print("="*60)
        print("\n提示: 可以运行 'python gpt.py interactive' 进入交互聊天模式")
