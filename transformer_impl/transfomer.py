import torch
import torch.nn as nn
import torch.nn.functional as F
try :
    from .encoder import TransformerEncoder
except ImportError:
    from encoder import TransformerEncoder
try :
    from .decoder import TransformerDecoder
except ImportError:
    from decoder import TransformerDecoder

class Transformer(nn.Module):
    """
    完整的Transformer模型（Encoder + Decoder）
    适用于序列到序列任务，如机器翻译
    """
    def __init__(self, src_vocab_size: int, tgt_vocab_size: int, 
                 num_layers: int = 6, d_model: int = 512, 
                 num_heads: int = 8, d_ff: int = 2048, 
                 dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()
        
        # 1. 源语言（输入）的嵌入层
        self.src_embedding = nn.Sequential(
            nn.Embedding(src_vocab_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # 2. 目标语言（输出）的嵌入层
        self.tgt_embedding = nn.Sequential(
            nn.Embedding(tgt_vocab_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # 3. 位置编码
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, d_model)
        )
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)
        
        # 4. Encoder
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # 5. Decoder
        self.decoder = TransformerDecoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # 6. 输出层（线性变换 + softmax）
        self.output_layer = nn.Linear(d_model, tgt_vocab_size)
        
        # 7. 初始化参数
        self._init_parameters()
        
    def _init_parameters(self):
        """初始化模型参数"""
        # Xavier初始化
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_padding_mask(self, x, pad_idx=0):
        """
        创建padding掩码
        参数:
            x: 输入序列 [batch_size, seq_len]
            pad_idx: padding的索引
        返回:
            mask: [batch_size, 1, 1, seq_len]
        """
        # 找出padding位置
        mask = (x != pad_idx).unsqueeze(1).unsqueeze(2)
        return mask.float()
    
    def create_lookahead_mask(self, seq_len):
        """
        创建lookahead掩码（因果掩码）
        用于Decoder的自注意力
        """
        mask = torch.tril(torch.ones(seq_len, seq_len))
        return mask  # [seq_len, seq_len]
    
    def encode(self, src, src_mask=None):
        """编码源序列"""
        # 1. 嵌入
        src_emb = self.src_embedding(src)
        
        # 2. 添加位置编码
        seq_len = src.size(1)
        src_emb = src_emb + self.positional_encoding[:, :seq_len, :]
        
        # 3. 通过Encoder
        encoder_output, _ = self.encoder(src_emb, src_mask)
        
        return encoder_output
    
    def decode(self, tgt, encoder_output, tgt_mask=None, src_tgt_mask=None):
        """解码目标序列"""
        # 1. 嵌入
        tgt_emb = self.tgt_embedding(tgt)
        
        # 2. 添加位置编码
        seq_len = tgt.size(1)
        tgt_emb = tgt_emb + self.positional_encoding[:, :seq_len, :]
        
        # 3. 通过Decoder
        decoder_output, _, _ = self.decoder(
            tgt_emb, encoder_output, tgt_mask, src_tgt_mask
        )
        
        return decoder_output
    
    def forward(self, src, tgt, src_pad_idx=0, tgt_pad_idx=0):
        """
        完整的前向传播
        
        参数:
            src: 源序列 [batch_size, src_seq_len]
            tgt: 目标序列 [batch_size, tgt_seq_len]
        
        返回:
            output: 预测结果 [batch_size, tgt_seq_len, tgt_vocab_size]
        """
        batch_size, src_seq_len = src.shape
        tgt_seq_len = tgt.shape[1]
        
        # 1. 创建掩码
        # 源序列padding掩码
        src_mask = self.create_padding_mask(src, src_pad_idx)
        src_mask = src_mask.expand(-1, -1, src_seq_len, -1).squeeze(1)  # [batch, src_seq_len, src_seq_len]
        
        # 目标序列padding掩码
        tgt_padding_mask = self.create_padding_mask(tgt, tgt_pad_idx)
        tgt_padding_mask = tgt_padding_mask.expand(-1, -1, tgt_seq_len, -1).squeeze(1)
        
        # 目标序列lookahead掩码（因果掩码）
        tgt_lookahead_mask = self.create_lookahead_mask(tgt_seq_len)
        tgt_lookahead_mask = tgt_lookahead_mask.unsqueeze(0)  # [1, tgt_seq_len, tgt_seq_len]
        
        # 结合两种目标掩码
        tgt_mask = tgt_padding_mask.bool() & tgt_lookahead_mask.bool()
        tgt_mask = tgt_mask.float()
        
        # 交叉注意力掩码（防止Decoder attend到Encoder的padding）
        src_tgt_mask = self.create_padding_mask(src, src_pad_idx)  # [batch, 1, 1, src_seq_len]
        src_tgt_mask = src_tgt_mask.expand(-1, -1, tgt_seq_len, -1)  # [batch, 1, tgt_seq_len, src_seq_len]
        src_tgt_mask = src_tgt_mask.squeeze(1)  # [batch, tgt_seq_len, src_seq_len]
        
        # 2. 编码
        encoder_output = self.encode(src, src_mask)
        
        # 3. 解码
        decoder_output = self.decode(tgt, encoder_output, tgt_mask, src_tgt_mask)
        
        # 4. 输出层
        output = self.output_layer(decoder_output)
        
        return output
    
    def generate(self, src, max_len=50, start_token=1, end_token=2, 
                 src_pad_idx=0, temperature=1.0):
        """
        生成序列（贪婪解码）
        
        参数:
            src: 源序列 [batch_size, src_seq_len]
            max_len: 最大生成长度
            start_token: 起始token
            end_token: 结束token
            temperature: 温度参数（用于随机采样）
        
        返回:
            generated: 生成的序列 [batch_size, generated_len]
        """
        self.eval()
        batch_size = src.shape[0]
        
        # 编码源序列
        encoder_output = self.encode(src)
        
        # 初始化目标序列（以start_token开始）
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=src.device)
        
        # 生成序列
        for i in range(max_len):
            # 解码
            output = self.decode(generated, encoder_output)
            
            # 获取最后一个位置的输出
            last_output = output[:, -1, :]  # [batch_size, d_model]
            
            # 通过输出层
            logits = self.output_layer(last_output) / temperature  # [batch_size, tgt_vocab_size]
            
            # 使用softmax获取概率
            probs = F.softmax(logits, dim=-1)
            
            # 贪婪选择（取概率最大的token）
            next_token = torch.argmax(probs, dim=-1, keepdim=True)  # [batch_size, 1]
            
            # 添加到生成的序列中
            generated = torch.cat([generated, next_token], dim=1)
            
            # 检查是否所有序列都生成了结束token
            if (next_token == end_token).all():
                break
        
        return generated
    
"""
    翻译测试
"""
def create_simple_translator():
    """创建一个简单的翻译模型示例"""
    
    print("=== 创建简单翻译模型 ===\n")
    
    # 配置参数
    config = {
        'src_vocab_size': 10000,  # 源语言词汇表大小
        'tgt_vocab_size': 10000,  # 目标语言词汇表大小
        'num_layers': 3,          # 为了演示，使用较少的层
        'd_model': 128,
        'num_heads': 4,
        'd_ff': 512,
        'dropout': 0.1,
        'max_seq_len': 50
    }
    
    # 创建模型
    model = Transformer(**config)
    
    print("模型配置:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n模型参数统计:")
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    return model, config


def test_transformer_forward():
    """测试Transformer的前向传播"""
    
    model, config = create_simple_translator()
    
    # 创建模拟数据
    batch_size = 2
    src_seq_len = 10
    tgt_seq_len = 12
    
    # 源序列和目标序列（随机生成）
    src = torch.randint(3, config['src_vocab_size'], (batch_size, src_seq_len))
    tgt = torch.randint(3, config['tgt_vocab_size'], (batch_size, tgt_seq_len))
    
    print(f"\n=== 测试前向传播 ===")
    print(f"源序列形状: {src.shape}")
    print(f"目标序列形状: {tgt.shape}")
    
    # 前向传播
    output = model(src, tgt)
    
    print(f"输出形状: {output.shape}")
    print(f"期望形状: [batch_size, tgt_seq_len, tgt_vocab_size]")
    
    # 检查输出
    print(f"\n输出验证:")
    print(f"  输出值范围: [{output.min():.3f}, {output.max():.3f}]")
    print(f"  输出均值: {output.mean():.3f}")
    print(f"  输出标准差: {output.std():.3f}")
    
    return model, src, tgt, output


def visualize_decoder_attention():
    """可视化Decoder的注意力权重"""
    
    # 创建一个小模型用于可视化
    small_config = {
        'src_vocab_size': 100,
        'tgt_vocab_size': 100,
        'num_layers': 2,
        'd_model': 64,
        'num_heads': 2,
        'd_ff': 128,
        'dropout': 0.0,
        'max_seq_len': 20
    }
    
    model = Transformer(**small_config)
    
    # 创建输入
    batch_size = 1
    src_seq = torch.randint(3, small_config['src_vocab_size'], (batch_size, 8))
    tgt_seq = torch.randint(3, small_config['tgt_vocab_size'], (batch_size, 6))
    
    # 手动运行以便获取注意力权重
    src_mask = model.create_padding_mask(src_seq, 0)
    src_mask = src_mask.expand(-1, -1, 8, -1).squeeze(1)
    
    tgt_padding_mask = model.create_padding_mask(tgt_seq, 0)
    tgt_padding_mask = tgt_padding_mask.expand(-1, -1, 6, -1).squeeze(1)
    tgt_lookahead_mask = model.create_lookahead_mask(6).unsqueeze(0)
    tgt_mask = tgt_padding_mask.bool() & tgt_lookahead_mask.bool()
    tgt_mask = tgt_mask.float()
    
    src_tgt_mask = model.create_padding_mask(src_seq, 0)
    src_tgt_mask = src_tgt_mask.expand(-1, -1, 6, -1).squeeze(1)
    
    # 编码
    encoder_output = model.encode(src_seq, src_mask)
    
    # 解码
    tgt_emb = model.tgt_embedding(tgt_seq)
    tgt_emb = tgt_emb + model.positional_encoding[:, :6, :]
    
    # 手动运行Decoder层以获取注意力权重
    all_self_attention_weights = []
    all_cross_attention_weights = []
    
    x = tgt_emb
    for i, layer in enumerate(model.decoder.layers):
        # 自注意力
        self_attn_output, self_attention_weights = layer.self_attention(
            x, x, x, tgt_mask
        )
        x = layer.norm1(x + self_attn_output)
        
        # 交叉注意力
        cross_attn_output, cross_attention_weights = layer.cross_attention(
            x, encoder_output, encoder_output, src_tgt_mask
        )
        x = layer.norm2(x + cross_attn_output)
        
        # 前馈网络
        ff_output = layer.feed_forward(x)
        x = layer.norm3(x + ff_output)
        
        all_self_attention_weights.append(self_attention_weights)
        all_cross_attention_weights.append(cross_attention_weights)
    
    # 可视化
    import matplotlib.pyplot as plt
    
    # 可视化第一层的自注意力权重
    if len(all_self_attention_weights) > 0:
        self_attn = all_self_attention_weights[0][0, 0].detach().numpy()
        
        plt.figure(figsize=(10, 8))
        plt.subplot(2, 2, 1)
        plt.imshow(self_attn, cmap='viridis')
        plt.colorbar()
        plt.title("Self-Attention (Head 0, Layer 0)")
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        
        # 可视化因果掩码
        plt.subplot(2, 2, 2)
        plt.imshow(tgt_mask[0].detach().numpy(), cmap='gray')
        plt.title("Causal Mask")
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
    
    # 可视化第一层的交叉注意力权重
    if len(all_cross_attention_weights) > 0:
        cross_attn = all_cross_attention_weights[0][0, 0].detach().numpy()
        
        plt.subplot(2, 2, 3)
        plt.imshow(cross_attn, cmap='viridis')
        plt.colorbar()
        plt.title("Cross-Attention (Head 0, Layer 0)")
        plt.xlabel("Encoder Position")
        plt.ylabel("Decoder Position")
        
        # 可视化源-目标掩码
        plt.subplot(2, 2, 4)
        plt.imshow(src_tgt_mask[0].detach().numpy(), cmap='gray')
        plt.title("Source-Target Mask")
        plt.xlabel("Encoder Position")
        plt.ylabel("Decoder Position")
        
        plt.tight_layout()
        plt.show()
    
    return all_self_attention_weights, all_cross_attention_weights


def demonstrate_masking():
    """演示各种掩码的作用"""
    
    print("=== 掩码演示 ===\n")
    
    seq_len = 5
    
    # 1. 因果掩码
    print("1. 因果掩码 (Causal Mask):")
    print("   防止Decoder看到未来信息")
    causal_mask = torch.tril(torch.ones(seq_len, seq_len))
    print(causal_mask)
    
    # 2. Padding掩码
    print("\n2. Padding掩码:")
    print("   防止注意力到padding位置")
    # 假设序列实际长度为3，后2个是padding
    padding_mask = torch.ones(1, seq_len, seq_len)
    padding_mask[:, :, 3:] = 0
    print(padding_mask[0])
    
    # 3. 组合掩码
    print("\n3. 组合掩码 (Causal + Padding):")
    print("   同时防止看到未来信息和padding")
    combined_mask = causal_mask.unsqueeze(0) * padding_mask
    print(combined_mask[0])
    
    # 4. 源-目标掩码
    print("\n4. 源-目标掩码 (Source-Target Mask):")
    print("   防止Decoder attend到Encoder的padding")
    src_len = 4
    tgt_len = 5
    # 假设源序列实际长度为3，目标序列实际长度为4
    src_tgt_mask = torch.ones(1, tgt_len, src_len)
    src_tgt_mask[:, :, 3:] = 0  # 源序列的padding
    src_tgt_mask[:, 4:, :] = 0  # 目标序列的padding
    print(src_tgt_mask[0])
    
    return {
        'causal': causal_mask,
        'padding': padding_mask,
        'combined': combined_mask,
        'src_tgt': src_tgt_mask
    }


# 运行演示
print("=== Transformer Decoder 演示 ===")
print("=" * 50 + "\n")

# 测试前向传播
model, src, tgt, output = test_transformer_forward()

print("\n" + "="*50 + "\n")

# 演示掩码
masks = demonstrate_masking()

print("\n" + "="*50 + "\n")

# 可视化注意力
print("正在可视化注意力权重...")
# 解开注释 可以运行可视化
# self_attn_weights, cross_attn_weights = visualize_decoder_attention()


"""
简单训练翻译任务
"""

def train_simple_translation_example():
    """训练一个简单的序列到序列任务"""
    
    print("=== 训练简单翻译示例 ===\n")
    
    # 创建简单数据集：反转序列
    def create_reversal_dataset(num_samples=1000, max_len=10, vocab_size=50):
        """创建数据集：输入序列和它的反转"""
        X = []
        y = []
        
        for _ in range(num_samples):
            # 随机生成长度
            length = torch.randint(2, max_len, (1,)).item()
            
            # 生成序列（从2开始，避免特殊token）
            seq = torch.randint(2, vocab_size-2, (length,))
            
            # 输入：添加起始和结束token
            input_seq = torch.cat([
                torch.tensor([1]),  # 起始token
                seq,
                torch.tensor([2])   # 结束token
            ])
            
            # 输出：反转序列并添加起始和结束token
            output_seq = torch.cat([
                torch.tensor([1]),  # 起始token
                seq.flip(0),
                torch.tensor([2])   # 结束token
            ])
            
            # 填充到最大长度
            input_padded = F.pad(input_seq, (0, max_len+2 - len(input_seq)), value=0)
            output_padded = F.pad(output_seq, (0, max_len+2 - len(output_seq)), value=0)
            
            X.append(input_padded)
            y.append(output_padded)
        
        return torch.stack(X), torch.stack(y)
    
    # 创建数据集
    vocab_size = 50
    max_len = 10
    num_samples = 500
    
    X_train, y_train = create_reversal_dataset(num_samples, max_len, vocab_size)
    X_val, y_val = create_reversal_dataset(100, max_len, vocab_size)
    
    print(f"数据集信息:")
    print(f"  训练样本: {len(X_train)}")
    print(f"  验证样本: {len(X_val)}")
    print(f"  输入形状: {X_train.shape}")
    print(f"  输出形状: {y_train.shape}")
    print(f"  词汇表大小: {vocab_size}")
    
    # 创建模型
    model = Transformer(
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_ff=128,
        dropout=0.1,
        max_seq_len=max_len+2
    )
    
    # 训练配置
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略padding
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 训练循环
    batch_size = 32
    epochs = 20
    
    print(f"\n开始训练...")
    print(f"  Batch大小: {batch_size}")
    print(f"  训练轮数: {epochs}")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total_tokens = 0
        
        # 随机打乱
        indices = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_X = X_train[batch_indices]
            batch_y = y_train[batch_indices]
            
            # 前向传播
            # 对于训练，我们使用"teacher forcing"：将目标序列向右移动一位作为Decoder输入
            decoder_input = batch_y[:, :-1]  # 移除最后一个token
            decoder_target = batch_y[:, 1:]  # 移除第一个token
            
            output = model(batch_X, decoder_input)
            
            # 计算损失
            loss = criterion(
                output.reshape(-1, vocab_size),
                decoder_target.reshape(-1)
            )
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算准确率
            predictions = torch.argmax(output, dim=-1)
            correct += (predictions == decoder_target).sum().item()
            total_tokens += decoder_target.numel()
        
        # 验证
        model.eval()
        val_loss = 0
        val_correct = 0
        val_tokens = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_X = X_val[i:i+batch_size]
                batch_y = y_val[i:i+batch_size]
                
                decoder_input = batch_y[:, :-1]
                decoder_target = batch_y[:, 1:]
                
                output = model(batch_X, decoder_input)
                
                loss = criterion(
                    output.reshape(-1, vocab_size),
                    decoder_target.reshape(-1)
                )
                
                val_loss += loss.item()
                
                predictions = torch.argmax(output, dim=-1)
                val_correct += (predictions == decoder_target).sum().item()
                val_tokens += decoder_target.numel()
        
        train_accuracy = correct / total_tokens
        val_accuracy = val_correct / val_tokens
        
        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  训练损失: {total_loss/len(X_train):.4f}, 训练准确率: {train_accuracy:.4f}")
        print(f"  验证损失: {val_loss/len(X_val):.4f}, 验证准确率: {val_accuracy:.4f}")
        
        scheduler.step()
    
    print("\n训练完成!")
    
    # 测试模型
    print("\n=== 测试模型 ===")
    
    # 创建测试数据
    test_inputs = [
        [1, 3, 4, 5, 6, 2, 0, 0, 0, 0, 0, 0],  # 1 3 4 5 6 2
        [1, 7, 8, 9, 2, 0, 0, 0, 0, 0, 0, 0],   # 1 7 8 9 2
    ]
    
    test_inputs = torch.tensor(test_inputs)
    
    print("测试输入:")
    for i in range(len(test_inputs)):
        seq = test_inputs[i]
        # 移除padding和特殊token用于显示
        seq_display = [str(x.item()) for x in seq if x not in [0, 1, 2]]
        print(f"  输入 {i+1}: {' '.join(seq_display)}")
    
    # 生成输出
    model.eval()
    with torch.no_grad():
        for i in range(len(test_inputs)):
            src = test_inputs[i:i+1]
            
            # 使用生成函数
            generated = model.generate(src, max_len=10, start_token=1, end_token=2)
            
            # 提取生成的序列（移除起始token和结束token）
            gen_seq = generated[0].tolist()
            
            # 找到结束token的位置
            try:
                end_idx = gen_seq.index(2)
                gen_seq = gen_seq[1:end_idx]  # 移除起始token和结束token
            except ValueError:
                gen_seq = gen_seq[1:]  # 没有结束token，只移除起始token
            
            # 移除padding
            gen_seq = [str(x) for x in gen_seq if x not in [0, 1, 2]]
            
            print(f"  输出 {i+1}: {' '.join(gen_seq)} (期望反转)")
    
    return model


# 运行训练示例
print("\n" + "="*60)
print("训练简单翻译模型")
print("="*60)

trained_model = train_simple_translation_example()