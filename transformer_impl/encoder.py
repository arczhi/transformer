"""
Encoder编码器
1.多头注意力
2.前馈神经网络/MLP多层感知机
3.残差连接 Add
4.层归一化 LayerNorm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
try :
    from .multi_head_attention import MultiHeadAttention
except ImportError:
    from multi_head_attention import MultiHeadAttention
try :
    from .feed_forward_network import MLP
except ImportError:
    from feed_forward_network import MLP

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers:int = 6 ,d_model: int = 512, num_heads: int = 8, 
                d_ff: int = 2048, dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        
        self.layers = nn.ModuleList([
            EncoderLayer(d_model,num_heads,d_ff,dropout)
            for _ in range(self.num_layers)
        ])
        
        # 最终的层归一化（有些实现会加）
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self,x,mask=None):
        """
        前向传播
        
        参数:
            x: 输入张量 [batch_size, seq_len, d_model]
            mask: 注意力掩码 [batch_size, seq_len, seq_len]
        
        返回:
            output: 编码后的张量 [batch_size, seq_len, d_model]
            all_attention_weights: 所有层的注意力权重列表
        """
        all_attention_weights = []
        for layer in self.layers:
            x,attention_weights = layer(x,mask)
            all_attention_weights.append(attention_weights)
        
        # 最终做一次层归一化 使得结果更加平衡与稳定
        output = self.norm(x)
        
        return output,attention_weights
    
    def get_attention_maps(self, x, mask=None):
        """获取所有层的注意力权重用于可视化"""
        _, all_attention_weights = self.forward(x, mask)
        return all_attention_weights

class EncoderLayer(nn.Module):
    def __init__(self, d_model: int = 512, num_heads: int = 8, 
                d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        #1.第一个子层 多头注意力+残差连接&层归一化
        self.self_attention = MultiHeadAttention(d_model,num_heads,dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        #2.第二个子层 前馈神经网络+残差连接&层归一化
        self.feed_forward = MLP(d_model,d_ff,d_model,dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self,x,mask=None):
        # 1.多头注意力 (自注意力：Q=K=V=x + 可选掩码)
        residual = x
        attention_output,attention_weights = self.self_attention(x, mask=mask)
        x = self.norm1(residual + self.dropout1(attention_output))
        
        # 2.前馈神经网络
        residual = x
        ffn_output = self.feed_forward(x)
        x = self.norm2(residual+self.dropout2(ffn_output))
        
        return x,attention_weights
    


def create_simple_encoder():
    """创建一个简单的Encoder示例"""
    
    # 配置参数（遵循论文）
    config = {
        'num_layers': 6,
        'd_model': 512,
        'num_heads': 8,
        'd_ff': 2048,
        'dropout': 0.1,
        'max_seq_len': 100
    }
    
    print("=== 创建Transformer Encoder ===")
    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 创建Encoder
    encoder = TransformerEncoder(**config)
    
    return encoder, config


def test_encoder_with_simple_input():
    """用简单的输入测试Encoder"""
    
    # 创建Encoder
    encoder, config = create_simple_encoder()
    
    # 创建模拟输入
    batch_size = 2
    seq_len = 10
    
    # 模拟经过嵌入层后的输入
    # 实际中应该先经过 TokenEmbedding + PositionalEncoding
    x = torch.randn(batch_size, seq_len, config['d_model'])
    
    print(f"\n=== 测试Encoder ===")
    print(f"输入形状: {x.shape}")
    print(f"输入值范围: [{x.min():.3f}, {x.max():.3f}]")
    
    # 创建注意力掩码（可选）
    # 这里创建一个简单的padding mask
    # 假设序列实际长度为8（后2个位置是padding）
    mask = torch.ones(batch_size, seq_len, seq_len)
    mask[:, :, 8:] = 0  # 屏蔽padding位置
    
    # 前向传播
    output, all_attention_weights = encoder(x, mask)
    
    print(f"\n输出形状: {output.shape}")
    print(f"注意力权重数量: {len(all_attention_weights)}")
    print(f"每个注意力权重形状: {all_attention_weights[0].shape}")
    
    # 检查输入输出维度是否一致
    print(f"输入输出维度一致: {x.shape == output.shape}")
    
    return encoder, x, output, all_attention_weights


def visualize_encoder_workflow():
    """可视化Encoder的工作流程"""
    
    print("=== Transformer Encoder 工作流程 ===")
    print("\n1. 输入: [batch_size, seq_len, d_model]")
    print("   - 已经包含词嵌入和位置编码")
    
    print("\n2. 通过多个编码器层:")
    print("   每个编码器层包含:")
    print("   a) 多头自注意力")
    print("       - Q, K, V线性变换")
    print("       - 分割为多个注意力头")
    print("       - 计算注意力分数")
    print("       - Softmax获取注意力权重")
    print("       - 应用到V值")
    print("       - 合并多头")
    print("       - 残差连接 + 层归一化")
    
    print("\n   b) 前馈神经网络")
    print("       - 两个线性层 + ReLU激活")
    print("       - 残差连接 + 层归一化")
    
    print("\n3. 输出: [batch_size, seq_len, d_model]")
    print("   - 与输入形状相同")
    print("   - 包含了序列的上下文信息")
    
    # 创建示意图
    encoder_structure = """
    Transformer Encoder 结构:
    
    输入 [batch, seq_len, d_model]
        │
        ▼
    ┌─────────────────────────────┐
    │   编码器层 1 (Encoder Layer 1) │
    │  ┌─────────────────────┐    │
    │  │ 多头自注意力          │    │
    │  │   ┌─残差连接─┐       │    │
    │  │   │         │       │    │
    │  └──►│ 层归一化 ├───────┘    │
    │       └─────────┘            │
    │       ┌─────────────────┐    │
    │       │ 前馈神经网络      │    │
    │       │   ┌─残差连接─┐   │    │
    │       │   │         │   │    │
    │       └─►│ 层归一化 ├───┘    │
    │           └─────────┘        │
    └─────────────────────────────┘
        │
        ▼
    ┌─────────────────────────────┐
    │   编码器层 2 (Encoder Layer 2) │
    │            ...               │
    └─────────────────────────────┘
        │
        ▼
            ...
        │
        ▼
    ┌─────────────────────────────┐
    │   编码器层 N (Encoder Layer N) │
    │            ...               │
    └─────────────────────────────┘
        │
        ▼
    输出 [batch, seq_len, d_model]
    """
    
    print(encoder_structure)


# 运行测试
encoder, x, output, all_attention_weights = test_encoder_with_simple_input()
# 解开代码注释 运行可视化
# visualize_encoder_workflow()


"""
仅编码器的transformer
"""

class SimpleTransformer(nn.Module):
    """
    完整的简单Transformer（只有Encoder部分）
    用于分类任务或特征提取
    """
    def __init__(self, vocab_size: int = 10000, num_layers: int = 6,
                 d_model: int = 512, num_heads: int = 8, 
                 d_ff: int = 2048, dropout: float = 0.1,
                 max_seq_len: int = 100, num_classes: int = 2):
        super().__init__()
        
        # 1. 嵌入层（词嵌入 + 位置编码）
        self.embedding = nn.Sequential(
            nn.Embedding(vocab_size, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # 位置编码（可学习的，简单版本）
        self.positional_encoding = nn.Parameter(
            torch.zeros(1, max_seq_len, d_model)
        )
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)
        
        # 2. Transformer编码器
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len
        )
        
        # 3. 分类头
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, input_ids, attention_mask=None):
        """
        前向传播
        
        参数:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] (可选)
        
        返回:
            logits: [batch_size, num_classes]
            attention_weights: 所有层的注意力权重
        """
        batch_size, seq_len = input_ids.shape
        
        # 1. 词嵌入
        embeddings = self.embedding(input_ids)  # [batch, seq_len, d_model]
        
        # 2. 添加位置编码
        embeddings = embeddings + self.positional_encoding[:, :seq_len, :]
        
        # 3. 创建注意力掩码（如果需要）
        if attention_mask is not None:
            # 扩展为[batch, seq_len, seq_len]的掩码矩阵
            mask = attention_mask.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1, seq_len]
            mask = mask.expand(-1, -1, seq_len, -1)  # [batch, 1, seq_len, seq_len]
            mask = mask.squeeze(1)  # [batch, seq_len, seq_len]
        else:
            mask = None
        
        # 4. 通过Transformer编码器
        encoded, attention_weights = self.encoder(embeddings, mask)
        
        # 5. 使用[CLS] token进行分类
        # 通常取第一个位置的输出（假设是[CLS] token）
        cls_output = encoded[:, 0, :]  # [batch_size, d_model]
        
        # 6. 分类
        logits = self.classifier(cls_output)  # [batch_size, num_classes]
        
        return logits, attention_weights
    
    def predict(self, input_ids, attention_mask=None):
        """预测方法"""
        with torch.no_grad():
            logits, _ = self.forward(input_ids, attention_mask)
            probabilities = F.softmax(logits, dim=-1)
            predictions = torch.argmax(probabilities, dim=-1)
        return predictions, probabilities


def demo_simple_transformer():
    """演示简单的Transformer"""
    
    print("=== 演示简单的Transformer（Encoder） ===\n")
    
    # 配置
    config = {
        'vocab_size': 5000,
        'num_layers': 3,  # 为了演示，使用较少的层
        'd_model': 128,
        'num_heads': 4,
        'd_ff': 512,
        'dropout': 0.1,
        'max_seq_len': 50,
        'num_classes': 3  # 3分类问题
    }
    
    # 创建模型
    model = SimpleTransformer(**config)
    print("模型参数:")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 创建模拟数据
    batch_size = 4
    seq_len = 20
    
    # 输入IDs（假设0是padding，1是[CLS]，2是[SEP]等）
    input_ids = torch.randint(3, config['vocab_size'], (batch_size, seq_len))
    
    # 注意力掩码（假设前15个是真实token，后5个是padding）
    attention_mask = torch.ones(batch_size, seq_len)
    attention_mask[:, 15:] = 0
    
    print(f"\n输入形状: {input_ids.shape}")
    print(f"注意力掩码: {attention_mask[0]}")
    
    # 前向传播
    logits, attention_weights = model(input_ids, attention_mask)
    
    print(f"\n输出:")
    print(f"  Logits形状: {logits.shape}")
    print(f"  预测类别: {torch.argmax(logits, dim=-1)}")
    print(f"  注意力权重数量: {len(attention_weights)}")
    
    # 可视化第一层第一个头的注意力权重
    if len(attention_weights) > 0:
        first_layer_attention = attention_weights[0]
        print(f"  注意力权重形状: {first_layer_attention.shape}")
        
        # 可视化
        model.encoder.layers[0].self_attention.attention_visualization(
            first_layer_attention, seq_len=10
        )
    
    return model, input_ids, logits


def train_simple_example():
    """训练一个简单的示例"""
    
    print("\n=== 训练简单示例 ===")
    
    # 创建简单的数据集
    vocab_size = 100
    seq_len = 10
    num_samples = 100
    
    # 生成随机数据
    X = torch.randint(3, vocab_size, (num_samples, seq_len))
    
    # 简单规则：如果序列中数字5出现次数大于3次，则为类别1，否则为类别0
    y = torch.zeros(num_samples, dtype=torch.long)
    for i in range(num_samples):
        if (X[i] == 5).sum() > 3:
            y[i] = 1
    
    print(f"数据集:")
    print(f"  样本数量: {num_samples}")
    print(f"  类别分布: {torch.bincount(y).tolist()}")
    
    # 创建模型
    model = SimpleTransformer(
        vocab_size=vocab_size,
        num_layers=2,
        d_model=64,
        num_heads=2,
        d_ff=128,
        dropout=0.1,
        max_seq_len=seq_len,
        num_classes=2
    )
    
    # 训练配置
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # 简单训练循环
    epochs = 10
    batch_size = 16
    
    print(f"\n开始训练...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        
        # 简单分批
        for i in range(0, num_samples, batch_size):
            batch_X = X[i:i+batch_size]
            batch_y = y[i:i+batch_size]
            
            # 前向传播
            logits, _ = model(batch_X)
            loss = criterion(logits, batch_y)
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == batch_y).sum().item()
        
        accuracy = correct / num_samples
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}, Accuracy: {accuracy:.4f}")
    
    print("训练完成!")
    
    # 测试
    model.eval()
    test_X = torch.randint(3, vocab_size, (20, seq_len))
    test_y = torch.zeros(20, dtype=torch.long)
    for i in range(20):
        if (test_X[i] == 5).sum() > 3:
            test_y[i] = 1
    
    with torch.no_grad():
        predictions, probabilities = model.predict(test_X)
        accuracy = (predictions == test_y).sum().item() / 20
        print(f"\n测试集准确率: {accuracy:.4f}")
    
    return model


# 运行演示
print("=== 开始演示 ===")
model1, input_ids1, logits1 = demo_simple_transformer()
print("\n" + "="*50 + "\n")
model2 = train_simple_example()


        
        
        
        