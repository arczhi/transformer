"""
解码器
1.掩码自注意力
*残差连接+层归一化
2.编码器-解码器注意力（交叉注意力）
*残差连接+层归一化
3.前馈神经网络
"""

import torch
import torch.nn as nn
try :
    from .multi_head_attention import MultiHeadAttention
except ImportError:
    from multi_head_attention import MultiHeadAttention
try :
    from .feed_forward_network import MLP
except ImportError:
    from feed_forward_network import MLP
    
class TransformerDecoder(nn.Module):
    """
    完整的Transformer解码器
    由多个DecoderLayer堆叠而成
    """
    def __init__(self, num_layers: int = 6, d_model: int = 512, 
                 num_heads: int = 8, d_ff: int = 2048, 
                 dropout: float = 0.1, max_seq_len: int = 5000):
        super().__init__()
        
        self.num_layers = num_layers
        self.d_model = d_model
        
        # 创建多个解码器层
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终的层归一化
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, encoder_output, self_attention_mask=None, 
                cross_attention_mask=None):
        """
        前向传播
        
        参数:
            x: Decoder输入 [batch_size, tgt_seq_len, d_model]
            encoder_output: Encoder输出 [batch_size, src_seq_len, d_model]
            self_attention_mask: 自注意力掩码
            cross_attention_mask: 交叉注意力掩码
        
        返回:
            output: Decoder输出 [batch_size, tgt_seq_len, d_model]
            all_self_attention_weights: 所有层的自注意力权重列表
            all_cross_attention_weights: 所有层的交叉注意力权重列表
        """
        all_self_attention_weights = []
        all_cross_attention_weights = []
        
        # 逐层处理
        for layer in self.layers:
            x, self_attn_weights, cross_attn_weights = layer(
                x, encoder_output, self_attention_mask, cross_attention_mask
            )
            all_self_attention_weights.append(self_attn_weights)
            all_cross_attention_weights.append(cross_attn_weights)
        
        # 最终层归一化
        output = self.norm(x)
        
        return output, all_self_attention_weights, all_cross_attention_weights

class DecoderLayer(nn.Module):
    """
    单个Transformer解码器层
    """
    def __init__(self, d_model: int = 512, num_heads: int = 8, 
                 d_ff: int = 2048, dropout: float = 0.1):
        super().__init__()
        
        # 掩码自注意力
        self.self_attention = MultiHeadAttention(d_model,num_heads,dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 交叉注意力
        self.cross_attention = MultiHeadAttention(d_model,num_heads,dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        
        # 前馈神经网络
        # 这里使用一个简单的实现
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model,d_ff), # 线性层 转高维度
            nn.ReLU(), # 激活
            nn.Dropout(dropout), # dropout
            nn.Linear(d_ff,d_model) # 线性层 转化为输入的维度
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        
    def forward(self,x,encoder_output,self_attention_mask=None,cross_attention_mask=None):
        
        # 掩码自注意力 (Q=K=V=x + 掩码)
        residual = x
        self_att_output,self_att_weights = self.self_attention(x, mask=self_attention_mask)
        x = self.norm1(residual+self.dropout1(self_att_output))
        
        # 交叉注意力 (Q来自解码器，K/V来自编码器)
        residual = x
        # Q=x(来自解码器), K=V=encoder_output(来自编码器)
        cross_att_output,cross_att_weights = self.cross_attention(x, encoder_output, encoder_output, mask=cross_attention_mask)
        x = self.norm2(residual+self.dropout2(cross_att_output))
        
        # 前馈神经网络
        residual = x
        fnn_output = self.feed_forward(x)
        x = self.norm3(residual + self.dropout3(fnn_output))
        
        return x,self_att_weights,cross_att_weights


def create_simple_decoder():
    """创建一个简单的Decoder示例"""
    
    # 配置参数（遵循论文）
    config = {
        'num_layers': 6,
        'd_model': 512,
        'num_heads': 8,
        'd_ff': 2048,
        'dropout': 0.1,
        'max_seq_len': 100
    }
    
    print("=== 创建Transformer Decoder ===")
    print("配置参数:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # 创建Decoder
    decoder = TransformerDecoder(**config)
    
    return decoder, config


def test_decoder_with_simple_input():
    """用简单的输入测试Decoder"""
    
    # 创建Decoder
    decoder, config = create_simple_decoder()
    
    # 创建模拟输入
    batch_size = 2
    src_seq_len = 15  # 编码器序列长度
    tgt_seq_len = 10  # 解码器序列长度
    
    # 模拟编码器输出（来自Encoder）
    # 实际中应该是真实的编码器输出
    encoder_output = torch.randn(batch_size, src_seq_len, config['d_model'])
    
    # 模拟解码器输入（已经包含词嵌入和位置编码）
    decoder_input = torch.randn(batch_size, tgt_seq_len, config['d_model'])
    
    print(f"\n=== 测试Decoder ===")
    print(f"编码器输出形状: {encoder_output.shape}")
    print(f"解码器输入形状: {decoder_input.shape}")
    print(f"解码器输入值范围: [{decoder_input.min():.3f}, {decoder_input.max():.3f}]")
    
    # 创建自注意力掩码（因果掩码）- 解码器必须有！
    # 下三角矩阵：每个位置只能看到当前及之前的位置
    # 更简洁的写法（效果完全相同）
    self_attention_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len))
    # 形状：[tgt_seq_len, tgt_seq_len]
    # MultiHeadAttention 内部会自动处理广播
    
    # 创建交叉注意力掩码（可选）
    # 假设编码器序列实际长度为13（后2个位置是padding）
    cross_attention_mask = torch.ones(batch_size, tgt_seq_len, src_seq_len)
    cross_attention_mask[:, :, 13:] = 0  # 屏蔽编码器序列的padding位置
    
    print(f"\n自注意力掩码形状: {self_attention_mask.shape}")
    print(f"自注意力掩码（因果性）:\n{self_attention_mask}")
    print(f"交叉注意力掩码形状: {cross_attention_mask.shape}")
    
    # 前向传播
    output, all_self_attn_weights, all_cross_attn_weights = decoder(
        decoder_input, encoder_output, 
        self_attention_mask=self_attention_mask,
        cross_attention_mask=cross_attention_mask
    )
    
    print(f"\n输出形状: {output.shape}")
    print(f"输出值范围: [{output.min():.3f}, {output.max():.3f}]")
    print(f"自注意力权重层数: {len(all_self_attn_weights)}")
    print(f"交叉注意力权重层数: {len(all_cross_attn_weights)}")
    print(f"每个自注意力权重形状: {all_self_attn_weights[0].shape}")
    print(f"每个交叉注意力权重形状: {all_cross_attn_weights[0].shape}")
    
    # 检查输入输出维度是否一致
    print(f"\n输入输出维度一致: {decoder_input.shape == output.shape}")
    
    # 验证因果性：位置 i 对位置 j>i 的自注意力应该为 0
    print(f"\n=== 因果性验证 ===")
    first_layer_self_attn = all_self_attn_weights[0]  # 第一层的自注意力
    sample_weights = first_layer_self_attn[0, 0, :, :]  # [batch=0, head=0, seq_len, seq_len]
    print(f"位置0对位置1的自注意力: {sample_weights[0, 1].item():.6f} (应该接近0)")
    print(f"位置1对位置0的自注意力: {sample_weights[1, 0].item():.6f} (可以非0)")
    print(f"位置2对位置3的自注意力: {sample_weights[2, 3].item():.6f} (应该接近0)")
    print(f"位置3对位置2的自注意力: {sample_weights[3, 2].item():.6f} (可以非0)")
    
    return decoder, encoder_output, decoder_input, output, all_self_attn_weights, all_cross_attn_weights


def visualize_decoder_workflow():
    """可视化Decoder的工作流程"""
    
    print("\n" + "=" * 60)
    print("=== Transformer Decoder 工作流程 ===")
    print("=" * 60)
    
    print("\n1. 输入:")
    print("   - 解码器输入: [batch_size, tgt_seq_len, d_model]")
    print("     (已包含词嵌入和位置编码)")
    print("   - 编码器输出: [batch_size, src_seq_len, d_model]")
    print("     (来自Encoder)")
    
    print("\n2. 通过多个解码器层:")
    print("   每个解码器层包含:")
    
    print("\n   a) 掩码自注意力 (Self-Attention with Mask)")
    print("       - Q, K, V 都来自解码器输入")
    print("       - 应用因果掩码（下三角矩阵）")
    print("       - 目的：防止每个位置看到未来的token")
    print("       - 残差连接 + 层归一化")
    
    print("\n   b) 交叉注意力 (Cross-Attention)")
    print("       - Q 来自解码器（上一层输出）")
    print("       - K, V 来自编码器输出")
    print("       - 目的：解码器关注编码器的信息")
    print("       - 残差连接 + 层归一化")
    
    print("\n   c) 前馈神经网络 (Feed-Forward)")
    print("       - 两个线性层 + ReLU激活")
    print("       - 残差连接 + 层归一化")
    
    print("\n3. 输出: [batch_size, tgt_seq_len, d_model]")
    print("   - 与解码器输入形状相同")
    print("   - 包含了编码器信息和解码器上下文")
    
    # 创建示意图
    decoder_structure = """
    Transformer Decoder 结构:
    
    解码器输入 [batch, tgt_seq_len, d_model]    编码器输出 [batch, src_seq_len, d_model]
         │                                            │
         ▼                                            │
    ┌──────────────────────────────────────┐        │
    │   解码器层 1 (Decoder Layer 1)        │        │
    │  ┌──────────────────────┐             │        │
    │  │ 掩码自注意力          │             │        │
    │  │  (因果掩码)           │             │        │
    │  │   ┌─残差连接─┐        │             │        │
    │  │   │         │        │             │        │
    │  └──►│ 层归一化 ├────────┤             │        │
    │       └─────────┘        │             │        │
    │       ┌──────────────────────────┐    │        │
    │       │ 交叉注意力              │    │        │
    │       │ (Q:解码器, K/V:编码器)  │    │        │
    │       │     ▲                    │    │        │
    │       │     │                    │    │        │
    │       │     └────────────────────┼────┴────────┤
    │       │        ┌─残差连接─┐     │             │
    │       │        │         │     │             │
    │       └───────►│ 层归一化 ├─────┤             │
    │               └─────────┘     │             │
    │               ┌─────────────────┐            │
    │               │ 前馈神经网络      │            │
    │               │   ┌─残差连接─┐   │            │
    │               │   │         │   │            │
    │               └──►│ 层归一化 ├───┤            │
    │                   └─────────┘   │            │
    └──────────────────────────────────────┘        │
         │                                            │
         ▼                                            ▼
    ┌──────────────────────────────────────┐
    │         解码器层 2, 3, ..., N        │
    │            (类似结构)                 │
    └──────────────────────────────────────┘
         │
         ▼
    输出 [batch, tgt_seq_len, d_model]
    """
    
    print(decoder_structure)
    
    print("\n关键差异总结:")
    print("┌────────────────┬──────────────────┬──────────────────┐")
    print("│    模块        │     编码器        │     解码器        │")
    print("├────────────────┼──────────────────┼──────────────────┤")
    print("│ 自注意力       │ 无掩码            │ 因果掩码          │")
    print("│ 交叉注意力     │ 无（只有自注意力）│ 有（编码器-解码器）│")
    print("│ 前馈网络       │ 有                │ 有                │")
    print("│ 层数           │ 6                 │ 6                 │")
    print("└────────────────┴──────────────────┴──────────────────┘")


# 运行测试
# if __name__ == "__main__":
decoder, encoder_output, decoder_input, output, all_self_attn_weights, all_cross_attn_weights = test_decoder_with_simple_input()
visualize_decoder_workflow()
        