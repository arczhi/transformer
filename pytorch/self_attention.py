import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def scaled_dot_product_attention(query,key,value,mask=None):
    """
    缩放点积注意力（单头版）

    参数:
        query: [batch_size, seq_len, d_k]
        key:   [batch_size, seq_len, d_k]
        value: [batch_size, seq_len, d_v]
        mask:  [seq_len, seq_len] 或 [batch_size, seq_len, seq_len]，元素为0或1（可选）
              1 表示允许关注，0 表示屏蔽（不允许关注）
    """

    # 1) 相关性评分 - 查询矩阵和键矩阵进行矩阵乘法
    #维度
    d_k = query.size(-1) 
    #key矩阵的最后两个维度（行列）进行倒置，以便进行矩阵乘法计算
    scores = torch.matmul(query,key.transpose(-2,-1))
    # 2）缩放 - 使用d_k的平方根，防止值过大，让softmax更稳定
    scores = scores / math.sqrt(d_k)
    # 3) 掩码 - 掩掉不需要关注的位置，例如未来的token或padding，避免对处理中及之前的产生影响
    if mask is not None:
        # 掩码为0的地方 替换成负无穷 -inf
        scores = scores.masked_fill(mask==0,float('-inf'))
    # 4) softmax进行归一化处理 确保各项之和为1 处理最后一维（对最后一维进行归一化处理）
    # 注意力权重整体就是“一批样本 × 查询位置 × 被关注位置”，形成 [batch_size, seq_len, seq_len] = [2, 4, 4]。
    attention_weights = F.softmax(scores,dim=-1)
    # 5) 乘以值矩阵 - 注意力权重加权求和 Value
    # attention_weights [batch_size, seq_len, seq_len]
    # value [batch_size, seq_len, d_v]
    # `softmax` 之后得到的 `attention_weights[b, i, :]`：就是第 i 个位置对整句所有位置的注意力分布，所有元素加起来 = 1。
    # - 再乘 `V`，就是拿这些权重，对所有 token 的 `value` 做加权求和。
    output = torch.matmul(attention_weights,value)
    return output,attention_weights

class SingleHeadSelfAttention(nn.Module):
    def __init__(self,d_model,d_k,d_v,dropout=0.1):
        """
            参数:
            d_model: 输入/输出特征维度（例如 512）
            d_k:     Q、K 的特征维度
            d_v:     V 的特征维度
        """
        super(SingleHeadSelfAttention,self).__init__()

        # 1.把输出投影成查询、键、值矩阵
        self.W_q = nn.Linear(d_model,d_k)
        self.W_k = nn.Linear(d_model,d_k)
        self.W_v = nn.Linear(d_model,d_v)

        # 2.输出层 把维度映射回输入的维度
        self.W_o = nn.Linear(d_v,d_model)

        # 3.dropout and layer_normal
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(self,x,mask=None):
        """
        前向传播

        参数:
            x: [batch_size, seq_len, d_model]
            mask: [seq_len, seq_len] 或 [batch_size, seq_len, seq_len]，0/1 掩码（可选）

        返回:
            out: [batch_size, seq_len, d_model]
            attention_weights: [batch_size, seq_len, seq_len]
        """

        # 保存输入 用于残差
        residual = x

        # 1.自注意力计算
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)
        atten_output,atten_weights = scaled_dot_product_attention(Q,K,V,mask)
        out = self.W_o(atten_output)

        # 2.dropout 防止过拟合 提高模型的泛化能力，处理未知数据的能力 （防止明星员工大包大揽）
        out = self.dropout(out)

        # 3.残差 本层的输入加上本层的输出，一起输出到下一层，防止信息在传播过程中失真，
        # 防止（反向传播backward时）链式法则引起的梯度消失或爆炸
        # 4.归一化 平衡每个神经元的输出结果，标准化，让训练更稳定
        out = self.layer_norm(residual + out)

        return out,atten_weights
    

def demo(batch=2,seq_len=4,d_model=8,d_k=8,d_v=8):
    input = torch.randn(batch,seq_len,d_model)
    attention_layer = SingleHeadSelfAttention(d_model,d_k,d_v)
    atten_output,atten_weights = attention_layer(input)

    print("输入形状 input: ",input.shape)
    print("注意力输出形状 atten_output: ",atten_output.shape)
    print("注意力权重形状 input: ",atten_weights.shape) 
    # 第 `i` 行表示**第 i 个 token 对句子中所有 token 的注意力分布**。

    mask = torch.tril(torch.ones(seq_len,seq_len)) #保留左下角的三角形 右上角三角形置为0
    print("因果掩码",mask) #防止看到未来
    input - torch.randn(batch,seq_len,d_model)
    output,atten_weights = attention_layer(input,mask)
    print("掩码 注意力形状",atten_weights.shape)
    print("第一个样本第一行",atten_weights[0,0])
    print("第一个样本第二行",atten_weights[0,1])
    print("第一个样本第三行",atten_weights[0,2])
    print("第一个样本第四行",atten_weights[0,3])

    x = torch.randn(1,5,10)
    mask = torch.ones(5,5)
    mask[3:,:3] = 0 #后两个token不能看前3个
    #部分位置不能关注（一个批次多个句子的长度不一样，存在空白,即padding，使用掩码避免模型关注这些位置）
    print("padding mask ",mask) 
    atten_layer = SingleHeadSelfAttention(10,5,5)
    output,atten_weights = atten_layer(x,mask)
    print("注意力权重 ",atten_weights) #每行和为1





demo()











