import torch
import torch.nn as nn
import torch.optim as optim

# nn.Linear：全连接层（线性变换）
# nn.ReLU：激活函数
# nn.Sequential：按顺序把各层“串”起来

# nn.Linear 就是全连接层 / 线性层，做的事情只有一句话：
# 对输入做一次：输出 = 输入 × 权重矩阵ᵀ + 偏置
# 也就是线性代数里常见的“矩阵乘法 + 加一个向量”。
# 你可以把它想象成一个“特征转换器”：
# 输入：一堆数字（特征）
# 里面：乘以一个可学习的矩阵，再加一个可学习的偏置
# 输出：另一堆数字（新的特征）
# 在神经网络里，MLP、分类头、回归输出等，基本都离不开它。


m = nn.Linear(20,30)
t = torch.randn(256,20) #关注最后一个维度的变化 20维->30维 
# 最后一维从 in_features 变成 out_features，前面所有维度原样保留。
output = m(t) #内部自动调用 forward()
# def forward(self, input):
#     return input @ weight.T + bias      # @ 是矩阵乘法 bias是偏置，即数学上的加上一个向量
print(output.shape)  # torch.Size([256, 30])


# nn.ReLU是一个激活函数模块，作用是对张量里的每一个元素做同样的一种简单变换：
# ReLU(x) = max(0, x)
# 用人话说，就是：
# 如果这个数是正数或 0 → 保持原样
# 如果是负数 → 直接变成 0
# 你可以把 ReLU 想象成一个“负数剪刀”：
# 凡是小于 0 的，一律剪掉变成 0；大于等于 0 的，原样留下。
# 类似于掩码

# m = nn.ReLU(inplace=False)
# inplace=False（默认）
# 不在原张量上改，而是生成一个新的张量作为输出
# 原始输入 input 里的值不变
# inplace=True
# 直接在原张量上修改：负数会被就地变成 0
# 这样可以省一点内存，有时也会稍微快一点
# 但缺点是：输入会被改掉，后面如果你还想用原始值，就拿不到了

m = nn.ReLU()
t = torch.randn(2)
print(t)
output = m(t)
print(output)
t2 = torch.tensor([-1.0, 0.0, 1.0, 2.0])
output2 = m(t2)
print(f"after: {output2} origin: {t2}")



# Sequential 到底是什么？
# 一句话：nn.Sequential 就是一个“按顺序串起来的网络层容器”。

# 可以把它想象成：

# 你有一条流水线，有很多工序（卷积、激活、全连接……）
# 你用 nn.Sequential 把这些工序按顺序排好队
# 数据一头进去，依次经过每一个工序，最后从另一头输出结果
# 所以：

# 你不需要自己写 forward() 里「x先过这一层再过那一层」的代码
# nn.Sequential 会自动帮你把前一层的输出当作下一层的输入


# 简单前馈网络（两层隐藏层）
model = nn.Sequential(
    nn.Linear(784, 256),  # 输入：784 维 → 隐藏层 1：256 维
    nn.ReLU(),            # 激活函数
    nn.Linear(256, 128),  # 隐藏层 1：256 维 → 隐藏层 2：128 维
    nn.ReLU(),            # 激活函数
    nn.Linear(128, 10)    # 隐藏层 2：128 维 → 输出层：10 类
)

print(model)

# 2. 模拟一批数据：batch_size = 32
x = torch.randn(32, 1, 28, 28)      # 假设是 32 张 28x28 灰度图
x = x.view(32, -1)                  # 展平为 (32, 784)
y = torch.randint(0, 10, (32,))     # 随机标签 0~9

# 3. 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()           # 多分类常用
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 4. 一次训练迭代示例
# 4.1 前向传播
logits = model(x)                  # 形状: (32, 10)

# 4.2 计算损失
loss = criterion(logits, y)

# 4.3 反向传播 + 更新参数
optimizer.zero_grad()              # 清空旧梯度
loss.backward()                    # 反向传播，计算梯度
optimizer.step()                   # 用梯度更新参数

print("loss =", loss.item())