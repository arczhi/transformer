import torch
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn
import torch.optim as optim

x = torch.randn(200,2) #200行2别的张量
y = (x[:,0]**2 + x[:,1]**2 > 1).long() # x**2 + y**2 > 1

dataset = TensorDataset(x,y)
data_loader = DataLoader(dataset,batch_size=10,shuffle=True)

class SimpleMLP(nn.Module):
    # 输入2维 隐藏层2维 输出2维
    def __init__(self,input_dim=2,hidden_dim=10,output_hidden=2):
        super().__init__() #调用父类的构造函数
        self.full_connect1 = nn.Linear(input_dim,hidden_dim) # 2->10 从原始 2 个特征，变成 10 个“更抽象的特征”。
        self.relu = nn.ReLU() # 激活函数 保留正数，让模型可以画出“弯弯曲曲”的决策边界，而不是直线。
        self.full_connect2 = nn.Linear(hidden_dim,hidden_dim) # 10->10 在10维空间里面继续加工
        self.full_connect3 = nn.Linear(hidden_dim,hidden_dim) # 10 -> 10 继续加工
        self.full_connect4 = nn.Linear(hidden_dim,output_hidden) # 10->2 10维变回2维，得到最终的2个类别的分数。

    def forward(self,x):
        #上一层的输出作为下一层的输入 
        x = self.full_connect1(x) #线性1
        x = self.relu(x) # 非线性1
        x = self.full_connect2(x) #线性2
        x = self.relu(x) # 非线性2
        x = self.full_connect3(x) #线性3
        x = self.relu(x) # 非线性3
        x = self.full_connect4(x) #线性4
        return x
    # 可以类比为：
    # 原材料进厂（输入 x）
    # 工序1：粗加工 → fc1
    # 加工后做一下非线性处理 → ReLU
    # 工序2：精加工 → fc2
    # 再做一次非线性 → ReLU
    # 最后一工序：输出最终成品种类 → fc3

model = SimpleMLP()
print(model)


criterion = nn.CrossEntropyLoss() #损失函数 看预测类别和真实类别差距有多大。差距越大，损失越大。
optimizer = optim.SGD(model.parameters(),lr=0.01) #优化器 用梯度下降来一步步调整模型的参数（权重和偏置）

for epoch in range(128): #训练128轮
    for batch_x,batch_y in data_loader:
        # 向前传播 算预测
        predict = model(batch_x)
        # 计算损失 看预测的predict和真实标签batch_y差距
        loss = criterion(predict,batch_y)
        # 反向传播：算梯度
        optimizer.zero_grad()  # 清空旧梯度
        loss.backward()        # 根据链式法则自动求导
        # 更新参数：沿着梯度方向走一步
        optimizer.step()


    if (epoch + 1) % 20 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# 5. 简单测试一下
with torch.no_grad():
    # test_X = torch.randn(5, 2)
    test_X = torch.tensor([[-1.1166, -2.4299],
        [ 0.6872,  0.7643],
        [ 0.6659, -0.0338],
        [ 0.0338, -1.0083],
        [-0.5733,  0.8034]])
    print("测试输入：", test_X)
    out = model(test_X)
    # 举例
    # 第1个样本 out[0] = [1.2, -0.3]
    # 第2个样本 out[1] = [-0.5, 2.0]
    pred = out.argmax(dim=1) # 在“第 1 维”上取最大值的下标，相当于对“每一行”找最大值的位置；
    print("测试点：", test_X)
    print("out: ",out)
    print("预测类别：", pred)

# nn.Linear 搭框架 + nn.ReLU 加非线性 + 训练循环里 loss.backward() 和 optimizer.step()。



