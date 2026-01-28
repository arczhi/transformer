import torch
from torch.utils.data import TensorDataset,DataLoader
import torch.nn as nn
import torch.optim as optim

class MLP(nn.Module):
    # 输入512维 隐藏层2048维 输出512维
    def __init__(self,input_dim=512,hidden_dim=2048,output_hidden=512,dropout=0.1):
        super().__init__() #调用父类的构造函数
        self.full_connect1 = nn.Linear(input_dim,hidden_dim) # 512->2048 从原始 512 个特征，变成 2048 个“更抽象的特征”。
        self.relu = nn.ReLU() # 激活函数 保留正数，让模型可以画出“弯弯曲曲”的决策边界，而不是直线。
        self.dropout = nn.Dropout(dropout) # dropout 防止过拟合
        self.full_connect2 = nn.Linear(hidden_dim,hidden_dim) # 2048->2048 在10维空间里面继续加工
        self.full_connect3 = nn.Linear(hidden_dim,hidden_dim) # 2048->2048 继续加工
        self.full_connect4 = nn.Linear(hidden_dim,output_hidden) # 2048->512 2048维变回512维

    def forward(self,x):
        #上一层的输出作为下一层的输入 
        x = self.full_connect1(x) #线性1
        x = self.relu(x) # 非线性1
        x = self.dropout(x) # dropout
        x = self.full_connect2(x) #线性2
        x = self.relu(x) # 非线性2
        x = self.dropout(x) # dropout
        x = self.full_connect3(x) #线性3
        x = self.relu(x) # 非线性3
        x = self.dropout(x) # dropout
        x = self.full_connect4(x) #线性4
        return x
    # 可以类比为：
    # 原材料进厂（输入 x）
    # 工序1：粗加工 → fc1
    # 加工后做一下非线性处理 → ReLU
    # 工序2：精加工 → fc2
    # 再做一次非线性 → ReLU
    # 最后一工序：输出最终成品种类 → fc3

model = MLP()
print(model)