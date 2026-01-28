"""Entry point for PyTorch experiments."""

import torch


def tensor() -> None:
    # """Print basic environment info and confirm torch import."""
    # print("PyTorch version:", torch.__version__)
    # if torch.cuda.is_available():
    #     print("CUDA available; device count:", torch.cuda.device_count())
    # else:
    #     print("CUDA not available; running on CPU.")
    t1 = torch.tensor([[1.,-1.],[-1.,1.]])
    print(t1)
    print(t1[1][0])
    t2 = torch.tensor([[1,2,3],[4,5,6]])
    print(t2)
    t3 = torch.zeros([6,6],dtype=torch.int32)
    print(t3)
    # 矩阵相加
    t4 = t2 + t2
    print(t4)
    t5 = t2 - t2
    print(t5)
    t6 = t2 * t2
    print(t6)
    t7 = t2 / t2    
    print(t7)

    n1 = torch.randn(2,3)
    n2 = torch.randn(3,6)
    n3 = torch.matmul(n1,n2)
    print(n1)
    print(n2)
    print(n3)

    print("------------------")
    n1 = torch.randn(2,3)
    n2 = torch.randn(3)
    n3 = torch.matmul(n1,n2)
    print(n1)
    print(n2)
    print(n3)

    print("\n")

    n1 = torch.randn(3)
    n2 = torch.randn(3,2)
    n3 = torch.matmul(n1,n2)
    print(n1)
    print(n2)
    print(n3)
    print(n3.size()) # 2 col

    n11 = torch.randn(2,9,6,3,4)
    n22 = torch.randn(6,4,5)
    n33 = n11 @ n22 #从右向左边对齐 最后两维进行矩阵乘法 第一个维度是批次
    print("n33",n33.size()) 

    n66 = torch.tensor([[3.,9.],[1.,2.]],requires_grad=True) #自动求导
    print(n66)
    #print(n66.pow(2))
    # x = n66.pow(2).sum() #向前求和
    x = n66.pow(2) + n66 + 1
    x = x.sum()
    print("测试求和",x)
    x.backward() #反向传播
    print(n66.grad)

    # 向量自动求导
    n77 = torch.tensor([3.,9.,2.],requires_grad=True)
    y = n77.pow(2) + n77 + 1
    print("y:",y)
    z = y.sum()
    print("z:",z)
    z.backward()
    print("n77.grad:",n77.grad)


if __name__ == "__main__":
    tensor()
