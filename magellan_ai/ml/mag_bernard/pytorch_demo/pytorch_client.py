# -*- coding: utf-8 -*-

from torch.autograd import Variable

import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import numpy as np

"""
    Author: huangning
    Date: 2020/10/29
    Function: 用Python调用Bernard的Pytorch服务
"""


# class Net(torch.nn.Module):
#
#     def __init__(self, n_input, n_hidden, n_output):
#
#         # 将Net类对象转换成nn.Module类的对象，并调用nn.Module类的构造方法
#         super(Net, self).__init__()
#         self.hidden1 = nn.Linear(n_input, n_hidden)
#         self.hidden2 = nn.Linear(n_hidden, n_hidden)
#         self.predict = nn.Linear(n_hidden, n_output)
#
#     def forward(self, input):
#
#         out = self.hidden1(input)
#         out = torch.relu(out)
#         out = self.hidden2(out)
#         out = torch.sigmoid(out)
#         out = self.predict(out)
#
#         return out

class MyRNNLoop(torch.nn.Module):
    def __init__(self):
        super(MyRNNLoop, self).__init__()
        self.cell = torch.jit.trace(MyCell(scripted_gate), (x, h))

    def forward(self, xs):
        h, y = torch.zeros(3, 4), torch.zeros(3, 4)
        for i in range(xs.size(0)):
            y, h = self.cell(xs[i], h)
        return y, h


def main():

    rnn_loop = torch.jit.script(MyRNNLoop())
    print(rnn_loop.code)

    # 加上随机噪声增加数据的复杂性
    # x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
    # y = x.pow(3)+0.1*torch.randn(x.size())
    # x, y = (Variable(x), Variable(y))

    # # 实例化1*20*20*1的简单神经网络
    # net = Net(1, 20, 1)
    #
    # # 模型训练
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    # loss_func = torch.nn.MSELoss()
    # for t in range(50):
    #     prediction = net(x)
    #     loss = loss_func(prediction, y)
    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    #
    # # 模型保存
    # model_path = "./pytorch_clf.pt"
    # torch.save(net.state_dict(), model_path)
    #
    # new_model = Net(1, 20, 1)
    # new_model.load_state_dict(torch.load(model_path))
    #
    # # for i in np.linspace(-1, 1, 100):
    # test_data = torch.unsqueeze(torch.linspace(-2, 0, 10), dim=1)
    # print(new_model.forward(test_data))

    # 将不属于与训练的模型参数替换掉
    # # 把不属于新模型 model_dict 的参数， 除掉。pretrained_dict是预训练参数
    # pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}

    # # 计算准确率
    # accuracy_rate = cal_accuracy(FLAGS.server, FLAGS.input_data_path, FLAGS.concurrency, FLAGS.num_tests)
    # print('\n准确率的计算结果为:{:.2%}'.format(accuracy_rate))


if __name__ == "__main__":
    main()
