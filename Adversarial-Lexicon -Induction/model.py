#!/usr/bin/env python
# encoding: utf-8


"""
@version: ??
@author: muyeby
@contact: bxf_hit@163.com
@site: http://muyeby.github.io
@software: PyCharm
@file: model.py
@time: 18-3-30 下午9:52
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.autograd import Variable

HALF_BATCH_SIZE = 128

class GaussianNoiseLayer(nn.Module):
    def __init__(self, sigma,shape):
        super(GaussianNoiseLayer,self).__init__()
        self.sigma = sigma
        self.noise = Variable(torch.zeros(HALF_BATCH_SIZE,shape).cuda())

    def forward(self, x):
        self.noise.data.normal_(1, std=self.sigma)
        return x*self.noise

class Generator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Generator, self).__init__()
        self.map = nn.Linear(input_size, output_size, bias=False)

    def forward(self, x):
        x = self.map(x)
        recon = F.linear(x, self.map.weight.t(), bias=None)
        return x, recon

class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Discriminator, self).__init__()


        self.input_noise = GaussianNoiseLayer(sigma=0.5,shape=input_size)
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, output_size)

        # self.hidden_noise = GaussianNoiseLayer(sigma=0.5,shape=hidden_size)
        self.sigmod = nn.Sigmoid()
        # self.relu = nn.ReLU()
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        # dropout 加在线性变换前面
        net = F.dropout(x, 0.5, training=self.training)
        # net = self.input_noise(x)         # 用高斯噪声和dropout差不多
        net = self.map1(net)
        net = self.relu(net)
        # net = F.dropout(net, 0.5, training=self.training)     #epoch比较大时加上隐层的dropout
        net = self.sigmod(self.map2(net))
        return net