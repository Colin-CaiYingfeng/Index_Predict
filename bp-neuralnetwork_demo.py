# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:07:26 2020

@author: Colin
"""
#配置环境
import numpy as np
import time
import random
from StochasticGradientDescent import *
#%%
# 计时器
def timer(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        func(*args, **kwargs)
        end_time = time.time()
        print('Training time is :{:.2f} s.'.format(end_time - start_time))
    return wrapper

#%%
# 两种不同的激活函数
def tanh(x):
    return np.tanh(x)
def tanh_deriv(x):
    return 1.0 - np.tanh(x)*np.tanh(x)
#********************************************
def sigmoid(x):
    return 1/(1 + np.exp(-x))
def sigmoid_deriv(x):
    return sigmoid(x)*(1-sigmoid(x))

#%%
#创建一个三层BP神经网络主体
#因为是做时间序列预测所以说隐藏层到输出层不需要额外的激活函数
#隐藏层的output即输出层的output
class BPnn(object):
    def __init__(self, activation_choose = 'sigmoid', learn_rate):
        if activation_choose == 'sigmoid':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        elif activation_choose == 'tanh':
            self.activation = tanh
            self.activation_deriv = tanh_deriv
        #设置学习率
        self.learn_rate = learn_rate
        # w_、b_初始化为正态分布随机数
        self.w_ = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.b_ = [np.random.randn(y, 1) for y in sizes[1:]]
        
    #正向传播过程
    def foraward(self, x):
        n = self.w_[0].shape[1]
        x = x.reshape(n, -1)
        for b, w in zip(self.b_, self.w_):
            x = self.activation(np.dot(w, x) + b)
        return x

    
    #反向传播过程
    def backpropagation(self, x, y):
        new_b = [np.zeros(b.shape) for b in self.b_]
        new_w = [np.zeros(w.shape) for w in self.w_]
        
        xi = x
        xis = [x] # 输出
        xos = [] # 每层的输入，即w*上层的输出
        for b, w in zip(self.b_, self.w_):
            xo = np.dot(w, xi) + b
            xos.append(xo)
    
        delta = self.cost_derivative(xis[-1], y) * self.activation_deriv(xos[-1])
        new_b[-1] = delta
        new_w[-1] = np.dot(delta, xis[-2].transpose())
    
        return (new_b, new_w)
    
    #更新权值和偏置
    def update(self, mini_batch, learn_rate):
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_new_b, delta_new_w = self.backprop(x, y)
            new_b = [nb+dnb for nb, dnb in zip(new_b, delta_new_b)]
            new_w = [nw+dnw for nw, dnw in zip(new_w, delta_new_w)]
        
        self.w_ = [w - lr * nw for w, nw in zip(self.w_, new_w)]
        self.b_ = [b - lr * nb for b, nb in zip(self.b_, new_b)]
    
    @timer
    #随机梯度下降法
    def SGD(self, training_data, epochs, mini_batch_size, learn_rate, test_data=None):
        return SGD_function(training_data, epochs, mini_batch_size, learn_rate, test_data=None)
    
    