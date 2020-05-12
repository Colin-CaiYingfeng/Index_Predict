# -*- coding: utf-8 -*-
"""
Created on Mon May 11 16:07:26 2020

@author: Colin
"""
#配置环境
import numpy as np
from numpy.linalg import norm
import random
import time
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
    def __init__(self,  learn_rate, sizes, activation = ''):
        self.sizes_ = sizes
        self.num_layers_ = len(sizes)  # 层数
        if activation == 'sigmoid':
            self.activation = sigmoid
            self.activation_deriv = sigmoid_deriv
        elif activation == 'tanh':
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
        
        for l in range(2, self.num_layers_):
            xo = xos[-l]
            sp = self.sigmoid_der(xo)
            delta = np.dot(self.w_[-l+1].transpose(), delta) * sp
            new_b[-l] = delta
            new_w[-l] = np.dot(delta, xis[-l-1].transpose())
            
        return (new_b, new_w)
    
    #梯度下降更新b，w的具体算法
    def update_mini_batch(self, mini_batch, learn_rate):
        new_b = [np.zeros(b.shape) for b in self.biases]
        new_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_new_b, delta_new_w = self.backprop(x, y)
            new_b = [nb+dnb for nb, dnb in zip(new_b, delta_new_b)]
            new_w = [nw+dnw for nw, dnw in zip(new_w, delta_new_w)]
        
        self.w_ = [w - learn_rate * nw for w, nw in zip(self.w_, new_w)]
        self.b_ = [b - learn_rate * nb for b, nb in zip(self.b_, new_b)]
    
    @timer
    #随机梯度下降法
    def SGD_function(self, training_data, epochs, mini_batch_size, learn_rate,
            test_data):
        """
        training_data: training_data 是一个 (x, y) 元组的列表，表⽰训练输⼊和其对应的期望输出。
        epochs: 变量 epochs为迭代期数量
        mini_batch_size: 变量mini_batch_size为采样时的⼩批量数据的⼤⼩
        learn_rate: 学习速率
        test_data: 如果给出了可选参数 test_data ，那么程序会在每个训练器后评估，并打印出部分进展。
        这对于追踪进度很有用，但相当拖慢执行速度。
        """
        training_data = list(training_data)#将训练数据集强转为list
        n = len(training_data)#n为训练数据总数，大小等于训练数据集的大小
        if test_data:# 如果有测试数据集
            test_data = list(test_data)# 将测试数据集强转为list
            n_test = len(test_data)# n_test为测试数据总数，大小等于测试数据集的大小
        for j in range(epochs):# 对于每一个迭代期
            random.shuffle(training_data)# shuffle() 方法将序列的所有元素随机排序。
            mini_batches = [
                training_data[k:k + mini_batch_size] for k in range(0, n, mini_batch_size)
            ]
            """
            对于下标为0到n-1中的每一个下标，最小数据集为从训练数据集中下标为k到下标为k+⼩批量数据的⼤⼩-1之间的所有元素
            这些最小训练集组成的集合为mini_batches
            mini_batches[0]=training_data[0:0+mini_batch_size]
            mini_batches[1]=training_data[mini_batch_size:mini_batch_size+mini_batch_size]
            """
        for mini_batch in mini_batches:
            # 对于最小训练集组成的集合mini_batches中的每一个最小训练集mini_batch
            self.update_mini_batch(mini_batch, learn_rate)
            # 调用梯度下降算法
        if test_data:
            # 如果有测试数据集
            print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test));
            # j为迭代期序号
            # evaluate(test_data)为测试通过的数据个数
            # n_test为测试数据集的大小
        else:
            print("Epoch {} complete".format(j))
    
    #评价函数
    def evaluate(self, test_data):
        """
        Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation.
        """
        test_results = [(np.argmax(self.feedforward(x)), y) 
                    for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    #均方误差MSE作为loss函数
    def mse_loss(self, training_data):
        x_t, x_label = training_data
        test_results = [.5 * norm(self.feedforward(x).flatten() - y)**2
                        for (x, y) in zip(list(x_t), list(x_label))]
        return np.array(test_results).mean()
    
    #loss函数的导数
    def cost_derivative(self, output_activations, y):
        return (output_activations - y)
