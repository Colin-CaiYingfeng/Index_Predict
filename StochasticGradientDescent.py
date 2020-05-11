# -*- coding: utf-8 -*-
"""
Created on Sat May  9 15:31:05 2020

@author: Colin
"""
import random
#随机梯度下降法
def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
    """
    :param training_data: training_data 是一个 (x, y) 元组的列表，表⽰训练输⼊和其对应的期望输出。
    :param epochs: 变量 epochs为迭代期数量
    :param mini_batch_size: 变量mini_batch_size为采样时的⼩批量数据的⼤⼩
    :param eta: 学习速率
    :param test_data: 如果给出了可选参数 test_data ，那么程序会在每个训练器后评估⽹络，并打印出部分进展。
    这对于追踪进度很有⽤，但相当拖慢执⾏速度。
    :return:
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
        ...
        """
        for mini_batch in mini_batches:
            # 对于最小训练集组成的集合mini_batches中的每一个最小训练集mini_batch
            self.update_mini_batch(mini_batch, eta)
            # 调用梯度下降算法
        if test_data:
            # 如果有测试数据集
            print("Epoch {} : {} / {}".format(j, self.evaluate(test_data), n_test));
            # j为迭代期序号
            # evaluate(test_data)为测试通过的数据个数
            # n_test为测试数据集的大小
        else:
            print("Epoch {} complete".format(j))