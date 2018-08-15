# coding:utf-8
"""
优化的感知机算法，提前算好gram矩阵
"""

import numpy as np
import algorithm.util.file as file
import os

# 计算gram矩阵
# print(np.outer([1, 2, 3], [4, 5, 6, 7]))
# data_x = np.outer([1, 2, 3,4], [4, 5, 6, 7])
path = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/data/iris.data'

data = file.readlines(path)


def train_model(data):
    gram, labels,data_x = pre_process_data(data)
    a, b, n = init_params(data_x)
    a, b = update_params_by_data(a, b, n, gram, labels)  # 更新参数方法，可以替换
    return a, b


def update_params_by_data(a, b, n, gram, labels):
    """
    更新迭代参数的方法,遍历数据集，循环判断结果，如果遇到不满足的点，更新w、b
    :return: 更新好的参数
    """
    i = 0
    while i < len(labels):
        if isUpdate(a, b, i, gram, labels):
            a = a + n
            b = b + n * labels[i]
            i = 0
        else:
            i += 1

    return a, b


def isUpdate(a, b, i, gram, labels):
    """
    :param w: 权重参数
    :param x: 输入
    :param b: 常量
    :param y: 分类标签
    :return: 判断是否更新
    """
    print(type(a),a)
    print(type(gram[:, i]),gram[:,i])
    print(b)
    print(np.dot(np.array(a)*labels, gram[:, i])+ b)
    if (np.dot(np.array(a), gram[:, i]) + b) * labels[i] <= 0:
        return True
    return False


def init_params(data_x):
    """
    从数据中分析出初始化参数
    :param data:
    :return:
    """
    a = np.array(np.zeros(len(data_x[0])))
    a = [float(x) for x in a]
    b = 0
    n = 0.1  # 学习速率
    return a, b, n


def pre_process_data(data):
    """
    :param data: 从文本读取的数据集
    :return:
    """
    data = data[:100]
    data = [line.strip().split(',') for line in data]
    data_x = [line[:-1] for line in data]

    gram = compute_garms(data_x)
    labels_str = [line[-1] for line in data]
    labels = []
    for i in range(labels_str.__len__()):
        if labels_str[i] == 'Iris-setosa':
            labels.append(-1)
        else:
            labels.append(1)
    print(labels)
    return gram, labels,data_x


def compute_garms(data_X):
    """
    :param data_X:  输入变量
    :return: 计算x
    """
    len_n = len(data_X[0])
    gram = np.ones((len_n, len_n))
    for i in range(len_n):
        for j in range(len_n):
            gram[i][j] = np.dot([float(x) for x in data_X[i]], [float(x) for x in data_X[j]])
    return gram


def predict(w, x, b):
    if np.dot(w, x) + b >= 0:
        return 1
    return -1


a, b = train_model(data)
# print(w,b)
if predict(a, np.array([4.9, 3.1, 1.5, 0.1]), b) == 1:
    print('Iris-versicolor')
else:
    print('Iris-setosa')

# print(compute_garms(data_x))
