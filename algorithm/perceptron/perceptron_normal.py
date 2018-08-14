# coding:utf-8
"""
               ┏┓      ┏┓
            ┏┛┻━━━┛┻┓
            ┃      ☃      ┃
            ┃  ┳┛  ┗┳  ┃
            ┃      ┻      ┃
            ┗━┓      ┏━┛
                ┃      ┗━━━┓
                ┃  神兽保佑    ┣┓
                ┃　永无BUG！   ┏┛
                ┗┓┓┏━┳┓┏┛
                  ┃┫┫  ┃┫┫
                 ┗┻┛  ┗┻┛
按照定义实现的简单感知机算法，数据集就用testSet,对数据要做一个预处理，将0全部换成1
"""
import numpy as np
import algorithm.util.file as file
import os


def isUpdate(w, x, b, y):
    """
    :param w: 权重参数
    :param x: 输入
    :param b: 常量
    :param y: 分类标签
    :return: 判断是否更新
    """
    x = np.array(x)
    if y * (np.dot(w, x) + b) <= 0:
        return True
    return False


def pre_process_data(data):
    """
    预处理文件
    :param path:
    :return:
    """
    # iris
    data = data[:100]
    data = [line.strip().split(',') for line in data]

    # testSet.txt
    # data = [line.strip().split('	') for line in data]
    for i in range(len(data)):
        # iris.txt
        if data[i][-1] == 'Iris-setosa':
            data[i][-1] = '-1.0'
        else:
            data[i][-1] = '1.0'
            # if data[i][-1] == '0':
            #     data[i][-1] = '-1.0'
            # else:
            #     data[i][-1] = '1.0'

    for i in range(len(data)):
        for j in range(len(data[i])):
            if '-' in data[i][j]:
                data[i][j] = -float(data[i][j][1:])
            else:
                data[i][j] = float(data[i][j])
    data_arr = np.array(data)
    # print(data_arr)
    return data_arr


def after_process_data(data):
    """
    后处理文件
    :param data:
    :return:
    """
    pass


def init_params(data):
    """
    从数据中分析出初始化参数
    :param data:
    :return:
    """
    w = np.array(np.zeros(len(data[0]) - 1))
    w = w.T
    b = 0
    n = 0.1  # 学习速率
    return w, b, n


def train_model(data):
    data = pre_process_data(data)
    w, b, n = init_params(data)
    param = {}
    param['w'] = w
    param['b'] = b
    param['n'] = n
    w, b = update_params_by_data(param, data)  # 更新参数方法，可以替换
    return w, b


def update_params_by_data(params, dataSet):
    """
    更新迭代参数的方法,遍历数据集，循环判断结果，如果遇到不满足的点，更新w、b
    :return: 更新好的参数
    """
    w = params.get('w')
    b = params.get('b')
    n = params.get('n')
    i = 0
    while i < len(dataSet):
        for line in dataSet:
            if isUpdate(w, line[:-1], b, line[-1]):
                w = w + n * (np.array(line[:-1]).T) * line[-1]
                b = b + n * line[-1]
                i = 0
                print(w, b)
            else:
                i += 1

    return w, b


def predict(w, x, b):
    if np.dot(w, x) + b >= 0:
        return 1
    return -1


path = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/data/iris.data'
# path = 'D:/github/nlp/algorithm/data/testSet.txt'
data = file.readlines(path)
w, b = train_model(data)
# print(w,b)
if predict(w, np.array([4.9, 3.1, 1.5, 0.1]), b) == 1:
    print('Iris-versicolor')
else:
    print('Iris-setosa')

"""
经过测试，貌似testSet线性不可分，需要实现线性可分的数据
"""
