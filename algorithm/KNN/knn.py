# coding:utf-8
"""
普通的knn算法，用欧式距离计算
"""

import numpy as np
import algorithm.util.file as file
import os

path = os.path.abspath(os.path.join(os.getcwd(), "..")) + '/data/iris.data'
data = file.readlines(path)


def pre_process_data(data):
    data = [line.split(',') for line in data]
    labels = [line[-1] for line in data]
    data_x = [line[:-1] for line in data]

    #字符串数据转成浮点型
    data_x = [list(map(lambda x: float(x), line)) for line in data_x]
    return data_x, labels


def compute_distance_from_data(test_x,data_x, labels):
    distance_label = []
    for i in range(data_x.__len__()):
        temp = []
        #计算欧式距离
        temp.append(np.linalg.norm(np.array(test_x) - np.array(data_x[i])))
        temp.append(labels[i].strip())
        distance_label.append(temp)
    print(distance_label[76])
    distance_label = sorted(distance_label,key=lambda x:x[0])
    print(distance_label[0])


data_x,labels= pre_process_data(data)
compute_distance_from_data([6.8,2.8,4.8,1.4],data_x,labels)