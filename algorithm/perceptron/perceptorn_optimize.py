#coding:utf-8
'''
优化的感知机算法，
提前算好gram矩阵
'''

import numpy as np
#计算gram矩阵
print(np.outer([1,2,3],[4,5,6,7]))

def compute_garms(data_X):
    """
    :param data_X:  输入变量
    :return: 计算变量
    """
    for line in data_X:
        line
