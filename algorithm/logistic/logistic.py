#coding:utf-8
'''
逻辑斯谛回归算法的实现及参数优化算法实现
'''
from numpy import *



def sigmoid(inX):
    """
    定义逻辑斯谛回归
    :param inX: wx之和
    :return:
    """
    return 1.0/(1+exp(-inX))


def loadDataSet():
    """
    加载数据集

    :return:输入向量矩阵和输出向量
    """
    dataMat = [];
    labelMat = []
    fr = open('D:/github/nlp/algorithm/data/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])  # X0设为1.0，构成拓充后的输入向量
        labelMat.append(int(lineArr[2]))
    return dataMat, labelMat


def plotBestFit(weights):
    """
    画出数据集和逻辑斯谛最佳回归直线
    :param weights:
    """
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    if weights is not None:
        x = arange(-3.0, 3.0, 0.1)
        y = (-weights[0]-weights[1]*x)/weights[2]   #令w0*x0 + w1*x1 + w2*x2 = 0，其中x0=1，解出x1和x2的关系
        ax.plot(x, y)                               #一个作为X一个作为Y，画出直线
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()

plotBestFit(None)