#coding;utf-8
import codecs
import random

def readlines(path):
    """
    :param path:
    :return:
    """
    with codecs.open(path,encoding="utf-8") as f:
        data = f.readlines()
        return data
def getRandom(begin,end):
    """
    函数返回数字 N ，N 为 a 到 b 之间的数字（a <= N <= b），包含 a 和 b。
    """
    return random.randint(begin,end)
