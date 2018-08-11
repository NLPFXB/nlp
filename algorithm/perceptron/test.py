#coding:utf-8
import numpy as np
positive = 'Iris-setosa'
negative = 'Iris-versicolor'
growth_rate=0.1
train_data_path ='D:/github/nlp/algorithm/data/iris.data'
def read_data(path):
    with open(path) as f:
        data = f.readlines()
    return data

def get_iris_data(path):
    data = read_data(path)
    items = []
    for line in data:
        if line.split(',')[-1].strip() == positive:
            temp = []
            for tem in line.split(','):
                try:
                    temp.append(float(tem))
                except:
                    temp.append(1)
            items.append(temp)
        elif line.split(',')[-1].strip() == negative:
            temp = []
            for tem in line.split(','):
                try:
                    temp.append(float(tem))
                except:
                    temp.append(-1)
            items.append(temp)
    items = np.array(items)
    return items
items =get_iris_data(train_data_path)

# w = np.zeros((1,(int(items.shape[1])-1)))[0]
w = np.array(np.zeros(len(items[0])-1)).T
print(w.shape[0])
b=0
# def update(items,w,b):
#     right_classify_total = 0
#     total = items.shape[0]
#     while right_classify_total < total:
#         for item in items:
#             if item[-1] * (item[:-1].dot(w) + b) <= 0:
#                 w = w + growth_rate * item[-1] * item[:-1]
#                 b = b + growth_rate * item[-1]
#                 right_classify_total = 0
#             else:
#                 right_classify_total+=1
#     return w,b
def update(items,w,b):
    right_classify_total = 0
    total = items.shape[0]
    while right_classify_total < total:
        for item in items:
            if item[-1] * (np.dot(item[:-1],w) + b) <= 0:
                w = w + growth_rate * item[-1] * item[:-1]
                b = b + growth_rate * item[-1]
                right_classify_total = 0
            else:
                right_classify_total += 1
    return w,b
res = update(items,w,b)
w = res[0]
b = res[1]
print (w,b)
path = "D:/github/nlp/algorithm/data/iris.data"
test_data =get_iris_data(path)
print (test_data[49][:-1].dot(w)+b)