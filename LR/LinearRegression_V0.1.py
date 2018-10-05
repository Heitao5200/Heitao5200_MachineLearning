#!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : LinearRegression_V0.1.py
# @Date    : 2018-09-29
# @Author  : 黑桃
# @Software: PyCharm
"""
简单线性回归
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from ML.SimpleLinearRegression import  SimpleLinearRegression2
from ML.SimpleLinearRegression import  SimpleLinearRegression1

"""=====================================================================================================================
0 数据集
"""
x_predict = 6

x = np.array([1,2,3,4,5])
y = np.array([1,3,2,3,5])

m = 100000
big_x = np.random.random(size=m)
big_y = big_x * 2 + 3 + np.random.normal(size=m)

"""=====================================================================================================================
1 实例化SimpleLinearRegression模型
"""
reg1 = SimpleLinearRegression1()
reg2 = SimpleLinearRegression2()#实例化一个模型

"""=====================================================================================================================
2.1 模型预测，预测使用数据集1
"""
reg1.fit(x, y)
reg1.predict(np.array([x_predict]))

"""=====================================================================================================================
2.2 模型训练，预测使用数据集2
"""
reg1.fit(x, y)
x1 = np.array([5,2,8,2,3,6,0,1.2])
y1 = reg1.predict(x1)
print(y1)

"""=====================================================================================================================
2.3 模型训练，预测使用数据集3
"""
reg1.fit(big_x,big_y)
big_x = np.random.random(size=m)
test_y = reg1.predict(np.array(big_x))

print(test_y,reg1.a_,reg1.b_)


"""=====================================================================================================================
2.4 性能测试
"""


time1_start = time.time()
reg1.fit(big_x,big_y)
time1_end = time.time()
time1 = time1_end - time1_start

time2_start = time.time()
reg2.fit(big_x,big_y)
time2_end = time.time()
time2 =  time2_end - time2_start

print("直接计算所用的时间：%s"%time1)
print("向量化后计算所用的时间：%s"%time2)
"""=====================================================================================================================
3 绘图
"""
plt.scatter(big_x,big_y)
plt.xlabel('X')
plt.ylabel('Y')
plt.title("LR")
plt.show()







