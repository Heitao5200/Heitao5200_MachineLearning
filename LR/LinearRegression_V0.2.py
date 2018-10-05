#!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : LinearRegression_V0.2.py.py
# @Date    : 2018-10-04
# @Author  : 黑桃
# @Software: PyCharm

"""
多元线性回归
"""
import numpy as np
from LR import  datasplit
from ML.LinearRegression import LinearRegression
from sklearn import datasets

"""=====================================================================================================================
0 读取数据 + 处理数据
"""
boston = datasets.load_boston()
x = boston.data
y = boston.target

x = x[y < 50]
y = y[y < 50]

x_train,x_test,y_train,y_test = datasplit.train_test_split(x,y,seed=666)
print(x_train.shape,x_test.shape)

"""=====================================================================================================================
1 实例化模型
"""
reg = LinearRegression()

"""=====================================================================================================================
2 模型预测
"""
reg.fit(x_train, y_train)
y_test = reg.predict(x_test)

print(reg.coef_)
print(reg.interception)
print(reg._theta)

print(y_test.shape,y_test.shape)

print(reg.score(x_test,y_test))









