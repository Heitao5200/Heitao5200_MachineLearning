#!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : SimpleLinearRegression.py
# @Date    : 2018-09-29
# @Author  : 黑桃
# @Software: PyCharm
import numpy as np

class SimpleLinearRegression1:#定义类

    def __init__(self):
        """ 初始化 Simple Linear Regression 模型 """
        self.a_ = None
        self.b_ = None
        #a b不是用户送进来的参数，只是用来存储算法经过给定的数据计算后的结果

    def fit(self, x_train, y_train): #定义训练方法
        """ 根据训练数据集 x_train,y_train 训练 Simple Linear Regression 模型 """
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = 0.0
        d = 0.0
        for x, y in zip(x_train, y_train):
            num += (x - x_mean) * (y - y_mean)
            d += (x - x_mean) ** 2

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """
        给定待预测数据集 x_predict，返回表示 x_predict 的结果向量
        :param x_predict:
        :return:
        """
        assert x_predict.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """ 给定单个待预测数据 x，返回 x 的预测结果值 """
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression1()"


class SimpleLinearRegression2:#定义类

    def __init__(self):
        """ 初始化 Simple Linear Regression 模型 """
        self.a_ = None
        self.b_ = None
        #a b不是用户送进来的参数，只是用来存储算法经过给定的数据计算后的结果

    def fit(self, x_train, y_train): #定义训练方法
        """ 根据训练数据集 x_train,y_train 训练 Simple Linear Regression 模型 """
        assert x_train.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert len(x_train) == len(y_train), \
            "the size of x_train must be equal to the size of y_train"

        x_mean = np.mean(x_train)
        y_mean = np.mean(y_train)

        num = (x_train - x_mean).dot(y_train - y_mean)# 使用向量化点乘计算分子和分母
        d = (x_train - x_mean).dot(x_train - x_mean)

        self.a_ = num / d
        self.b_ = y_mean - self.a_ * x_mean

        return self

    def predict(self, x_predict):
        """
        给定待预测数据集 x_predict，返回表示 x_predict 的结果向量
        :param x_predict:
        :return:
        """
        assert x_predict.ndim == 1, \
            "Simple Linear Regressor can only solve single feature training data."
        assert self.a_ is not None and self.b_ is not None, \
            "must fit before predict!"

        return np.array([self._predict(x) for x in x_predict])

    def _predict(self, x_single):
        """ 给定单个待预测数据 x，返回 x 的预测结果值 """
        return self.a_ * x_single + self.b_

    def __repr__(self):
        return "SimpleLinearRegression2()"