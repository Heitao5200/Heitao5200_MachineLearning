 #!/usr/bin/env python 3.6
#-*- coding:utf-8 -*-
# @File    : LinearRegression_V0.3.py
# @Date    : 2018-10-04
# @Author  : 黑桃
# @Software: PyCharm 
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from LR.datasplit import train_test_split
# from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsRegressor
"""
【报错一】解决：使用 if __name__=='__main__':
【Error】joblib Attempting to do parallel computing without protecting your import on a system that does not support forking.
To use parallel-computing in a script, you must protect your main loop using "if __name__ == "__main__".
Please see the joblib documentation on Parallel for more informationjoblib%5D Attempting to do parallel 
computing without protecting your import on a system that does not support forking. To use parallel-computing in a script,
you must protect your main loop using "if __name__ == %27__main__%27". Please see the joblib documentation on Parallel 
for more information

【报错二】解决：使用from LR.datasplit import train_test_split
【Error】Invalid parameters passed: {'seed': 666}
"""
def main():

    """=====================================================================================================================
    0 读取数据 + 处理数据
    """
    boston = datasets.load_boston()

    X = boston.data
    y = boston.target

    X = X[y < 50.0]
    y = y[y < 50.0]

    X_train, X_test, y_train, y_test = train_test_split(X, y, seed = 666)

    """=====================================================================================================================
    1 实例化线性回归模型 + 模型训练 + 打印系数、截距、R2分数
    """
    lin_reg = LinearRegression()
    lin_reg.fit(X_train, y_train)
    print(lin_reg.coef_)
    print(lin_reg.intercept_.shape)
    print(lin_reg.score(X_test, y_test))

    """=================================================================================================================
    2 去均值和方差归一化。且是针对每一个特征维度来做的，而不是针对样本
    """
    standardScaler = StandardScaler()
    standardScaler.fit(X_train, y_train)

    X_train_standard = standardScaler.transform(X_train)
    X_test_standard = standardScaler.transform(X_test)

    """=================================================================================================================
    3 设置【超参数】 
    """
    param_grid = [
        {
            "weights": ["uniform"],
            "n_neighbors": [i for i in range(1, 11)]
        },
        {
            "weights": ["distance"],
            "n_neighbors": [i for i in range(1, 11)],
            "p": [i for i in range(1,6)]
        }
    ]
    """=====================================================================================================================
    4  实例KNN模型 
    """
    knn_reg = KNeighborsRegressor()

    """=================================================================================================================
    5 网格搜索最优【超参数】+ 模型训练 + 打印系数、截距、R2分数
    n_jobs = -1 并行处理 使用计算机所有的核
    verbose=1 输出内容
    """
    grid_search = GridSearchCV(knn_reg, param_grid, n_jobs=-1, verbose=1)
    grid_search.fit(X_train_standard, y_train)

    params = grid_search.best_params_
    best_score = grid_search.best_score_# 这里的best_score和lin_reg.score的求解方式是不一样的
    best_estimator_score = grid_search.best_estimator_.score(X_test_standard, y_test)# 这里的best_estimator_.score和lin_reg.score的求解方式是一样的

    print(params)
    print(best_score)
    print(best_estimator_score)

if __name__ == '__main__':
    main()
