
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/12/19 18:51
# @Author  : Ryu
# @Site    :
# @File    : data_process.py
# @Software: PyCharm

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def load_dataset(fname):
    # fname = 'ensemble_study/dataset/weatherHistory.csv'
    data = pd.read_csv(fname, index_col=0)
    return data


def process_data(data: pd.core.frame.DataFrame):
    x = data['密度'].values.reshape(-1, 1)
    y = data['含糖率'].values.reshape(-1, 1)
    return x, y


def split_train_test_set(x, y):
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=0.3)
    return xtrain, xtest, ytrain, ytest


if __name__ == '__main__':

    file_name = 'D:\MachineLearning\watermelon3_0_Ch.csv'
    data = load_dataset(file_name)
    x, y = process_data(data)
    xtrain, xtest, ytrain, ytest = split_train_test_set(x, y)
    svr = svm.SVR(kernel='sigmoid', degree=3, gamma='auto', coef0=0, C=0.5)
    gauss_svr = svr.fit(xtrain, ytrain)
    y_pred = gauss_svr.predict(xtest)


    plt.scatter(x, y, c='k', label='data', zorder=1)
    # plt.hold(True)
    plt.plot(xtest, y_pred, c='r', label='SVR_fit')
    plt.xlabel('data')
    plt.ylabel('target')
    plt.title('SVR versus Kernel Ridge')
    plt.legend()
    plt.show()

