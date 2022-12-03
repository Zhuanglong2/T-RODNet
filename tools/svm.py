#!/usr/bin/python3
# -*- coding: utf8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


iris_feature = u'花萼长度', u'花萼宽度', u'花瓣长度', u'花瓣宽度'

if __name__ == "__main__":
    path = 'iris.data'  # 数据文件路径
    data = pd.read_csv(path, header=None)
    x, y = data[range(4)], data[4]
    y = pd.Categorical(y).codes
    x = x[[0, 1]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1, train_size=0.6)

    # 分类器
    clf = svm.SVC(kernel='linear',C=1)
    # clf = svm.SVC(kernel='poly', degree=3)
    # clf = svm.SVC(kernel='rbf',C=1)
    # clf = svm.SVC(kernel='sigmoid')
    # clf = svm.SVC(C=0.3, kernel='linear', decision_function_shape='ovo')
    clf.fit(x_train, y_train.ravel())

    # 准确率
    # print(clf.score(x_train, y_train))  # 精度
    print('训练集准确率：', accuracy_score(y_train, clf.predict(x_train)))
    # print(clf.score(x_test, y_test))
    print('测试集准确率：', accuracy_score(y_test, clf.predict(x_test)))
    x1_min, x2_min = x.min()
    x1_max, x2_max = x.max()
    x1, x2 = np.mgrid[x1_min:x1_max:500j, x2_min:x2_max:500j]  # 生成网格采样点
    grid_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    print('grid_test = \n', grid_test)
    Z = clf.decision_function(grid_test)
    Z = Z[:, 0].reshape(x1.shape)
    print("decision_function:", Z)
    grid_hat = clf.predict(grid_test)
    grid_hat = grid_hat.reshape(x1.shape)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    cm_light = mpl.colors.ListedColormap(['#A0FFA0', '#FFA0A0', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    plt.figure(facecolor='w')
    plt.pcolormesh(x1, x2, grid_hat, cmap=cm_light)
    plt.scatter(x[0], x[1], c=y, edgecolors='k', s=50, cmap=cm_dark)  # 样本
    plt.scatter(x_test[0], x_test[1], s=120, facecolors='none', zorder=10)  # 圈中测试集样本
    plt.xlabel(iris_feature[0], fontsize=13)
    plt.ylabel(iris_feature[1], fontsize=13)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)
    plt.title(u'线性核 C=1', fontsize=16)
    plt.grid(b=True, ls=':')
    plt.show()