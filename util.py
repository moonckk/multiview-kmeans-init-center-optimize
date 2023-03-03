"""
工具方法
"""
from datetime import date

import numpy as np
import configparser

# L2,1范数
from typing import Dict


def norm21(X):
    res = 0
    for i in range(X.shape[0]):
        res += np.linalg.norm(X[i])
    return res


# 样本标签转换成G
def onehot(labels, k):
    n_samples = len(labels)  # 样本点数量
    G = np.zeros((n_samples, k))  # 指示矩阵G,默认都是0
    for i in range(len(labels)):
        G[i, labels[i]] = 1
    return G


# 将G转换成标签
def deonehot(G):
    n_samples = G.shape[0]
    labels = [0] * n_samples
    for i in range(n_samples):
        labels[i] = np.nonzero(G[i])[0][0]
    return labels


def parse_cfg(filename):
    cf = configparser.ConfigParser()
    cf.read(filename)
    train = cf.items("rmkmc")
    dic = {}
    for key, val in train:
        dic[key] = val
    init_way = dic['init_way']
    k = int(dic['k'])
    gamma = int(dic['gamma'])
    n_iterations = int(dic['n_iterations'])
    init_path = dic['init_path']
    cluster_path = dic['cluster_path']
    return init_way, k, gamma, n_iterations, init_path, cluster_path
