# encoding=utf-8
from math import *

from numpy import *
import numpy as np


class kmeans:
    def __init__(self):
        self = self

    # coding=utf-8
    # 计算两个向量的距离，用的是欧几里得距离
    def distEclud(self, vecA, vecB):
        return sqrt(sum(power(vecA - vecB, 2)))

    # 随机生成初始的质心（初始方式是随机选K个点）
    def randCent(self, dataSet, k):
        n = shape(dataSet)[1]
        centroids = mat(zeros((k, n)))
        for j in range(n):
            minJ = min(dataSet[:, j])
            rangeJ = float(max(array(dataSet)[:, j]) - minJ)
            centroids[:, j] = minJ + rangeJ * random.rand(k, 1)
        return centroids

    def kMeans(self, dataSet, k, pop, n_iter, distMeas=distEclud, createCent=randCent):
        for i in dataSet:
            if np.nan in i:
                print('false1')
        m = shape(dataSet)[0]  # 样本点个数
        clusterAssment = mat(zeros((m, 2)))  # (N,2)   分配每个样本点
        # to a centroid, also holds SE of each point
        centroids = pop  # 中心点  (k,NP+)
        clusterChanged = True

        while clusterChanged and n_iter > 0:
            clusterChanged = False
            for i in range(m):  # for each data point assign it to the closest centroid   分配样本点到距离最近的簇
                minDist = inf
                minIndex = -1
                for j in range(k):
                    distJI = distMeas(self, centroids[j, :], dataSet[i, :])  # 样本点到中心的距离
                    if distJI < minDist:
                        minDist = distJI
                        minIndex = j
                if clusterAssment[i, 0] != minIndex:  # 如果当前样本点的标签发生变化,继续聚类.否则,聚类完成
                    clusterChanged = True
                clusterAssment[i, :] = minIndex, minDist ** 2  # 更新样本点的所属簇和minDist^2
            for cent in range(k):  # 重新计算每个簇的中心点,可能会有空簇
                ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]  # 获取当前簇的所有样本点,含有所有列属性
                # 如果是空簇,则使用原先的中心点
                if ptsInClust.size == 0:
                    centroids[cent, :] = pop[cent, :]
                else:
                    centroids[cent, :] = mean(ptsInClust, axis=0)  # 求列属性的均值,更新当前簇的中心点
            n_iter = n_iter - 1
        J = []
        center = mean(np.array(centroids), axis=0)
        B = np.sum([distMeas(self, center, i) for i in centroids])
        for cent in range(k):  # recalculate centroids
            jk = 0
            ptsInClust = dataSet[nonzero(clusterAssment[:, 0].A == cent)[0]]
            for point in ptsInClust:
                jk += distMeas(self, point, centroids[cent])
            J.append(jk)
        return centroids, clusterAssment, J, B


def main():
    dataMat = mat(data)
    myCentroids, clustAssing, J, B = kmeans('_', '_').kMeans(dataMat, 4)
    print(myCentroids, clustAssing, np.sum(J), B)
    return B / np.sum(J)
