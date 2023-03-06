"""
初始化方法:
- 随机初始化方式
- kmeans++
- 遗传优化GA
"""
import random

import numpy as np

# 随机初始化
from codec import code
from kmeans import kmeans


def random_init(Xs, k, gamma, n_iter, init_path):
    n_samples = Xs[0].shape[0]
    labels = np.random.randint(0, k, size=n_samples)
    return labels


# kmeans++初始化
def kmeans_pp_init(Xs, k, gamma, n_iter, init_path):
    n_views = len(Xs)  # 视图数量
    n_samples = Xs[0].shape[0]  # 样本点数量
    Fs = [None for _ in range(n_views)]  # 初始化每个视图的中心点Fs
    for v in range(n_views):
        Fs[v] = kmeans_pp(Xs[v], k)  # kmeans++算法初始化每个视图的中心点,对每个视图进行kmeans
    # 寻找每个样本点最合适的标签
    labels = []
    for j in range(n_samples):
        cur_min = float("inf")
        cur_ind = 0
        for m in range(k):  # 比较当前样本点j和中心点m的距离, 选择距离最小的,作为当前样本点的初始簇
            cur_sum = sum(
                [np.linalg.norm(Xs[v][j] - Fs[v][m]) ** 2 for v in range(n_views)])  # 一个样本点和中心包含多个视图,此处计算多个视图,然后求和
            if cur_sum <= cur_min:  # 更新最小值
                cur_min = cur_sum
                cur_ind = m
        labels.append(cur_ind)  # label存储和当前样本点最匹配的簇(cur_sum最小的簇)
    return labels


def kmeans_pp(X, k):
    n_samples = X.shape[0]
    first_cent = np.random.choice(n_samples, 1)
    centroids = [X[first_cent]]
    for i in range(1, k):
        min_dist = [min([np.linalg.norm(x - cent) ** 2 for cent in centroids]) for x in X]
        next_cent = np.random.choice(n_samples, 1, min_dist)
        centroids.append(X[next_cent])
    return np.array(centroids)


# 遗传算法优化初始化中心点
def ga_init(Xs, k, gamma, n_iter, init_path):
    n_views = len(Xs)
    n_samples = Xs[0].shape[0]
    Fs = [None for _ in range(n_views)]
    for v in range(n_views):
        Fs[v] = ga(Xs[v], k)
    labels = []
    for j in range(n_samples):
        cur_min = float("inf")
        cur_ind = 0
        for m in range(k):  # 比较当前样本点j和中心点m的距离, 选择距离最小的,作为当前样本点的初始簇
            cur_sum = sum(
                [np.linalg.norm(Xs[v][j] - Fs[v][m]) ** 2 for v in range(n_views)])  # 一个样本点和中心包含多个视图,此处计算多个视图,然后求和
            if cur_sum <= cur_min:  # 更新最小值
                cur_min = cur_sum
                cur_ind = m
        labels.append(cur_ind)  # label存储和当前样本点最匹配的簇(cur_sum最小的簇)
    return labels


def ga(X, k):
    popsize = 50  # 种群大小
    factor_num = len(X[0])  # 变量个数(列数)
    pc = 0.5
    pm = 0.05
    # 计算各属性最小值和全距，以确定随机种群取值范围
    fc_min, fc_d = [], []
    for i in range(factor_num):
        fc_min.append(np.min(X[i]))  # 取每列的最小值
        fc_d.append(np.max(X[i]) - np.min(X[i]))  # 每列最大值-最小值
    # 标准化数据
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    results = []
    bestfit = 0
    fitvalue = []
    tempop = []
    pop = []
    # 对个体编码
    for i in range(popsize):
        p = ''
        for _ in range(k):
            for j in range(factor_num):
                p += mybin(np.random.rand())
        pop.append(p)
    # 迭代
    for i in range(5):
        objvalue = calobjvalue(X, pop, factor_num, k)  # 计算目标函数值
        fitvalue = calfitvalue(objvalue)  # 计算个体的适应值
        [bestindividual, bestfit] = best(pop, fitvalue)  # 选出最好的个体和最好的函数值
        if len(bestindividual) != 0:
            results.append([bestfit, split_dec(bestindividual)])
        pop = selection(pop, fitvalue)  # 自然选择，淘汰掉一部分适应性低的个体
        pop = crossover(pop, pc)  # 交叉繁殖
        pop = mutation(pop, pm)  # 基因突变
    return np.array(split_dec(bestindividual)).reshape(k, -1)


def mybin(floatingPoint):
    return code().floatToBinary64(floatingPoint)


def calobjvalue(data, pop, factor_num, k):  # 计算目标函数值
    objvalue = []
    for i in pop:  # pop是遗传样本数量
        objvalue.append(main(data, split_dec(i), k))
    return objvalue  # 目标函数值objvalue[m] 与个体基因 pop[m] 对应


def main(data, center, k):
    print(np.shape(center), np.shape(np.mat(data)), k)
    myCentroids, clustAssing, J, B = kmeans().kMeans(np.mat(data), k, np.array(center).reshape(k, -1), 5)
    return B / np.sum(J)


# 将多个属性值连起来的浮点数编码化为属性值列表
def split_dec(code, code_len=64):
    temp = []
    n = len(code) // 64

    okl = code

    if n != 1:
        for l in range(n - 1):
            temp.append(mydec(okl[l * code_len:(l + 1) * code_len]))
        temp.append(mydec(okl[(n - 1) * code_len:]))
    else:
        temp.apppend(mydec(okl))
    return temp


def mydec(binary):
    return code().binaryToFloat(binary)


def calfitvalue(objvalue):
    return objvalue


def best(pop, fitvalue):  # 找出适应函数值中最大值，和对应的个体
    px = len(pop)
    bestindividual = []
    bestfit = fitvalue[0]
    for i in range(1, px):
        if (fitvalue[i] >= bestfit):
            bestfit = fitvalue[i]
            bestindividual = pop[i]
    return [bestindividual, bestfit]


def selection(pop, fitvalue):  # 自然选择（轮盘赌算法）
    newfitvalue = []
    totalfit = sum(fitvalue)
    for i in range(len(fitvalue)):
        newfitvalue.append(fitvalue[i] / totalfit)
    cumsum(newfitvalue)

    ms = [];
    ms.sort()
    poplen = len(pop)
    for i in range(poplen):
        ms.append(random.random())  # random float list ms
    ms.sort()
    fitin = 0
    newin = 0
    newpop = pop
    newfitvalue = cumsum(newfitvalue)
    while newin < poplen:
        if (ms[newin] < newfitvalue[fitin]):
            newpop[newin] = pop[fitin]
            newin = newin + 1
        else:
            fitin = fitin + 1
    pop = newpop
    return pop


def cumsum(fit_value):
    fit = []
    for i in range(len(fit_value) - 1):
        fit.append(np.sum(fit_value[:i + 1]))
    fit.append(np.sum(fit_value))
    return fit


def crossover(pop, pc):  # 个体间交叉，实现基因交换
    poplen = len(pop)
    for i in range(poplen - 1):
        if (random.random() < pc):
            cpoint = random.randint(0, len(pop[0]))
            temp1 = ''
            temp2 = ''
            temp1 += (pop[i][0: cpoint])
            temp1 += (pop[i + 1][cpoint: len(pop[i])])
            temp2 += (pop[i + 1][0: cpoint])
            temp2 += (pop[i][cpoint: len(pop[i])])
            pop[i] = temp1
            pop[i + 1] = temp2
    return pop


def mutation(pop, pm):  # 基因突变
    px = len(pop)
    py = len(pop[0])
    for i in range(px):
        if (random.random() < pm):
            mpoint = random.randint(0, py - 1)
            if (pop[i][mpoint] == 1):
                pop[i][mpoint] = 0
            elif (pop[i][mpoint] == 0):
                pop[i][mpoint] = 1
            else:
                pass
    return pop
