"""
初始化方法:
- 随机初始化方式
- kmeans++
- 遗传优化GA
"""

import numpy as np
import pandas as pd


def random_init(Xs, k, gamma, n_iter, init_path):
    n_samples = Xs[0].shape[0]
    labels = np.random.randint(0, k, size=n_samples)
    df = pd.DataFrame(labels)
    df.to_csv(init_path, sep='\t', index=0)  # 不保留列名和行索引


# kmeans++初始化
def kmeans_pp_init(Xs, k, gamma, n_iter, init_path):
    n_views = len(Xs)
    n_samples = Xs[0].shape[0]
    Fs = [None for _ in range(n_views)]
    for v in range(n_views):
        Fs[v] = kmeans_pp(Xs[v], k)
    # Update G by finding the best label for each data point
    labels = []
    for j in range(n_samples):
        # We to a brute force search
        cur_min = float("inf")
        cur_ind = 0
        for m in range(k):
            cur_sum = sum([np.linalg.norm(Xs[v][j] - Fs[v][m]) ** 2 for v in range(n_views)])
            if cur_sum <= cur_min:
                cur_min = cur_sum
                cur_ind = m
        labels.append(cur_ind)
    df = pd.DataFrame(labels)
    df.to_csv(init_path, sep='\t', index=0)  # 不保留列名和行索引


def kmeans_pp(X, k):
    n_samples = X.shape[0]
    first_cent = np.random.choice(n_samples, 1)
    centroids = [X[first_cent]]
    for i in range(1, k):
        min_dist = [min([np.linalg.norm(x - cent) ** 2 for cent in centroids]) for x in X]
        next_cent = np.random.choice(n_samples, 1, min_dist)
        centroids.append(X[next_cent])
    return np.array(centroids)
