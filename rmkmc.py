import numpy as np
import pandas as pd

from util import onehot, norm21


def rmkmc(Xs, k, gamma, n_iter, G):
    '''
    Parameters
    ----------
    Xs: a list of matrices. For each of them, the dimention is (n_samples, n_features),
        so every row correpsonds to a data point.

    k: the expected number of clusters.

    gamma: the parameter controling the weights. It needs to be strictly larger than 1.

    n_iter: maximum number of iterations, default is 300.

    Returns
    -------
    G: common indicator matrix of dimension (n_samples, k).

    Fs: a list of cluster centroids matrices, each of dimention (k, n_features).

    aa: 1darray of weights for the views.
    '''
    n_views = len(Xs)
    n_samples = Xs[0].shape[0]

    if n_views == 0:
        print("No data")
        return
    if k <= 1:
        print("k less than 1")
        return
    if gamma == 1:
        print("gamma cannot be 1")
        return

    Ds = [np.eye(n_samples) for _ in range(n_views)]

    Fs = [None for _ in range(n_views)]

    aa = [1 / n_views for _ in range(n_views)]

    # 迭代
    for i in range(n_iter):
        print("loop", i)

        # Calculate tildeD for each view
        tildeDs = [(aa[v] ** gamma) * Ds[v] for v in range(n_views)]

        # Update the centroids matrix F for each view.
        for v in range(n_views):
            Ftrans = Xs[v].T @ tildeDs[v] @ G @ np.linalg.inv(G.T @ tildeDs[v] @ G)
            Fs[v] = Ftrans.T

        # Update G by finding the best label for each data point
        labels = []
        for j in range(n_samples):
            # We to a brute force search
            cur_min = float("inf")
            cur_ind = 0
            for m in range(k):
                cur_sum = sum([tildeDs[v][j, j] * np.linalg.norm(Xs[v][j] - Fs[v][m]) ** 2 for v in range(n_views)])
                if cur_sum <= cur_min:
                    cur_min = cur_sum
                    cur_ind = m
            labels.append(cur_ind)
        G = onehot(labels, k)

        # Update D for each view
        for v in range(n_views):
            for j in range(n_samples):
                Ds[v][j, j] = 1 / (2 * np.linalg.norm((Xs[v] - G @ Fs[v])[j]))

        # Update weights aa
        numerator = [(gamma * norm21(Xs[v] - G @ Fs[v])) ** (1 / (1 - gamma)) for v in range(n_views)]
        denominator = sum(numerator)

        for v in range(n_views):
            aa[v] = numerator[v] / denominator
            # Security check
            if np.isnan(aa[v]):
                print("linalg error")
                return

    return (G, Fs, aa)
