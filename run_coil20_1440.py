import os
from datetime import datetime

import pandas as pd
import scipy
from scipy import io

from logger import get_logger
from measure import my_output
from rmkmc import rmkmc
from rmkmc_init import *
from util import deonehot, onehot

logger = get_logger("logger")

"""
COIL20数据集: 样本点1440个, v1包含30个变量,v2包含19个变量,v3包含30个变量,分成20类
"""
if __name__ == '__main__':

    # 运行参数
    # random_init,kmeans_pp_init,ga_init
    init_way = 'kmeans_pp_init'
    k = 20
    gamma = 3
    n_iterations = 3

    # 运行文件配置
    test_name = 'coil20_1440'
    date_str = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    save_dir = f'./test/{test_name}_{date_str}/'
    os.mkdir(save_dir)
    init_labels_file = save_dir + 'init_labels.txt'
    test_labels_file = save_dir + 'test_labels.txt'
    real_labels_file = save_dir + 'real_labels.txt'
    result_file = save_dir + 'result.txt'

    # 读取mat文件
    mat = scipy.io.loadmat("data/COIL20.mat", matlab_compatible=True)
    # 视图数据X
    X = np.array(mat['X'])
    v1 = X[0, 0].T  # 需要转置,样本点为行,变量为列
    v2 = X[0, 1].T
    v3 = X[0, 2].T
    Xs = [v1, v2, v3]  # 将3个视图组合在一起
    # 真实标签Y,N*1的list
    real_labels = (np.array(mat['Y'], dtype=int) - 1).flatten().tolist()  # 需要讲每个真实标签-1,从0开始
    df = pd.DataFrame(real_labels)
    df.to_csv(real_labels_file, sep='\t', index=False)  # 不保留列名和行索引

    # 初始化中心G
    init_labels = []
    if init_way == 'random_init':
        init_labels = random_init(Xs, k, gamma, n_iterations, init_labels_file)
    elif init_way == 'kmeans_pp_init':
        init_labels = kmeans_pp_init(Xs, k, gamma, n_iterations, init_labels_file)
    elif init_way == 'ga_init':
        init_labels = ga_init(Xs, k, gamma, n_iterations, init_labels_file)
    df = pd.DataFrame(init_labels)
    df.to_csv(init_labels_file, sep='\t', index=False)  # 不保留列名和行索引
    G = onehot(np.array(pd.read_csv(init_labels_file)), k)

    # 运行rmkmc
    ind_matrix, Fs, aa = rmkmc(Xs, k, gamma, n_iterations, G)
    test_labels = deonehot(ind_matrix)
    df = pd.DataFrame(test_labels)
    df.to_csv(test_labels_file, sep='\t', index=False)  # 不保留列名和行索引

    # 评价精确度,纯度,nmi
    accuracy, puri, nmi = my_output("digits", gamma, real_labels, test_labels)

    # 记录日志
    report = f'init ways:{init_way}\nk:{k}\ngamma:{gamma}\nn_iterations:{n_iterations}\nweight distribution:{aa}\n' \
             f'accuracy:{accuracy}\npurity:{puri}\nnmi:{nmi}'
    with open(result_file, mode='w', encoding='utf-8') as f:
        f.write(report)
