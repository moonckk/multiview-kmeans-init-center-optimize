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
HW 数据集: 样本点2000个, 包含6个视图
"""
if __name__ == '__main__':

    # 运行参数
    # random_init,kmeans_pp_init,ga_init
    init_way = 'random_init'
    k = 10
    gamma = 3
    n_iterations = 3

    # 运行文件配置
    test_name = 'hw_2000'
    date_str = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    save_dir = f'./test/{test_name}_{date_str}/'
    os.mkdir(save_dir)
    init_labels_file = save_dir + 'init_labels.txt'
    test_labels_file = save_dir + 'test_labels.txt'
    real_labels_file = save_dir + 'real_labels.txt'
    result_file = save_dir + 'result.txt'

    # 读取mat文件
    fou = pd.read_csv('data/hw/mfeat-fou', sep=' +', header=None)
    fac = pd.read_csv('data/hw/mfeat-fac', sep=' +', header=None)
    kar = pd.read_csv('data/hw/mfeat-kar', sep=' +', header=None)
    pix = pd.read_csv('data/hw/mfeat-pix', sep=' +', header=None)
    zer = pd.read_csv('data/hw/mfeat-zer', sep=' +', header=None)
    mor = pd.read_csv('data/hw/mfeat-mor', sep=' +', header=None)
    Xs = [fou.values, fac.values, kar.values, pix.values, zer.values, mor.values]
    # 每个样本对应的标签
    real_labels = [0] * 200 + [1] * 200 + [2] * 200 + [3] * 200 + [4] * 200 + [5] * 200 + [6] * 200 + [7] * 200 + [
        8] * 200 + [9] * 200
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
