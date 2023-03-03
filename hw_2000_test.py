import os
from datetime import datetime

from logger import get_logger
from measure import my_output
from rmkmc import rmkmc
from rmkmc_init import *
from util import parse_cfg, deonehot, onehot

logger = get_logger("logger")

if __name__ == '__main__':
    # 读取每个视图的数据，并组合成一个mat
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

    # 读取运行配置
    date_str = str(datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    os.mkdir('./test/test_' + date_str)
    init_way, k, gamma, n_iterations, init_path, cluster_path = parse_cfg('./cfg.txt')
    init_path = init_path.replace('<$>', date_str)
    cluster_path = cluster_path.replace('<$>', date_str)

    # 初始化G
    if init_way == 'random_init':
        random_init(Xs, k, gamma, n_iterations, init_path)
    elif init_way == 'kmeans_pp_init':
        kmeans_pp_init(Xs, k, gamma, n_iterations, init_path)
    G = onehot(np.array(pd.read_csv(init_path)), k)

    # 运行rmkmc
    ind_matrix, Fs, aa = rmkmc(Xs, k, gamma, n_iterations, G)
    my_labels = deonehot(ind_matrix)
    df = pd.DataFrame(my_labels)
    df.to_csv(cluster_path, sep='\t', index=0)  # 不保留列名和行索引

    # 评价精确度,纯度,nmi
    accuracy, puri, nmi = my_output("digits", gamma, real_labels, my_labels)

    # 记录日志
    logger.info(f"=================test_{date_str}====================")
    logger.info(f"init ways:{init_way}")
    logger.info(f"k:{k}")
    logger.info(f"gamma:{gamma}")
    logger.info(f"n_iterations:{n_iterations}")
    logger.info(f"weight distribution:{aa}")
    logger.info(f"accuracy:{accuracy}")
    logger.info(f"puri:{puri}")
    logger.info(f"nmi:{nmi}")
    logger.info("=====================================================")
