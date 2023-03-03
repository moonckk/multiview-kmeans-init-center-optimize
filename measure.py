"""
Implement the measurements 'Accuracy' and 'Purity'. NMI is already provided in sklearn.
"""
import pandas as pd
from rmkmc import *

from sklearn.metrics import confusion_matrix
from sklearn.utils.linear_assignment_ import linear_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score


# Accuracy (inspired by https://smorbieu.gitlab.io/accuracy-from-classification-to-clustering-evaluation/)
def accu(y_true, y_pred):
    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)

    cm = confusion_matrix(y_true, y_pred)
    indexes = linear_assignment(_make_cost_m(cm))
    js = [e[1] for e in sorted(indexes, key=lambda x: x[0])]
    cm2 = cm[:, js]
    print(cm2)
    return np.trace(cm2) / np.sum(cm)


# Purity
def purity(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return np.sum(np.max(cm, axis=0)) / np.sum(cm)


# Utility function to output the measurement results
def my_output(data_name, gamma, y_true, y_pred):
    accuracy = accu(y_true, y_pred)
    print("Accuracy of RMKMC on", data_name, "with gamma=", gamma, "is", accuracy)

    puri = purity(y_true, y_pred)
    print("Purity of RMKMC on", data_name, "with gamma=", gamma, "is", puri)

    nmi = normalized_mutual_info_score(y_true, y_pred)
    print("NMI of RMKMC on", data_name, "with gamma=", gamma, "is", nmi)
    return accuracy, puri, nmi
