# -*- coding: utf-8 -*-
# @Time : 2020/12/22 6:07 下午
# @Author : quanbing
# @Email : quanbinks@sina.com
# @File : metrics.py

import numpy as np
from ..sharper_utils import Utils as ut
from sklearn.metrics import roc_auc_score, roc_curve


class Metrics:
    @staticmethod
    def ks(actual, pred):
        fpr, tpr, thresholds = roc_curve(actual, pred)
        return max(abs(tpr - fpr))

    @staticmethod
    def auc(actual, pred):
        return roc_auc_score(actual, pred)

    @staticmethod
    def ar(actual, pred):
        return Metrics.gini(actual, pred) / Metrics.gini(actual, actual)

    @staticmethod
    def gini(actual, pred):
        all = np.asarray(np.c_[actual, pred, np.arange(len(actual))], dtype=np.float)
        all = all[np.lexsort((all[:, 2], -1 * all[:, 1]))]
        totalLosses = all[:, 0].sum()
        giniSum = all[:, 0].cumsum().sum() / totalLosses
        giniSum -= (len(actual) + 1) / 2.

        return giniSum / len(actual)

    @staticmethod
    def probability(target, mask=None):
        """get probability of target by mask
        """
        if mask is None:
            return 1, 1

        counts_0 = ut.np_count(target, 0, default=1)
        counts_1 = ut.np_count(target, 1, default=1)

        sub_target = target[mask]

        sub_0 = ut.np_count(sub_target, 0, default=1)
        sub_1 = ut.np_count(sub_target, 1, default=1)

        y_prob = sub_1 / counts_1
        n_prob = sub_0 / counts_0

        return y_prob, n_prob



