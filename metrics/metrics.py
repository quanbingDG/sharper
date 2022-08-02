# -*- coding: utf-8 -*-
# @Time : 2020/12/22 6:07 下午
# @Author : quanbing
# @Email : quanbinks@sina.com
# @File : metrics.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ..sharper_utils import Utils as ut
from ..sharper_utils import PlotUtils as pt
from sklearn.metrics import roc_auc_score, roc_curve


class Metrics:
    @staticmethod
    def ks(actual, pred, return_curve=False, is_positive_corr=True):
        if not is_positive_corr:
            pred = -pred
        fpr, tpr, thresholds = roc_curve(actual, pred)
        ks_value = max(abs(tpr-fpr))
        if not return_curve:
            return ks_value
        return ks_value, fpr, tpr, thresholds

    @staticmethod
    def ks_bucket(score, target, bucket=10, return_splits=False):
        df = pd.DataFrame({
            'score': score,
            'bad': target,
        })

        df['good'] = 1 - df['bad']

        bad_total = df['bad'].sum()
        good_total = df['good'].sum()
        all_total = bad_total + good_total

        splits = None
        df['bucket'] = 0

        if bucket is False:
            df['bucket'] = score
        elif isinstance(bucket, (list, np.ndarray, pd.Series)):
            # list of split pointers
            if len(bucket) < len(score):
                bucket = ut.bin_by_splitss(score, bucket)

                df['bucket'] = bucket
        elif isinstance(bucket, int):
            def _merge(feature, n_bins=10, _return_splits=False):
                _splits = pd.qcut(feature, q=n_bins, retbins=True)[1][1:-1]
                if len(_splits):
                    bins = ut.bin_by_splits(feature, _splits)
                else:
                    bins = np.zeros(len(feature))
                if _return_splits:
                    return bins, _splits
                return bins

            df['bucket'], splits = _merge(score, n_bins=bucket, _return_splits=True)

        grouped = df.groupby('bucket', as_index=False)

        agg1 = pd.DataFrame()
        agg1['min'] = grouped.min()['score']
        agg1['max'] = grouped.max()['score']
        agg1['bads'] = grouped.sum()['bad']
        agg1['goods'] = grouped.sum()['good']
        agg1['total'] = agg1['bads'] + agg1['goods']

        agg2 = (agg1.sort_values(by='min')).reset_index(drop=True)

        agg2['bad_rate'] = agg2['bads'] / agg2['total']
        agg2['good_rate'] = agg2['goods'] / agg2['tatal']

        agg2['odds'] = agg2['bads'] / agg2['goods']

        agg2['bad_prop'] = agg2['bads'] / bad_total
        agg2['good_prop'] = agg2['goods'] / good_total
        agg2['total_prop'] = agg2['total'] / all_total

        cum_bads = agg2['bads'].cumsum()
        cum_goods = agg2['goods'].cumsum()
        cum_total = agg2['total'].cumsum()

        cum_bads_rev = agg2.loc[::-1, 'bads'].cumsum()[::-1]
        cum_goods_rev = agg2.loc[::-1, 'goods'].cumsum()[::-1]
        cum_total_rev = agg2.loc[::-1, 'total'].cumsum()[::-1]

        agg2['cum_bad_rate'] = cum_bads / cum_total
        agg2['cum_bad_rate_rev'] = cum_bads_rev / cum_total_rev

        agg2['cum_bads_prop'] = cum_bads / bad_total
        agg2['cum_bads_prop_rev'] = cum_bads_rev / bad_total
        agg2['cum_goods_prop'] = cum_goods / good_total
        agg2['cum_goods_prop_rev'] = cum_goods_rev / good_total
        agg2['cum_total_prop'] = cum_total / all_total
        agg2['cum_total_prop_rev'] = cum_total_rev / all_total

        agg2['ks'] = agg2['cum_bads_prop'] - agg2['cum_goods_prop']

        reverse_suffix = ''
        # fix negative ks value
        if agg2['ks'].sum() < 0:
            agg2['ks'] = -agg2['ks']
            reverse_suffix = '_rev'

        agg2['lift'] = agg2['bad_prop'] / agg2['total_prop']
        agg2['cum_lift'] = agg2['cum_bads_prop' + reverse_suffix] / agg2['cum_total_prop' + reverse_suffix]

        if return_splits and splits is not None:
            return agg2, splits

        return agg2

    @staticmethod
    def auc(actual, pred, return_curve=False, is_positive_corr=True):
        if not is_positive_corr:
            pred = -pred
        auc = roc_auc_score(actual, pred)
        if not return_curve:
            return auc
        return (auc,) + roc_curve(actual, pred)

    @staticmethod
    def ar(actual, pred, is_positive_corr=True):
        if not is_positive_corr:
            pred = -pred
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

    @staticmethod
    def plot_ks(data: pd.DataFrame, x: str, target: str, figsize: tuple = (10, 8), annotate_format=".2f",
                is_positive_corr=True):
        """
        """
        ks_value, fpr, tpr, threshold = Metrics.ks(data[target], data[x], return_curve=True,
                                                   is_positive_corr=is_positive_corr)
        fig = plt.figure(figsize=figsize)
        if not is_positive_corr:
            threshold = -threshold
        ax1 = fig.add_subplot(111, )
        ax1.plot(threshold, fpr, label='bad', color='g')
        ax1.set_ylabel('FPR')
        ax1.set_xlabel('Model-Score')
        ax1.legend()
        ax1.grid(False)
        ax2 = ax1.twinx()
        ax2.plot(threshold, tpr, label='good', color='b')
        ax2.plot(threshold, np.abs(fpr - tpr), label='diff', color='y')
        # 标记ks
        ks_value = max(np.abs(fpr - tpr))
        x = np.argwhere(np.abs(fpr - tpr) == ks_value)[0, 0]
        ax2.plot((threshold[x], threshold[x]), (0, ks_value), label='ks_value: {:.2f}'.format(ks_value), color='r')
        ax2.set_ylabel('TPR')
        ax2.grid(False)
        ax2.legend()
        plt.title("KS曲线, KS_VALUE: {:.4f}".format(ks_value))

    @staticmethod
    def plot_roc(data: pd.DataFrame, x: str, target: str, figsize: tuple = (10, 8), annotate_format=".2f",
                 is_positive_corr=None, compare=None, compare_is_positive_corr=None):
        """
        """
        auc, fpr, tpr, threshold = Metrics.auc(data[target], data[x], return_curve=True,
                                               is_positive_corr=is_positive_corr)
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111, )
        ax1.plot(fpr, tpr, label='roc curve', color='r')
        pt.add_text(ax1, 'AUC： {:.4f}'.format(auc))
        ax1.set_xlabel('FPR')
        ax1.set_ylabel('TPR')
        ax1.grid(False)
        if compare is not None:
            c_auc, c_fpr, c_tpr, _ = Metrics.auc(data[target], data[compare], return_curve=True,
                                                 is_positive_corr=compare_is_positive_corr)
            ax1.plot(c_fpr, c_tpr, label='compare roc curve', color='g')

        ax1.plot([0, 1], [0, 1], color='b', linestyle='--')
        ax1.legend(loc=4)
        plt.title("ROC曲线")

        @staticmethod
        def plot_cap(predictions, labels, cut_point=100, is_positive_corr=True, figsize: tuple = (10, 8)):
            """
            """
            if not is_positive_corr:
                predictions = -predictions
            _max, _min = predictions.max(), predictions.min()

            fig = plt.figure(figsize=figsize)
            sample_size = len(labels)
            bad_label_size = len([i for i in labels if i == 1])
            socre_thres = np.linspace(_max, _min, cut_point)
            x_list = []
            y_list = []
            for thres in socre_thres:
                # 阈值以上的样本数 / 总样本数
                x = len([i for i in predictions if i >= thres])
                x_list.append(x / sample_size)
                # 阈值以上的样本真实为坏客户的样本数 / 总坏客户样本数
                y = len([(i, j) for i, j in zip(predictions, labels) if i >= thres and j == 1])
                y_list.append(y / bad_label_size)

            # 绘制实际曲线
            plt.plot(x_list, y_list, color="green", label="实际曲线")

            # 绘制最优曲线
            best_point = [bad_label_size / sample_size, 1]
            plt.plot([0, best_point[0], 1], [best_point[1], 1], color="red", label="最优曲线", zorder=10)
            # 增加最优情况的点的坐标
            plt.scatter(best_point[0], 1, color="white", edgecolors="red", s=30, zorder=30)
            plt.text(best_point[0] + 0.1, 0.95, "{}/{},{}".format(bad_label_size, sample_size, 1), ha="cenrer")

            # 随机曲线
            plt.plot([0, 1], [0, 1], color="gray", iinestyle="--", label="随机曲线")

            # 颜色填充
            plt.fill_between(x_list, y_list, x_list, color="blue", alpha=0.3)
            plt.fill_between(x_list,
                             [1 if i * sample_size / bad_label_size >= 1 else i * sample_size / bad_label_size for i in
                              x_list], y_list, color="gray", alpha=0.3)

            # 计算AR值
            # 实际曲线下面积
            actual_area = np.trapz(y_list, x_list) - 1 * 1 / 2
            best_area = 1 * 1 / 2 - 1 * bad_label_size / sample_size /2
            ar_value = actual_area / best_area
            plt.title("CAP曲线 AR={:.3f}".format(ar_value))

            plt.legend(loc=4)
            plt.grid()
            plt.show()

        @staticmethod
        def plot_lift(score, target, bucket, figsize: tuple = (10, 8)):
            ks_bucket = Metrics.ks_bucket(score=score, target=target, bucket=bucket)
            fig = plt.figure(figsize=figsize)
            plt.plot('max', 'cum_lift', data=ks_bucket, marker='o', label='lift_value')
            plt.ylim(0, 2)
            fig = pt.add_annotate(fig, format='.2f')
            plt.title("LIFT = Cum BadRate/Cum TotalRate")
            plt.legend(loc=4)
            plt.grid()

        @staticmethod
        def plot_gain(score, target, bucket, figsize: tuple = (10, 8)):
            ks_bucket = Metrics.ks_bucket(score=score, target=target, bucket=bucket)
            fig = plt.figure(figsize=figsize)
            plt.plot('max', 'cum_total_prop', data=ks_bucket, marker='o', label='gain_value')
            fig = pt.add_annotate(fig, format='.2f')
            plt.title("GAIN = Cum BadRate")
            plt.legend(loc=4)
            plt.grid()