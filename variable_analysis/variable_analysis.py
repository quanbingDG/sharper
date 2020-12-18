# -*- coding: utf-8 -*-
# @Time : 2020/12/15 6:22 下午
# @Author : quanbing
# @Email : quanbinks@sina.com
# @File : variable_analysis.py


import pandas as pd
import matplotlib.pyplot as plt
from ..sharper_utils import Utils as ut
from ..sharper_utils import PlotUtils as put


class VariableAnalysis:
    def __init__(self):
        pass

    @staticmethod
    def statistical(df_: pd.DataFrame, ignore: list = [], infer_cat=False):
        '''
        :param infer_type:
        :param df_: dataframe
        :param ignore: special value need be exclude
        :return: a dataframe for the statistical of df_
        '''
        rows = []
        for name, series in df_.items():
            if infer_cat:
                series = ut.cat_infer(series)

            series = series[-series.isin(ignore)]
            numeric_index = ['mean', 'std', 'min', '1%', '10%', '50%', '75%', '90%', '99%', 'max']
            discrete_index = ['top1', 'top2', 'top3', 'top4', 'top5', 'bottom5', 'bottom4', 'bottom3', 'bottom2',
                              'bottom1']

            details_index = [numeric_index[i] + '_or_' + discrete_index[i] for i in range(len(numeric_index))]
            details = []

            if ut.is_numeric(series):
                desc = ut.get_describe(
                    series,
                    percentiles=[.01, .1, .5, .75, .9, .99],
                    drop_count=True
                )
                details = desc.tolist()
            else:
                top5 = ut.get_top_values(series)
                bottom5 = ut.get_top_values(series, reverse=True)
                details = top5.tolist() + bottom5[::-1].tolist()

            nblank, pblank = ut.count_null(series, re_all=True)

            row = pd.Series(
                index=['type', 'size', 'missing', 'unique'] + details_index,
                data=[series.dtype, series.size, pblank, series.nunique()] + details
            )

            row.name = name
            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def plot_distribute(df_, include=[], exclude=[], save_path=None):
        re_ = {}
        plot_cols = set(df_.columns) if include.__len__() == 0 else set(include)
        plot_cols = list(plot_cols - set(exclude))

        for col in plot_cols:
            re_[col] = put.plot_distribute_hist(df_[col], title=col)

        if save_path:
            [put.save_fig(re_[i], i, 'distribute') for i in re_.keys()]

    @staticmethod
    def plot_distribute_with_target(df_, target, include=[], exclude=[], save_path=None):
        re_ = {}
        plot_cols = set(df_.columns) if include.__len__() == 0 else set(include)
        plot_cols = list(plot_cols - set(exclude))

        for col in plot_cols:
            if ut.is_numeric(df_[col]):
                re_[col] = put.plot_distribute_with_target(df_, col, target=target)
            else:
                re_[col] = put.plot_distribute_with_target_norminal(df_, col, target=target)

        if save_path:
            [put.save_fig(re_[i], i, 'distribute_with_target') for i in re_.keys()]

    @staticmethod
    def statistical_slice(df_, by, cols='all', infer_cat=False):
        if cols != 'all':
            df_ = df_[cols]

        if infer_cat:
            df_ = df_.apply(ut.cat_infer)

        if df_[by].dtype.kind != 'M':
            df_[by] = pd.to_datetime(df_[by])

        continues, norminal = ut.split_numeric_norminal(df_)

        df_ = df_.set_index(by).resample('M')

        _tmp = []
        for col in norminal:
            _tmp.append(df_[col].apply(ut.get_top_values))
        return df_.agg(ut.get_describe), pd.DataFrame(_tmp).T

    @staticmethod
    def plot_stable(df_, by, cols='all', infer_cat=False):
        """丑陋实现，亟待优化"""
        if cols != 'all':
            df_ = df_[cols]

        re_ = VariableAnalysis.statistical_slice(df_, by, cols, infer_cat=infer_cat)
        re = []
        for i in re_:
            for j in i.columns:
                for col in i.index.to_frame()[1].unique():
                    plt.figure(figsize=(9, 6))
                    try:
                        re.append(i.loc[(slice(None), col), :][j].plot(kind='line', marker='o',
                                                                       markerfacecolor='r', markersize=12))
                    except:
                        print(i,j,col)
                        print(i.loc[(slice(None), col), :][j])
                        raise SystemError
                    plt.close()

        return re

    @staticmethod
    def correlation_y(df_, target, x='all', method='pearson'):
        """
        :param df_:
        :param target:
        :param x:
        :param method: {'pearson', 'kendall', 'spearman'}, default:pearson
        :return:
        """
        if df_ != 'all':
            df_ = df_[x]
        return df_.corr(method=method)[target].to_dict()

    @staticmethod
    def mutual_information_y(df_, target, x='all'):
        if df_ != 'all':
            df_ = df_[x]
        return dict([(i, normalized_mutual_info_score(df_[i], df_[target])) for i in df_.columns])

    @staticmethod
    def calc_auc():
        pass

    @staticmethod
    def calc_ks():
        pass



