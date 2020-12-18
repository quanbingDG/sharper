# -*- coding: utf-8 -*-
# @Time : 2020/12/15 6:59 下午
# @Author : quanbing
# @Email : quanbinks@sina.com
# @File : utils.py

from pandas import Series
import pandas as pd
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


class Utils:
    @staticmethod
    def is_numeric(series):
        """
        :param series: pandas.series
        :return: boolean
        """
        return series.dtype.kind in 'ifc'

    @staticmethod
    def get_describe(series, percentiles=[.25, .5, .75], drop_count=False):
        """
        :param series: pandas.series
        :param percentiles: percentiles list
        :return:
        """
        d = series.describe(percentiles)
        return d.drop('count') if drop_count else d

    @staticmethod
    def get_top_values(series, top=5, reverse=False):
        """
        :param series: pandas series
        :param top: num of top
        :param reverse: bottom
        :return:
        """
        v_type = 'top'
        counts = series.value_counts()
        counts = list(zip(counts.index, counts, counts.divide(series.size)))

        if reverse:
            counts.reverse()
            v_type = 'bottom'

        template = "{0[0]}:{0[2]:.2%}"
        indexs = [v_type + str(i + 1) for i in range(top)]
        values = [template.format(counts[i]) if i < len(counts) else None for i in range(top)]

        return Series(values, index=indexs)

    @staticmethod
    def count_null(series, re_all=False):
        """
        :param re_all:
        :param series: pandas series
        :return: tuple with (num of null, ratio of null)
        """
        n = series.isnull().sum()
        return (n, "{0:.2%}".format(n / series.size)) if re_all else (n, "{0:.2%}".format(n / series.size))[1]

    @staticmethod
    def dtype_infer(series):
        return pd.api.types.infer_dtype(series)

    @staticmethod
    def cat_infer(series, n=6):
        if series.nunique() < n:
            return series.astype('O')
        return series

    @staticmethod
    def split_numeric_norminal(df_):
        tmp = df_.dtypes
        continues = tmp[-(tmp == np.dtype('O'))].index.to_list()
        norminal = tmp[tmp == np.dtype('O')].index.to_list()
        return continues, norminal

    @staticmethod
    def csi():
        pass

    @staticmethod
    def ks(x, y, d_type):
        pass

    @staticmethod
    def auc(x, y, d_type):
        pass
