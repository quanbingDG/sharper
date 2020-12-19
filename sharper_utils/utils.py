# -*- coding: utf-8 -*-
# @Time : 2020/12/15 6:59 下午
# @Author : quanbing
# @Email : quanbinks@sina.com
# @File : utils.py

from pandas import Series
import pandas as pd
import numpy as np
from tqdm import tqdm
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

    @staticmethod
    def diff_dict(a:dict, b: dict, re_type='key'):
        """
        :param a: dict a
        :param b: dict b
        :param re_type: {key: "default key:return diff key list", all: "return diff element"}
        :return: list
        """
        if re_type == 'all':
            return set(a.items()) ^ set(b.items())
        return [i for i in a.keys() if a.get(i, None) != b.get(i, None)]

    @staticmethod
    def minus_list(a: list, b: list) -> list:
        return [i for i in a if i not in b]

    @staticmethod
    def combin_list(a: list, include: list =[], exclude: list =[]):
        re = set(a) if include.__len__() == 0 else set(include)
        re = list(re - set(exclude))
        return re

    @staticmethod
    def mutual_info_matrix(df_: pd.DataFrame):
        re = pd.DataFrame(index=df_.columns)

        for i in df_.columns:
            re[i] = [normalized_mutual_info_score(df_[i], df_[j]) for j in df_.columns]

        return re

