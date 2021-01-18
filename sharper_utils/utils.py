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

    LEGAL_TYPE = ['Categorical', 'Continues', 'Ordinal', 'Datetime', 'Other']

    @staticmethod
    def is_numeric(series):
        """
        Determine whether it is numeric
        :param series: pandas.series
        :return: boolean
        """
        return series.dtype.kind in 'iufc'

    @staticmethod
    def is_ordinal(series: Series):
        """
        Determine whether it is ordinal
        :param series:
        :return:
        """
        if pd.Series(sorted(series.dropna().unique())).diff().dropna().nunique() == 1:
            return True
        else:
            return False

    @staticmethod
    def is_categorical(series: Series):
        return series.dtype.kind in 'OUSb'

    @staticmethod
    def is_datetime(series: Series):
        return series.dtype.kind == 'M'

    @staticmethod
    def is_legal_type(l: list):
        return all(map(lambda x: x in Utils.LEGAL_TYPE, l))

    @staticmethod
    def convert_type(series: Series, types: str):
        if types == 'Categorical':
            return series.astype(str)
        elif not Utils.is_numeric(series) and types in ['Ordinal', 'Continues']:
            return series.astype(float)
        elif types == 'Datetime':
            return pd.datetime(series)
        else:
            return series

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
        n = series.isnull().sum() + (series == 'nan').sum()
        return (n, "{0:.2%}".format(n / series.size)) if re_all else (n, "{0:.2%}".format(n / series.size))[1]

    @staticmethod
    def dtype_infer(series: Series):
        return pd.api.types.infer_dtype(series)

    @staticmethod
    def data_type_classifier(series: Series):
        if Utils.is_categorical(series):
            return 'Categorical'
        elif Utils.is_numeric(series):
            if Utils.is_ordinal(series):
                return 'Ordinal'
            else:
                return 'Continues'
        elif Utils.is_datetime(series):
            return 'Datetime'
        else:
            return 'Other'

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
        p_bar = tqdm(df_.columns)

        for i in p_bar:
            p_bar.set_description("Calc mutual_info, var %s" % i)
            re[i] = [normalized_mutual_info_score(df_[i], df_[j]) for j in df_.columns]

        return re

    @staticmethod
    def normalization(series: pd.Series):
        _max = series.max()
        _min = series.min()
        return pd.Series([(i-_min) / (_max - _min) for i in series])

    @staticmethod
    def np_count(series, value, default=None):
        count = (series == value).sum()

        if default is not None and count == 0:
            return default

        return count

    @staticmethod
    def flat_dict(x: dict):
        for key, value in x.items():
            if isinstance(value, dict):
                for k, v in Utils.flat_dict(value):
                    k = f'{key}_{k}'
                    yield k, v
            else:
                yield key, value

    @staticmethod
    def infer_float_round(series: Series, num=2):
        if series.dtype.kind == 'f':
            return series.apply(lambda x: round(x, num))
        return series
