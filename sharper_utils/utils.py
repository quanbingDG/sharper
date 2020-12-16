# -*- coding: utf-8 -*-
# @Time : 2020/12/15 6:59 下午
# @Author : quanbing
# @Email : quanbinks@sina.com
# @File : utils.py

from pandas import Series


class Utils:
    @staticmethod
    def is_numeric(series):
        """
        :param series: pandas.series
        :return: boolean
        """
        return series.dtype.kind in 'ifc'

    @staticmethod
    def get_describe(series, percentiles=[.25, .5, .75]):
        """
        :param series: pandas.series
        :param percentiles: percentiles list
        :return:
        """
        d = series.describe(percentiles)
        return d.drop('count')

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
    def count_null(series):
        """
        :param series: pandas series
        :return: tuple with (num of null, ratio of null)
        """
        n = series.isnull().sum()
        return (n, "{0:.2%}".format(n / series.size))
