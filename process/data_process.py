# -*- coding: utf-8 -*-
# @Time : 2021/1/9 10:34 下午
# @Author : quanbing
# @Email : quanbinks@sina.com
# @File : data_process.py

from pandas import Series


class DataProcess:
    @staticmethod
    def cut_head(series: Series, per=.01):
        return series[series >= series.quantile(per)]

    @staticmethod
    def cut_tail(series: Series, per=.99):
        return series[series <= series.quantile(per)]

    @staticmethod
    def cut_both(series: Series, hper=.01, tper=.99):
        return series[(series <= series.quantile(tper)) & (series >= series.quantile(hper))]

    @staticmethod
    def exclude():
        pass

    @staticmethod
    def cut_outer(k=1.5):
        pass
