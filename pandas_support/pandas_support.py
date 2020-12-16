# -*- coding: utf-8 -*-
# @Time : 2020/11/9 8:51 下午
# @Author : quanbing
# @Email : quanbinks@sina.com
# @File : pandas_support.py

import numpy as np
import pandas as pd


class PandasSupport:
    @staticmethod
    def check_cols(columns: list, cols: list):
        '''
        :param columns: list composed of str
        :param cols: list composed of str
        :return: bool， cols'element is all contained with columns
        '''
        if isinstance(cols, list):
            return all(map(lambda x: x in columns, cols))
        return cols in columns

    @staticmethod
    def add_ratio(df_: pd.DataFrame, cols: list, csum=False, check_flag=True):
        '''
        :param df_: pandas dataframe
        :param cols: list composed of str
        :param csum: whether to add cumsum
        :param check_flag: whether to check cols legaled
        :return: df_ with the ratio
        '''
        if check_flag and PandasSupport.check_cols(df_.columns, cols):
            for col in cols:
                df_ = PandasSupport.add_ratio(df_, col, csum=csum, check_flag=False)
            return df_
        else:
            _sum = df_[cols].sum()
            df_['{0}_ratio'.format(cols)] = df_[cols] / _sum
            if csum:
                df_ = PandasSupport.add_csum(df_, '{0}_ratio'.format(cols))
            return df_

    @staticmethod
    def add_csum(df_: pd.DataFrame, cols: str):
        '''
        :param df_: pandas dataframe
        :param cols: list composed of str
        :return: df_ with the cumsum
        '''
        df_['{0}_cumsum'.format(cols)] = df_[cols].cumsum()
        return df_

    @staticmethod
    def type_infer(df_: pd.DataFrame, col=None):
        '''
        :param df_: pandas dataframe
        :param col: str
        :return: dict of dataframe's columns types
        '''
        re_ = {}
        for i in df_.columns:
            re_[i] = pd.api.types.infer_dtype(df_[i], skipna=True)
        return re_[col] if col else re_

    @staticmethod
    def add_sum(df_: pd.DataFrame, axis=0):
        '''
        :param df_: pandas dataframe
        :param axis: default 0, support 1
        :return:  new dataframe with sum
        '''
        sum_list = []
        for i in df_.columns:
            try: sum_list.append(sum(df_[i]))
            except: sum_list.append(np.nan)
        df_.loc['sum'] = sum_list
        return df_


if __name__ == '__main__':
    PS = PandasSupport()
    a = pd.DataFrame(np.array([1, 2, 3, 4]).reshape(2, 2), columns=['i1', 'i2'])
    b = a.pipe(PS.add_ratio, cols=['i1', 'i2'], csum=True)
    print(b.pipe(PS.add_sum, axis=0))
    print(b.pipe(PS.type_infer))
    print(PS.type_infer(a))