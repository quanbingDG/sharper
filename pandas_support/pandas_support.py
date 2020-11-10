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
        if isinstance(cols, list):
            return all(map(lambda x: x in columns, cols))
        return cols in columns

    @staticmethod
    def add_ratio(df_: pd.DataFrame, cols: list, csum=False, check_flag=True):
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
        df_['{0}_cumsum'.format(cols)] = df_[cols].cumsum()
        return df_


if __name__ == '__main__':
    PS = PandasSupport()
    print(PS.add_ratio(pd.DataFrame(np.array([1, 2, 3, 4]).reshape(2, 2), columns=['i1', 'i2']), ['i1'],csum=True).columns.__len__())