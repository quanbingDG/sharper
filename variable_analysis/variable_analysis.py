# -*- coding: utf-8 -*-
# @Time : 2020/12/15 6:22 下午
# @Author : quanbing
# @Email : quanbinks@sina.com
# @File : variable_analysis.py


import pandas as pd
from sharper_utils import Utils as ut
from sharper_utils import PlotUtils as put


class VariableAnalysis:
    @staticmethod
    def statistical(df_: pd.DataFrame, ignore: list = []):
        '''
        :param df_: dataframe
        :param ignore: special value need be exclude
        :return: a dataframe for the statistical of df_
        '''
        rows = []
        for name, series in df_.items():
            series = series[-series.isin(ignore)]
            numeric_index = ['mean', 'std', 'min', '1%', '10%', '50%', '75%', '90%', '99%', 'max']
            discrete_index = ['top1', 'top2', 'top3', 'top4', 'top5', 'bottom5', 'bottom4', 'bottom3', 'bottom2',
                              'bottom1']

            details_index = [numeric_index[i] + '_or_' + discrete_index[i] for i in range(len(numeric_index))]
            details = []

            if ut.is_numeric(series):
                desc = ut.get_describe(
                    series,
                    percentiles=[.01, .1, .5, .75, .9, .99]
                )
                details = desc.tolist()
            else:
                top5 = ut.get_top_values(series)
                bottom5 = ut.get_top_values(series, reverse=True)
                details = top5.tolist() + bottom5[::-1].tolist()

            nblank, pblank = ut.count_null(series)

            row = pd.Series(
                index=['type', 'size', 'missing', 'unique'] + details_index,
                data=[series.dtype, series.size, pblank, series.nunique()] + details
            )

            row.name = name
            rows.append(row)

        return pd.DataFrame(rows)

    @staticmethod
    def plot_distribute(df_, include=None, exclude=None):
        pass


if __name__ == '__main__':
    data = pd.read_csv(r'/Users/quanbing/Downloads/workspace/competition/binary classification/交通事故理赔审核/data/'
                       r'train.csv', index_col=[0])
    import random
    import numpy as np
    data['性别'] = np.array([random.choice(['男', '女', '女', '女', '跨性别']) for i in range(data.shape[0])])
    print(VariableAnalysis.statistical(data))