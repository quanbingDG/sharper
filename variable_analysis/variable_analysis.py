# -*- coding: utf-8 -*-
# @Time : 2020/12/15 6:22 下午
# @Author : quanbing
# @Email : quanbinks@sina.com
# @File : variable_analysis.py


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ..sharper_utils import Utils as ut
from ..sharper_utils import PlotUtils as put


class VariableAnalysis:
    def __init__(self, data: pd.DataFrame, target: str) -> None:
        self._ori_data = data
        self._target = target
        self._data = data.copy()
        self._X = data.drop(target, axis=1).copy()
        self._y = data[target].copy()
        self._infer_dict = data.dtypes.to_dict()
        self._desc = None
        self._desc_slice = None
        self._distribute = None
        self._distribute_with_target = None
        self._stable = None
        self._corr_matrix = None

    @property
    def distribute(self):
        if self._distribute is None:
            return self.plot_distribute()
        return self._distribute

    @property
    def corr_matrix(self):
        if self._corr_matrix is None:
            return self._gen_corr_matrix()
        return self._corr_matrix

    @property
    def distribute_with_target(self):
        if self._distribute_with_target is None:
            return self.plot_distribute_with_target()
        return self._distribute_with_target

    @property
    def stable(self):
        if self._stable is None:
            return self.plot_stable()
        return self._stable

    @property
    def ori_data(self):
        return self._ori_data

    @property
    def data(self):
        return self._data

    @property
    def X(self):
        return self._X

    @property
    def y(self):
        return self._y

    @property
    def _data_type(self):
        return self._infer_dict

    @_data_type.setter
    def _data_type(self, value: dict):
        if isinstance(value, dict):
            self._infer_dict.update(value)
        else:
            raise TypeError("please input the value like {'col1': int, 'col2': float}")

    @property
    def infer_data_type(self, infer_norminal=True, threshold=6):
        modify_cols = {}
        if infer_norminal:
            for col in self._data.columns:
                if self._data[col].nunique() < threshold:
                    modify_cols[col] = np.dtype('O')

        self._data_type = modify_cols
        return self._infer_dict

    @property
    def modify_dtype(self):
        self._data = self._data.astype(self._infer_dict)
        self._X = self._data.drop(self._target, axis=1).copy()
        self._y = self._data[self._target].copy()

        return "modify succeed"

    @property
    def desc(self):
        if self._desc is None:
            return self.gen_desc()
        return self._desc

    @property
    def desc_slice(self):
        if self._desc_slice is None:
            return "please call the method slice_desc() first !"
        return self._desc_slice

    def gen_desc(self, exclude: list = None, na: list = [np.nan], percentiles=[.01, .1, .5, .75, .9, .99]):
        rows = []
        cols = self._data.columns.to_list()

        if exclude:
            cols = ut.minus_list(cols, exclude)

        for name, series in self._data[cols].items():
            series = series[-series.isin(na)]
            numeric_index = ['mean', 'std', 'min', '1%', '10%', '50%', '75%', '90%', '99%', 'max']
            discrete_index = ['top1', 'top2', 'top3', 'top4', 'top5', 'bottom5', 'bottom4', 'bottom3', 'bottom2',
                              'bottom1']

            details_index = [numeric_index[i] + '_or_' + discrete_index[i] for i in range(len(numeric_index))]
            details = []

            if ut.is_numeric(series):
                desc = ut.get_describe(
                    series,
                    percentiles=percentiles,
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

        self._desc = pd.DataFrame(rows)

        return self._desc

    def plot_distribute(self, include=[], exclude=[], save_path=None):
        re = {}
        plot_cols = ut.combin_list(self._data.columns, include=include, exclude=exclude)

        for col in plot_cols:
            try:
                re[col] = put.plot_distribute_hist(self._data[col], title=col)
            except:
                print("the {0} columns plot failed".format(col))

        if save_path:
            [put.save_fig(re[i], i, 'distribute') for i in re.keys()]

        self._distribute = re

        return self._distribute

    def plot_distribute_with_target(self, include=[], exclude=[], save_path=None):
        re = {}
        plot_cols = ut.combin_list(self._data.columns, include=include, exclude=exclude)

        for col in plot_cols:
            if ut.is_numeric(self._data[col]):
                re[col] = put.plot_distribute_with_target(self._data, col, target=self._target)
            else:
                re[col] = put.plot_distribute_with_target_norminal(self._data, col, target=self._target)

        if save_path:
            [put.save_fig(re[i], i, 'distribute_with_target') for i in re.keys()]

        self._distribute_with_target = re

        return self._distribute_with_target

    def slice_desc(self, by, include=[], exclude=[]):
        re = {}
        cols = ut.combin_list(self._data.columns, include=include, exclude=exclude)

        if self._data[by].dtype.kind != 'M':
            self._data[by] = pd.to_datetime(self._data[by])

        continues, norminal = ut.split_numeric_norminal(self._data[cols])

        df_ = self._data[cols].set_index(by).resample('M')

        _tmp = []
        for col in norminal:
            _tmp.append(df_[col].apply(ut.get_top_values))

        re['norminal'] = pd.DataFrame(_tmp).T
        re['continues'] = df_.agg(ut.get_describe)

        self._desc_slice = re

        return self._desc_slice

    def plot_stable(self, include=[], exclude=[]):
        if self._desc_slice is None:
            return "please call the method slice_desc() first !"

        # plot_cols = ut.combin_list(self._data.columns, include=include, exclude=exclude)

        re = {}
        # continues
        for col in self._desc_slice['continues'].columns:
            for var in self._desc_slice['continues'].index.to_frame()[1].unique():
                re[col] = {var: self._desc_slice['continues'].loc[(slice(None), var), :][col].
                    plot(kind='line', marker='o', markerfacecolor='r', markersize=12)}

        self._stable = re

        return self._stable

    def _gen_corr_matrix(self):
        self._corr_matrix = {'pearson': self._data.corr(method='pearson'),
                             'kendall': self._data.corr(method='kendall'),
                             'spearman': self._data.corr(method='spearman'),
                             'mutual_info': ut.mutual_info_matrix(self._data)}

        return self._corr_matrix
