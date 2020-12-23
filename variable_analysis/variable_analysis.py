# -*- coding: utf-8 -*-
# @Time : 2020/12/15 6:22 下午
# @Author : quanbing
# @Email : quanbinks@sina.com
# @File : variable_analysis.py

import sys
import pandas as pd
import numpy as np
import ipywidgets as wg
from copy import deepcopy
from IPython.display import display
from ipywidgets import Layout
from ..metrics import Metrics
from ..sharper_utils import Utils as ut
from ..sharper_utils import PlotUtils as put


class VariableAnalysis:
    """
        Univariate analysis tool with the following functions：
            1. Indicator value statistics [missing rate, variance, extreme value, variable stability]
            2. Correlation between the indicator and Y label
            3. Single indicator AUC/KS calculation
            4. IV value calculation
            5. AR value calculation
            6. Single index logistic regression parameters and P value output
    """
    def __init__(self, data: pd.DataFrame, target: str) -> None:
        """
            VariableAnalysis init method

            Parameters
            ----------
            data : analyis data include target
            target: target columns

        """

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
        self._include = None
        self._exclude = None
        self._cols = data.columns.drop(target)
        self._ks = None
        self._auc = None
        self._ar = None
        self._iv = None
        self._lr_param = None

    @property
    def ori_data(self):
        """
        :return: Dataframe, Incoming raw data
        """
        return self._ori_data

    @property
    def data(self):
        """
        :return: Dataframe, Target data processed
        """
        return self._data

    @property
    def X(self):
        """
        :return: Dataframe, Set of variables
        """
        return self._X

    @property
    def y(self):
        """
        :return: Series, Set of y variable
        """
        return self._y

    @property
    def include(self):
        """
        Set which columns need to be processed, default all
        :return: array like
        """
        return self._include

    @include.setter
    def include(self, value: list):
        if not isinstance(value, list):
            raise TypeError("please give me list type")
        self._include = value
        self._update_cols()

    @property
    def exclude(self):
        """
        Set the columns that do not need to be processed, default None
        :return: array like
        """
        return self._exclude

    @exclude.setter
    def exclude(self, value: list):
        if not isinstance(value, list):
            raise TypeError("please give me list type")
        self._exclude = value
        self._update_cols()

    @property
    def distribute(self):
        """
        Draw distribution plot based on variable data

        Parameters
        ----------
        :return: {Distribution of variables}
        :sample: {'var_1': subplots, ... , 'var_n': subplots}
        """
        if self._distribute is None:
            return self.plot_distribute()
        return self._distribute

    @property
    def corr_matrix(self):
        """
        Calculate correlation matrix based on variables

        Parameters
        ----------
        :return: Variable correlation matrix
        :sample {'pearson': dataframe, 'kendall': dataframe, 'spearman': dataframe , 'mutual_info': dataframe}

        """
        if self._corr_matrix is None:
            return self._gen_corr_matrix()
        return self._corr_matrix

    @property
    def distribute_with_target(self):
        """
        Calculate correlation matrix based on X and y

        Parameters
        ----------
        :return: Distribution 2d with (var, y)
        """
        if self._distribute_with_target is None:
            return self.plot_distribute_with_target()
        return self._distribute_with_target

    @property
    def stable(self):
        """
        generate the plots which stability of variables
        :return: {'var':subplot}
        """
        if self._stable is None:
            return self.plot_stable()
        return self._stable

    @property
    def cols(self):
        """
        Set of columns to be processed
        :return: array like
        """
        return self._cols

    @property
    def univariat_ks(self):
        """
        :return: Univariate ks value collection
        """
        if self._ks is None:
            return self._gen_ks()
        return self._ks

    @property
    def univariat_auc(self):
        """
        :return: Univariate auc value collection
        """
        if self._auc is None:
            return self._gen_auc()
        return self._auc

    @property
    def univariat_ar(self):
        """
        :return: Univariate ar value collection
        """
        if self._ar is None:
            return self._gen_ar()
        return self._ar

    @property
    def univariat_iv(self):
        """
        :return: Univariate iv value collection, user sklearn dt merge in order to achieve the max iv
        """
        if self._iv is None:
            return self._gen_iv()
        return self._iv

    @property
    def univariat_lr(self):
        """
        :return: Univariate lr parame and p_value collection
        """
        if self._lr_param is None:
            self._gen_lr_param()
        return self._lr_param

    def _update_cols(self):
        """
        modify property cols
        """
        _include = self._include if self._include else []
        _exclude = self._exclude if self._exclude else []
        self._cols = ut.combin_list(self._data.columns, _include, _exclude)

    @property
    def _data_type(self):
        """
        :return: Type of data
        """
        return self._infer_dict

    @_data_type.setter
    def _data_type(self, value: dict):
        if isinstance(value, dict):
            self._infer_dict.update(value)
        else:
            raise TypeError("please input the value like {'col1': int, 'col2': float}")

    @property
    def infer_data_type(self, infer_flag=False, threshold=6):
        """
        Infer the data type according to the data range

        Parameters
        ----------
        :param infer_flag: Whether to speculate based on the variable unique, default False
        :param threshold: When infer_flag is True, it's will determine the threshold
        :return: infer data type
        """
        modify_cols = {}
        infer_dtype = deepcopy(self._infer_dict)

        if infer_flag:
            for col in self._data.columns:
                if self._data[col].nunique() < threshold:
                    modify_cols[col] = np.dtype('O')
            infer_dtype.update(modify_cols)

        display(wg.Text(
            value='Following data types have been inferred automatically, '
                  'if they are correct press enter to continue or type "quit" otherwise.',
            layout=Layout(width='100%')
        ), display_id='m1')

        display(pd.DataFrame.from_dict(infer_dtype, orient='index',columns=['infer_type']))

        if input() in ['quit', 'Quit', 'exit', 'EXIT', 'q', 'Q', 'e', 'E', 'QUIT', 'Exit']:
            sys.exit("Call the 'update_infer_type' method and "
                     "assign the dtype with dict just like {col_a: float, col_b:str}")
        else:
            self._data_type = modify_cols
            return self._infer_dict

    @property
    def modify_dtype(self):
        """
        Convert primitive data type according to infer dict
        """
        self._data = self._data.astype(self._infer_dict)
        self._X = self._data.drop(self._target, axis=1).copy()
        self._y = self._data[self._target].copy()

        return "modify succeed"

    @property
    def desc(self):
        """
        :return: Variable description
        """
        if self._desc is None:
            return self.gen_desc()
        return self._desc

    @property
    def desc_slice(self):
        """
        :return: Variable description base on slice
        """
        if self._desc_slice is None:
            return "please call the method slice_desc() first !"
        return self._desc_slice

    def update_infer_type(self, mapping: dict):
        """
        Manually modify the variable type

        Parameters
        ----------
        :param mapping: {'var_1': type_1,.., 'var_n': type_n}
        :return: Modified infer_dict
        """
        self._data_type = mapping
        return self._infer_dict

    def gen_desc(self, na: list = [], percentiles=[.01, .1, .5, .75, .9, .99]):
        """
        Generate variable statistical description

        Parameters
        ----------
        :param na: Set of values not to be counted, default = []
        :param percentiles: Percentile points to be counted, default = [.01, .1, .5, .75, .9, .99]
        :return:
        """
        rows = []

        for name, series in self._data[self._cols].items():
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

    def plot_distribute(self, save_path=None):
        """
        Draw distribution plot based on variable data

        Parameters
        ----------
        :param save_path: Path to save fig, if None ignore save
        :return:
        """
        re = {}

        for col in self._cols:
            try:
                re[col] = put.plot_distribute_hist(self._data[col], title=col)
            except:
                print("the {0} columns plot failed".format(col))

        if save_path:
            [put.save_fig(re[i], i, 'distribute') for i in re.keys()]

        self._distribute = re

        return self._distribute

    def plot_distribute_with_target(self, save_path=None):
        """
        Draw distribution plot based on X and y

        Parameters
        ----------
        :param save_path: Path to save fig, if None ignore save
        :return:
        """
        re = {}

        for col in self._cols:
            if ut.is_numeric(self._data[col]):
                re[col] = put.plot_distribute_with_target(self._data, col, target=self._target)
            else:
                re[col] = put.plot_distribute_with_target_norminal(self._data, col, target=self._target)

        if save_path:
            [put.save_fig(re[i], i, 'distribute_with_target') for i in re.keys()]

        self._distribute_with_target = re

        return self._distribute_with_target

    def slice_desc(self, by):
        """
        Variable description base on slice

        Parameters
        ----------
        :param by: date columns
        :return:
        """
        re = {}

        if self._data[by].dtype.kind != 'M':
            self._data[by] = pd.to_datetime(self._data[by])

        continues, norminal = ut.split_numeric_norminal(self._data[self._cols])

        df_ = self._data[self._cols].set_index(by).resample('M')

        _tmp = []
        for col in norminal:
            _tmp.append(df_[col].apply(ut.get_top_values))

        re['norminal'] = pd.DataFrame(_tmp).T
        re['continues'] = df_.agg(ut.get_describe)

        self._desc_slice = re

        return self._desc_slice

    def plot_stable(self):
        """
        generate the plots which stability of variables
        """
        if self._desc_slice is None:
            return "please call the method slice_desc() first !"

        re = {}
        # continues
        for col in self._desc_slice['continues'].columns:
            for var in self._desc_slice['continues'].index.to_frame()[1].unique():
                re[col] = {var: self._desc_slice['continues'].loc[(slice(None), var), :][col].
                    plot(kind='line', marker='o', markerfacecolor='r', markersize=12)}

        self._stable = re

        return self._stable

    def _gen_corr_matrix(self):
        """
        generate the corr matrix
        """
        self._corr_matrix = {'pearson': self._ori_data.corr(method='pearson'),
                             'kendall': self._ori_data.corr(method='kendall'),
                             'spearman': self._ori_data.corr(method='spearman'),
                             'mutual_info': ut.mutual_info_matrix(self._ori_data)}

        return self._corr_matrix

    def _gen_ks(self):
        """
        Single variable ks value calculation
        """
        re = {}
        for col in self._cols:
            if ut.is_numeric(self._data[col]) and col != self._target:
                re[col] = Metrics.ks(self._y, ut.normalization(self._X[col]))
        self._ks = pd.DataFrame.from_dict(re, orient='index', columns=['ks'])
        return self._ks

    def _gen_auc(self):
        """
        Single variable auc value calculation
        """
        re = {}
        for col in self._cols:
            if ut.is_numeric(self._data[col]) and col != self._target:
                re[col] = Metrics.auc(self._y, ut.normalization(self._X[col]))
        self._auc = pd.DataFrame.from_dict(re, orient='index', columns=['auc'])
        return self._auc

    def _gen_ar(self):
        """
        Single variable ar value calculation
        """
        re = {}
        for col in self._cols:
            if ut.is_numeric(self._data[col]) and col != self._target:
                re[col] = Metrics.ar(self._y, ut.normalization(self._X[col]))
        self._ar = pd.DataFrame.from_dict(re, orient='index', columns=['ar'])
        return self._ar

    def _gen_iv(self):
        """
        Single variable iv value calculation
        """
        from toad import quality
        self._iv = quality(self._data, self._target, iv_only=True)[['iv']]
        return self._iv

    def _gen_lr_param(self):
        """
        Single variable lr param and p_value calculation
        """
        from scipy import stats
        from sklearn.linear_model import LogisticRegressionCV

        re = {}
        lr_cv = LogisticRegressionCV(random_state=9527, max_iter=10000)
        for col in self._cols:

            X = np.array(self._X[col]).reshape(-1,1)
            y = self._y
            lr_cv.fit(X, y)
            params = np.append(lr_cv.intercept_, lr_cv.coef_)
            pred = lr_cv.predict(X)
            newX = pd.DataFrame({"Constant": np.ones(len(X))}).join(pd.DataFrame(X))
            MSE = (sum((y - pred) ** 2)) / (len(newX) - len(newX.columns))

            var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
            sd_b = np.round(np.sqrt(var_b), 4)
            ts_b = np.round(params / sd_b, 4)

            p_values = np.round([2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - 1))) for i in ts_b], 4)
            re[col] = {'coef_':lr_cv.coef_[0][0], 'intercept':lr_cv.intercept_[0], 'p_value':p_values}

        self._lr_param = pd.DataFrame.from_dict(re, orient='index')

        return self._lr_param
