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
from tqdm import tqdm
from IPython.display import display
from ipywidgets import Layout
from ..metrics import Metrics
from ..sharper_utils import Utils as ut
from ..sharper_utils import PlotUtils as put
from ..pandas_support import PandasSupport as ps


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
        self._dtypes_dict = data.dtypes.to_dict()
        self._infer_dict = None
        self._desc = None
        self._desc_slice = None
        self._distribute = None
        self._distribute_with_target = None
        self._distribute_group_target = None
        self._distribute_badrate = None
        self._stable = None
        self._corr_matrix = None
        self._include = None
        self._exclude = None
        self._cols = data.columns.drop(target)
        self._fill_continuous = -9999
        self._fill_nominal = 'X'
        self._plot_bins = None
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
    def fill_na(self):
        """
        """
        if self._infer_dict is None:
            print(" please use the 'data_type_infer' property to conform datatype")
            return self._infer_dict
        else:
            for ind, ind_type in self._infer_dict.items():
                if ind_type == "Categorical":
                    self._data[ind].replace({-9999: 'X', np.nan: 'X', 'Null': 'X'}, inplace=True)
                elif ind_type == 'Ordinal':
                    self._data[ind].replace({-9999: 'X', np.nan: 'X'}, inplace=True)
                elif ind_type == 'Continues':
                    self._data[ind].replace({np.nan: -9999}, inplace=True)
            print("fill finished")

    @property
    def data(self):
        """
        :return: Dataframe, Target data processed
        """
        return self._data

    @property
    def plot_bins(self):
        """
        """
        if self._plot_bins is None:
            print("you should set the first, like plot_bins = dict(xx)")
        return self._plot_bins

    @plot_bins.setter
    def plot_bins(self, value):
        """
        """
        if not isinstance(value, dict):
            raise TypeError("up")
        self._plot_bins = value

    @property
    def fill_nominal(self):
        """
        """
        return self._fill_nominal

    @fill_nominal.setter
    def fill_nominal(self, value):
        """
        """
        if not isinstance(value, str):
            raise TypeError("up")
        self._fill_nominal = value
        return f"modify finished, fill_nominal value is {self._fill_nominal} now"

    @property
    def fill_continuous(self):
        """
        """
        return self._fill_continuous

    @fill_continuous.setter
    def fill_continuous(self, value):
        """
        """
        if not isinstance(value, str):
            raise TypeError("up")
        self._fill_continuous = value
        print(f"modify finished, fill_continuous value is {self._fill_continuous} now")

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
    def X(self):
        return self._data[self._cols].copy()

    @property
    def y(self):
        return self._data[self._target].copy()

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
        print('Use command line to show the figure: ".distribute.get("var").figure" ')

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
        print('Use command line to show the figure: ".distribute_with_target.get("var").fig" ')

        return self._distribute_with_target

    @property
    def distribute_group_target(self):
        """
        Calculate correlation matrix based on X and y with grouped data

        Parameters
        ----------
        :return: Distribution 2d with (var, y)
        """
        if self._distribute_group_target is None:
            return self.plot_distribute_group_target()
        print('Use command line to show the figure: ".distribute_group_target.get("var").fig" ')

        return self._distribute_group_target

    @property
    def distribute_badrate(self):
        """
        Calculate correlation matrix based on X and y with grouped data

        Parameters
        ----------
        :return: Distribution 2d with (var, y)
        """
        if self.plot_bins is None:
            raise TypeError("you should set this first, like plot_bins = dict(xx)")

        if self._distribute_badrate is None:
            return self.plot_distribute_badrate()
        print('Use command line to show the figure: ".distribute_badrate.get("var").fig" ')

        return self._distribute_badrate

    @property
    def stable(self):
        """
        generate the plots which stability of variables
        :return: {'var':subplot}
        """
        if self._stable is None:
            return f"please call the fuction .plot_stable() firsr!"
        print('Use command line to show the figure: ".stable.get("var").fig" ')

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

    @property
    def _data_type(self):
        """
        :return: Type of data
        """
        if self._infer_dict is None:
            print("please use the 'data_type_infer' property to conform datatype")
        return self._infer_dict

    @_data_type.setter
    def _data_type(self, value: dict):
        if isinstance(value, dict):
            self._infer_dict = value
        else:
            raise TypeError("please input the value like {'col1': int, 'col2': float}")

    @property
    def data_type_infer(self, infer_flag=True):
        """
        Infer the data type according to the data range

        Parameters
        ----------
        :param infer_flag: Whether to speculate based on the variable unique, default False
        :return: infer data type
        """
        modify_cols = {}
        infer_dtype = deepcopy(self._dtypes_dict)

        if infer_flag:
            for col in self._data.columns:
                modify_cols[col] = ut.data_type_classifier(self._data[col])
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
    def data_type_convert(self):
        """
        Convert primitive data type according to set data type
        """
        data = self._data.copy()
        for col, types in self._data_type.items():
            data[col] = ut.convert_type(data[col], types)
        self.__modify_data(data)

        return "Modify successfully"

    @property
    def check_balance(self):
        display(self.y.value_counts().to_frame().pipe(ps.add_ratio, cols=[self._target])
                .style.bar(color=['lightblue'], align='zero'))

        return put.plot_distribute(self.y.to_frame(), self._target, 'Ordinal').get_figure()

    @property
    def desc(self):
        """
        :return: Variable description
        """
        if self._desc is None:
            return self.gen_desc()
        return self._desc

    def _update_cols(self):
        """
        modify property cols
        """
        _include = self._include if self._include else []
        _exclude = self._exclude if self._exclude else []
        self._cols = ut.combin_list(self._data.columns, _include, _exclude)

    def data_type_update(self, mapping: dict):
        """
        Manually modify the variable type

        Parameters
        ----------
        :param mapping: {'var_1': type_1,.., 'var_n': type_n}
        :return: Modified infer_dict
        """
        if ut.is_legal_type(mapping.values()):
            self._data_type = mapping
            return "Modify successfully, {0}".format(self._data_type)
        else:
            raise ValueError("Only the following types are supported, {0}".format(ut.LEGAL_TYPE))

    def __modify_data(self, data: pd.DataFrame):
        self._data = data

    def gen_desc(self, ignore: list = [], percentiles=[.01, .1, .5, .75, .9, .99]):
        """
        Generate variable statistical description

        Parameters
        ----------
        :param ignore: Set of values not to be counted, default = []
        :param percentiles: Percentile points to be counted, default = [.01, .1, .5, .75, .9, .99]
        :return:
        """
        rows = []

        for name, series in self.X.items():
            series = series[-series.isin(ignore)]
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

    def plot_stable(self, by=None, resample=None):
        """
        """
        if self._infer_dict is None:
            raise TypeError("need to call method 'data_type_infer' first")
        if by is None or resample is None:
            raise TypeError("columns by/resample must give")

        re = {}
        p_bar = tqdm(self.cols)
        for col in p_bar:
            p_bar.set_description(f"Plotting distribute var {col}")
            re[col] = put.plot_var_stable(self._ori_data, col=col, dtypes=self._infer_dict[col], by=by, resample=resample)

        self._stable = re

        return 'Plotting finished, use command line to show the figure: ".distribute.get("var").figure"'

    def plot_stable_by_x(self, x=None, by=None, resample=None, bins=None):
        """
        """
        if self._infer_dict is None:
            raise TypeError("need to call method 'data_type_infer' first")
        if by is None or resample is None:
            raise TypeError("columns by/resample must give")

        if x not in self._cols:
            raise TypeError("x should in the cols")
        return put.plot_var_stable(self._ori_data, col=x, dtypes=self._infer_dict[x], by=by, resample=resample, usr_bins=bins)

    def plot_distribute(self):
        """
        Draw distribution plot based on variable data

        Parameters
        ----------
        :param save_path: Path to save fig, if None ignore save
        :return:
        """
        if self._infer_dict is None:
            raise TypeError("need to call method 'data_type_infer' first")

        re = {}
        p_bar = tqdm(self.cols)

        for col in p_bar:
            p_bar.set_description("Plotting distribute var %s" % col)
            re[col] = put.plot_distribute(self.X, col=col, dtypes=self._infer_dict[col])

        self._distribute = re

        return 'Plotting finished, use command line to show the figure: ".distribute.get("var").figure" '

    def plot_distribute_with_target(self):
        """
        Draw distribution plot based on X and y

        Parameters
        ----------
        :return:
        """
        if self._infer_dict is None:
            raise TypeError("need to call method 'data_type_infer' first")

        re = {}
        p_bar = tqdm(self.cols)

        for col in p_bar:
            p_bar.set_description("Plotting distribute with target var %s" % col)
            re[col] = put.plot_distribute_class(self.data, x=col, target=self._target, dtypes=self._infer_dict[col])

        self._distribute_with_target = re

        return 'Plotting finished, use command line to show the figure: ".distribute_with_target.get("var").fig" '

    def plot_distribute_group_target(self):
        """
        Draw distribution plot based on X and y

        Parameters
        ----------
        :return:
        """
        if self._infer_dict is None:
            raise TypeError("need to call method 'data_type_infer' first")

        re = {}
        p_bar = tqdm(self.cols)

        for col in p_bar:
            p_bar.set_description("Plotting distribute with target var %s" % col)
            re[col] = put.plot_distribute_group(self.data, x=col, target=self._target, dtypes=self._infer_dict[col])

        self._distribute_group_target = re

        return 'Plotting finished, use command line to show the figure: ".distribute_group.get("var").fig" '

    def plot_distribute_badrate(self, x=None, usr_bins=None):
        """
        Draw distribution plot based on X and y

        Parameters
        ----------
        :return:
        """
        if self._infer_dict is None:
            raise TypeError("need to call method 'data_type_infer' first")

        if x is not None and usr_bins is not None:
            if self._data_type.get(x) not in ['Categorical', 'Ordinal']:
                return put.plot_default_con(self.data, x=x, target=self._target,
                                            bins=usr_bins)
            else:
                return put.plot_default_cat(self.data, x=x, target=self._target)

        re = {}
        p_bar = tqdm(self._plot_bins.keys())

        for col in p_bar:
            p_bar.set_description("Plotting distribute with target var %s" % col)
            if self._plot_bins[col]['var_type'] == 'Continuous' and self._data_type.get(col) not in ['Categorical', 'Ordinal']:
                re[col] = put.plot_default_con(self.data, x=col, target=self._target, bins=self._plot_bins[col].get('bins'))
            else:
                re[col] = put.plot_default_cat(self.data, x=col, target=self._target)

        self._distribute_badrate = re

        return 'Plotting finished, use command line to show the figure: "distribute_badrate.get("var").fig" '

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
                re[col] = Metrics.ks(self._data[self._target], ut.normalization(self._data[col]))
        self._ks = pd.DataFrame.from_dict(re, orient='index', columns=['ks'])
        return self._ks

    def _gen_auc(self):
        """
        Single variable auc value calculation
        """
        re = {}
        for col in self._cols:
            if ut.is_numeric(self._data[col]) and col != self._target:
                re[col] = Metrics.auc(self._data[self._target], ut.normalization(self._data[col]))
        self._auc = pd.DataFrame.from_dict(re, orient='index', columns=['auc'])
        return self._auc

    def _gen_ar(self):
        """
        Single variable ar value calculation
        """
        re = {}
        for col in self._cols:
            if ut.is_numeric(self._data[col]) and col != self._target:
                re[col] = Metrics.ar(self._data[self._target], ut.normalization(self._data[col]))
        self._ar = pd.DataFrame.from_dict(re, orient='index', columns=['ar'])
        return self._ar

    def _gen_iv(self):
        """
        Single variable iv value calculation
        """
        from toad import quality
        self._iv = quality(self._data[self._cols], self._target, iv_only=False)[['iv', 'gini', 'entropy', 'unique']]
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

            X = np.array(self.X[col]).reshape(-1,1)
            y = self.y
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

