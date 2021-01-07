# -*- coding: utf-8 -*-
# @Time : 2020/12/16 10:09 上午
# @Author : quanbing
# @Email : quanbinks@sina.com
# @File : plot_utils.py


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class PlotUtils:
    @staticmethod
    def plot_distribute(data: pd.DataFrame, col, dtypes, **kwargs):
        if dtypes in ['Categorical', 'Ordinal']:
            return PlotUtils.plot_distribute_bar(data, col, **kwargs)
        elif dtypes == 'Continues':
            return PlotUtils.plot_distribute_hist(data, col, **kwargs)
        return None

    @staticmethod
    def plot_distribute_hist(data: pd.DataFrame, x: str, stat='count', **kwargs):
        """
        :param data: data
        :param x: x
        :param stat: {“count”, “frequency”, “density”, “probability”}
        :param kwargs:
        :return:
        """
        # weights = np.ones_like(series) / float(len(series))
        fig = sns.histplot(data=data, x=x, stat=stat, **kwargs)
        plt.title(kwargs.get('title') if kwargs.get('title') else x)
        plt.close('all')
        return fig

    @staticmethod
    def plot_distribute_bar(data: pd.DataFrame, x: str, **kwargs):
        '''
        :param data: pandas series
        :param title: plot name
        :param nbin: num of bins
        :return: matplot figure
        '''
        fig = sns.countplot(data=data, x=x, **kwargs)
        plt.title(kwargs.get('title') if kwargs.get('title') else x)
        plt.close('all')
        return fig

    @staticmethod
    def plot_distribute_with_target(df_: pd.DataFrame, x: str, target: str, fig_size=(9, 6)):
        '''
        :param df_: pandas dataframe
        :param x: columns name
        :param target: target columns name
        :param fig_size: fig_size tuple
        :return: figure
        '''
        fig = plt.figure(figsize=fig_size)
        bins = len(df_[x].unique()) if len(df_[x].unique()) < 10 else 10
        df_[str(x) + '_bins'] = pd.cut(df_[x], bins=bins, right=False)
        fig = df_.groupby([str(x) + '_bins', target]).size().unstack().plot(kind='bar', stacked=False).get_figure()
        plt.title(x)
        plt.close('all')
        return fig

    @staticmethod
    def plot_distribute_with_target_norminal(df_: pd.DataFrame, x: str, target: str, fig_size=(9, 6)):
        '''
        :param df_: pandas dataframe
        :param x: columns name
        :param target: target columns name
        :param fig_size: fig_size tuple
        :return: figure
        '''
        fig = plt.figure(figsize=fig_size)
        fig = df_.groupby([x, target]).size().unstack().plot(kind='bar', stacked=False).get_figure()
        plt.title(x)
        plt.close('all')
        return fig

    @staticmethod
    def plots(x: pd.Series, title: str, kind: str, fig_size=(9, 6)):
        """
        :param x: pandas series
        :param title: fig title
        :param kind: fig type
        :param fig_size:
        :return:
        """
        fig = plt.figure(figsize=fig_size)
        fig = x.plot(kind=kind)
        plt.title(title)
        plt.close('all')
        return fig

    @staticmethod
    def save_fig(fig, name, save_path, type_str=None):
        """
        :param type_str:
        :param fig:
        :param name:
        :param save_path:
        :return: None
        """
        fig.savefig(save_path / '{0}_{1}.png'.format(name, type_str))
