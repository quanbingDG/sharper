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
    def plot_distribute_hist(x: pd.Series, title: str=None, nbin=10, fig_size=(9, 6)):
        '''
        :param x: pandas series
        :param title: plot name
        :param nbin: num of bins
        :param fig_size:
        :return: matplot figure
        '''
        fig = plt.figure(figsize=fig_size)
        weights = np.ones_like(x) / float(len(x))
        fig = x.hist(color=sns.desaturate("indianred", .8), bins=nbin, weights=weights).get_figure()
        plt.title(title if title else x.name)
        plt.close('all')
        return fig

    @staticmethod
    def plot_distribute_bar(x: pd.Series, title: str, nbin=10, figsize=(9, 6)):
        '''
        :param x: pandas series
        :param title: plot name
        :param nbin: num of bins
        :return: matplot figure
        '''
        fig = plt.figure(figsize=figsize)
        fig = x.plot(kind='bar', color=sns.desaturate("indianred", .8), bins=nbin).get_figure()
        plt.title(title)
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
