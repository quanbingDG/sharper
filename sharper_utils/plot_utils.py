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
    def annotate(ax, x, y, space=5, format=".2f"):
        """
        """
        va = 'bottom'

        if y < 0:
            space *= -1
            va = 'top'

        ax.annotate(
            ("{:" + format + "}").format(y),
            (x, y),
            xytext=(0, space),
            textcoords="offset points",
            ha="center",
            va=va,
        )

    @staticmethod
    def add_bar_annotate(ax, **kwargs):
        """
        """
        for rect in ax.patches:
            y_value = rect.get_height()
            x_value = rect.get_x() + rect.get_width() / 2

            PlotUtils.annotate(ax, x_value, y_value, **kwargs)
        return ax

    @staticmethod
    def add_line_annotate(ax, **kwargs):
        """
        """
        for line in ax.lines:
            points = line.get_xydata()

            for point in points:
                PlotUtils.annotate(ax, point[0], point[1], **kwargs)

        return ax

    @staticmethod
    def add_annotate(ax, **kwargs):
        if len(ax.lines) > 0:
            PlotUtils.add_line_annotate(ax, **kwargs)

        if len(ax.patches) > 0:
            PlotUtils.add_bar_annotate(ax, **kwargs)

        return ax

    @staticmethod
    def add_text(ax, text, loc='top left', offset=(0.01, 0.04)):
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        x_offset = (x_max - x_min) * offset[0]
        y_offset = (y_max - y_min) * offset[1]

        if loc == 'top left':
            loc = (x_min + x_offset, y_max - y_offset)
        elif loc == 'top right':
            loc = (x_max - x_offset, y_max - y_offset)
        elif loc == 'bottom left':
            loc = (x_min + x_offset, y_min + y_offset)
        elif loc == 'bottom right':
            loc = (x_max - x_offset, y_min + y_offset)

        ax.text(*loc, text, fontsize='x-large')

        return ax

    @staticmethod
    def plot_distribute(data: pd.DataFrame, col, dtypes, **kwargs):
        """
        :param data:
        :param col:
        :param dtypes:
        :param kwargs:
        :return:
        """
        if dtypes in ['Categorical', 'Ordinal']:
            return PlotUtils.plot_distribute_bar(data, col, **kwargs)
        elif dtypes == 'Continues':
            return PlotUtils.plot_distribute_hist(data, col, **kwargs)
        return 'Unsupported data type'

    @staticmethod
    def plot_distribute_class(data, x, target, dtypes, **kwargs):
        """
        :param data:
        :param x:
        :param target:
        :param dtypes:
        :param y:
        :param hue:
        :param kwargs:
        :return:
        """
        if dtypes in ['Categorical', 'Ordinal']:
            return PlotUtils.plot_distribute_cat(data=data, x=x, col=target, **kwargs)
        elif dtypes == 'Continues':
            return PlotUtils.plot_distribute_dis(data=data, x=x, col=target, **kwargs)
        return 'Unsupported data type'

    @staticmethod
    def plot_distribute_group(data, x, target, dtypes, **kwargs):
        """
        """
        if dtypes in ['Categorical', 'Ordinal']:
            return PlotUtils.plot_distribute_cat_group(data=data, x=x, target=target, **kwargs)
        elif dtypes == 'Continues':
            return PlotUtils.plot_distribute_his_group(data=data, x=x, target=target, **kwargs)
        return 'Unsupported data type'

    @staticmethod
    def plot_distribute_dis(data, x, y=None, col=None, hue=None, **kwargs):
        """
        :param data:
        :param x:
        :param y:
        :param hue:
        :param col:
        :param kwargs:
        :return:
        """
        fig = plt.figure(figsize=(16, 8))
        fig = sns.displot(data=data, x=x, y=y, hue=hue, col=col, bins=10)
        plt.title(kwargs.get('title') if kwargs.get('title') else x)
        plt.close('all')
        return fig

    @staticmethod
    def plot_distribute_cat(data, x, y=None, hue=None, col=None, **kwargs):
        """
        :param data:
        :param x:
        :param y:
        :param hue:
        :param col:
        :param kwargs:
        :return:
        """
        fig = plt.figure(figsize=(16, 8))
        fig = sns.catplot(data=data, x=x, y=y, hue=hue, col=col, kind='count', **kwargs)
        plt.title(kwargs.get('title') if kwargs.get('title') else x)
        plt.close('all')
        return fig

    @staticmethod
    def plot_distribute_cat_group(data, x, target, y=None, hue=None, col=None, **kwargs):
        """
        """
        grouped = data.groupby(target)
        target_unique = data[target].unique()
        fig, ax_arr = plt.subplots(1, len(target_unique), figsize=(16, 8))
        for n, i in enumerate(target_unique):
            ax_arr[n].set_title('when {0} = {1}'.format(target, i))
            sns.countplot(data=grouped.get_group(i), x=x, ax=ax_arr[n])
            ax_arr[n] = PlotUtils.add_annotate(ax_arr[n], format='.0f')

        plt.close('all')
        return fig

    @staticmethod
    def plot_distribute_his_group(data, x, target, y=None, hue=None, col=None, **kwargs):
        """
        """
        grouped = data.groupby(target)
        target_unique = data[target].unique()
        fig, ax_arr = plt.subplots(1, len(target_unique), figsize=(16, 8))
        for n, i in enumerate(target_unique):
            ax_arr[n].set_title('when {0} = {1}'.format(target, i))
            sns.histplot(data=grouped.get_group(i), x=x, stat='count', bins=10, ax=ax_arr[n])
            ax_arr[n] = PlotUtils.add_annotate(ax_arr[n], format='.0f')

        plt.close('all')
        return fig

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
        fig = plt.figure(figsize=(16, 8))
        fig = sns.histplot(data=data, x=x, stat=stat, bins=10, **kwargs)
        fig = PlotUtils.add_annotate(fig, format='.0f')
        plt.title(kwargs.get('title') if kwargs.get('title') else x + ' distributions')
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
        fig = plt.figure(figsize=(16, 8))
        fig = sns.countplot(data=data, x=x, **kwargs)
        fig = PlotUtils.add_annotate(fig, format='.0f')
        counts = data[x].value_counts(normalize=True).values
        pches = fig.patches

        for i, j in zip(pches, counts):
            h = i.get_height()
            fig.text(i.get_x() + i.get_width() / 2, h * 1.005, round(j, 4), ha='center')

        plt.title(kwargs.get('title') if kwargs.get('title') else x + ' distributions')
        plt.close('all')
        return fig

    @staticmethod
    def plot_default_con(data: pd.DataFrame, x: str, target: str, bins: list, figsize: tuple = (10, 8),
                         annotate_format=".2f"):
        """
        """
        all_bad = data[target].sum()
        total = data[target].count()
        all_default_rate = all_bad * 1.0 / total
        df_ = data[[x, target]].copy()
        df_['bins'] = pd.cut(df_[x], bins=bins)
        d1 = df_['bins'].value_counts().sort_index().to_frame(name='counts').reset_index()
        gp = df_.groupby('bins')
        d2 = pd.DataFrame()
        d2['total'] = gp[target].count()
        d2['bad'] = gp[target].sum()
        d2['default_rate'] = d2['bad'] / d2['total']
        d2 = d2.reset_index()
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111, )
        ax1.bar(d1['index'].astype(str), d1['counts'], color='#82C6E2')
        ax1 = PlotUtils.add_annotate(ax1, format=annotate_format)
        ax1.set_ylabel('Counts')
        ax1.grid(False)
        plt.xticks(rotation=45)
        ax2 = ax1.twinx()
        ax2.plot(d2['bins'].astype(str), d2['default_rate'].fillna(0), color='#D65F5F')
        ax2 = PlotUtils.add_annotate(ax2, format=annotate_format)
        ax2.grid(False)
        ax2.set_ylabel('Bad rate')

        plt.title("Var:{0}, Bad_rate:{1:.4f}".format(x, all_default_rate))
        plt.close()

        return fig

    @staticmethod
    def plot_default_cat(data: pd.DataFrame, x: str, target: str, figsize: tuple = (10, 8), annotate_format=".2f"):
        """
        """
        all_bad = data[target].sum()
        total = data[target].count()
        all_default_rate = all_bad * 1.0 / total
        df_ = data[[x, target]].copy()
        d1 = df_[x].value_counts().to_frame(name='counts').reset_index()
        gp = df_.groupby(x)
        d2 = pd.DataFrame()
        d2['total'] = gp[target].count()
        d2['bad'] = gp[target].sum()
        d2['default_rate'] = d2['bad'] / d2['total']
        d2 = d2.reset_index()
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111, )
        ax1.bar(d1['index'].astype(str), d1['counts'], color='#82C6E2')
        ax1 = PlotUtils.add_annotate(ax1, format=annotate_format)
        ax1.set_ylabel('Counts')
        ax1.grid(False)
        plt.xticks(rotation=45)
        ax2 = ax1.twinx()
        ax2.plot(d2[x].astype(str), d2['default_rate'].fillna(0), color='#D65F5F')
        ax2 = PlotUtils.add_annotate(ax2, format=annotate_format)
        ax2.grid(False)
        ax2.set_ylabel('Bad_rate')
        plt.title("Var:{0}, Bad_rate:{1:.4f}".format(x, all_default_rate))
        plt.close()

        return fig

    @staticmethod
    def plot_var_stable(data: pd.DataFrame, col, dtypes, by=None, resample='3M', figsize=(20, 8), usr_bins=None):
        """
        """
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_subplot(111, )
        ax1.grid(False)
        plt.xticks(rotation=45)
        plt.title("Var:{0} stable distribute".format(col))
        if not isinstance(data[by], np.datetime64):
            try:
                data[by] = pd.to_datetime(data[by])
            except:
                raise TypeError("the columns by must be datetime")
        _tmp = data.set_index(by).resample(resample)

        if dtypes in ['Categorical', 'Ordinal']:
            for var in data[col].unique():
                ax1.plot(_tmp[col].apply(lambda x: x.value_counts(normalize=1))[:, var], label=f"{var}", marker='o')
        elif dtypes == 'Continues' and usr_bins is None:
            if data[col].min() == -9999:
                _bins = np.linspace(0, data[col].max(), 4)
                _bins = np.insert(_bins, 0, -10000)
            else:
                _bins = np.linspace(data[col].min() - 0.001, data[col].max(), 5)
            _tmp_data = pd.cut(data[col], bins=_bins)
            for var in _tmp_data.factorize()[1].categories:
                ax1.plot(_tmp[col].apply(lambda x: pd.cut(x, bins=_bins).value_counts(normalize=1))[:, var], label=var, marker='o')
        elif dtypes == 'Continues' and usr_bins is not None:
            _tmp_data = pd.cut(data[col], bins=usr_bins)
            for var in _tmp_data.factorize()[1].categories:
                ax1.plot(_tmp[col].apply(lambda x: pd.cut(x, bins=usr_bins).value_counts(normalize=1))[:, var], label=var,
                         marker='o')

        ax1.legend(loc=1)
        plt.ylim(0, 1)
        plt.close()
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
