#!/usr/bin/env python
#-*- coding:utf8 -*-

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')

import os, sys


def kpi_catcher(dataFrame, dateColName, KPIColName, plot_type='l', tick_interval = 1, groupby=None,
              savedir='./img/', perspective='overall', plot_show=True, save_fig=True):
    """
    "
    <usage> 
    this function use to catch kpi with different time perspective and group, and support plot
    and save figure.Finally return a dict of kpi.
    "
    [DataFrame]<pandas.DataFrame>: "DataFrame"
    [savedir]<str>: "directort to save plot, default save in "./img/". (dirname must end with '/')"
    [sep]<str>: "segmentation format of file, 'tsv' use '\t'.(default None, means read 'csv')"
    [weekday]<bool>: "whether generates plot by weekday (default False)"
    [dateColName]<str>: "column name of datetime"
    [KPIColName]<str>: "column name of KPI"
    [groupby]<str>: "column name of which plot by (hour or by min or others) ( default by datetime)"
    [plot_type]<str>: "'l': line plot, 's': scatter plot"
    [tick_interval]<int>: "show the xticklabel after a tick_interval (default 1)"
    [perspective]<str>: "'overall':datetime, 'weekday':weekday, 'day':day"
    [plot_show]<bool>: "show plot or not(default True)"
    [save_fig]<bool>: "save figure or not"
    """
    if groupby is None:
        if perspective == 'weekday' or perspective == 'day':
            raise ValueError("in perspective of {}, cannot group by 'day', choose column name of 'hour'\
                             or column name of 'minite' please".format(perspective))
        groupby = dateColName
    path = savedir
    if not os.path.exists(path):
        os.mkdir(path)
    df = dataFrame
    data_dict = {}
    datetime = pd.to_datetime(df[dateColName])
    df['weekday'] = datetime.apply(lambda x: (x.weekday()+1,x.day_name()))
    if perspective == 'weekday':
        for day in list(set(df['weekday'])):
            data_dict[day] = df.loc[df['weekday']==day,:].groupby([groupby],as_index=False)[groupby, KPIColName].mean()
    elif perspective == 'day':
        for day in list(set(df[dateColName])):
            data_dict[day] =df.loc[df[dateColName]==day,:].\
                    groupby([groupby],as_index=False)[groupby, KPIColName].mean()
    else:
        data_dict['datetime'] = df.groupby([groupby],as_index=False)[groupby,KPIColName].mean()
    # plot KPI along with the datetime
    for name, value in sorted(data_dict.items()):
        fig = plt.figure(figsize=(20,10))
        ax = fig.add_subplot(111)
        x = value[groupby]
        y = value[KPIColName]
        if plot_type == 'l':
            ax.plot(x, y)
        elif plot_type == 's':
            ax.scatter(x, y)
        else:
            raise ValueError("plot_type only support 'l'(line) or 's'(scatter)")
        ax.set_xticks(np.arange(0, len(x), tick_interval))
        if groupby != dateColName:
            ax.set_xticklabels([groupby+str(i+1) for i in range(len(x))])
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_rotation('vertical')
        ax.set_xlabel(groupby)
        ax.set_ylabel(KPIColName)
        ax.set_title("{} plot of {}".format(KPIColName, name))
        if save_fig:
            saveName = "{}_by_{}".format(str(KPIColName),str(name))
            fig.savefig(path + saveName )
        if plot_show == False or len(data_dict) > 10:
            plt.close(fig)
    return data_dict
        
def trend_plot( data, window_size = 10, save_fig = False, save_dir = "./img/" ):
    """
    "
    <usage>
    this function use to plot trend against raw data, note that plot start from index of "window_size"
    "
    [data]<pandas.series>: "Time Series"
    [window_size]<int>: "use how many data points to predict the trend (default 10)"
    [save_fig]<bool>: "save figure or not"
    [savedir]<str>: "directort to save plot, default save in "./img/". (dirname must end with '/')"
    """
    from sklearn.linear_model import LinearRegression
    clf = LinearRegression()
    x = np.arange(len(data)).reshape(-1,1)
    y = np.array(data).reshape(-1,1)
    
    coefs = []
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(211)
    ax1.plot(x, y)
    ax1.set_title("origin data")
    ax1.set_ylabel(data.name)
    ax1.set_xlim(0, len(data))
    ax1.set_xticks(np.arange(len(data)))
    for i in range(window_size, len(data)):
        left = i - window_size
        clf.fit(x[left:i], y[left:i])
        pred = clf.predict(x[i:min(i+window_size, len(data))])
        coefs.append(clf.coef_[0][0])
        ax1.plot(x[i:min(i+window_size, len(data))],pred, c='blue')


    ax2 = fig.add_subplot(212)
    ax2.plot(np.arange(len(coefs)), coefs)
    ax2.hlines(0, xmin=0, xmax=len(coefs)-1)
    ax2.set_title("trend plot")
    ax2.set_xlim(-window_size, len(data)-window_size)
    ax2.set_xticks(np.arange(-window_size, len(data)-window_size))
    ax2.set_ylabel('slope')
    if save_fig:
        figname = "trend_{}_ws_{}".format(data.name, window_size)
        fig.savefig(save_dir+figname)
    