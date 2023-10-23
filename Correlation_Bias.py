# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:26:17 2022

@author: HIDRAULICA-Dani
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.lines as mlines
import matplotlib.patheffects as pe
from matplotlib import colors
#import spotpy
import netCDF4 as nc
from datetime import datetime, timedelta
# import pyextremes
from pyextremes import EVA
import gc

os.chdir('D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS')

ensemble = [0,1,2,3,4,5,6,7,8,9,10] #with control
#slt = [*range(2,17)]
slt = [*range(2,17,3)]

for lt in slt:
    # start_lt = 2
    start_lt = lt
    end_lt = start_lt + 3
    print('lt', start_lt, end_lt)
    
    Q = pd.read_csv('../PYR/HMS/Results/csv/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.csv', parse_dates=['Unnamed: 0'])
    Q.index = Q['Unnamed: 0']
    Q.index.name = None
    Q = Q.drop(['Unnamed: 0'], axis=1)
    
    if not os.path.exists('./VARIOS/Figures/discharge/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'/'):
        os.mkdir('./VARIOS/Figures/discharge/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'/')
    if not os.path.exists('./VARIOS/Figures/discharge/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'/scatter/'):
        os.mkdir('./VARIOS/Figures/discharge/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'/scatter/')

    #Q_corlt = pd.concat([Q_obs_id, df_Qlt], axis=1)
    Q_corlt = Q
    #Q_sumlt = Q_corlt.resample('365.25D').sum()
    Q_meanlt = Q_corlt.resample('Y').mean()
    Q_maxlt = Q_corlt.resample('Y').max()
    # plt.plot(Q_max)
    #Q_maxlt

    Q_reslt = {'Q medio diario':Q_corlt, 'Q medio anual':Q_meanlt, 'Q max anual':Q_maxlt}
    Q_res = 'Q max anual'
    
    #Pearson correlation between different ensemble members
    df_corr_res = Q_reslt[Q_res].corr(method='pearson')
    np_res = df_corr_res.to_numpy()
    
    fig, ax = plt.subplots(1)
    # divnorm=colors.TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
    c = ax.matshow(df_corr_res, cmap='Greens') #, norm=divnorm)
    plt.xticks(ensemble)
    plt.yticks(ensemble)
    ax.set_title('Correlación '+Q_res)
    ax.set_xlabel('Miembro, '+str(start_lt)+' a '+str(end_lt)+' días')
    ax.set_ylabel('Miembro, '+str(start_lt)+' a '+str(end_lt)+' días')
    ax.xaxis.tick_bottom()
    fig.colorbar(c, ax=ax)
    for (i, j), z in np.ndenumerate(np_res):
        # print(i,j,z)
        ax.text(i, j, '{:0.2f}'.format(z), color='k', weight='normal', ha='center', va='center', fontsize=8)
            # path_effects=[pe.withStroke(linewidth=0.1, foreground='k')])
    #            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    name = 'Q_corr_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)
    fig.savefig('./VARIOS/Figures/discharge/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'/'+name+'.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    plt.close()
    
#########################################################################################
#Scatter in one plot

    for ens in ensemble:
        for enss in ensemble:
            plt.scatter(Q_reslt[Q_res][str(ens)], Q_reslt[Q_res][str(enss)], label=str(enss), marker='.')
            plt.title(Q_res+', Miembro '+str(ens)+' vs Todos')
            plt.xlabel(Q_res+', Miembro '+str(ens))
            plt.ylabel(Q_res+', Miembros')
            plt.legend(ncol=2, prop={'size': 8})
        name = 'Q_scat_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'_'+str(ens)+'_vs_All'
        plt.savefig('./VARIOS/Figures/discharge/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'/scatter/'+name+'.jpg', format='jpg', dpi=1000, bbox_inches='tight')
        plt.close()
        
        
######################################################################################################
######################################################################################################




