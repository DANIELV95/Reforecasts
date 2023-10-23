# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:07:05 2021

@author: villarre
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.lines as mlines
import spotpy
import netCDF4 as nc
from datetime import datetime, timedelta
# import pyextremes
from pyextremes import EVA
import gc

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

os.chdir('D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS')
os.listdir()

#Variables
ensemble = [0,1,2,3,4,5,6,7,8,9,10] #with control
#slt = [*range(2,17)]
slt = [*range(2,17,3)]
Tr = [2,5,10,25,50,100,300,1000,1250]

for lt in slt:
    # start_lt = 2
    start_lt = lt
    end_lt = start_lt + 3
    print('lt', start_lt, end_lt)
    
    Q = pd.read_csv('../PYR/HMS/Results/csv/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.csv', parse_dates=['Unnamed: 0'])
    Q.index = Q['Unnamed: 0']
    Q.index.name = None
    Q = Q.drop(['Unnamed: 0'], axis=1)
    Q_maxlt = Q.resample('Y').max()
    
    if not os.path.exists('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/'):
        os.mkdir('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/')
    
    eva_obs = EVA(Q_obs[id])
    eva_obs.get_extremes(method='BM', block_size='365.2425D', errors='ignore')
    eva_obs.fit_model(distribution = 'gumbel_r') #forced to follow gumbel distribution
    summary_obs = eva_obs.get_summary(return_period=Tr, alpha=0.95, n_samples=1000)
    #Tr_res = summary_obs.drop(['upper ci', 'lower ci'], axis=1)
    summary_obs.columns = ['Qobs', 'lower ci', 'upper ci']
    #summary_obs
    
    df_Qlt = pd.read_csv('C:/Users/villarre/DataAnalysis/Reforecasts/Results/Q'+str(start_lt)+str(end_lt)+'_'+str(model)+'_'+id+'.csv')
    df_Qlt['Date'] = pd.to_datetime(df_Qlt['Date']) #format='%Y-%m-%d'
    df_Qlt.index = df_Qlt['Date']
    df_Qlt.index.name = None
    df_Qlt = df_Qlt.drop(df_Qlt.columns[0], axis=1)
    start_date = df_Qlt.index >= start_day
    end_date = df_Qlt.index <= end_day
    df_Qlt = df_Qlt.loc[start_date & end_date]
    df_Qlt = df_Qlt.replace(-999,np.nan)
    df_Qlt[df_Qlt < 0] = np.nan
    df_Qlt = df_Qlt.dropna()
    #df_Qlt

#########################################################################################
#Extreme value analysis for each ensemble member
    df_QQAll = pd.DataFrame()
    
    for ens in ensemble:
        print(ens+1)
        
        eva = EVA(df_Qlt[str(ens+1)])
        eva.get_extremes(method='BM', block_size='365.2425D', errors='ignore') #365.2425D
        eva.plot_extremes()
        plt.title('Extreme Value Analysis - Block Maxima, ' + id + '_KsHF='+str(model)+'_Reforecast='+str(ens+1))
        plt.xlabel('Year')
        plt.ylabel('Discharge [$m^3$/s]')
        frmt='jpg'
        name_plot='fig_EVA_BM_' + id + '_KsHF='+str(model)+'_Reforecast='+str(ens+1)+'.' + frmt
        plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/'+name_plot, format=frmt, dpi=1000, bbox_inches='tight')
        plt.close()
        
        eva.fit_model(distribution = 'gumbel_r')
        summary = eva.get_summary(return_period=Tr, alpha=0.95, n_samples=1000)
        fig, ax = eva.plot_diagnostic(return_period=Tr, alpha=0.95)
        ax[0].set_ylabel('Reforecast='+str(ens+1))
        ax[1].set_xlabel('Reforecast='+str(ens+1))
        ax[2].set_ylabel('Reforecast='+str(ens+1))
        ax[3].set_ylabel('Reforecast='+str(ens+1))
        #ax[0].plot(summary_obs.index,summary_obs['return value'].values, color='y', ls='dotted', alpha=0.8)
        #sim_lg = mlines.Line2D([], [], label='sim', color='r')
        #obs_lg = mlines.Line2D([], [], label='obs', color='y', ls='dotted', alpha=0.8)
        #ax[0].legend(handles=[sim_lg, obs_lg])
        QQ_data = ax[2].collections[0].get_offsets().data
        #QQ_line = ax[2].lines[0].get_data()
        df_QQ = pd.DataFrame(QQ_data, columns=['T'+str(ens+1), 'O'+str(ens+1)])
        df_QQAll = pd.concat([df_QQAll,df_QQ], axis=1)
        name_plot='fig_EVA_diag_' + id + '_KsHF='+str(model)+'_Reforecast='+str(ens+1)+'.' + frmt
        plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/'+name_plot, format=frmt, dpi=1000, bbox_inches='tight')
        extent = ax[0].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
        plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/sp1'+name_plot, format=frmt, dpi=1000, bbox_inches=extent.expanded(1.05,1.05))
        plt.close()

        f = open('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/eva_'+id+'_'+str(ens+1)+'_KsHF='+str(model)+'.txt', 'w')
        print(eva, file=f)
        f.close()
        
    for ens in ensemble:
        #r2 = round(spotpy.objectivefunctions.rsquared(df_QQAll['T'+str(ens+1)], df_QQAll['O'+str(ens+1)]),3)
        plt.scatter(df_QQAll['T'+str(ens+1)], df_QQAll['O'+str(ens+1)], label=str(ens+1), marker='.')
    plt.title('Q-Q plot Reforecast=All')
    plt.legend(ncol=2, prop={'size': 8})
    # plt.xlim(25,225)
    # plt.ylim(25,300)
    plt.xlabel('Theoretical')
    plt.ylabel('Reforecast=All')
    frmt='jpg'
    name_plot='fig_EVA_QQplot_'+id+'_KsHF='+str(model)+'_Reforecast=All.' + frmt
    plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/'+name_plot, format=frmt, dpi=1000, bbox_inches='tight')
    plt.close()
    
#########################################################################################        
#EVA for combined time series

    Q_ref = df_Qlt['1'].iloc[:1]
    for ens in ensemble:
        Q_ref = Q_ref.append(df_Qlt[str(ens+1)])
    Q_ref = Q_ref.iloc[1:]
    Q_ref = Q_ref.reset_index()
    Q_ref = Q_ref.drop(['index'], axis=1)
    #Q_ref = Q_ref.drop(Q_ref.index[Q_ref[0] == max(Q_ref[0])])
    df_date = pd.date_range(datetime(1900,1,1), periods=len(Q_ref))
    Q_ref.index = df_date
    #Q_ref
    
    eva = EVA(Q_ref[0])
    eva.get_extremes(method='BM', block_size='365.2425D', errors='ignore') #365.2425D
        
    eva.plot_extremes()
    plt.title('Extreme Value Analysis - Block Maxima, ' + id + '_KsHF='+str(model)+'_Reforecast=All')
    plt.xlabel('Year')
    plt.ylabel('Discharge [$m^3$/s]')
    frmt='jpg'
    name_plot='fig_EVA_BM_' + id + '_KsHF='+str(model)+'_Reforecast=All.' + frmt
    plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/'+name_plot, format=frmt, dpi=1000, bbox_inches='tight')
    plt.close()
    
    eva.fit_model(distribution = 'gumbel_r')
    summary = eva.get_summary(return_period=Tr, alpha=0.95, n_samples=1000)
    summary.columns = ['Qref', 'lower ci', 'upper ci']
    
    fig, ax = eva.plot_diagnostic(return_period=Tr, alpha=0.95)
    ax[0].set_ylabel('Reforecast=All')
    ax[1].set_xlabel('Reforecast=All')
    ax[2].set_ylabel('Reforecast=All')
    ax[3].set_ylabel('Reforecast=All')
    ax[0].plot(summary_obs.index,summary_obs['Qobs'].values, color='y', ls='dotted', alpha=0.8)
    ref_lg = mlines.Line2D([], [], label='Qref', color='r')
    #sim_lg = mlines.Line2D([], [], label='sim', color='r')
    obs_lg = mlines.Line2D([], [], label='Qobs', color='y', ls='dotted', alpha=0.8)
    ax[0].legend(handles=[ref_lg, obs_lg])
    QQ_data = ax[2].collections[0].get_offsets().data
    #QQ_line = ax[2].lines[0].get_data()
    df_QQ = pd.DataFrame(QQ_data, columns=['TAll', 'OAll'])
    df_QQAll = pd.concat([df_QQAll,df_QQ], axis=1)
    name_plot='fig_EVA_diag_' + id + '_KsHF='+str(model)+'_Reforecast=All.' + frmt
    plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/'+name_plot, format=frmt, dpi=1000, bbox_inches='tight')
    extent = ax[0].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
    plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/sp1'+name_plot, format=frmt, dpi=1000, bbox_inches=extent.expanded(1.05,1.05))
    plt.close()

    f = open('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/eva_'+id+'_All_KsHF='+str(model)+'.txt', 'w')
    print(eva, file=f)
    f.close()
    
    Tr_res = pd.concat([summary_obs,summary], axis=1)
    Tr_res.to_csv('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/Tr_res_' + id + '_Qref.csv')
    
    for ens in ensemble:
        #r2 = round(spotpy.objectivefunctions.rsquared(df_QQAll['T'+str(ens+1)], df_QQAll['O'+str(ens+1)]),3)
        plt.scatter(df_QQAll['T'+str(ens+1)], df_QQAll['O'+str(ens+1)], label=str(ens+1), marker='.')
    plt.scatter(df_QQAll['TAll'], df_QQAll['OAll'], label='All', marker='*')
    plt.title('Q-Q plot Reforecast=All, '+id)
    plt.legend(ncol=3, prop={'size': 8})
    # plt.xlim(25,225)
    # plt.ylim(25,300)
    plt.xlabel('Theoretical')
    plt.ylabel('Reforecast=All')
    frmt='jpg'
    name_plot='fig_EVA_QQplotAll_'+id+'_KsHF='+str(model)+'_Reforecast=All.' + frmt
    plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/'+name_plot, format=frmt, dpi=1000, bbox_inches='tight')
    plt.close()
    
#########################################################################################
#Return period plot

    Tr_res = pd.read_csv('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/Tr_res_'+id+'_Qref.csv')
    Tr_res.index = Tr_res['return period']
    Tr_res.index.name = None
    Tr_res = Tr_res.drop(['return period'], axis=1)
    Tr_res.columns = ['Qobs','Qobs_lci','Qobs_uci','Qref','Qref_lci','Qref_uci']
    #Tr_res
    
    Qobs_rng = abs(Tr_res['Qobs_lci'] - Tr_res['Qobs_uci'])
    Qref_rng = abs(Tr_res['Qref_lci'] - Tr_res['Qref_uci'])
    UnRed = Qobs_rng/Qref_rng
    Un_mean = round(UnRed.mean(),2)
    #(Qref_rng/Qobs_rng*100).mean()
    Un_1250 = round(Qobs_rng[1250]/Qref_rng[1250], 2)

    fig, ax = plt.subplots(1)
    ax.plot(Tr_res.index, Tr_res['Qobs'], marker='.', color='b')
    ax.plot(Tr_res.index, Tr_res['Qref'], marker='.', color='g')
    ax.plot(Tr_res.index, Tr_res['Qobs_lci'], ls='--', color='b', alpha=0.5)
    ax.plot(Tr_res.index, Tr_res['Qobs_uci'], ls='--', color='b', alpha=0.5)
    ax.plot(Tr_res.index, Tr_res['Qref_lci'], ls='--', color='g', alpha=0.5)
    ax.plot(Tr_res.index, Tr_res['Qref_uci'], ls='--', color='g', alpha=0.5)
    ax.fill_between(Tr_res.index, Tr_res['Qobs_lci'], Tr_res['Qobs_uci'], color='b', alpha=0.2)
    ax.fill_between(Tr_res.index, Tr_res['Qref_lci'], Tr_res['Qref_uci'], color='g', alpha=0.2)
    ax.set_xscale('log')
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xticks(Trx)
    ax.set_xticklabels(Trx) #, rotation=90)
    ax.set_xlabel('Returnn Period [Years]')
    ax.set_ylabel('Discharge [$m^3$/s]')
    ax.set_title('Return Period Plot, '+id)
    ax.text(1250, min(min(Tr_res['Qobs']),min(Tr_res['Qref'])),
            'Uncertainty reduction:\n'
            'Average by '+str(Un_mean)+' times\n'
            'Tr=1250 by '+str(Un_1250)+' times', ha='right', linespacing=1.3)
    obs_lg = mlines.Line2D([], [], label='Qobs', color='b')
    ref_lg = mlines.Line2D([], [], label='Qref', color='g')
    obs_ci_lg = mlines.Line2D([], [], label='', color='g')
    ax.legend(handles=[ref_lg, obs_lg])
    fig.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/Tr_res_'+id+'_Qref.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    plt.close()



  
#########################################################################################
#Comparison of extreme values for different lead times

for model in models:
    for id in gid:
        # model = 2
        # id = '9286162'
        
        if not os.path.exists('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/EVAcomparison/'):
                os.mkdir('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/EVAcomparison/')

        Tr_res = pd.read_csv('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q58/'+id+'/EVAmax/Tr_res_'+id+'_Qref.csv')
        Tr_res.index = Tr_res['return period']
        Tr_res.index.name = None
        Tr_res = Tr_res.drop(['return period'], axis=1)
        Tr_res.columns = ['Qobs','Qobs_lci','Qobs_uci','Qref','Qref_lci','Qref_uci']
        Tr_Q = Tr_res*1
        Tr_Q = Tr_Q.drop(['Qref','Qref_lci','Qref_uci'], axis=1)
        
        for lt in slt:
            start_lt = lt
            end_lt = start_lt + 3
            print(id, start_lt, end_lt)

            Tr_res = pd.read_csv('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/Tr_res_'+id+'_Qref.csv')
            Tr_res.index = Tr_res['return period']
            Tr_res.index.name = None
            Tr_res = Tr_res.drop(['return period'], axis=1)
            Tr_res.columns = ['Qobs','Qobs_lci','Qobs_uci','Qref_'+str(start_lt)+str(end_lt),'Qref_lci_'+str(start_lt)+str(end_lt),'Qref_uci_'+str(start_lt)+str(end_lt)]
            Tr_res = Tr_res.drop(['Qobs','Qobs_lci','Qobs_uci'], axis=1)
            Tr_Q = pd.concat([Tr_Q,Tr_res], axis=1)
            # Tr_Q
            
        idx=['Qobs','Qref, 5-8','Qref, 8-11','Qref, 11-14']
        Q = pd.DataFrame([Tr_Q['Qobs'], Tr_Q['Qref_58'], Tr_Q['Qref_811'], Tr_Q['Qref_1114']], index=idx)
        Qu = pd.DataFrame([Tr_Q['Qobs_uci'], Tr_Q['Qref_uci_58'], Tr_Q['Qref_uci_811'], Tr_Q['Qref_uci_1114']], index=idx)
        Ql = pd.DataFrame([Tr_Q['Qobs_lci'], Tr_Q['Qref_lci_58'], Tr_Q['Qref_lci_811'], Tr_Q['Qref_lci_1114']], index=idx)
        Qerr = (Qu - Ql)/2
        
        Trx = [[2,25,300], [5,50,1000], [10,100,1250]]
        colors = {2:'b', 5:'b', 10:'b', 25:'y', 50:'y', 100:'y', 300:'g', 1000:'g', 1250:'g'}
        axvl = [0.5,1.5,2.5,3.5]
        
        for trx in Trx:
            print(trx[0])
            for tr in trx:
                plt.errorbar(Q.index, Q[tr], yerr=Qerr[tr], marker='_', ms=50, ls='', color=colors[tr], elinewidth=50, alpha=0.5)
                plt.axhline(Q[tr]['Qobs'], ls='--', color=colors[tr], alpha=0.5)
                plt.scatter(Q.index, Q[tr], marker='_', s=200, color=colors[tr] , label=str(tr)+' years')
            [plt.axvline(_axvl, ls='--', lw=1, color='k', alpha=0.5) for _axvl in axvl]
            plt.xlim(-0.5,3.5)
            plt.xlabel('Lead time (days)')
            plt.ylabel('Discharge [$m^3$/s]')
            plt.title(id+', Comparison of discharge return periods, KsHF='+str(model))
            plt.legend(fontsize=8, loc='best', ncol=3)
            # plt.show()
            plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/EVAcomparison/'+id+'_Qobs_vs_Qref_'+str(trx[0])+'.jpg', format='jpg', dpi=1000, bbox_inches='tight')
            plt.close()
            
        
######################################################################################
#EVA for Qobs

Q_obs = pd.read_csv('C:/Users/villarre/DataAnalysis/Measurements/Discharge/Q_obs.csv')
Q_obs['Unnamed: 0'] = pd.to_datetime(Q_obs['Unnamed: 0'])
Q_obs.index = Q_obs['Unnamed: 0']
Q_obs.index.name = None
Q_obs = Q_obs.drop(['Unnamed: 0'], axis=1)
# start_date = Q_obs.index >= start_day_obs
# end_date = Q_obs.index <= end_day_obs
# Q_obs = Q_obs.loc[start_date & end_date]
#Q_obs

for id in gid:
    Q = Q_obs[id]
    Q = Q.dropna()
    eva_obs = EVA(Q)
    eva_obs.get_extremes(method='BM', block_size='365.2425D', errors='ignore')
    eva_obs.fit_model(distribution = 'gumbel_r') #forced to follow gumbel distribution
    summary_obs = eva_obs.get_summary(return_period=Tr, alpha=0.95, n_samples=1000)
    summary_obs.columns = ['Qobs', 'lower ci', 'upper ci']
    Tr_res = summary_obs
    #summary_obs
    
    eva_obs.plot_extremes()
    plt.title('Extreme Value Analysis - Block Maxima, ' + id + '_Observed')
    plt.xlabel('Year')
    plt.ylabel('Discharge [$m^3$/s]')
    frmt = 'jpg'
    name_plot='C:/Users/villarre/DataAnalysis/figures/EVA/QobsAll/fig_EVA_BM_' + id + '_Observed.' + frmt
    plt.savefig(name_plot, format=frmt, dpi=1000, bbox_inches='tight')
    plt.close()

    fig, ax = eva_obs.plot_diagnostic(return_period=Tr, alpha=0.95)
    ax[0].set_ylabel('Observed')
    ax[1].set_xlabel('Observed')
    name_plot='C:/Users/villarre/DataAnalysis/figures/EVA/QobsAll/fig_EVA_diag_' + id + '_Observed.' + frmt
    plt.savefig(name_plot, format=frmt, dpi=1000, bbox_inches='tight')
    extent = ax[0].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
    name_plot='C:/Users/villarre/DataAnalysis/figures/EVA/QobsAll/sp1fig_EVA_diag_' + id + '_Observed.' + frmt
    plt.savefig(name_plot, format=frmt, dpi=1000, bbox_inches=extent.expanded(1.05,1.05))
    plt.close()

    f = open('C:/Users/villarre/DataAnalysis/figures/EVA/QobsAll/eva_' + id + '.txt', 'w')
    print(eva_obs, file=f)
    f.close()
    
    Tr_res.to_csv('C:/Users/villarre/DataAnalysis/figures/EVA/QobsAll/Tr_res_' + id + '.csv')
            

    
#########################################################################################
#Return period plot All
        
for model in models:
    for lt in slt:
        start_lt = lt
        end_lt = start_lt + 3
        
        for id in gid:
            print(id)

            Tr_res_obs = pd.read_csv('C:/Users/villarre/DataAnalysis/figures/EVA/QobsAll/Tr_res_' + id + '.csv')
            Tr_res_obs.index = Tr_res_obs['return period']
            Tr_res_obs.index.name = None
            Tr_res_obs = Tr_res_obs.drop(['return period'], axis=1)
            Tr_res_obs.columns = ['Qobs','Qobs_lci','Qobs_uci']
            #Tr_res_obs
            
            Tr_res_ref = pd.read_csv('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/Tr_res_'+id+'_Qref.csv')
            Tr_res_ref.index = Tr_res_ref['return period']
            Tr_res_ref.index.name = None
            Tr_res_ref = Tr_res_ref.drop(['return period'], axis=1)
            Tr_res_ref.columns = ['Qobs','Qobs_lci','Qobs_uci','Qref','Qref_lci','Qref_uci']
            Tr_res_ref = Tr_res_ref.drop(['Qobs','Qobs_lci','Qobs_uci'], axis=1)
            #Tr_res_ref
            
            Tr_res = pd.concat([Tr_res_obs,Tr_res_ref], axis=1)
            
            Qobs_rng = abs(Tr_res['Qobs_lci'] - Tr_res['Qobs_uci'])
            Qref_rng = abs(Tr_res['Qref_lci'] - Tr_res['Qref_uci'])
            UnRed = Qobs_rng/Qref_rng
            Un_mean = round(UnRed.mean(),2)
            #(Qref_rng/Qobs_rng*100).mean()
            Un_1250 = round(Qobs_rng[1250]/Qref_rng[1250], 2)
        
            fig, ax = plt.subplots(1)
            ax.plot(Tr_res.index, Tr_res['Qobs'], marker='.', color='b')
            ax.plot(Tr_res.index, Tr_res['Qref'], marker='.', color='g')
            ax.plot(Tr_res.index, Tr_res['Qobs_lci'], ls='--', color='b', alpha=0.5)
            ax.plot(Tr_res.index, Tr_res['Qobs_uci'], ls='--', color='b', alpha=0.5)
            ax.plot(Tr_res.index, Tr_res['Qref_lci'], ls='--', color='g', alpha=0.5)
            ax.plot(Tr_res.index, Tr_res['Qref_uci'], ls='--', color='g', alpha=0.5)
            ax.fill_between(Tr_res.index, Tr_res['Qobs_lci'], Tr_res['Qobs_uci'], color='b', alpha=0.2)
            ax.fill_between(Tr_res.index, Tr_res['Qref_lci'], Tr_res['Qref_uci'], color='g', alpha=0.2)
            ax.set_xscale('log')
            ax.get_xaxis().set_major_formatter(ScalarFormatter())
            ax.set_xticks(Trx)
            ax.set_xticklabels(Trx) #, rotation=90)
            ax.set_xlabel('Returnn Period [Years]')
            ax.set_ylabel('Discharge [$m^3$/s]')
            ax.set_title('Return Period Plot, '+id)
            ax.text(1250, min(min(Tr_res['Qobs']),min(Tr_res['Qref'])),
                    'Uncertainty reduction:\n'
                    'Average by '+str(Un_mean)+' times\n'
                    'Tr=1250 by '+str(Un_1250)+' times', ha='right', linespacing=1.3)
            obs_lg = mlines.Line2D([], [], label='Qobs', color='b')
            ref_lg = mlines.Line2D([], [], label='Qref', color='g')
            obs_ci_lg = mlines.Line2D([], [], label='', color='g')
            ax.legend(handles=[ref_lg, obs_lg])
            fig.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/Tr_res_'+id+'_Qref_All.jpg', format='jpg', dpi=1000, bbox_inches='tight')
            plt.close()
  
#########################################################################################
#Comparison of extreme values for different lead times All

for model in models:
    for id in gid:
        # model = 2
        # id = '9286162'
        
        if not os.path.exists('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/EVAcomparisonAll/'):
                os.mkdir('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/EVAcomparisonAll/')

        Tr_res_obs = pd.read_csv('C:/Users/villarre/DataAnalysis/figures/EVA/QobsAll/Tr_res_' + id + '.csv')
        Tr_res_obs.index = Tr_res_obs['return period']
        Tr_res_obs.index.name = None
        Tr_res_obs = Tr_res_obs.drop(['return period'], axis=1)
        Tr_res_obs.columns = ['Qobs','Qobs_lci','Qobs_uci']
        Tr_Q = Tr_res_obs*1
        #Tr_res_obs
        
        for lt in slt:
            start_lt = lt
            end_lt = start_lt + 3
            print(id, start_lt, end_lt)

            Tr_res = pd.read_csv('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/EVAmax/Tr_res_'+id+'_Qref.csv')
            Tr_res.index = Tr_res['return period']
            Tr_res.index.name = None
            Tr_res = Tr_res.drop(['return period'], axis=1)
            Tr_res.columns = ['Qobs','Qobs_lci','Qobs_uci','Qref_'+str(start_lt)+str(end_lt),'Qref_lci_'+str(start_lt)+str(end_lt),'Qref_uci_'+str(start_lt)+str(end_lt)]
            Tr_res = Tr_res.drop(['Qobs','Qobs_lci','Qobs_uci'], axis=1)
            Tr_Q = pd.concat([Tr_Q,Tr_res], axis=1)
            # Tr_Q
            
        idx=['Qobs','Qref, 5-8','Qref, 8-11','Qref, 11-14']
        Q = pd.DataFrame([Tr_Q['Qobs'], Tr_Q['Qref_58'], Tr_Q['Qref_811'], Tr_Q['Qref_1114']], index=idx)
        Qu = pd.DataFrame([Tr_Q['Qobs_uci'], Tr_Q['Qref_uci_58'], Tr_Q['Qref_uci_811'], Tr_Q['Qref_uci_1114']], index=idx)
        Ql = pd.DataFrame([Tr_Q['Qobs_lci'], Tr_Q['Qref_lci_58'], Tr_Q['Qref_lci_811'], Tr_Q['Qref_lci_1114']], index=idx)
        Qerr = (Qu - Ql)/2
        
        Trx = [[2,25,300], [5,50,1000], [10,100,1250]]
        colors = {2:'b', 5:'b', 10:'b', 25:'y', 50:'y', 100:'y', 300:'g', 1000:'g', 1250:'g'}
        axvl = [0.5,1.5,2.5,3.5]
        
        for trx in Trx:
            print(trx[0])
            for tr in trx:
                plt.errorbar(Q.index, Q[tr], yerr=Qerr[tr], marker='_', ms=50, ls='', color=colors[tr], elinewidth=50, alpha=0.5)
                plt.axhline(Q[tr]['Qobs'], ls='--', color=colors[tr], alpha=0.5)
                plt.scatter(Q.index, Q[tr], marker='_', s=200, color=colors[tr] , label=str(tr)+' years')
            [plt.axvline(_axvl, ls='--', lw=1, color='k', alpha=0.5) for _axvl in axvl]
            plt.xlim(-0.5,3.5)
            plt.xlabel('Lead time (days)')
            plt.ylabel('Discharge [$m^3$/s]')
            plt.title(id+', Comparison of discharge return periods, KsHF='+str(model))
            plt.legend(fontsize=8, loc='best', ncol=3)
            # plt.show()
            plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/EVAcomparisonAll/'+id+'_Qobs_vs_Qref_'+str(trx[0])+'_All.jpg', format='jpg', dpi=1000, bbox_inches='tight')
            plt.close()
            
        







