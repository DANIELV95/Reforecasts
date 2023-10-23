# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:26:17 2022

@author: HIDRAULICA-Dani
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import spotpy
import csv
import gc

os.chdir('D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS')

#Compare Qsim vs Qobs (CLICOM and ERA5 vs River gauge station)
gauges = ['CLICOM', 'ERA5']
# gauge = 'CLICOM'
CLICOM = pd.read_csv('../PYR/HMS/Results/csv/Q_CLICOM.csv', parse_dates=['Unnamed: 0'])
CLICOM.index = CLICOM['Unnamed: 0']
CLICOM.index.name = None
CLICOM = CLICOM.drop(['Unnamed: 0'], axis=1)

ERA5 = pd.read_csv('../PYR/HMS/Results/csv/Q_ERA5.csv', parse_dates=['Unnamed: 0'])
ERA5.index = ERA5['Unnamed: 0']
ERA5.index.name = None
ERA5 = ERA5.drop(['Unnamed: 0'], axis=1)

Qsim = pd.concat([CLICOM['CLICOM'], ERA5['ERA5']], axis=1)

#pbserved discharge from river gauge
Q_mean = pd.read_csv('./VARIOS/Tables/Q_mean.csv', parse_dates=['Unnamed: 0'])
Q_mean.index = Q_mean['Unnamed: 0']
Q_mean.index.name = None
Q_mean = Q_mean.drop(['Unnamed: 0'], axis=1)
Q_mean.columns = ['Qobs']


Qclicom = CLICOM['CLICOM'][(CLICOM['CLICOM'].index>=Q_mean.index[0]) & (CLICOM['CLICOM'].index<=Q_mean.index[-1])]
Qclicom
Qobs = Q_mean['Qobs']
Qobs
spotpy.objectivefunctions.nashsutcliffe(Qobs,Qclicom)

#Monthly NSE
Qobs_m = Qobs.resample('M').mean()
Qclicom_m = Qclicom.resample('M').mean()
spotpy.objectivefunctions.nashsutcliffe(Qobs_m,Qclicom_m)

#High Flows NSE
Qobs_hf = Qobs[Qobs>2]
Qclicom_hf = Qclicom[Qobs>2]
spotpy.objectivefunctions.nashsutcliffe(Qobs_hf,Qclicom_hf)



for gauge in gauges:
        
    if not os.path.exists('./VARIOS/Figures/discharge/' + gauge):
        os.mkdir('./VARIOS/Figures/discharge/' + gauge)

    f1=open('./VARIOS/Figures/discharge/'+gauge+'/'+gauge+'.csv','w')
    writer=csv.writer(f1,delimiter=',',lineterminator='\n',)
    
    Q_sim = pd.DataFrame(Qsim[gauge].dropna().values, Qsim[gauge].dropna().index, columns=[gauge])
    Q_mean.columns = [gauge]

    z1 = []
    z2 = []
    z3 = []
    years = []

    for i in range(0,len(Q_sim.resample('Y').max())+1):
        year= Q_sim[gauge].index[0].year-1 + i
        mask1 = (Q_sim[gauge].index > str(year)+'-12-31') & (Q_sim[gauge].index<= str(year+1)+'-12-31')
        mask2 = (Q_mean[gauge].index > str(year)+'-12-31') & (Q_mean[gauge].index<= str(year+1)+'-12-31')
        A=pd.concat([Q_sim[gauge][mask1],Q_mean[gauge][mask2]],axis=1)
        B=A.dropna()
        B.columns=[gauge,'Qobs']

        if not B.empty:
            kge_wflow,cc,alpha,beta=spotpy.objectivefunctions.kge(B['Qobs'],B[gauge],return_all=True)
            nse_wflow=spotpy.objectivefunctions.nashsutcliffe(B['Qobs'],B[gauge])
            pbias_wflow=spotpy.objectivefunctions.pbias(B['Qobs'],B[gauge])
            rmse_wflow=spotpy.objectivefunctions.rmse(B['Qobs'],B[gauge])
            logp_wflow=spotpy.objectivefunctions.log_p(B['Qobs'],B[gauge])
            row=[year+1,round(pbias_wflow,2),round(rmse_wflow,2),round(logp_wflow,2),round(nse_wflow,2),
                 round(kge_wflow,2),round(cc,2),round(alpha,2),round(beta,2)]

            print(row)

            writer.writerow(row)
            years.append(year+1)
            z1.append(nse_wflow)
            z2.append(kge_wflow)
            z3.append(rmse_wflow)

            beingsaved=plt.figure(i)
            plt.figure(i)
            plt.subplot(1,1,1)
            plt.plot(Q_sim[gauge].index[mask1], Q_sim[gauge][mask1], color='b', alpha=0.8, label=gauge)
            plt.plot(Q_mean[gauge].index[mask2], Q_mean[gauge][mask2], color='y', alpha=0.8, label='Qobs')
            plt.title(gauge+' NSE='+str(round(nse_wflow,2))+' KGE='+str(round(kge_wflow,2))+' ('+str(round(cc,2))+','+
                      str(round(alpha,2))+','+str(round(beta,2))+')')
            # ax=plt.gca()
            # Q_sim[gauge].loc[mask1].plot(ax=plt.gca(),color='b', alpha=0)
            # Q_mean[gauge].loc[mask2].plot(ax=plt.gca(),color='k', alpha=0) #,linestyle='dotted')
            # ax.set_ylim(0,max(Q_sim[gauge].loc[mask1].max(),Q_mean[gauge].loc[mask2].max()))
            # ax.set_ylabel(r'Discharge [$m^3$/s]')
            # colors = {gauge:'b', 'Qobs':'y'}
            # labels = colors.keys()
            # handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
            plt.legend() #handles, labels)
            frmt='jpg'
            name_plot='fig_'+str(year+1)+'_'+gauge+'_Q_sim_vs_Qobs.'+frmt
            beingsaved.savefig('./VARIOS/Figures/discharge/'+gauge+'/'+name_plot, format=frmt, dpi=1000)
            plt.close(i)
            del mask1,mask2

    f1.close()

######################################################################################################################

    year_i= Q_mean[gauge].index[0].year-1
    year_f= Q_mean[gauge].index[-1].year
    
    mask1 = (Q_sim[gauge].index > str(year_i)+'-12-31') & (Q_sim[gauge].index<= str(year_f)+'-12-31')
    mask2 = (Q_mean[gauge].index > str(year_i)+'-12-31') & (Q_mean[gauge].index<= str(year_f)+'-12-31')
    A=pd.concat([Q_sim[gauge][mask1],Q_mean[gauge][mask2]],axis=1)
    B=A.dropna()
    B.columns=[gauge,'Qobs']
    B=B[:-1]

    if not B.empty:
        kge_wflow,cc,alpha,beta=spotpy.objectivefunctions.kge(B['Qobs'],B[gauge],return_all=True)
        nse_wflow=spotpy.objectivefunctions.nashsutcliffe(B['Qobs'],B[gauge])
        pbias_wflow=spotpy.objectivefunctions.pbias(B['Qobs'],B[gauge])
        rmse_wflow=spotpy.objectivefunctions.rmse(B['Qobs'],B[gauge])
        logp_wflow=spotpy.objectivefunctions.log_p(B['Qobs'],B[gauge])
        row=[year+1,round(pbias_wflow,2),round(rmse_wflow,2),round(logp_wflow,2),round(nse_wflow,2),
             round(kge_wflow,2),round(cc,2),round(alpha,2),round(beta,2)]

        print(row)

        years.append(year+1)
        z1.append(nse_wflow)
        z2.append(kge_wflow)   
        z3.append(rmse_wflow)

        plt.figure(year)
        beingsaved=plt.figure(year)
        plt.subplot(3,1,1)
        plt.title(gauge+' NSE='+str(round(z1[-1],2))+' RMSE='+str(round(z3[-1],2))+' KGE='+str(round(z2[-1],2))+
                  ' ('+str(round(cc,2))+','+str(round(alpha,2))+','+str(round(beta,2))+')')
        plt.plot(years[:-1],z1[:-1],color='r',marker='+', linestyle='none')
        plt.hlines(z1[-1],year_i,year_f,color='k',linestyle='--',label='NSE= '+str(z1[-1]))
        plt.xticks(range(year_i, year_f,4))
        plt.ylabel('NSE')
        plt.ylim(0,1)
        plt.subplot(3,1,2)
        plt.plot(years[:-1],z3[:-1],color='r',marker='+', linestyle='none')
        plt.hlines(z3[-1],year_i,year_f,color='k',linestyle='--')
        plt.xticks(range(year_i, year_f, 4))
        plt.ylabel('RMSE')
        plt.ylim(0,10)
        plt.subplot(3,1,3)
        plt.plot(years[:-1],z2[:-1],color='r',marker='+', linestyle='none')
        plt.hlines(z2[-1],year_i,year_f,color='k',linestyle='--',label='KGE= '+str(z2[-1]))
        plt.xticks(range(year_i, year_f, 4))
        plt.ylabel('KGE')
        plt.xlabel('Year')
        plt.ylim(0.5,1)
        frmt='jpg'
        name_plot='fig_'+str(year+1)+'_'+gauge+'_Q_sim_vs_Qobs.'+frmt
        beingsaved.savefig('./VARIOS/Figures/discharge/'+gauge+'/'+name_plot, format=frmt, dpi=1000)
        
        plt.close()

######################################################################################################################

        plt.figure(1)
        beingsaved=plt.figure(1)

        plt.subplot(2,1,1)
        plt.plot(B.groupby(B.index.year).max())
        plt.legend([gauge,'Qobs'], loc='best',bbox_to_anchor=(0.5, 1.05))
        plt.ylabel('Annual Max. [$m^3$/s]')
        plt.xticks(range(year_i, year_f, 4))

        plt.subplot(2,1,2)
        plt.scatter(B.groupby(B.index.year).max()[gauge],B.groupby(B.index.year).max()['Qobs'],color='k')
        y_lim = (min(min(B.groupby(B.index.year).max()[gauge]),min(B.groupby(B.index.year).max()['Qobs'])),
                 max(max(B.groupby(B.index.year).max()[gauge]),max(B.groupby(B.index.year).max()['Qobs'])))
        x_lim = (min(min(B.groupby(B.index.year).max()[gauge]),min(B.groupby(B.index.year).max()['Qobs'])),
                 max(max(B.groupby(B.index.year).max()[gauge]),max(B.groupby(B.index.year).max()['Qobs'])))
        plt.plot(x_lim, y_lim, color = 'r')
        plt.ylim(y_lim)
        plt.xlim(x_lim)
        plt.ylabel('Observed [$m^3$/s]')

        frmt='jpg'
        name_plot='fig_'+str(year+1)+'_'+gauge+'_Q_sim_vs_Qobs_maxmin.'+frmt
        beingsaved.savefig('./VARIOS/Figures/discharge/'+gauge+'/'+name_plot, format=frmt, dpi=1000)
        
        plt.close()

        monthly=B.resample('M').mean()

        from scipy.stats import ttest_ind

        # if p>0.05 reject hypothesis of equal average
        for i in range(1,12):
            TT=ttest_ind(monthly[monthly.index.month==i]['Qobs'],monthly[monthly.index.month==i][gauge])
            print(TT)

        yearly=B.resample('Y').mean()
        ttest_ind(yearly['Qobs'],yearly[gauge])

        yearly=B.resample('Y').max()
        ttest_ind(yearly['Qobs'],yearly[gauge])

        yearly=B.resample('Y').min()
        ttest_ind(yearly['Qobs'],yearly[gauge])
                            
print('Task completed')

gc.collect()
#f1.close()



maskQ = (CLICOM.index >= '1973-02-01') & (CLICOM.index <= '1994-12-31')
CLICOM = CLICOM['CLICOM'][maskQ]
CLICOM = pd.DataFrame(CLICOM.values, CLICOM.index, columns=['CLICOM'])

len(CLICOM['CLICOM'])
nse=spotpy.objectivefunctions.nashsutcliffe(Q_mean['Qobs'],CLICOM['CLICOM'])
