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

gauges = ['P']

CLICOM = pd.read_csv('./VARIOS/Tables/clicom_malla_data.csv', parse_dates=['Unnamed: 0'])
CLICOM.index = CLICOM['Unnamed: 0']
CLICOM.index.name = None
CLICOM = CLICOM.drop(['Unnamed: 0', 'T'], axis=1)
CLICOM.columns = ['P']

#precipitation from individual rain gauge
Pobs_m = pd.read_csv('./VARIOS/Tables/precip_est.csv', parse_dates=['Unnamed: 0'])
Pobs_m.index = Pobs_m['Unnamed: 0']
Pobs_m.index.name = None
maskP = (Pobs_m.index >= '1929-01-01') & (Pobs_m.index <= '2001-12-10')
Pobs_m = Pobs_m['19049'][maskP]
Pobs_m = pd.DataFrame(Pobs_m.values, Pobs_m.index, columns=['P'])

#weighted average precipitation from rain gauges
Pobs_m = pd.read_csv('./VARIOS/Tables/precip_wa.csv', parse_dates=['Unnamed: 0'])
Pobs_m.index = Pobs_m['Unnamed: 0']
Pobs_m.index.name = None
Pobs_m = Pobs_m.drop(['Unnamed: 0'], axis=1)
Pobs_m.columns = ['P']

for gauge in gauges:
        
    if not os.path.exists('./VARIOS/Figures/precipitation/' + gauge):
        os.mkdir('./VARIOS/Figures/precipitation/' + gauge)

    f1=open('./VARIOS/Figures/precipitation/'+gauge+'/'+gauge+'.csv','w')
    writer=csv.writer(f1,delimiter=',',lineterminator='\n',)

    z1 = []
    z2 = []
    z3 = []
    years = []

    for i in range(0,50):
        year=1959+i
        mask1 = (CLICOM[gauge].index > str(year)+'-12-31') & (CLICOM[gauge].index<= str(year+1)+'-12-31')
        mask2 = (Pobs_m[gauge].index > str(year)+'-12-31') & (Pobs_m[gauge].index<= str(year+1)+'-12-31')
        A=pd.concat([CLICOM[gauge][mask1],Pobs_m[gauge][mask2]],axis=1)
        B=A.dropna()
        B.columns=['CLICOM','Pobs']

        if not B.empty:
            kge_wflow,cc,alpha,beta=spotpy.objectivefunctions.kge(B['Pobs'],B['CLICOM'],return_all=True)
            nse_wflow=spotpy.objectivefunctions.nashsutcliffe(B['Pobs'],B['CLICOM'])
            pbias_wflow=spotpy.objectivefunctions.pbias(B['Pobs'],B['CLICOM'])
            rmse_wflow=spotpy.objectivefunctions.rmse(B['Pobs'],B['CLICOM'])
            logp_wflow=spotpy.objectivefunctions.log_p(B['Pobs'],B['CLICOM'])
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
            plt.bar(CLICOM[gauge].index[mask1], CLICOM[gauge][mask1], color='b', alpha=0.8)
            plt.bar(Pobs_m[gauge].index[mask2], Pobs_m[gauge][mask2], color='y', alpha=0.8)
            plt.title(gauge+' NSE='+str(round(nse_wflow,2))+' KGE='+str(round(kge_wflow,2))+' ('+str(round(cc,2))+','+
                      str(round(alpha,2))+','+str(round(beta,2))+')')
            ax=plt.gca()
            CLICOM[gauge].loc[mask1].plot(ax=plt.gca(),color='b', alpha=0)
            Pobs_m[gauge].loc[mask2].plot(ax=plt.gca(),color='k', alpha=0) #,linestyle='dotted')
            ax.set_ylim(0,max(CLICOM[gauge].loc[mask1].max(),Pobs_m[gauge].loc[mask2].max()))
            ax.set_ylabel(r'Precipitation [mm/day]')
            colors = {'CLICOM':'b', 'Pobs':'y'}
            labels = colors.keys()
            handles = [plt.Rectangle((0,0),1,1, color=colors[label]) for label in labels]
            plt.legend(handles, labels)
            frmt='jpg'
            name_plot='fig_'+str(year+1)+'_'+gauge+'_CLICOM_vs_Pobs.'+frmt
            beingsaved.savefig('./VARIOS/Figures/precipitation/'+gauge+'/'+name_plot, format=frmt, dpi=1000)
            plt.close(i)
            del mask1,mask2

    f1.close()

######################################################################################################################

    year_i=1960
    year_f=2008
    
    mask1 = (CLICOM[gauge].index > str(year_i)+'-12-31') & (CLICOM[gauge].index<= str(year_f)+'-12-31')
    mask2 = (Pobs_m[gauge].index > str(year_i)+'-12-31') & (Pobs_m[gauge].index<= str(year_f)+'-12-31')
    A=pd.concat([CLICOM[gauge][mask1],Pobs_m[gauge][mask2]],axis=1)
    B=A.dropna()
    B.columns=['CLICOM','Pobs']
    B=B[:-1]

    if not B.empty:
        kge_wflow,cc,alpha,beta=spotpy.objectivefunctions.kge(B['Pobs'],B['CLICOM'],return_all=True)
        nse_wflow=spotpy.objectivefunctions.nashsutcliffe(B['Pobs'],B['CLICOM'])
        pbias_wflow=spotpy.objectivefunctions.pbias(B['Pobs'],B['CLICOM'])
        rmse_wflow=spotpy.objectivefunctions.rmse(B['Pobs'],B['CLICOM'])
        logp_wflow=spotpy.objectivefunctions.log_p(B['Pobs'],B['CLICOM'])
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
        plt.hlines(z1[-1],years[0],years[-1],color='k',linestyle='--',label='NSE= '+str(z1[-1]))
        plt.xticks(range(years[0], years[len(years)-1],4))
        plt.ylabel('NSE')
        plt.ylim(0,1)
        plt.subplot(3,1,2)
        plt.plot(years[:-1],z3[:-1],color='r',marker='+', linestyle='none')
        plt.hlines(z3[-1],years[0],years[-1],color='k',linestyle='--')
        plt.xticks(range(years[0], years[len(years)-1],4))
        plt.ylabel('RMSE')
        plt.ylim(0,10)
        plt.subplot(3,1,3)
        plt.plot(years[:-1],z2[:-1],color='r',marker='+', linestyle='none')
        plt.hlines(z2[-1],years[0],years[-1],color='k',linestyle='--',label='KGE= '+str(z2[-1]))
        plt.xticks(range(years[0], years[len(years)-1],4))
        plt.ylabel('KGE')
        plt.xlabel('Year')
        plt.ylim(0.5,1)
        frmt='jpg'
        name_plot='fig_'+str(year+1)+'_'+gauge+'_CLICOM_vs_Pobs.'+frmt
        beingsaved.savefig('./VARIOS/Figures/precipitation/'+gauge+'/'+name_plot, format=frmt, dpi=1000)
        
        plt.close()

######################################################################################################################

        plt.figure(1)
        beingsaved=plt.figure(1)

        plt.subplot(2,1,1)
        plt.plot(B.groupby(B.index.year).max())
        plt.legend(['CLICOM','Pobs'], loc='best',bbox_to_anchor=(0.5, 1.05))
        plt.ylabel('Annual Max. [mm/day]')
        plt.xticks(range(years[0], years[len(years)-1],4))

        plt.subplot(2,1,2)
        plt.scatter(B.groupby(B.index.year).max()['CLICOM'],B.groupby(B.index.year).max()['Pobs'],color='k')
        y_lim = (min(min(B.groupby(B.index.year).max()['CLICOM']),min(B.groupby(B.index.year).max()['Pobs'])),
                 max(max(B.groupby(B.index.year).max()['CLICOM']),max(B.groupby(B.index.year).max()['Pobs'])))
        x_lim = (min(min(B.groupby(B.index.year).max()['CLICOM']),min(B.groupby(B.index.year).max()['Pobs'])),
                 max(max(B.groupby(B.index.year).max()['CLICOM']),max(B.groupby(B.index.year).max()['Pobs'])))
        plt.plot(x_lim, y_lim, color = 'r')
        plt.ylim(y_lim)
        plt.xlim(x_lim)
        plt.ylabel('Observed [mm/day]')

        frmt='jpg'
        name_plot='fig_'+str(year+1)+'_'+gauge+'_CLICOM_vs_Pobs_maxmin.'+frmt
        beingsaved.savefig('./VARIOS/Figures/precipitation/'+gauge+'/'+name_plot, format=frmt, dpi=1000)
        
        plt.close()

        monthly=B.resample('M').mean()

        from scipy.stats import ttest_ind

        # if p>0.05 reject hypothesis of equal average
        for i in range(1,12):
            TT=ttest_ind(monthly[monthly.index.month==i]['Pobs'],monthly[monthly.index.month==i]['CLICOM'])
            print(TT)

        yearly=B.resample('Y').mean()
        ttest_ind(yearly['Pobs'],yearly['CLICOM'])

        yearly=B.resample('Y').max()
        ttest_ind(yearly['Pobs'],yearly['CLICOM'])

        yearly=B.resample('Y').min()
        ttest_ind(yearly['Pobs'],yearly['CLICOM'])
                            
print('Task completed')

gc.collect()
#f1.close()