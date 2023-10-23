# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 16:26:17 2022

@author: HIDRAULICA-Dani
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spotpy
from scipy.stats.stats import pearsonr

os.chdir('D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS')

#Variables
ensemble = [0,1,2,3,4,5,6,7,8,9,10] #with control
#slt = [*range(2,17)]
slt = [*range(2,17,3)]
Tr = [2,5,10,25,50,100,300,1000,1250]
years = range(2000, 2021, 2)


#Datasets precipitation
Pobs = pd.read_csv('./VARIOS/Tables/precip_wa.csv', parse_dates=['Unnamed: 0'])
Pobs.index = Pobs['Unnamed: 0']
Pobs = Pobs.drop(['Unnamed: 0'], axis=1)
Pobs.index.name = None
Pobs.columns = ['Pobs']
Pobs_m = Pobs.resample('Y').max()
Pobs_m['Pobs'][:15] = np.nan
Pobs_m = Pobs_m.dropna()[6:]
Pobs_s = Pobs.resample('Y').sum()
Pobs_s['Pobs'][:15] = np.nan
Pobs_s = Pobs_s.dropna()[6:]
Pobs_s = Pobs_s[Pobs_s['Pobs']>10]
Pobs_s[40:]
plt.plot(Pobs[-3287+170:-3287+365-150])

Pera = pd.read_csv('./VARIOS/Tables/era5_all_wa.csv', parse_dates=['Unnamed: 0'])
Pera.index = Pera['Unnamed: 0']
Pera = Pera.drop(['Unnamed: 0'], axis=1)
Pera.index.name = None
Pera = pd.DataFrame(Pera['tp'].values, Pera['tp'].index, columns=['ERA5'])
Pera = Pera.resample('D').sum()
Pera.to_csv('./VARIOS/Tables/era5_all_wa_d.csv')
Pera_m = Pera.resample('Y').max()
Pera_s = Pera.resample('Y').sum()
Pera_s[Pera_s['ERA5']>1100]
Pera

Pclic = pd.read_csv('./VARIOS/Tables/clicom_malla_data.csv', parse_dates=['Unnamed: 0'])
Pclic.index = Pclic['Unnamed: 0']
Pclic = Pclic.drop(['Unnamed: 0', 'T'], axis=1)
Pclic.index.name = None
Pclic.columns = ['CLICOM']
Pclic_m = Pclic.resample('Y').max()[:-5]
Pclic_s = Pclic.resample('Y').sum()[:-5]
Pclic

#Maximos de preciptiacion anual
plt.plot(Pobs_m.index.year, Pobs_m, label='Pobs', alpha=0.7, color='r')
plt.plot(Pclic_m.index.year, Pclic_m, label='CLICOM', alpha=0.7, color='y')
plt.plot(Pera_m.index.year, Pera_m, label='ERA5', alpha=0.7, color='b')
plt.legend()
plt.title('Precipitación máxima anual acumulada en 24 horas')
plt.ylabel('Precipitación [mm]')
# plt.savefig('../SSP/P_max.jpg', format='jpg', dpi=1000)
plt.close()


#Datasets discharge
Qobs = pd.read_csv('../PYR/HMS/Results/csv/Q_mean.csv', parse_dates=['Unnamed: 0'])
Qobs.index = Qobs['Unnamed: 0']
Qobs = Qobs.drop(['Unnamed: 0'], axis=1)
Qobs.index.name = None
Qobs.columns = ['Qobs']
Qobs_m = Qobs.resample('Y').max()
Qobs_p = Qobs.resample('Y').mean()
Qobs

Qera = pd.read_csv('../PYR/HMS/Results/csv/Q_ERA5.csv', parse_dates=['Unnamed: 0'])
Qera.index = Qera['Unnamed: 0']
Qera = Qera.drop(['Unnamed: 0'], axis=1)
Qera.index.name = None
Qera_m = Qera.resample('Y').max()
Qera_p = Qera.resample('Y').mean()
Qera

Qclic = pd.read_csv('../PYR/HMS/Results/csv/Q_CLICOM.csv', parse_dates=['Unnamed: 0'])
Qclic.index = Qclic['Unnamed: 0']
Qclic = Qclic.drop(['Unnamed: 0'], axis=1)
Qclic.index.name = None
Qclic_m = Qclic.resample('Y').max()[:-5]
Qclic_p = Qclic.resample('Y').mean()[:-5]
Qclic

Qpobs = pd.read_csv('../PYR/HMS/Results/csv/Q_Pobs.csv', parse_dates=['Unnamed: 0'])
Qpobs.index = Qpobs['Unnamed: 0']
Qpobs = Qpobs.drop(['Unnamed: 0'], axis=1)
Qpobs.index.name = None
Qpobs_m = Qpobs.resample('Y').max()
Qpobs_p = Qpobs.resample('Y').mean()
Qpobs_p = Qpobs_p[Qpobs_p['Pobs']>0.1]
Qpobs

#Maximos de gasto anual
plt.plot(Qobs_m.index.year, Qobs_m, label='Qobs', alpha=0.7, color='k')
# plt.plot(Qpobs_m.index.year, Qpobs_m, label='Qpobs', alpha=0.7, color='r')
plt.plot(Qclic_m.index.year, Qclic_m, label='CLICOM', alpha=0.7, color='y')
# plt.plot(Qera_m.index.year, Qera_m, label='ERA5', alpha=0.7, color='b')
plt.legend()
plt.title('Gasto medio diario máximo anual')
plt.ylabel('Gasto [$m^3$/s]')
# plt.savefig('../SSP/Q_max.jpg', format='jpg', dpi=1000)
plt.close()

################################################################################
#Correlation coefficient Qobs vs Qclicom max, mean yearly and daily
mask = ((Qclic.index >= '1973-02-01') & (Qclic.index <= '1994-12-31'))
x = Qobs.values.flatten()
y = Qclic[mask].values.flatten()
corrp = pearsonr(x,y)
qmax = max([max(x),max(y)])
f = np.polyfit(x,y,1)
z = np.poly1d(f)
plt.scatter(x,y, marker='o', s=2, label='Data')
plt.plot([0,qmax],[0,qmax], ls='--', lw=1, color='k', label='1:1')
plt.plot(x,z(x), ls='-', lw=1, color='b', label='Reg')
# plt.plot(x.sort_values(), y.sort_values(), ls='-', lw=1, color='g', label='Q-Q')
plt.title('Comparación de Q obs vs Q clicom')
plt.xlabel('Q observado [$m^3$/s]')
plt.ylabel('Q simulado [$m^3$/s]')
plt.text(0, max([max(x),max(y)]), ha='left', va='top',
         s= 'Cc = '+str(round(corrp[0],3))+'\np-value = '+str(round(corrp[1],3)))
plt.legend(loc='lower right')
# plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/correlation/scatter2/'+id+'_Ref='+str(ens+1)+'vs'+str(enss+1)+'_Q mean daily_wr.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.close()

mask = ((Qclic_m.index >= '1973-02-01') & (Qclic_m.index <= '1994-12-31'))
x = Qobs_m.values.flatten()
y = Qclic_m[mask].values.flatten()
corrp = pearsonr(x,y)
qmax = max([max(x),max(y)])
f = np.polyfit(x,y,1)
z = np.poly1d(f)
plt.scatter(x,y, marker='o', s=5, label='Data')
plt.plot([0,qmax],[0,qmax], ls='--', lw=1, color='k', label='1:1')
plt.plot(x,z(x), ls='-', lw=1, color='b', label='Reg')
# plt.plot(x.sort_values(), y.sort_values(), ls='-', lw=1, color='g', label='Q-Q')
plt.title('Comparación de Q obs vs Q clicom')
plt.xlabel('Q observado [$m^3$/s]')
plt.ylabel('Q simulado [$m^3$/s]')
plt.text(0, max([max(x),max(y)]), ha='left', va='top',
         s= 'Cc = '+str(round(corrp[0],3))+'\np-value = '+str(round(corrp[1],3)))
plt.legend(loc='lower right')
# plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/correlation/scatter2/'+id+'_Ref='+str(ens+1)+'vs'+str(enss+1)+'_Q mean daily_wr.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.close()

mask = ((Qclic_p.index >= '1973-02-01') & (Qclic_p.index <= '1994-12-31'))
x = Qobs_p.values.flatten()
y = Qclic_p[mask].values.flatten()
corrp = pearsonr(x,y)
qmax = max([max(x),max(y)])
f = np.polyfit(x,y,1)
z = np.poly1d(f)
plt.scatter(x,y, marker='o', s=5, label='Data')
plt.plot([0,qmax],[0,qmax], ls='--', lw=1, color='k', label='1:1')
plt.plot(x,z(x), ls='-', lw=1, color='b', label='Reg')
# plt.plot(x.sort_values(), y.sort_values(), ls='-', lw=1, color='g', label='Q-Q')
plt.title('Comparación de Q obs vs Q clicom')
plt.xlabel('Q observado [$m^3$/s]')
plt.ylabel('Q simulado [$m^3$/s]')
plt.text(0, max([max(x),max(y)]), ha='left', va='top',
         s= 'Cc = '+str(round(corrp[0],3))+'\np-value = '+str(round(corrp[1],3)))
plt.legend(loc='lower right')
# plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/correlation/scatter2/'+id+'_Ref='+str(ens+1)+'vs'+str(enss+1)+'_Q mean daily_wr.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.close()


#Correlation coefficient Qobs vs Qera5 max, mean yearly and daily
mask = ((Qera.index >= '1973-02-01') & (Qera.index <= '1994-12-31'))
x = Qobs.values.flatten()
y = Qera[mask].values.flatten()
corrp = pearsonr(x,y)
qmax = max([max(x),max(y)])
f = np.polyfit(x,y,1)
z = np.poly1d(f)
plt.scatter(x,y, marker='o', s=2, label='Data')
plt.plot([0,qmax],[0,qmax], ls='--', lw=1, color='k', label='1:1')
plt.plot(x,z(x), ls='-', lw=1, color='b', label='Reg')
# plt.plot(x.sort_values(), y.sort_values(), ls='-', lw=1, color='g', label='Q-Q')
plt.title('Comparación de Q obs vs Q clicom')
plt.xlabel('Q observado [$m^3$/s]')
plt.ylabel('Q simulado [$m^3$/s]')
plt.text(0, max([max(x),max(y)]), ha='left', va='top',
         s= 'Cc = '+str(round(corrp[0],3))+'\np-value = '+str(round(corrp[1],3)))
plt.legend(loc='lower right')
# plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/correlation/scatter2/'+id+'_Ref='+str(ens+1)+'vs'+str(enss+1)+'_Q mean daily_wr.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.close()

mask = ((Qera_m.index >= '1973-02-01') & (Qera_m.index <= '1994-12-31'))
x = Qobs_m.values.flatten()
y = Qera_m[mask].values.flatten()
corrp = pearsonr(x,y)
qmax = max([max(x),max(y)])
f = np.polyfit(x,y,1)
z = np.poly1d(f)
plt.scatter(x,y, marker='o', s=5, label='Data')
plt.plot([0,qmax],[0,qmax], ls='--', lw=1, color='k', label='1:1')
plt.plot(x,z(x), ls='-', lw=1, color='b', label='Reg')
# plt.plot(x.sort_values(), y.sort_values(), ls='-', lw=1, color='g', label='Q-Q')
plt.title('Comparación de Q obs vs Q clicom')
plt.xlabel('Q observado [$m^3$/s]')
plt.ylabel('Q simulado [$m^3$/s]')
plt.text(0, max([max(x),max(y)]), ha='left', va='top',
         s= 'Cc = '+str(round(corrp[0],3))+'\np-value = '+str(round(corrp[1],3)))
plt.legend(loc='lower right')
# plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/correlation/scatter2/'+id+'_Ref='+str(ens+1)+'vs'+str(enss+1)+'_Q mean daily_wr.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.close()

mask = ((Qera_p.index >= '1973-02-01') & (Qera_p.index <= '1994-12-31'))
x = Qobs_p.values.flatten()
y = Qera_p[mask].values.flatten()
corrp = pearsonr(x,y)
qmax = max([max(x),max(y)])
f = np.polyfit(x,y,1)
z = np.poly1d(f)
plt.scatter(x,y, marker='o', s=5, label='Data')
plt.plot([0,qmax],[0,qmax], ls='--', lw=1, color='k', label='1:1')
plt.plot(x,z(x), ls='-', lw=1, color='b', label='Reg')
# plt.plot(x.sort_values(), y.sort_values(), ls='-', lw=1, color='g', label='Q-Q')
plt.title('Comparación de Q obs vs Q clicom')
plt.xlabel('Q observado [$m^3$/s]')
plt.ylabel('Q simulado [$m^3$/s]')
plt.text(0, max([max(x),max(y)]), ha='left', va='top',
         s= 'Cc = '+str(round(corrp[0],3))+'\np-value = '+str(round(corrp[1],3)))
plt.legend(loc='lower right')
# plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/correlation/scatter2/'+id+'_Ref='+str(ens+1)+'vs'+str(enss+1)+'_Q mean daily_wr.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.close()

################################################################################
#NSE monthly
mask_c = ((Qclic.index >= '1973-02-01') & (Qclic.index <= '1994-12-31'))
mask_e = ((Qera.index >= '1973-02-01') & (Qera.index <= '1994-12-31'))

Qobs_mon = Qobs.resample('M').mean()
Qclic_mon = Qclic[mask_c].resample('M').mean()
Qera_mon = Qera[mask_e].resample('M').mean()

help(spotpy.objectivefunctions.nashsutcliffe)

spotpy.objectivefunctions.nashsutcliffe(Qobs, Qclic[mask_c])
spotpy.objectivefunctions.nashsutcliffe(Qobs, Qera[mask_e])
spotpy.objectivefunctions.nashsutcliffe(Qobs_mon, Qclic_mon)
spotpy.objectivefunctions.nashsutcliffe(Qobs_mon, Qera_mon)

################################################################################


for lt in slt:
    # start_lt = 5
    start_lt = lt
    end_lt = start_lt + 3
    print('lt', start_lt, end_lt)
    
    #Precipitation
    Pc = pd.read_csv('./ECMWF/nc/csv/wa/c_P_228228_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=['Unnamed: 0'])
    Pc.index = Pc['Unnamed: 0']
    Pc = Pc.drop(['Unnamed: 0'], axis=1)
    Pc.index.name = None
    Pc.columns = ['0']
    Pc.index.freq = '1D'
    
    Pe = pd.read_csv('./ECMWF/nc/csv/wa/P_228228_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=['Unnamed: 0'])
    Pe.index = Pe['Unnamed: 0']
    Pe = Pe.drop(['Unnamed: 0'], axis=1)
    Pe.index.name = None
    Pe.index.freq = '1D'
    
    P = pd.concat([Pc, Pe], axis=1)
    P.to_csv('./ECMWF/nc/csv/wa/all/P_228228_lt'+str(start_lt)+str(end_lt)+'_wa.csv')
    P_maxlt = P.resample('Y').max()
    P_maxlt = P_maxlt[:-1]
    
    plt.plot(P_maxlt.index.year, P_maxlt, label=P_maxlt.columns)
    plt.legend(ncol=2, prop={'size': 8})
    plt.title('Precipitación máxima anual acumulada en 24 horas, '+str(start_lt)+' a '+str(end_lt)+' días')
    plt.ylabel('Precipitación [mm]')
    plt.xticks(years)
    # plt.savefig('../SSP/P_max_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.jpg', format='jpg', dpi=1000)
    plt.close()
    
    plt.plot(Pobs_m.index.year, Pobs_m, label='Pobs', alpha=0.7, color='r')
    plt.plot(Pclic_m.index.year, Pclic_m, label='CLICOM', alpha=0.7, color='y')
    plt.plot(Pera_m.index.year, Pera_m, label='ERA5', alpha=0.7, color='b')
    plt.plot(P_maxlt.index.year, P_maxlt, label=P_maxlt.columns)
    plt.legend(ncol=3, prop={'size': 8})
    plt.title('Precipitación máxima anual acumulada en 24 horas, '+str(start_lt)+' a '+str(end_lt)+' días')
    plt.ylabel('Precipitación [mm]')
    plt.xlim(years[0], years[-1])
    plt.xticks(years)
    # plt.savefig('../SSP/P_max_Obs_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.jpg', format='jpg', dpi=1000)
    plt.close()
    
    #Discharge
    Q = pd.read_csv('../PYR/HMS/Results/csv/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.csv', parse_dates=['Unnamed: 0'])
    Q.index = Q['Unnamed: 0']
    Q.index.name = None
    Q = Q.drop(['Unnamed: 0'], axis=1)
    Q_maxlt = Q.resample('Y').max()
    Q_maxlt = Q_maxlt[:-1]
    
    plt.plot(Q_maxlt.index.year, Q_maxlt, label=Q_maxlt.columns)
    plt.plot(Qobs_m.index.year, Qobs_m, label='Qobs', alpha=0.7, color='k')
    plt.legend(ncol=3, prop={'size': 8})
    plt.title('Gasto medio diario máximo anual, '+str(start_lt)+' a '+str(end_lt)+' días')
    plt.ylabel('Gasto [$m^3$/s]')
    plt.xticks(years)
    # plt.savefig('../SSP/Q_max_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    # plt.savefig('../SSP/Q_max_obsLT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    plt.close()
    
    plt.plot(Qobs_m.index.year, Qobs_m, label='Qobs', alpha=0.7, color='k')
    plt.plot(Qpobs_m.index.year, Qpobs_m, label='Qpobs', alpha=0.7, color='r')
    plt.plot(Qclic_m.index.year, Qclic_m, label='CLICOM', alpha=0.7, color='y')
    plt.plot(Qera_m.index.year, Qera_m, label='ERA5', alpha=0.7, color='b')
    plt.plot(Q_maxlt.index.year, Q_maxlt, label=Q_maxlt.columns)
    plt.legend(ncol=3, prop={'size': 8})
    plt.title('Gasto medio diario máximo anual, '+str(start_lt)+' a '+str(end_lt)+' días')
    plt.ylabel('Gasto [$m^3$/s]')
    plt.xlim(years[0], years[-1])
    plt.xticks(years)
    # plt.savefig('../SSP/Q_max_Obs_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    plt.close()
    
#     #Temperature
#     Tc = pd.read_csv('./ECMWF/nc/csv/wa/c_T_121-122_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=['Unnamed: 0'])
#     Tc.index = Tc['Unnamed: 0']
#     Tc = Tc.drop(['Unnamed: 0'], axis=1)
#     Tc.index.name = None
#     Tc.columns = ['0']
#     Tc.index.freq = '1D'
    
#     Te = pd.read_csv('./ECMWF/nc/csv/wa/T_121-122_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=['Unnamed: 0'])
#     Te.index = Te['Unnamed: 0']
#     Te = Te.drop(['Unnamed: 0'], axis=1)
#     Te.index.name = None
#     Te.index.freq = '1D'
    
#     T = pd.concat([Tc, Te], axis=1)
#     T.to_csv('./ECMWF/nc/csv/wa/all/T_121-122_lt'+str(start_lt)+str(end_lt)+'_wa.csv')
#     T_maxlt = T.resample('Y').max()
#     T_maxlt = T_maxlt[:-1]
#     T_minlt = T.resample('Y').min()
#     T_minlt = T_minlt[:-1]

#     plt.plot(T_maxlt.index.year, T_maxlt, label=T_maxlt.columns)
#     plt.legend(ncol=2, prop={'size': 8})
#     plt.title('Temperatura media diaria máxima anual, '+str(start_lt)+' a '+str(end_lt)+' días')
#     plt.ylabel('Temperatura [°C]')
#     plt.xticks(years)
#     plt.savefig('../SSP/T_max_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.jpg', format='jpg', dpi=1000)
#     plt.close()
    
#     plt.plot(T_minlt.index.year, T_minlt, label=T_minlt.columns)
#     plt.legend(ncol=2, prop={'size': 8})
#     plt.title('Temperatura media diaria mínima anual, '+str(start_lt)+' a '+str(end_lt)+' días')
#     plt.ylabel('Temperatura [°C]')
#     plt.xticks(years)
#     plt.savefig('../SSP/T_min_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.jpg', format='jpg', dpi=1000)
#     plt.close()
    
#     #Wind
#     Wc = pd.read_csv('./ECMWF/nc/csv/wa/c_Ww_165-166_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=['Unnamed: 0'])
#     Wc.index = Wc['Unnamed: 0']
#     Wc = Wc.drop(['Unnamed: 0'], axis=1)
#     Wc.index.name = None
#     Wc.columns = ['0']
#     Wc.index.freq = '1D'
    
#     We = pd.read_csv('./ECMWF/nc/csv/wa/Ww_165-166_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=['Unnamed: 0'])
#     We.index = We['Unnamed: 0']
#     We = We.drop(['Unnamed: 0'], axis=1)
#     We.index.name = None
#     We.index.freq = '1D'
    
#     W = pd.concat([Wc, We], axis=1)
#     W.to_csv('./ECMWF/nc/csv/wa/all/Ww_165-166_lt'+str(start_lt)+str(end_lt)+'_wa.csv')
#     W_maxlt = W.resample('Y').max()
#     W_maxlt = W_maxlt[:-1]
    
#     plt.plot(W_maxlt.index.year, W_maxlt, label=W_maxlt.columns)
#     plt.legend(ncol=2, prop={'size': 8})
#     plt.title('Velocidad de viento máxima anual, '+str(start_lt)+' a '+str(end_lt)+' días')
#     plt.ylabel('Velocidad de viento [m/s]')
#     plt.xticks(years)
#     plt.savefig('../SSP/Ww_max_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.jpg', format='jpg', dpi=1000)
#     plt.close()
    
#     #Radiation
#     Rsc = pd.read_csv('./ECMWF/nc/csv/wa/c_Rs_169-175_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=['Unnamed: 0'])
#     Rsc.index = Rsc['Unnamed: 0']
#     Rsc = Rsc.drop(['Unnamed: 0'], axis=1)
#     Rsc.index.name = None
#     Rsc.columns = ['0']
#     Rsc.index.freq = '1D'
    
#     Rse = pd.read_csv('./ECMWF/nc/csv/wa/Rs_169-175_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=['Unnamed: 0'])
#     Rse.index = Rse['Unnamed: 0']
#     Rse = Rse.drop(['Unnamed: 0'], axis=1)
#     Rse.index.name = None
#     Rse.index.freq = '1D'
    
#     Rs = pd.concat([Rsc, Rse], axis=1)
#     Rs.to_csv('./ECMWF/nc/csv/wa/all/Rs_169-175_lt'+str(start_lt)+str(end_lt)+'_wa.csv')
    
#     Rlc = pd.read_csv('./ECMWF/nc/csv/wa/c_Rl_169-175_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=['Unnamed: 0'])
#     Rlc.index = Rlc['Unnamed: 0']
#     Rlc = Rlc.drop(['Unnamed: 0'], axis=1)
#     Rlc.index.name = None
#     Rlc.columns = ['0']
#     Rlc.index.freq = '1D'
    
#     Rle = pd.read_csv('./ECMWF/nc/csv/wa/Rl_169-175_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=['Unnamed: 0'])
#     Rle.index = Rle['Unnamed: 0']
#     Rle = Rle.drop(['Unnamed: 0'], axis=1)
#     Rle.index.name = None
#     Rle.index.freq = '1D'
    
#     Rl = pd.concat([Rlc, Rle], axis=1)
#     Rl.to_csv('./ECMWF/nc/csv/wa/all/Rl_169-175_lt'+str(start_lt)+str(end_lt)+'_wa.csv')
    
    

# from pyextremes import EVA

# data = Pobs['Pobs'].dropna()[3994:]
# data = data[data>10]

# eva_obs = EVA(data)
# eva_obs.get_extremes(method='BM', block_size='365.2425D', errors='ignore')
# eva_obs.fit_model() #distribution = 'lognorm') #forced to follow gumbel distribution
# summary_obs = eva_obs.get_summary(return_period=Tr, alpha=0.95, n_samples=10)
# #Tr_res = summary_obs.drop(['upper ci', 'lower ci'], axis=1)
# summary_obs.columns = ['Qobs', 'lower ci', 'upper ci']
# #summary_obs

# eva_obs.plot_diagnostic(return_period=Tr, alpha=0.95)



for lt in slt:
    # start_lt = 5
    start_lt = lt
    end_lt = start_lt + 3
    print('lt', start_lt, end_lt)
    
    Pcols = ['Pobs', 'ERA5', 'CLICOM']
    Qcols = ['Qobs', 'Qpobs', 'ERA5', 'CLICOM']
    Pprom = [Pobs_m.mean().values[0], Pera_m.mean().values[0], Pclic_m.mean().values[0]]
    Pmax = [Pobs_m.max().values[0], Pera_m.max().values[0], Pclic_m.max().values[0]]
    Pmin = [Pobs_m.min().values[0], Pera_m.min().values[0], Pclic_m.min().values[0]]
    Pproms = [Pobs_s.mean().values[0], Pera_s.mean().values[0], Pclic_s.mean().values[0]]
    Pmaxs = [Pobs_s.max().values[0], Pera_s.max().values[0], Pclic_s.max().values[0]]
    Pmins = [Pobs_s.min().values[0], Pera_s.min().values[0], Pclic_s.min().values[0]]
    Qprom = [Qobs_m.mean().values[0], Qpobs_m.mean().values[0], Qera_m.mean().values[0], Qclic_m.mean().values[0]]
    Qmax = [Qobs_m.max().values[0], Qpobs_m.max().values[0], Qera_m.max().values[0], Qclic_m.max().values[0]]
    Qmin = [Qobs_m.min().values[0], Qpobs_m.min().values[0], Qera_m.min().values[0], Qclic_m.min().values[0]]
    Qpromp = [Qobs_p.mean().values[0], Qpobs_p.mean().values[0], Qera_p.mean().values[0], Qclic_p.mean().values[0]]
    Qmaxp = [Qobs_p.max().values[0], Qpobs_p.max().values[0], Qera_p.max().values[0], Qclic_p.max().values[0]]
    Qminp = [Qobs_p.min().values[0], Qpobs_p.min().values[0], Qera_p.min().values[0], Qclic_p.min().values[0]]
        
    P = pd.read_csv('./ECMWF/nc/csv/wa/all/P_228228_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=['Unnamed: 0'])
    P.index = P['Unnamed: 0']
    P = P.drop(['Unnamed: 0'], axis=1)
    P.index.name = None
    P_m = P.resample('Y').max()[:-1]
    P_s = P.resample('Y').sum()[:-1]
    
    Q = pd.read_csv('./ECMWF/nc/csv/wa/all/P_228228_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=['Unnamed: 0'])
    Q.index = Q['Unnamed: 0']
    Q = Q.drop(['Unnamed: 0'], axis=1)
    Q.index.name = None
    Q_m = Q.resample('Y').max()[:-1]
    Q_p = Q.resample('Y').mean()[:-1]
    
    for ens in ensemble:
        Pcols = np.append(Pcols, str(ens))
        Pprom = np.append(Pprom, P_m[str(ens)].mean())
        Pmax = np.append(Pmax, P_m[str(ens)].max())
        Pmin = np.append(Pmin, P_m[str(ens)].min())
        Pproms = np.append(Pproms, P_s[str(ens)].mean())
        Pmaxs = np.append(Pmaxs, P_s[str(ens)].max())
        Pmins = np.append(Pmins, P_s[str(ens)].min())
        
        Qcols = np.append(Qcols, str(ens))
        Qprom = np.append(Qprom, Q_m[str(ens)].mean())
        Qmax = np.append(Qmax, Q_m[str(ens)].max())
        Qmin = np.append(Qmin, Q_m[str(ens)].min())
        Qpromp = np.append(Qpromp, Q_p[str(ens)].mean())
        Qmaxp = np.append(Qmaxp, Q_p[str(ens)].max())
        Qminp = np.append(Qminp, Q_p[str(ens)].min())
    
    Pdf = pd.DataFrame([Pprom, Pmax, Pmin, Pproms, Pmaxs, Pmins], columns=Pcols)
    Pdf.index = ['PMA.P', 'PMA.M', 'PMA.m', 'PAA.P', 'PAA.M', 'PAA.m']
    #Promedio de la precipitacion máxima anual acumulada en 24 hr
    Pdf.to_csv('../SSP/comparison/P_summary_'+str(start_lt)+str(end_lt)+'.csv')
    
    Qdf = pd.DataFrame([Qprom, Qmax, Qmin, Qpromp, Qmaxp, Qminp], columns=Qcols)
    Qdf.index = ['QMA.P', 'QMA.M', 'QMA.m', 'QPA.P', 'QPA.M', 'QPA.m']
    Qdf.to_csv('../SSP/comparison/Q_summary_'+str(start_lt)+str(end_lt)+'.csv')
    
    plt.scatter(Pdf.index, Pdf['Pobs'], marker='X', label='Pobs')
    plt.scatter(Pdf.index, Pdf['ERA5'], marker='P', label='ERA5')
    plt.scatter(Pdf.index, Pdf['CLICOM'], marker='*', label='CLICOM')
    for ens in ensemble:
        plt.scatter(Pdf.index, Pdf[str(ens)], marker='.', label=str(ens))
    plt.legend(ncol=3, prop={'size': 8}) #loc='center left', bbox_to_anchor=(1, 0.5), 
    plt.title('Comparación de PMA y PAA, '+str(start_lt)+' a '+str(end_lt)+' días')
    plt.ylabel('Precipitación [mm]')
    plt.savefig('../SSP/comparison/P_summary_'+str(start_lt)+str(end_lt)+'.jpg', format='jpg', dpi=1000)
    plt.close()
    
    plt.scatter(Qdf.index, Qdf['Qobs'], marker='X', label='Qobs')
    # plt.scatter(Qdf.index, Qdf['Qpobs'], marker=',', label='Qpobs')
    plt.scatter(Qdf.index, Qdf['ERA5'], marker='P', label='ERA5')
    plt.scatter(Qdf.index, Qdf['CLICOM'], marker='*', label='CLICOM')
    for ens in ensemble:
        plt.scatter(Qdf.index, Qdf[str(ens)], marker='.', label=str(ens))
    plt.legend(ncol=3, prop={'size': 8}) #loc='center left', bbox_to_anchor=(1, 0.5), 
    plt.title('Comparación de QMA y QPA, '+str(start_lt)+' a '+str(end_lt)+' días')
    plt.ylabel('Gasto [$m^3$/s]')
    plt.savefig('../SSP/comparison/Q_summary_'+str(start_lt)+str(end_lt)+'.jpg', format='jpg', dpi=1000)
    plt.close()
    
    
