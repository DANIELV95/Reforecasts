# -*- coding: utf-8 -*-
"""
Created on Tue Aug  2 09:25:17 2022

@author: HIDRAULICA-Dani
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyhecdss
import spotpy

os.chdir('D:/DANI/2021/TEMA4_PRONOSTICOS/HMS/RioLaSilla_CNH410')
os.listdir()


qest = pd.read_csv('../../DATOS/VARIOS/Tables/Q_mean.csv', parse_dates=['Unnamed: 0'])
qest.index = qest['Unnamed: 0']
qest.index.name = None
qest = qest.drop(['Unnamed: 0'], axis=1)
qest.columns = ['Q']
qestd = qest*1
qestm = qestd.resample('M').mean()
qesty = qestd.resample('Y').mean()


# CLICOM Comparison NSE
fname=r'CLICOM_Calibration.dss'
d = pyhecdss.DSSFile('./'+fname)
catdf=d.read_catalog()
plist1=d.get_pathnames(catdf)
dfr1,units1,ptype1=d.read_rts(plist1[0])
dfr1.columns = ['Qcc']
qc_cal = dfr1*1

fname=r'CLICOM_Validation.dss'
d = pyhecdss.DSSFile('./'+fname)
catdf=d.read_catalog()
plist1=d.get_pathnames(catdf)
dfr1,units1,ptype1=d.read_rts(plist1[0])
dfr1.columns = ['Qcv']
qc_val = dfr1*1

fname=r'CLICOM_Run.dss'
d = pyhecdss.DSSFile('./'+fname)
catdf=d.read_catalog()
plist1=d.get_pathnames(catdf)
dfr1,units1,ptype1=d.read_rts(plist1[0])
dfr1.columns = ['Qc']
# dfr1.index = pd.PeriodIndex(dfr1.index, freq='D').to_timestamp()
qc = dfr1*1

# ERA5 Comparison NSE
fname=r'ERA5_Calibration.dss'
d = pyhecdss.DSSFile('./'+fname)
catdf=d.read_catalog()
plist1=d.get_pathnames(catdf)
dfr1,units1,ptype1=d.read_rts(plist1[0])
dfr1.columns = ['Qec']
qe_cal = dfr1*1

fname=r'ERA5_Validation.dss'
d = pyhecdss.DSSFile('./'+fname)
catdf=d.read_catalog()
plist1=d.get_pathnames(catdf)
dfr1,units1,ptype1=d.read_rts(plist1[0])
dfr1.columns = ['Qev']
qe_val = dfr1*1

fname=r'ERA5_Run.dss'
d = pyhecdss.DSSFile('./'+fname)
catdf=d.read_catalog()
plist1=d.get_pathnames(catdf)
dfr1,units1,ptype1=d.read_rts(plist1[0])
dfr1.columns = ['Qe']
# dfr1.index = pd.PeriodIndex(dfr1.index, freq='D').to_timestamp()
qe = dfr1*1


Qcal = pd.concat([qestd, qc_cal, qe_cal], axis=1)
Qcald = Qcal.dropna()
Qcalm = Qcald.resample('M').mean()
Qcaly = Qcald.resample('Y').mean()
Qcalmm = Qcald.resample('M').max()
Qcalym = Qcald.resample('Y').max()

Qval = pd.concat([qestd, qc_val, qe_val], axis=1)
Qvald = Qval.dropna()
Qvalm = Qvald.resample('M').mean()
Qvaly = Qvald.resample('Y').mean()
Qvalmm = Qvald.resample('M').max()
Qvalym = Qvald.resample('Y').max()

Qall = pd.concat([qestd, qc, qe], axis=1)
mask_d = ((Qall.index >= '1988-01-01') & (Qall.index <= '1988-12-31'))
Qall = Qall.drop(Qall[mask_d].index)
Qalld = Qall.dropna()
Qallm = Qalld.resample('M').mean()
Qallm = Qallm.dropna()
Qally = Qalld.resample('Y').mean()
Qally = Qally.dropna()
Qallmm = Qalld.resample('M').max()
Qallmm = Qallmm.dropna()
Qallym = Qalld.resample('Y').max()
Qallym = Qallym.dropna()

# QALL = pd.concat([Qcald, Qcalm, Qcaly, Qcalmm, Qcalym], axis=1)
# QALL.mean()

# Qalld.std(ddof=0)
# spotpy.objectivefunctions.calculate_all_functions(Qalld['Q'], Qalld['Qc'])
# qadc = [spotpy.objectivefunctions.calculate_all_functions(Qalld['Q'], Qalld['Qc'])[i][1] for i in ids]

# help(Qalld.std)

ids = [2,3,5,9,10,11,12,15]
objfun = [spotpy.objectivefunctions.calculate_all_functions(Qcalm['Q'], Qcalm['Qcc'])[i][0] for i in ids]

qcdc = [spotpy.objectivefunctions.calculate_all_functions(Qcald['Q'], Qcald['Qcc'])[i][1] for i in ids]
qcmc = [spotpy.objectivefunctions.calculate_all_functions(Qcalm['Q'], Qcalm['Qcc'])[i][1] for i in ids]
qcyc = [spotpy.objectivefunctions.calculate_all_functions(Qcaly['Q'], Qcaly['Qcc'])[i][1] for i in ids]
qcmcm = [spotpy.objectivefunctions.calculate_all_functions(Qcalmm['Q'], Qcalmm['Qcc'])[i][1] for i in ids]
qcycm = [spotpy.objectivefunctions.calculate_all_functions(Qcalym['Q'], Qcalym['Qcc'])[i][1] for i in ids]

qvdc = [spotpy.objectivefunctions.calculate_all_functions(Qvald['Q'], Qvald['Qcv'])[i][1] for i in ids]
qvmc = [spotpy.objectivefunctions.calculate_all_functions(Qvalm['Q'], Qvalm['Qcv'])[i][1] for i in ids]
qvyc = [spotpy.objectivefunctions.calculate_all_functions(Qvaly['Q'], Qvaly['Qcv'])[i][1] for i in ids]
qvmcm = [spotpy.objectivefunctions.calculate_all_functions(Qvalmm['Q'], Qvalmm['Qcv'])[i][1] for i in ids]
qvycm = [spotpy.objectivefunctions.calculate_all_functions(Qvalym['Q'], Qvalym['Qcv'])[i][1] for i in ids]

qadc = [spotpy.objectivefunctions.calculate_all_functions(Qalld['Q'], Qalld['Qc'])[i][1] for i in ids]
qamc = [spotpy.objectivefunctions.calculate_all_functions(Qallm['Q'], Qallm['Qc'])[i][1] for i in ids]
qayc = [spotpy.objectivefunctions.calculate_all_functions(Qally['Q'], Qally['Qc'])[i][1] for i in ids]
qamcm = [spotpy.objectivefunctions.calculate_all_functions(Qallmm['Q'], Qallmm['Qc'])[i][1] for i in ids]
qaycm = [spotpy.objectivefunctions.calculate_all_functions(Qallym['Q'], Qallym['Qc'])[i][1] for i in ids]

qcde = [spotpy.objectivefunctions.calculate_all_functions(Qcald['Q'], Qcald['Qec'])[i][1] for i in ids]
qcme = [spotpy.objectivefunctions.calculate_all_functions(Qcalm['Q'], Qcalm['Qec'])[i][1] for i in ids]
qcye = [spotpy.objectivefunctions.calculate_all_functions(Qcaly['Q'], Qcaly['Qec'])[i][1] for i in ids]
qcmem = [spotpy.objectivefunctions.calculate_all_functions(Qcalmm['Q'], Qcalmm['Qec'])[i][1] for i in ids]
qcyem = [spotpy.objectivefunctions.calculate_all_functions(Qcalym['Q'], Qcalym['Qec'])[i][1] for i in ids]

qvde = [spotpy.objectivefunctions.calculate_all_functions(Qvald['Q'], Qvald['Qev'])[i][1] for i in ids]
qvme = [spotpy.objectivefunctions.calculate_all_functions(Qvalm['Q'], Qvalm['Qev'])[i][1] for i in ids]
qvye = [spotpy.objectivefunctions.calculate_all_functions(Qvaly['Q'], Qvaly['Qev'])[i][1] for i in ids]
qvmem = [spotpy.objectivefunctions.calculate_all_functions(Qvalmm['Q'], Qvalmm['Qev'])[i][1] for i in ids]
qvyem = [spotpy.objectivefunctions.calculate_all_functions(Qvalym['Q'], Qvalym['Qev'])[i][1] for i in ids]

qade = [spotpy.objectivefunctions.calculate_all_functions(Qalld['Q'], Qalld['Qe'])[i][1] for i in ids]
qame = [spotpy.objectivefunctions.calculate_all_functions(Qallm['Q'], Qallm['Qe'])[i][1] for i in ids]
qaye = [spotpy.objectivefunctions.calculate_all_functions(Qally['Q'], Qally['Qe'])[i][1] for i in ids]
qamem = [spotpy.objectivefunctions.calculate_all_functions(Qallmm['Q'], Qallmm['Qe'])[i][1] for i in ids]
qayem = [spotpy.objectivefunctions.calculate_all_functions(Qallym['Q'], Qallym['Qe'])[i][1] for i in ids]

OF_cal_clic = pd.DataFrame(data=[qcdc, qcmc, qcyc, qcmcm, qcycm], index=['cD', 'cM', 'cA', 'cMm', 'cAm'], columns=objfun).T
OF_val_clic = pd.DataFrame(data=[qvdc, qvmc, qvyc, qvmcm, qvycm], index=['vD', 'vM', 'vA', 'vMm', 'vAm'], columns=objfun).T
OF_all_clic = pd.DataFrame(data=[qadc, qamc, qayc, qamcm, qaycm], index=['aD', 'aM', 'aA', 'aMm', 'aAm'], columns=objfun).T

OF_cal_era = pd.DataFrame(data=[qcde, qcme, qcye, qcmem, qcyem], index=['cD', 'cM', 'cA', 'cMm', 'cAm'], columns=objfun).T
OF_val_era = pd.DataFrame(data=[qvde, qvme, qvye, qvmem, qvyem], index=['vD', 'vM', 'vA', 'vMm', 'vAm'], columns=objfun).T
OF_all_era = pd.DataFrame(data=[qade, qame, qaye, qamem, qayem], index=['aD', 'aM', 'aA', 'aMm', 'aAm'], columns=objfun).T

OF_clic = pd.concat([OF_cal_clic, OF_val_clic, OF_all_clic], axis=1).T
OF_era = pd.concat([OF_cal_era, OF_val_era, OF_all_era], axis=1).T

OF_clic['pbias']

OF_clic.to_csv('D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/VARIOS/Tables/summary_OF_clicom.csv')
OF_era.to_csv('D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/VARIOS/Tables/summary_OF_era.csv')

OF_clic.to_csv('D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/VARIOS/Tables/summary_OF_clicom_wo88.csv')
OF_era.to_csv('D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/VARIOS/Tables/summary_OF_era_wo88.csv')

plt.plot(OF_val_clic)

plt.scatter(Qcalym['Q'], Qcalym['Qcc'])

spotpy.objectivefunctions.rmse(Qcalm['Q'], Qcalm['Qcc'])
spotpy.objectivefunctions.mse(Qcalm['Q'], Qcalm['Qcc'])
spotpy.objectivefunctions.rsr(Qcalm['Q'], Qcalm['Qcc'])
spotpy.objectivefunctions.covariance(Qcalm['Q'], Qcalm['Qcc'])
spotpy.objectivefunctions.kge(Qcalm['Q'], Qcalm['Qcc'])
spotpy.objectivefunctions.correlationcoefficient(Qcalm['Q'], Qcalm['Qcc'])

spotpy.objectivefunctions.rsr(Qvalm['Q'], Qvalm['Qcv'])

help(spotpy.objectivefunctions)

#NSE Daily Mean
spotpy.objectivefunctions.nashsutcliffe(Qcal['Q'], Qcal['Qcc'])
spotpy.objectivefunctions.nashsutcliffe(Qval['Q'], Qval['Qcv'])
spotpy.objectivefunctions.nashsutcliffe(Qall['Q'], Qall['Qc'])

spotpy.objectivefunctions.nashsutcliffe(Qcal['Q'], Qcal['Qec'])
spotpy.objectivefunctions.nashsutcliffe(Qval['Q'], Qval['Qev'])
spotpy.objectivefunctions.nashsutcliffe(Qall['Q'], Qall['Qe'])

#NSE Monthly Mean
spotpy.objectivefunctions.nashsutcliffe(Qcalm['Q'], Qcalm['Qcc'])
spotpy.objectivefunctions.nashsutcliffe(Qvalm['Q'], Qvalm['Qcv'])
spotpy.objectivefunctions.nashsutcliffe(Qallm['Q'], Qallm['Qc'])

spotpy.objectivefunctions.nashsutcliffe(Qcalm['Q'], Qcalm['Qec'])
spotpy.objectivefunctions.nashsutcliffe(Qvalm['Q'], Qvalm['Qev'])
spotpy.objectivefunctions.nashsutcliffe(Qallm['Q'], Qallm['Qe'])

#NSE Yearly Mean
spotpy.objectivefunctions.nashsutcliffe(Qcaly['Q'], Qcaly['Qcc'])
spotpy.objectivefunctions.nashsutcliffe(Qvaly['Q'], Qvaly['Qcv'])
spotpy.objectivefunctions.nashsutcliffe(Qally['Q'], Qally['Qc'])

spotpy.objectivefunctions.nashsutcliffe(Qcaly['Q'], Qcaly['Qec'])
spotpy.objectivefunctions.nashsutcliffe(Qvaly['Q'], Qvaly['Qev'])
spotpy.objectivefunctions.nashsutcliffe(Qally['Q'], Qally['Qe'])

#NSE Monthly Max
spotpy.objectivefunctions.nashsutcliffe(Qcalmm['Q'], Qcalmm['Qcc'])
spotpy.objectivefunctions.nashsutcliffe(Qvalmm['Q'], Qvalmm['Qcv'])
spotpy.objectivefunctions.nashsutcliffe(Qallmm['Q'], Qallmm['Qc'])

spotpy.objectivefunctions.nashsutcliffe(Qcalmm['Q'], Qcalmm['Qec'])
spotpy.objectivefunctions.nashsutcliffe(Qvalmm['Q'], Qvalmm['Qev'])
spotpy.objectivefunctions.nashsutcliffe(Qallmm['Q'], Qallmm['Qe'])

#NSE Yearly Max
spotpy.objectivefunctions.nashsutcliffe(Qcalym['Q'], Qcalym['Qcc'])
spotpy.objectivefunctions.nashsutcliffe(Qvalym['Q'], Qvalym['Qcv'])
spotpy.objectivefunctions.nashsutcliffe(Qallym['Q'], Qallym['Qc'])

spotpy.objectivefunctions.nashsutcliffe(Qcalym['Q'], Qcalym['Qec'])
spotpy.objectivefunctions.nashsutcliffe(Qvalym['Q'], Qvalym['Qev'])
spotpy.objectivefunctions.nashsutcliffe(Qallym['Q'], Qallym['Qe'])






# fdf1=catdf[(catdf.C=='PRECIP-INC') & (catdf.F=='LT 10-13')] #(catdf.A=='ECMWF-REF-ENS-0') &
# display("Catalog filtered for B == 'ITS1' & C=='RANDOM'")
# display(fdf1)
# display("Catalog filtered for B == 'SIN'")
# fdf2=catdf[catdf.C=='PRECIP-INC']
# display(fdf2.head())


# with pyhecdss.DSSFile('./HMS/DSS/'+fname) as d:
#     plist1=d.get_pathnames(fdf1)
#     dfr1,units1,ptype1=d.read_rts(plist1[0])
#     print('Units: %s'%units1, 'Period Type: %s'%ptype1)
#     print('Sample values: ')
#     display(dfr1) #.head())
#     plist2=d.get_pathnames(fdf2)
#     dfi1,units2,ptype2=d.read_rts(plist2[0])
#     print('Sample values: ')
#     display(dfi1.head())

# d.close()
