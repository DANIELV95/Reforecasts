# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 12:12:52 2022

@author: HIDRAULICA-Dani
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
import pyhecdss
import spotpy

os.chdir('D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/VARIOS/Tables')
os.listdir()

est = ['19015','19017','19018','19038','19049','19052','19058','19061','19096','19150','19200']
est = ['19049']

pest = pd.read_csv('./precip_est.csv', parse_dates=['Unnamed: 0'])
pest.index = pest['Unnamed: 0']
pest.index.name = None
pest = pest[est]
pest = pest.dropna(how='all')
test = pd.read_csv('./Tmean_est.csv', parse_dates=['Unnamed: 0'])
test.index = test['Unnamed: 0']
test.index.name = None
test = test[est]
test = test.dropna(how='all')
pestd = pest.T.mean().to_frame('P')
testd = test.T.mean().to_frame('T')
pestm = pestd.resample('M').sum()
testm = testd.resample('M').mean()
pesty = pestd.resample('Y').sum()
testy = testd.resample('Y').mean()

plt.plot(testd, color='r')
plt.plot(testm, color='b')
plt.plot(testy, color='g')

fig, ax = plt.subplots()
# ax3 = ax.twinx()
# rspine = ax3.spines['right']
# rspine.set_position(('axes', 1.1))
# pestd.Pe.plot(ax=ax3, style='r-')
pestm.Pe.plot(ax=ax, style='b-')
pesty.Pe.plot(ax=ax, style='g-', secondary_y=True)

pesty.mean()
# plt.plot(pesty)

clic = pd.read_csv('./clicom_malla_data.csv', parse_dates=['Unnamed: 0'])
clic.index = clic['Unnamed: 0']
clic = clic.drop(['Unnamed: 0'], axis=1)
clic.index.name = None
tclicd = clic['T'].to_frame('Tc')
pclicd = clic['P'].to_frame('Pc')
tclicm = tclicd.resample('M').mean()
pclicm = pclicd.resample('M').sum()
tclicy = tclicd.resample('Y').mean()
pclicy = pclicd.resample('Y').sum()

plt.plot(tclicd, color='r')
plt.plot(tclicm, color='b')
plt.plot(tclicy, color='g')

fig, ax = plt.subplots()
# ax3 = ax.twinx()
# rspine = ax3.spines['right']
# rspine.set_position(('axes', 1.1))
# pclicd.Pc.plot(ax=ax3, style='r-')
pclicm.Pc.plot(ax=ax, style='b-')
pclicy.Pc.plot(ax=ax, style='g-', secondary_y=True)

pclicy.mean()
# plt.plot(pesty)
# plt.plot(pclicy)


era5 = pd.read_csv('./era5_all_wa.csv', parse_dates=['Unnamed: 0'])
era5.index = era5['Unnamed: 0']
era5.index.name = None
era5 = era5.drop(['Unnamed: 0'], axis=1)
tera5 = era5['t2m'].to_frame('Te')
pera5 = era5['tp'].to_frame('Pe')
tera5d = tera5.resample('D').mean()
pera5d = pera5.resample('D').sum()
tera5m = tera5.resample('M').mean()
pera5m = pera5.resample('M').sum()
tera5y = tera5.resample('Y').mean()
pera5y = pera5.resample('Y').sum()

plt.plot(tera5d, color='r')
plt.plot(tera5m, color='b')
plt.plot(tera5y, color='g')

fig, ax = plt.subplots()
# ax3 = ax.twinx()
# rspine = ax3.spines['right']
# rspine.set_position(('axes', 1.1))
# pera5d.Pe.plot(ax=ax3, style='r-')
pera5m.Pe.plot(ax=ax, style='b-')
pera5y.Pe.plot(ax=ax, style='g-', secondary_y=True)

pera5y.mean()
# plt.plot(pesty)
# plt.plot(pclicy)
# plt.plot(pera5y)


prefs = pd.read_csv('./Refs/P_228228_lt25_wa.csv', parse_dates=['Unnamed: 0'])
prefs.index = prefs['Unnamed: 0']
prefs.index.name = None
prefs = prefs.drop(['Unnamed: 0'], axis=1)
prefsd = prefs*1
prefsm = prefsd.resample('M').sum()
prefsy = prefsd.resample('Y').sum()

prefd = prefs.T.mean()
prefd = pd.DataFrame(prefd.values, prefd.index)
prefd.columns = ['Pr']
prefm = prefd.resample('M').sum()
prefy = prefd.resample('Y').sum()

plt.plot(pesty)
plt.plot(pclicy)
# plt.plot(pera5y)
plt.plot(prefsy)

# pesty.mean()
# pclicy.mean()
# pera5y.mean()
# prefsy[:-1].mean().mean()



qest = pd.read_csv('./Q_mean.csv', parse_dates=['Unnamed: 0'])
qest.index = qest['Unnamed: 0']
qest.index.name = None
qest = qest.drop(['Unnamed: 0'], axis=1)
qest.columns = ['Q']
qestd = qest*1
qestm = qestd.resample('M').mean()
qesty = qestd.resample('Y').mean()
qestmm = qestd.resample('M').max()
qestym = qestd.resample('Y').max()

x = list(qestd.index.values.flatten())
y = list(qestd.values.flatten())
ymax = list(qestd.resample('Y').max().values.flatten())
xpos = [y.index(i) for i in ymax]
xmax = [x[i] for i in xpos]
qestd_max = qestd['Q'][xmax].to_frame()
qestd_max.mean()

dates = pd.date_range('1972-01-01', '1995-01-01', freq='2Y')

#Gráfica caudal obs
plt.plot(qestd, color='b', lw=0.75, label='Q')
plt.plot(qestd_max, color='g', marker='.', ls='', ms=5, label='Qmax')
# plt.hlines(qestd.mean().values[0], qestd.index[0], qestd.index[-1])
# plt.hlines(40, qestd.index[0], qestd.index[-1])
# plt.hlines(90, qestd.index[0], qestd.index[-1])
plt.title('Cuadal observado en estación hidrométrica 24387')
plt.ylabel('Caudal [$m^3$/s]')
plt.xlim(qestd.index[0], qestd.index[-1])
plt.ylim(0, 225) #qestd.max().values[0]
# plt.xticks(qestd_max.index, rotation=90)
plt.xticks(dates, dates.year+1)
plt.legend()
plt.savefig('../Figures/discharge/Bandas_24387_Los-Lermas.png', dpi=300, format='png', bbox_inches='tight')


mask = ((pestd.index >= '1973-02-01') & (pestd.index <= '1994-12-31'))
pestd = pestd[mask]

x = list(pestd.index.values.flatten())
y = list(pestd.values.flatten())
ymax = list(pestd.resample('Y').max().values.flatten())
xpos = [y.index(i) for i in ymax]
xmax = [x[i] for i in xpos]
pestd_max = pestd['P'][xmax].to_frame()
pestd_max.mean()


plt.plot(pestd, color='b', lw=0.75, label='P')
plt.plot(pestd_max, color='g', marker='.', ls='', ms=5, label='Pmax')
plt.title('Precipitación observada en estación climatológica 19049')
plt.ylabel('Precipitación [mm]')
plt.xlim(pestd.index[0], pestd.index[-1])
plt.ylim(0, 225) #qestd.max().values[0]
# plt.xticks(qestd_max.index, rotation=90)
plt.xticks(dates, dates.year+1)
plt.legend()
plt.savefig('../Figures/precipitation/SMN-CNA_19049_Monterrey.png', dpi=300, format='png', bbox_inches='tight')


#Gráfica precipitación y caudal obs
fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(pestd, color='b', lw=0.75, label='P')
ax1.plot(pestd_max, color='g', marker='.', ls='', ms=5, label='Pmax')
ax2.plot(qestd, color='b', lw=0.75, label='Q')
ax2.plot(qestd_max, color='g', marker='.', ls='', ms=5, label='Qmax')
ax1.invert_yaxis()
# plt.hlines(qestd.mean().values[0], qestd.index[0], qestd.index[-1])
# plt.hlines(40, qestd.index[0], qestd.index[-1])
# plt.hlines(90, qestd.index[0], qestd.index[-1])
ax1.set_title('Precipitación y caudal observados')
ax1.set_ylabel('Precipitación [mm]')
ax2.set_ylabel('Caudal [$m^3$/s]')
ax1.set_xlim(qestd.index[0], qestd.index[-1])
ax2.set_xlim(qestd.index[0], qestd.index[-1])
ax1.set_ylim(0, 225) #qestd.max().values[0]
ax2.set_ylim(0, 225) #qestd.max().values[0]
# plt.xticks(qestd_max.index, rotation=90)
plt.xticks(dates, dates.year+1)
plt.legend()
plt.savefig('../Figures/discharge/Bandas_24387_Los-Lermas.png', dpi=300, format='png', bbox_inches='tight')






delta = dt.timedelta(days=365/2)
fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.plot(qestd, color='b', lw=0.75, label='Q')
ax2.plot(qesty.index-delta, qesty, color='r', lw=0.75, label='Qmed')
ax1.plot(qestd_max, color='g', marker='.', ls='', ms=3, label='Qmax')
# plt.hlines(qestd.mean().values[0], qestd.index[0], qestd.index[-1])
# plt.hlines(40, qestd.index[0], qestd.index[-1])
# plt.hlines(90, qestd.index[0], qestd.index[-1])
ax1.set_title('Cuadal observado en estación hidrométrica 24387')
ax1.set_ylabel('Caudal [$m^3$/s]')
ax1.set_xlim(qestd.index[0], qestd.index[-1])
ax1.set_ylim(0, 225) #qestd.max().values[0]
# plt.xticks(qestd_max.index, rotation=90)
ax1.set_xticks(dates, dates.year+1)
ax1.legend()
ax2.legend()
# plt.savefig('../Figures/discharge/Bandas_24387_Los-Lermas.jpg', dpi=1000, format='jpg', bbox_inches='tight')



##############
fig = plt.figure()
ax = fig.add_subplot(111)

x=[1,2,3,4,5,6,7,8,9,10]
y=[1,1,1,2,10,2,1,1,1,1]
line, = ax.plot(x, y)

ymax = max(y)
xpos = y.index(ymax)
xmax = x[xpos]

ax.annotate('local max', xy=(xmax, ymax), xytext=(xmax, ymax+5),
            arrowprops=dict(facecolor='black', shrink=0.05),
            )
ax.set_ylim(0,20)
plt.show()
##############

plt.plot(qestm, color='g')
plt.plot(qestmm, color='g')
plt.plot(qesty, color='r')

fig, ax = plt.subplots()
ax3 = ax.twinx()
rspine = ax3.spines['right']
rspine.set_position(('axes', 1.1))
qestd.Q.plot(ax=ax, style='b-')
qestm.Q.plot(ax=ax, style='g-', secondary_y=True)
qesty.Q.plot(ax=ax3, style='r-')


fname=r'CLICOM_Run.dss'
d = pyhecdss.DSSFile('../../../HMS/RioLaSilla_CNH410/'+fname)
catdf=d.read_catalog()
plist1=d.get_pathnames(catdf)
dfr1,units1,ptype1=d.read_rts(plist1[0])
dfr1.columns = ['Qc']
# dfr1.index = pd.PeriodIndex(dfr1.index, freq='D').to_timestamp()
qclicd = dfr1*1
qclicm = qclicd.resample('M').mean()
qclicy = qclicd.resample('Y').mean()
qclicmm = qclicd.resample('M').max()
qclicym = qclicd.resample('Y').max()

fname=r'ERA5_Run.dss'
d = pyhecdss.DSSFile('../../../HMS/RioLaSilla_CNH410/'+fname)
catdf=d.read_catalog()
plist1=d.get_pathnames(catdf)
dfr1,units1,ptype1=d.read_rts(plist1[0])
dfr1.columns = ['Qe']
# dfr1.index = pd.PeriodIndex(dfr1.index, freq='D').to_timestamp()
qera5d = dfr1*1
qera5m = qera5d.resample('M').mean()
qera5y = qera5d.resample('Y').mean()
qera5mm = qera5d.resample('M').max()
qera5ym = qera5d.resample('Y').max()



##############
#Comparison P CLICOM, P ERA5 and Q bandas

data_all = pd.concat([pestd, pera5d, pclicd, qestd, qera5d, qclicd], axis=1)
data_all = data_all.dropna()

p = pd.concat([pestd, pera5d, pclicd], axis=1)
mask_o = ((p.index >= '1973-02-01') & (p.index <= '1994-12-31'))
p = p[mask_o]

pdm = p.mean()
pdmx = p.max()
# pdmn = p.min()
pdstd = p.std()

pmsm = p.resample('M').sum().mean()
pmsmx = p.resample('M').sum().max()
# pmsmn = p.resample('M').sum().min()
pmsstd = p.resample('M').sum().std()

pysm = p.resample('Y').sum().mean()
pysmx = p.resample('Y').sum().max()
# pysmn = p.resample('Y').sum().min()
pysstd = p.resample('Y').sum().std()

q = pd.concat([qestd, qera5d, qclicd], axis=1)
q = q.dropna()
mask_o = ((q.index >= '1973-02-01') & (q.index <= '1994-12-31'))
q = q[mask_o]
mask_d = ((q.index >= '1988-01-01') & (q.index <= '1988-12-31'))
q = q.drop(q[mask_d].index)

qdm = q.mean()
qdmx = q.max()
# qdmn = q.min()
qdstd = q.std()

qmmm = q.resample('M').mean().mean()
qmmmx = q.resample('M').mean().max()
# qmmmn = q.resample('M').mean().min()
qmmstd = q.resample('M').mean().std()

qymm = q.resample('Y').mean().mean()
qymmx = q.resample('Y').mean().max()
# qymmn = q.resample('Y').mean().min()
qymstd = q.resample('Y').mean().std()

p_all_sum = pd.concat([pdm, pdmx, pdstd, 
                       pmsm, pmsmx, pmsstd, 
                       pysm, pysmx, pysstd], 
                      keys=['pdm', 'pdmx', 'pdstd', 
                            'pmsm', 'pmsmx', 'pmsstd', 
                            'pysm', 'pysmx', 'pysstd'], axis=1)
p_all_sum.to_csv('p_summary_all_data.csv')

q_all_sum = pd.concat([qdm, qdmx, qdstd, 
                       qmmm, qmmmx, qmmstd, 
                       qymm, qymmx, qymstd], 
                      keys=['qdm', 'qdmx', 'qdstd', 
                            'qmmm', 'qmmmx', 'qmmstd', 
                            'qymm', 'qymmx', 'qymstd'], axis=1)
q_all_sum.to_csv('q_summary_all_data.csv')
q_all_sum.to_csv('q_summary_all_data_wo88.csv')

# data_all_sum = pd.concat([dm, dmx, dstd, 
#                           msm, msstd, mmm, mmstd, mmxm, mmxstd, 
#                           ysm, ymm, ysstd, ymstd, ymxm, ymxstd], 
#                          keys=['dm', 'dmx', 'dstd', 
#                                'msm', 'msstd', 'mmm', 'mmstd', 'mmxm', 'mmxstd', 
#                                'ymm', 'ysstd', 'ymm', 'ymstd', 'ymxm', 'ymxstd'], axis=1)
# data_all_sum.to_csv('summary_all_data.csv')

p.resample('Y').max().plot()
q.resample('Y').max().plot()


mask_c = ((data_all.index >= '1973-06-01') & (data_all.index <= '1973-08-31'))
data = data_all[mask_c]
data.max()


plt.plot(data_all['P'].resample('M').sum())
plt.plot(data_all['Pe'].resample('M').sum())
plt.plot(data_all['Pc'].resample('M').sum())

# data = data_all

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6.5,7))
ax1.plot(data.index, data['P'], label='Pobs', color='k', ls='-', lw=1, alpha=0.75, zorder=2)
ax1.plot(data.index, data['Pe'], label='Pera5', color='g', ls='--', lw=1, alpha=1, zorder=1)
ax1.plot(data.index, data['Pc'], label='Pclicom', color='b', ls='--', lw=1, alpha=1, zorder=1)
ax2.plot(data.index, data['Q'], label='Qobs', color='k', ls='-', lw=1, alpha=0.75, zorder=2)
ax2.plot(data.index, data['Qe'], label='Qera5', color='g', ls='--', lw=1, alpha=1, zorder=1)
ax2.plot(data.index, data['Qc'], label='Qclicom', color='b', ls='--', lw=1, alpha=1, zorder=1)
# ax.plot(qobs_m, label=qref_m.columns, alpha=0.75)
# maj_loc = mdates.MonthLocator(bymonth=np.arange(1,13,2))
# ax.xaxis.set_major_locator(maj_loc)
# min_loc = mdates.MonthLocator()
# ax.xaxis.set_minor_locator(min_loc)
# Set major date tick formatter
zfmts = ['', '%b\n%Y', '%b', '%b-%d', '%H:%M', '%H:%M']
maj_loc = mdates.MonthLocator() #bymonth=np.arange(1,13,2))
maj_fmt = mdates.ConciseDateFormatter(maj_loc) #maj_loc, zero_formats=zfmts)
ax2.xaxis.set_major_formatter(maj_fmt)
min_loc = mdates.MonthLocator()
ax2.xaxis.set_minor_locator(min_loc)
# ax2.set_xlabel('1973')
ax1.set_title('Comparación de hietograma e hidrograma en periodo de calibración') #, x=0.45, size=11)
ax1.invert_yaxis()
ax1.xaxis.set_ticklabels([])
# ax1.xaxis.set_visible(False)
plt.subplots_adjust(wspace=0, hspace=0.04)
ax1.set_ylabel('Precipitación [mm]')
ax2.set_ylabel('Caudal [$m^3$/s]')
ax1.legend(loc='lower right')
ax2.legend(loc='upper right')
# plt.savefig('./VARIOS/Figures/Cal_1973-1974_PQ.png', format='png', dpi=300, bbox_inches='tight')
plt.savefig('../Figures/Cal_1973_PQ.png', format='png', dpi=300, bbox_inches='tight')


mask_v = ((data_all.index >= '1978-08-01') & (data_all.index <= '1978-11-30'))
data = data_all[mask_v]

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(6.5,7))
ax1.plot(data.index, data['P'], label='Pobs', color='k', ls='-', lw=1, alpha=0.75, zorder=2)
ax1.plot(data.index, data['Pe'], label='Pera5', color='g', ls='--', lw=1, alpha=1, zorder=1)
ax1.plot(data.index, data['Pc'], label='Pclicom', color='b', ls='--', lw=1, alpha=1, zorder=1)
ax2.plot(data.index, data['Q'], label='Qobs', color='k', ls='-', lw=1, alpha=0.75, zorder=2)
ax2.plot(data.index, data['Qe'], label='Qera5', color='g', ls='--', lw=1, alpha=1, zorder=1)
ax2.plot(data.index, data['Qc'], label='Qclicom', color='b', ls='--', lw=1, alpha=1, zorder=1)
# ax.plot(qobs_m, label=qref_m.columns, alpha=0.75)
# maj_loc = mdates.MonthLocator(bymonth=np.arange(1,13,2))
# ax.xaxis.set_major_locator(maj_loc)
# min_loc = mdates.MonthLocator()
# ax.xaxis.set_minor_locator(min_loc)
# Set major date tick formatter
zfmts = ['', '%b\n%Y', '%b', '%b-%d', '%H:%M', '%H:%M']
maj_loc = mdates.MonthLocator() #bymonth=np.arange(1,13,2))
maj_fmt = mdates.ConciseDateFormatter(maj_loc) #maj_loc, zero_formats=zfmts)
ax2.xaxis.set_major_formatter(maj_fmt)
min_loc = mdates.MonthLocator()
ax2.xaxis.set_minor_locator(min_loc)
# ax2.set_xlabel('1978')
ax1.set_title('Comparación de hietograma e hidrograma en periodo de validación') #, x=0.45, size=11)
ax1.invert_yaxis()
ax1.xaxis.set_ticklabels([])
# ax1.xaxis.set_visible(False)
plt.subplots_adjust(wspace=0, hspace=0.04)
ax1.set_ylabel('Precipitación [mm]')
ax2.set_ylabel('Caudal [$m^3$/s]')
ax1.legend(loc='lower right')
ax2.legend(loc='upper right')
# plt.savefig('./VARIOS/Figures/Val_1977-1978_PQ.png', format='png', dpi=300, bbox_inches='tight')
plt.savefig('../Figures/Val_1978_PQ.png', format='png', dpi=300, bbox_inches='tight')


# 1XX = Temperature
# 2XX = Precipitation
# 3XX = Discharge
# X1X = Estation
# X2X = CLICOM
# X3X = ERA5
# XX1 = Daily
# XX2 = Monthly
# XX3 = Annualy
# XX4 = Max Daily per Month
# XX5 = Max Daily per Year

dfs = {111:testd, 112:testm, 113:testy, 121:tclicd, 122:tclicm, 123:tclicy, 131:tera5d, 132:tera5m, 133:tera5y,
       211:pestd, 212:pestm, 213:pesty, 221:pclicd, 222:pclicm, 223:pclicy, 231:pera5d, 232:pera5m, 233:pera5y,
       311:qestd, 312:qestm, 313:qesty, 321:qclicd, 322:qclicm, 323:qclicy, 331:qera5d, 332:qera5m, 333:qera5y,
       314:qestmm, 324:qclicmm, 334:qera5mm, 315:qestym, 325:qclicym, 335:qera5ym}

dfsplot = [212,222,232]

df1 = dfs[dfsplot[0]]
df2 = dfs[dfsplot[1]]
df3 = dfs[dfsplot[2]]

# bias = abs(df2.mean().values - df3.mean().values)
# df3 = df3 - bias

plt.plot(df1, color='r', label=df1.columns[0])
plt.plot(df2, color='g', label=df2.columns[0])
plt.plot(df3, color='b', label=df3.columns[0])
plt.legend()

start_date = qestd.index[0]
end_date = qestd.index[-1]

mask1 = ((df1.index >= start_date) & (df1.index <= end_date))
mask2 = ((df2.index >= start_date) & (df2.index <= end_date))
mask3 = ((df3.index >= start_date) & (df3.index <= end_date))
df1m = df1[mask1]
df2m = df2[mask2]
df3m = df3[mask3]

# plt.plot(df1m, color='r', label=df1.columns[0])
# plt.plot(df2m, color='g', label=df2.columns[0])
# plt.plot(df3m, color='b', label=df3.columns[0])
# plt.legend()

# df = pd.concat([df1m,df2m,df3m], axis=1)

fig, ax = plt.subplots()
ax3 = ax.twinx()
rspine = ax3.spines['right']
rspine.set_position(('axes', 1.1))
df1m[df1m.columns].plot(ax=ax3, style='r-', lw=1, label=df1m.columns[0])
df2m[df2m.columns].plot(ax=ax, style='g-', lw=1, label=df2m.columns[0], secondary_y=True)
df3m[df3m.columns].plot(ax=ax, style='b-', lw=1, label=df3m.columns[0])
# plt.legend()

plt.plot(df1m, color='r', label=df1m.columns[0])
plt.plot(df2m, color='g', label=df2m.columns[0])
plt.plot(df3m, color='b', label=df3m.columns[0])
plt.legend()



TPQed = pd.concat([testd, pestd, qobsd], axis=1)
TPQed = TPQed.dropna()
TPQem = pd.concat([testm, pestm, qobsm], axis=1)
TPQem = TPQem.dropna()
TPQey = pd.concat([testy, pesty, qobsy], axis=1)
TPQey = TPQey.dropna()



fig, ax = plt.subplots()
ax3 = ax.twinx()
rspine = ax3.spines['right']
rspine.set_position(('axes', 1.1))
TPQed.Q.plot(ax=ax3, style='r-')
TPQed.Q.plot(ax=ax, style='b-')
TPQed.Q.plot(ax=ax, style='g-', secondary_y=True)

plt.plot(TPQed)
plt.plot(TPQem)
plt.plot(TPQey)

TPQ = pd.concat([tobs,pobs,qobs], axis=1)
TPQ = TPQ.dropna()
TPQy = TPQ.resample('Y').mean()


PQ = pd.concat([pobs,qobs], axis=1)
PQ = PQ.dropna()
PQy = PQ.resample('Y').mean()


tobsm = TPQ['T'].resample('M').mean()
pobsm = TPQ['P'].resample('M').sum()
qobsm = TPQ['Q'].resample('M').mean()
TPQm = pd.concat([tobsm,pobsm,qobsm], axis=1)


fig, ax = plt.subplots()
ax2 = ax.twinx()
ax.scatter(TPQm['Q'], TPQm['P'], color='r')
ax2.scatter(TPQm['Q'], TPQm['T'], color='b')
ax.set_xlabel('Pobs')
ax.set_ylabel('Qobs')

plt.plot(tobs)
plt.plot(tobsm)
plt.plot(pobs)
plt.plot(pobsm)
plt.plot(qobs)
plt.plot(qobsm)

plt.plot(PQ)
plt.plot(TPQm[:])
plt.plot(PQy)

fig, ax = plt.subplots()
ax2 = ax.twinx()
PQm.P.plot(ax=ax, style='b-')
PQm.Q.plot(ax=ax2, style='r-')



fig, ax = plt.subplots()
# ax3 = ax.twinx()
# rspine = ax3.spines['right']
# rspine.set_position(('axes', 1.1))
# TPQm['T'].plot(ax=ax3, style='r-')
TPQm.Q.plot(ax=ax, style='b-')
TPQm.P.plot(ax=ax, style='g-', secondary_y=True)

PQ.corr(method='pearson')
PQm.corr(method='pearson')
PQy.corr(method='pearson')


prefs = pd.read_csv('P_228228_lt58_wa.csv', parse_dates=['Unnamed: 0'])
prefs.index = prefs['Unnamed: 0']
prefs.index.name = None
prefs = prefs.drop(['Unnamed: 0'], axis=1)
Prefd = prefs.T.mean()
Prefd = pd.DataFrame(Prefd.values, Prefd.index)
Prefd.columns = ['Pr']
Prefm = Prefd.resample('M').sum()
Prefy = Prefd.resample('Y').sum()



plt.plot(pesty)
plt.plot(pclicy)
# plt.plot(pera5y)
plt.plot(Prefy)



start_date = pref.index[0]
end_date = pref.index[-1]

P = pd.concat([pobs,Prm], axis=1)
P = P.dropna()
Pm = P.resample('M').mean()

Pm['P'].corr(Pm['Pr'])
Pm['P']
Pm.corr(method='pearson')

mask = ((pobs.index >= start_date) & (pobs.index <= end_date))
pobs = pobs[mask]


plt.plot(Prm)
plt.scatter(P['P'], P['Pr'])
plt.xlabel('Pobs')
plt.ylabel('Pref')