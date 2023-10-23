# -*- coding: utf-8 -*-
"""
Created on Fri May 20 09:57:44 2022

@author: HIDRAULICA-Dani
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime as dt

os.chdir("D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/")
os.listdir()

ensemble = [0,1,2,3,4,5,6,7,8,9]
ensembles = [0,1,2,3,4,5,6,7,8,9,10]
slt = [*range(2,17,3)] #1,17
variables = ['228228'] #, '121-122', '169-175', '165-166']
# variables24 = ['165-166', '169-175', '130']

months = {1:'JAN', 2:'FEB', 3:'MAR', 4:'APR', 5:'MAY', 6:'JUN', 7:'JUL', 8:'AGO', 9:'SEP', 10:'OCT', 11:'NOV', 12:'DIC'}
months_id = [1,2,3,4,5,6,7,8,9,10,11,12]
months_ids = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AGO','SEP','OCT','NOV','DIC']
months_idsp = ['ENE','FEB','MAR','ABR','MAY','JUN','JUL','AGO','SEP','OCT','NOV','DIC']



for lt in slt:
# start_lt = 5
    start_lt = lt
    end_lt = start_lt + 3
    print(start_lt, end_lt)
    
    
    pref = pd.read_csv('./ECMWF/nc/csv/wa/all/P_228228_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=["Unnamed: 0"])
    pref.index = pref["Unnamed: 0"]
    pref.index.name = None
    pref = pref.drop(["Unnamed: 0"], axis=1)
    pref
        
    tref = pd.read_csv('./ECMWF/nc/csv/wa/all/T_121-122_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=["Unnamed: 0"])
    tref.index = tref["Unnamed: 0"]
    tref.index.name = None
    tref = tref.drop(["Unnamed: 0"], axis=1)
    tref
    
    # tpe = pd.DataFrame()
    # for ens in ensembles:
    # tpc = tp.resample("1M").count()
    # tp = pref['0']
    tp = pref*1
    tp = tp.resample("1M").sum()
    tp = tp.mean(axis=1)
    # tpc[:50]
    # plt.plot(tp[:-6])
    
    # t2m = tref['0']
    t2m = tref*1
    t2m = t2m.resample("1M").mean()
    t2m = t2m.mean(axis=1)
    # t2m = t2m.dropna()
    # plt.plot(t2m[2:-6])
    
    P = pd.DataFrame(tp.values, tp.index.month, columns=["P"])
    P = P.groupby(P.index)["P"].mean()
    # plt.bar(P.index, P)
    
    T = pd.DataFrame(t2m.values, t2m.index.month, columns=["T"])
    Tmin = T.groupby(T.index)["T"].min()
    Tmax = T.groupby(T.index)["T"].max()
    Tmed = T.groupby(T.index)["T"].mean()
    # plt.plot(Tmin)
    # plt.plot(Tmax)
    # plt.plot(Tmed)
    
    
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.bar(months_id, P, color='b', label='Pmed')
    ax2.plot(months_id, Tmax, color='r', label='Tmax')
    ax2.plot(months_id, Tmed, color='y', label='Tmed')
    ax2.plot(months_id, Tmin, color='g', label='Tmin')
    # ax1.set_xlabel('Mes')
    ax1.set_ylabel('Precipitación (mm)')
    ax2.set_ylabel('Temperatura (°C)')
    ax1.set_xticks(months_id)
    ax1.set_xticklabels(months_idsp) # rotation=45)
    # fig.suptitle('Climate graph')
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc=0)
    ax1.set_title('Clima de repronósticos Monterrey, '+str(start_lt)+' a '+str(end_lt)+' días')
    fig.savefig('./VARIOS/Figures/climate/Refs_Monterrey_SP'+str(start_lt)+str(end_lt)+'.jpg', dpi=1000, format='jpg', bbox_inches='tight')
    

################################################################################

ecm = pd.read_csv("./VARIOS/Tables/clicom_malla_data.csv", parse_dates=["Unnamed: 0"])
ecm.index = ecm["Unnamed: 0"]
ecm.index.name = None
ecm = ecm.drop(["Unnamed: 0"], axis=1)
ecm

tp = ecm["P"]
# tpc = tp.resample("1M").count()
tp = tp.resample("1M").sum()
# tp.resample('Y').sum().mean()
# tpc[:50]
plt.plot(tp[:-6])

t2m = ecm["T"]
t2m = t2m.resample("1M").mean()
# t2m.resample('Y').mean().mean()
# t2m.resample('Y').max().mean()
# t2m.resample('Y').min().mean()
# t2m = t2m.dropna()
plt.plot(t2m[2:-6])
# plt.plot(ecm["T"])

P = pd.DataFrame(tp[:-1].values, tp.index[:-1].month, columns=["P"])
P = P.groupby(P.index)["P"].mean()
plt.bar(P.index, P)

T = pd.DataFrame(t2m[:-1].values, t2m.index[:-1].month, columns=["T"])
Tmin = T.groupby(T.index)["T"].min()
Tmax = T.groupby(T.index)["T"].max()
Tmed = T.groupby(T.index)["T"].mean()
plt.plot(Tmin)
plt.plot(Tmax)
plt.plot(Tmed)

# (Tmax - Tmin).mean()
# (Tmed - Temed).mean()
# (Tmax - Temax).mean()
# (Tmin - Temin).mean()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(months_id, P, color='b', label='Pmed')
ax2.plot(months_id, Tmax, color='r', label='Tmax')
ax2.plot(months_id, Tmed, color='y', label='Tmed')
ax2.plot(months_id, Tmin, color='g', label='Tmin')
# ax1.set_xlabel('Mes')
ax1.set_ylabel('Precipitación (mm)')
ax2.set_ylabel('Temperatura (°C)')
ax1.set_xticks(months_id)
ax1.set_xticklabels(months_idsp) # rotation=45)
# fig.suptitle('Climate graph')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
ax1.set_title('Clima CLICOM-MALLA Monterrey')
fig.savefig('./VARIOS/Figures/climate/CLICOM_Monterrey_SP.jpg', dpi=1000, format='jpg', bbox_inches='tight')


################################################################################

ecm = pd.read_csv("./VARIOS/Tables/era5_all_wa.csv", parse_dates=["Unnamed: 0"])
ecm.index = ecm["Unnamed: 0"]
ecm.index.name = None
ecm = ecm.drop(["Unnamed: 0"], axis=1)
ecm

tp = ecm["tp"]
# tpc = tp.resample("1M").count()
tp = tp.resample("1M").sum()
# tp.resample('Y').sum().mean()
# tpc[:50]
# plt.plot(tp[:-6])
plt.plot(ecm["tp"])

t2m = ecm["t2m"]
t2m = t2m.resample("D").mean()
t2m = t2m.resample("1M").mean()
# t2m.resample('Y').mean().mean()
# t2m.resample('Y').max().mean()
# t2m.resample('Y').min().mean()
# t2m = t2m.dropna()
# plt.plot(t2m[2:-6])
plt.plot(ecm["t2m"])

Pe = pd.DataFrame(tp[:-1].values, tp.index[:-1].month, columns=["Pe"])
Pe = Pe.groupby(Pe.index)["Pe"].mean()
plt.bar(Pe.index, Pe)

Te = pd.DataFrame(t2m[:-1].values, t2m.index[:-1].month, columns=["Te"])
Temin = Te.groupby(Te.index)["Te"].min()
Temax = Te.groupby(Te.index)["Te"].max()
Temed = Te.groupby(Te.index)["Te"].mean()
plt.plot(Temin)
plt.plot(Temax)
plt.plot(Temed)

# (Temax - Temin).mean()

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(months_id, Pe, color='b', label='Pmed')
ax2.plot(months_id, Temax, color='r', label='Tmax')
ax2.plot(months_id, Temed, color='y', label='Tmed')
ax2.plot(months_id, Temin, color='g', label='Tmin')
# ax1.set_xlabel('Mes')
ax1.set_ylabel('Precipitación (mm)')
ax2.set_ylabel('Temperatura (°C)')
ax1.set_xticks(months_id)
ax1.set_xticklabels(months_idsp) # rotation=45)
# fig.suptitle('Climate graph')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
ax1.set_title('Clima ERA5-Land Monterrey')
fig.savefig('./VARIOS/Figures/climate/ERA5Land_Monterrey_SP.jpg', dpi=1000, format='jpg', bbox_inches='tight')



################################################################################
#Comparison ERA5-Land y CLICLOM-MALLA climate graphs
fig, [ax1,ax2] = plt.subplots(2,1, figsize=(6,8))
plt.subplots_adjust(wspace=0, hspace=0.05)
ax3 = ax1.twinx()
ax1.bar(months_id, Pe, color='b', label='P')
ax3.plot(months_id, Temax, color='r', label='Tmax')
ax3.plot(months_id, Temed, color='y', label='Tmed')
ax3.plot(months_id, Temin, color='g', label='Tmin')
ax3.set_ylim(0,35)
# ax1.set_xlabel('Mes')
ax1.set_ylabel('Precipitación (mm)')
ax3.set_ylabel('Temperatura (°C)')
ax1.set_xticks([])
# ax1.set_xticks(months_id)
# ax1.set_xticklabels(months_idsp) # rotation=45)
ax1.set_title('Clima Monterrey a) ERA5-Land y b) CLICOM-MALLA')
ax1.text(0.2, 148, 'a)')
ax4 = ax2.twinx()
ax2.bar(months_id, P, color='b', label='P')
ax4.plot(months_id, Tmax, color='r', label='Tmax')
ax4.plot(months_id, Tmed, color='y', label='Tmed')
ax4.plot(months_id, Tmin, color='g', label='Tmin')
ax4.set_ylim(0,35)
# ax1.set_xlabel('Mes')
ax2.set_ylabel('Precipitación (mm)')
ax4.set_ylabel('Temperatura (°C)')
ax2.set_xticks(months_id)
ax2.set_xticklabels(months_idsp) # rotation=45)
# fig.suptitle('Climate graph')
ax2.text(0.2, 145, 'b)')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax3.get_legend_handles_labels()
ax4.legend(lines + lines2, labels + labels2, loc='center', ncol=4, bbox_to_anchor=(0.5, -0.125))
# ax2.set_title('Clima CLICOM-MALLA Monterrey')
fig.savefig('./VARIOS/Figures/climate/ERA5Land_CLICOM_Monterrey_SP.png', dpi=300, format='png', bbox_inches='tight')



################################################################################
#estacion 19049

ecm = pd.read_csv("./CLICOM/19049_data.csv", parse_dates=["Unnamed: 0"])
ecm.index = ecm["Unnamed: 0"]
ecm.index.name = None
ecm = ecm.drop(["Unnamed: 0"], axis=1)
mask = (ecm.index >= '1970-01-01')
ecm = ecm[mask]

ecm = ecm.dropna()

tp = ecm["PRECIP"]
# tpc = tp.resample("1M").count()
tp = tp.resample("1M").sum()
# tp.resample('Y').sum().mean()
# tpc[:50]
plt.plot(tp[:-6])

tmax = ecm["TMAX"]
tmin = ecm["TMIN"]
t2m = (tmax + tmin)/2
t2m = t2m.resample("1M").mean()
# t2m.resample('Y').mean().mean()
# t2m.resample('Y').max().mean()
# t2m.resample('Y').min().mean()
# t2m = t2m.dropna()
plt.plot(t2m[2:-6])
# plt.plot(ecm["T"])

# Pe = pd.DataFrame(tp[:-1].values, tp.index[:-1].month, columns=["Pe"])
# Pe = Pe.groupby(Pe.index)["Pe"].mean()
# plt.bar(Pe.index, Pe)

# Te = pd.DataFrame(t2m[:-1].values, t2m.index[:-1].month, columns=["Te"])
# Temin = Te.groupby(Te.index)["Te"].min()
# Temax = Te.groupby(Te.index)["Te"].max()
# Temed = Te.groupby(Te.index)["Te"].mean()
# plt.plot(Temin)
# plt.plot(Temax)
# plt.plot(Temed)

tmax = [20.7,23.2,26.9,30.0,32.2,33.8,34.8,34.5,31.5,27.6,24.1,21.2]
tmin = [8.2,10.0,13.2,16.7,20.2,22.0,22.3,22.5,20.9,17.2,12.7,9.1]
tmed = [14.4,16.6,20.0,23.4,26.2,27.9,28.6,28.5,26.2,22.4,18.4,15.1]
p = [16.6,16.5,19.9,29.7,52.3,68.4,43.0,81.6,150.6,75.1,23.0,14.1]

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
ax1.bar(months_id, p, color='b', label='Pmed')
ax2.plot(months_id, tmax, color='r', label='Tmax')
ax2.plot(months_id, tmed, color='y', label='Tmed')
ax2.plot(months_id, tmin, color='g', label='Tmin')
# ax1.set_xlabel('Mes')
ax2.set_ylim(0,38)
ax1.set_ylabel('Precipitación (mm)')
ax2.set_ylabel('Temperatura (°C)')
ax1.set_xticks(months_id)
ax1.set_xticklabels(months_idsp) # rotation=45)
# fig.suptitle('Climate graph')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc=0)
ax1.set_title('Clima Estación 19049 Monterrey')
fig.savefig('./VARIOS/Figures/climate/Estacion19049_Monterrey_SP.png', dpi=300, format='png', bbox_inches='tight')
