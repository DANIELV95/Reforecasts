# -*- coding: utf-8 -*-
"""
Created on Wed May  4 08:43:36 2022

@author: HIDRAULICA-Dani
"""

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from dateutil.rrule import rrule, WEEKLY, MO, TH
import matplotlib.pyplot as plt
import matplotlib.dates as mdates	
import matplotlib.patches as mpatches
# import matplotlib.patheffects as pe
from scipy.stats.stats import pearsonr

os.chdir("D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/ECMWF/nc")
os.listdir()

ensemble = [0,1,2,3,4,5,6,7,8,9]
ensembles = [0,1,2,3,4,5,6,7,8,9,10]
refs_list = os.listdir()
ilon = [6,7,8]
ilat = [3,4,5]
start = datetime(1900,1,1)
slt = [*range(2,17,3)] #1,17
# slt = [2,5,8,11,14] #17
rtf_dates = list(rrule(WEEKLY, byweekday=[MO,TH], dtstart=datetime(2019,12,31), until=datetime(2020,12,31)))
variables = ['228228'] #, '121-122', '169-175', '165-166']
# variables24 = ['165-166', '169-175', '130']



# Control forecasts

# for var in variables:
var = '228228'
    # for lt in slt:
start_lt = 5
        # start_lt = lt
end_lt = start_lt + 3
print(start_lt, end_lt)

# drop_list = [*range(0,start_lt*4), *range(end_lt*4-1,1620)] #for lt =! 1

# np.set_printoptions(suppress=True) # print w/o scientific notation

# for lat in ilat:
#     for lon in ilon:
lat = 6
lon = 3

# if var == '228228': #Precipitation
df_Plt = pd.DataFrame()

# for rtf_day in rtf_dates:
rtf_day = rtf_dates[53] #50,51,52,53
date = str(rtf_day)[:10]
ref = nc.Dataset('c_param_'+var+'_'+date+'.nc')

# start = datetime(int(date[0:4])-year,int(date[5:7]),int(date[8:10]))
dates = ref.variables['time'][:].data/24
delta = dates*timedelta(days=1)
day = start + delta
df_day = pd.DataFrame(pd.to_datetime(day), columns=['Date'])
# df_day = df_day.iloc[keep_list]
# df_day_drop = df_day.drop(drop_list, axis=0)
# df_gid = df_day

df_gid = df_day
# df_gid = df_day[df_day['Date']<'2001-01-01']
# df_gid.to_csv('days.csv')
print(lat, lon, rtf_day)
  
# for ens in ensemble:
    # print(ens+1)
TP = ref.variables['tp'][:].data[:,lat,lon]
TP_1 = np.append([0],TP[:-1])
P = np.round(TP, 1) - np.round(TP_1, 1)
P[P <= 0] = 0
df_P = pd.DataFrame(P, columns=['0']) #put data in dataframe
# df_P = df_P.iloc[keep_list]
# df_P = df_P.drop(drop_list, axis=0)
# df_P = df_P[:len(df_gid)]
df_gid = pd.concat([df_gid,df_P], axis=1)
                        
  
# df_Plt = df_Plt.sort_index()
# df_Plt = df_Plt[~df_Plt.index.duplicated(keep='last')]
# df_Plt.to_csv('./csv/c_P_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'_6h.csv')
# df_Plt = df_Plt.resample('D').sum()
# df_Plt.to_csv('./csv/c_P_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'.csv')
# df_Plt
# plt.plot(df_Plt)

#################################################################################
# Perturbed forecasts

ref = nc.Dataset('param_'+var+'_'+date+'.nc')
for ens in ensemble:
    # print(ens+1)
    TP = ref.variables['tp'][:].data[:,ens,lat,lon]
    TP_1 = np.append([0],TP[:-1])
    P = np.round(TP, 1) - np.round(TP_1, 1)
    P[P <= 0] = 0
    df_P = pd.DataFrame(P, columns=[str(ens+1)]) #put data in dataframe
    # df_P = df_P.iloc[keep_list]
    # df_P = df_P.drop(drop_list, axis=0)
    # df_P = df_P[:len(df_gid)]
    df_gid = pd.concat([df_gid,df_P], axis=1)


mask = ((df_gid['Date']>'2010-01-01') & (df_gid['Date']<='2011-01-01'))
df_gid = df_gid[mask]
df_gid.index = df_gid['Date']
df_gid = df_gid.drop(['Date'], axis=1)
# df_Plt = df_Plt.append(df_gid)

df_gidd = df_gid.resample('D').sum()

# df_gidd1 = df_gidd*1
# df_gidd2 = df_gidd*1
# df_gidd3 = df_gidd*1
# df_gidd4 = df_gidd*1


# plt.plot(df_gid.index[:], df_gid[:])

ensembles = [0,1,2,3,4,5,6,7,8,9,10]


fig, ax = plt.subplots()
plt.plot(df_gidd.index[:12], df_gidd[:12], alpha=0.7, zorder=1)
for ens in ensembles:
    plt.scatter(df_gidd.index[4:8], df_gidd[4:8][str(ens)], marker='o', s=10, c='purple', zorder=2)
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.set_xlabel('Fecha')
ax.set_ylabel('Precipitación [mm]')
ax.set_title('Predicción en conjunto de precipitación acumulada en 24 horas\n a partir de repronósticos climatológicos')
startx = int(round(df_gidd.index[0].timestamp()))/86400
left, bottom, width, height = (startx+3.5, -5, 3.9, 140)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        fill=False,
                        color="purple",
                       linewidth=2)
                       #facecolor="red")
plt.gca().add_patch(rect)

################################################################################

fig, ax = plt.subplots()
plt.plot(df_gidd1.index[:10], df_gidd1[:10], alpha=0.7, zorder=1, label=df_gidd1.columns)
plt.scatter(df_gidd1.index[4:8], df_gidd1[4:8]['10'], marker='o', s=10, c='purple', zorder=2, label='5-8d')
for ens in ensemble:
    plt.scatter(df_gidd1.index[4:8], df_gidd1[4:8][str(ens)], marker='o', s=10, c='purple', zorder=2)
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.set_xlabel('Fecha')
ax.set_ylabel('Precipitación [mm]')
ax.set_title('Predicción en conjunto de precipitación diaria')
startx = int(round(df_gidd1.index[0].timestamp()))/86400
left, bottom, width, height = (startx+3.75, -5, 3.5, 140)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        fill=False,
                        color="purple",
                       linewidth=2)
                       #facecolor="red")
plt.gca().add_patch(rect)
plt.legend(ncol=2)
# plt.savefig('../figs/Alex2010_25jun-04jul_reforecasts.jpg', format='jpg', dpi=1000, bbox_inches='tight')

################################################################################


fig, ax = plt.subplots()
plt.plot(df_gidd2.index[:12], df_gidd2[:12], alpha=0.7, zorder=1)
for ens in ensembles:
    plt.scatter(df_gidd2.index[4:7], df_gidd2[4:7][str(ens)], marker='o', s=10, c='purple', zorder=2)
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.set_xlabel('Fecha')
ax.set_ylabel('Precipitación [mm]')
ax.set_title('Predicción en conjunto de precipitación acumulada en 24 horas\n a partir de repronósticos climatológicos')
startx = int(round(df_gidd2.index[0].timestamp()))/86400
left, bottom, width, height = (startx+3.5, -5, 2.9, 160)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        fill=False,
                        color="purple",
                       linewidth=2)
                       #facecolor="red")
plt.gca().add_patch(rect)

################################################################################

fig, ax = plt.subplots()
plt.plot(df_gidd1.index[:15], df_gidd1[:15], alpha=0.5, zorder=1)
plt.plot(df_gidd2.index[:11], df_gidd2[:11], alpha=0.5, zorder=1)
for ens in ensembles:
    plt.scatter(df_gidd1.index[4:8], df_gidd1[4:8][str(ens)], marker='o', s=10, c='purple', zorder=2)
    plt.scatter(df_gidd2.index[4:7], df_gidd2[4:7][str(ens)], marker='o', s=10, c='purple', zorder=2)
ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.set_xlabel('Fecha')
ax.set_ylabel('Precipitación [mm]')
ax.set_title('Predicción en conjunto de precipitación acumulada en 24 horas\n a partir de repronósticos climatológicos')
startx = int(round(df_gidd1.index[0].timestamp()))/86400
left, bottom, width, height = (startx+3.5, -5, 4, 160)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        fill=False,
                        color="purple",
                       linewidth=2)
                       #facecolor="red")
plt.gca().add_patch(rect)
startx = int(round(df_gidd1.index[0].timestamp()))/86400
left, bottom, width, height = (startx+7.5, -5, 3, 160)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        fill=False,
                        color="purple",
                       linewidth=2)
                       #facecolor="red")
plt.gca().add_patch(rect)




################################################################################

fig, ax = plt.subplots()
plt.plot(df_gidd1.index[:], df_gidd1[:], color='purple', alpha=0.25)
plt.plot(df_gidd2.index[:-4], df_gidd2[:-4], color='red', alpha=0.25)
plt.plot(df_gidd3.index[:-7], df_gidd3[:-7], color='green', alpha=0.25)
plt.plot(df_gidd4.index[:-11], df_gidd4[:-11], color='orange', alpha=0.25)

# plt.plot(df_gidd1.index[4:8], df_gidd1[4:8], color='purple', marker='.', markerfacecolor='blue', markersize=10)
# plt.plot(df_gidd2.index[4:7], df_gidd2[4:7], color='red', marker='.', markerfacecolor='blue', markersize=10)
# plt.plot(df_gidd3.index[4:8], df_gidd3[4:8], color='green', marker='.', markerfacecolor='blue', markersize=10)
# plt.plot(df_gidd4.index[4:7], df_gidd4[4:7], color='orange', marker='.', markerfacecolor='blue', markersize=10)

for ens in ensembles:
    plt.scatter(df_gidd1.index[4:8], df_gidd1[4:8][str(ens)], marker='o', s=10, c='blue')
    plt.scatter(df_gidd2.index[4:7], df_gidd2[4:7][str(ens)], marker='o', s=10, c='blue')
    plt.scatter(df_gidd3.index[4:8], df_gidd3[4:8][str(ens)], marker='o', s=10, c='blue')
    plt.scatter(df_gidd4.index[4:7], df_gidd4[4:7][str(ens)], marker='o', s=10, c='blue')


ax.xaxis.set_major_formatter(
    mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
ax.set_xlabel('Fecha')
ax.set_ylabel('Precipitación [mm]')
ax.set_title('Precipitación acumulada en 24 horas de repronóstico climatológico')
startx = int(round(df_gidd1.index[0].timestamp()))/86400

left, bottom, width, height = (startx+3.5, 0, 3.9, 160)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        fill=False,
                        color="purple",
                        linewidth=2)
                        #facecolor="red")
plt.gca().add_patch(rect)
left, bottom, width, height = (startx+7.5, 0, 2.9, 160)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        fill=False,
                        color="red",
                       linewidth=2)
                       #facecolor="red")
plt.gca().add_patch(rect)
left, bottom, width, height = (startx+10.5, 0, 3.9, 160)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        fill=False,
                        color="green",
                       linewidth=2)
                       #facecolor="red")
plt.gca().add_patch(rect)
left, bottom, width, height = (startx+14.5, 0, 2.9, 160)
rect=mpatches.Rectangle((left,bottom),width,height, 
                        fill=False,
                        color="orange",
                       linewidth=2)
                       #facecolor="red")
plt.gca().add_patch(rect)


# daysFmt = mdates.DateFormatter('%d')
# days = mdates.DayLocator(interval=2)
# ax.xaxis.set_major_formatter(
#     mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
# ax.xaxis.set_major_formatter(
#     mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
# monthsFmt = mdates.DateFormatter('%d\n%b')
# # months = mdates.MonthLocator()
# ax.xaxis.set_major_formatter(daysFmt)
# ax.xaxis.set_major_locator(days)
# ax.xaxis.set_minor_formatter(monthsFmt)
# # ax.xaxis.set_minor_locator(months)

# df_gid.index = df_gid['Date']
# df_gid = df_gid.drop(['Date'], axis=1)
# df_Plt = df_Plt.append(df_gid)

# df_Plt = df_Plt.sort_index()
# df_Plt = df_Plt[~df_Plt.index.duplicated(keep='last')]
# df_Plt.to_csv('./csv/P_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'_6h.csv')
# df_Plt = df_Plt.resample('D').sum()
# df_Plt.to_csv('./csv/P_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'.csv')
# df_Plt
# plt.plot(df_Plt)

################################################################################
#Prefs
# os.chdir("D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/ECMWF/nc")

start_lt = 5
# start_lt = lt
end_lt = start_lt + 3
print(start_lt, end_lt)

pref = pd.read_csv('./csv/wa/all/P_228228_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=["Unnamed: 0"])
pref.index = pref["Unnamed: 0"]
pref.index.name = None
pref = pref.drop(["Unnamed: 0"], axis=1)
pref

# mask = ((pref.index>='2010-01-01') & (pref.index<'2011-01-01'))
mask = ((pref.index>='2010-06-01') & (pref.index<'2010-08-01'))
pref_m = pref[mask]

fig, ax = plt.subplots()
ax.plot(pref_m, label=pref_m.columns, alpha=0.75)
# maj_loc = mdates.MonthLocator(bymonth=np.arange(1,13,2))
# ax.xaxis.set_major_locator(maj_loc)
# min_loc = mdates.MonthLocator()
# ax.xaxis.set_minor_locator(min_loc)
# Set major date tick formatter
zfmts = ['', '%b\n%Y', '%b', '%b-%d', '%H:%M', '%H:%M']
maj_loc = mdates.MonthLocator() #bymonth=np.arange(1,13,2))
maj_fmt = mdates.ConciseDateFormatter(maj_loc) #maj_loc, zero_formats=zfmts)
ax.xaxis.set_major_formatter(maj_fmt)
min_loc = mdates.DayLocator()
ax.xaxis.set_minor_locator(min_loc)
ax.set_xlabel('Fecha')
ax.set_ylabel('Precipitación [mm]')
ax.set_title('Series de tiempo sintéticas de precipitación')
ax.legend(ncol=2)
plt.savefig('../figs/Julio2010_01jun-01ago_reforecasts.jpg', format='jpg', dpi=1000, bbox_inches='tight')


################################################################################
#Obs discharge and model
os.chdir("D:/DANI/2021/TEMA4_PRONOSTICOS/PYR/HMS/Results/csv")

pobs = pd.read_csv('D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/VARIOS/Tables/clicom_malla_data.csv', parse_dates=["Unnamed: 0"])
pobs.index = pobs["Unnamed: 0"]
pobs.index.name = None
pobs = pobs.drop(["Unnamed: 0", "T"], axis=1)
pobs.columns = ['Pobs']
pobs

qobs = pd.read_csv('Q_mean.csv', parse_dates=["Unnamed: 0"])
qobs.index = qobs["Unnamed: 0"]
qobs.index.name = None
qobs = qobs.drop(["Unnamed: 0"], axis=1)
qobs.columns = ['Qobs']
qobs

qsim = pd.read_csv('Q_CLICOM.csv', parse_dates=["Unnamed: 0"])
qsim.index = qsim["Unnamed: 0"]
qsim.index.name = None
qsim = qsim.drop(["Unnamed: 0"], axis=1)
qsim.columns = ['Qsim']
qsim

# mask = ((pref.index>='2010-01-01') & (pref.index<'2011-01-01'))
mask_p = ((pobs.index>='1973-06-01') & (pobs.index<'1974-01-01'))
mask_o = ((qobs.index>='1973-06-01') & (qobs.index<'1974-01-01'))
mask_s = ((qsim.index>='1973-06-01') & (qsim.index<'1974-01-01'))
pobs_m = pobs[mask_p]
qobs_m = qobs[mask_o]
qsim_m = qsim[mask_s]


fig, (ax1, ax2) = plt.subplots(2,1)
ax1.bar(pobs_m.index, pobs_m['Pobs'], label='PCLICOM')
ax2.plot(qobs_m, label='Qobs', alpha=0.9, ls='-')
ax2.plot(qsim_m, label='Qsim', alpha=0.9, ls='--')
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
ax2.set_xlabel('Fecha')
ax1.set_title('Hietograma de PCLICOM e hidrograma simulado y observado')
ax1.invert_yaxis()
# ax1.xaxis.set_ticklabels([])
ax1.xaxis.set_visible(False)
plt.subplots_adjust(wspace=0, hspace=0.04)
ax1.set_ylabel('Precipitación [mm]')
ax2.set_ylabel('Caudal [$m^3$/s]')
ax1.legend(loc=4)
ax2.legend()
plt.savefig('../images/Jun-Dic1973_PQobs_Qsim.jpg', format='jpg', dpi=1000, bbox_inches='tight')

################################################################################
#Qrefs
# os.chdir("D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/ECMWF/nc")
os.chdir("D:/DANI/2021/TEMA4_PRONOSTICOS/PYR/HMS/Results/csv")
os.chdir("D:/DANI/2021/TEMA4_PRONOSTICOS/")

start_lt = 5
# start_lt = lt
end_lt = start_lt + 3
print(start_lt, end_lt)

qref = pd.read_csv('./PYR/HMS/Results/csv/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.csv', parse_dates=["Unnamed: 0"])
qref.index = qref["Unnamed: 0"]
qref.index.name = None
qref = qref.drop(["Unnamed: 0"], axis=1)
qref

pref = pd.read_csv('./DATOS/ECMWF/nc/csv/wa/all/P_228228_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=["Unnamed: 0"])
pref.index = pref["Unnamed: 0"]
pref.index.name = None
pref = pref.drop(["Unnamed: 0"], axis=1)
pref

mask_p = ((pref.index>='2010-06-01') & (pref.index<'2010-08-01'))
mask_q = ((qref.index>='2010-06-01') & (qref.index<'2010-08-01'))
pref_m = pref[mask_p]
qref_m = qref[mask_q]

fig, (ax1, ax2) = plt.subplots(2,1)
ax1.plot(pref_m, label=qref_m.columns, alpha=0.75)
ax2.plot(qref_m, label=qref_m.columns, alpha=0.75)
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
ax2.set_xlabel('Fecha')
ax1.set_title('Series de tiempo sintéticas de precipitación y caudal')
ax1.invert_yaxis()
# ax1.xaxis.set_ticklabels([])
ax1.xaxis.set_visible(False)
plt.subplots_adjust(wspace=0, hspace=0.04)
ax1.set_ylabel('Precipitación [mm]')
ax2.set_ylabel('Caudal [$m^3$/s]')
ax2.legend(ncol=3, loc=2, prop={'size': 9})
# ax2.legend()
plt.savefig('./PYR/HMS/Results/images/jun-ago2010_PQreforecasts.jpg', format='jpg', dpi=1000, bbox_inches='tight')


################################################################################
#Qcorrelation for Q daily and Q max
# os.chdir("D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/ECMWF/nc")
os.chdir("D:/DANI/2021/TEMA4_PRONOSTICOS/PYR/HMS/Results/csv")
os.chdir("D:/DANI/2021/TEMA4_PRONOSTICOS/")

for lt in slt:
    # start_lt = 5
    start_lt = lt
    end_lt = start_lt + 3
    print(start_lt, end_lt)

    qref = pd.read_csv('./PYR/HMS/Results/csv/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.csv', parse_dates=["Unnamed: 0"])
    qref.index = qref["Unnamed: 0"]
    qref.index.name = None
    qref = qref.drop(["Unnamed: 0"], axis=1)
    qref
    
    qcorr = qref.corr(method='pearson')
    np_res = qcorr.to_numpy()
    
    fig, ax = plt.subplots(1)
    # divnorm=colors.TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
    c = ax.matshow(qcorr, cmap='Greens') #, norm=divnorm)
    plt.xticks(ensembles)
    plt.yticks(ensembles)
    ax.set_title('Correlación entre miembros de caudal medio diario')
    ax.set_xlabel('Miembro, '+str(start_lt)+' a '+str(end_lt)+' días')
    ax.set_ylabel('Miembro, '+str(start_lt)+' a '+str(end_lt)+' días')
    ax.xaxis.tick_bottom()
    fig.colorbar(c, ax=ax)
    for (i, j), z in np.ndenumerate(np_res):
        # print(i,j,z)
        if i==j:
            ax.text(i, j, '{:0.2f}'.format(z), color='w', weight='normal', ha='center', va='center', fontsize=8)
        else:
            ax.text(i, j, '{:0.2f}'.format(z), color='k', weight='normal', ha='center', va='center', fontsize=8)
            # path_effects=[pe.withStroke(linewidth=0.5, foreground='k')])
    #            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    name = 'Qd_corr_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)
    fig.savefig('./DATOS/VARIOS/Figures/discharge/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'/'+name+'.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    plt.close()
    
#Qmax yearly
for lt in slt:
    # start_lt = 5
    start_lt = lt
    end_lt = start_lt + 3
    print(start_lt, end_lt)

    qref = pd.read_csv('./PYR/HMS/Results/csv/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.csv', parse_dates=["Unnamed: 0"])
    qref.index = qref["Unnamed: 0"]
    qref.index.name = None
    qref = qref.drop(["Unnamed: 0"], axis=1)
    # qref
    qref_max = qref.resample('Y').max()
    qref_max = qref_max[:-1]
    
    qcorr = qref_max.corr(method='pearson')
    np_res = qcorr.to_numpy()
    
    fig, ax = plt.subplots(1)
    # divnorm=colors.TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
    c = ax.matshow(qcorr, cmap='Greens') #, norm=divnorm)
    plt.xticks(ensembles)
    plt.yticks(ensembles)
    ax.set_title('Correlación entre miembros de caudal máximo anual')
    ax.set_xlabel('Miembro, '+str(start_lt)+' a '+str(end_lt)+' días')
    ax.set_ylabel('Miembro, '+str(start_lt)+' a '+str(end_lt)+' días')
    ax.xaxis.tick_bottom()
    fig.colorbar(c, ax=ax)
    for (i, j), z in np.ndenumerate(np_res):
        # print(i,j,z)
        if i==j:
            ax.text(i, j, '{:0.2f}'.format(z), color='w', weight='normal', ha='center', va='center', fontsize=8)
        else:
            ax.text(i, j, '{:0.2f}'.format(z), color='k', weight='normal', ha='center', va='center', fontsize=8)
            # path_effects=[pe.withStroke(linewidth=0.5, foreground='k')])
    #            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    name = 'Qm_corr_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)
    fig.savefig('./DATOS/VARIOS/Figures/discharge/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'/'+name+'.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    plt.close()
    
################################################################################
#Qcorrelation for Q daily and Q max
# os.chdir("D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/ECMWF/nc")
# os.chdir("D:/DANI/2021/TEMA4_PRONOSTICOS/PYR/HMS/Results/csv")
os.chdir("D:/DANI/2021/TEMA4_PRONOSTICOS/")

def ensemble1(x):
    return x
def ensemble2(x):
    return x

for lt in slt:
    # start_lt = 5
    start_lt = lt
    end_lt = start_lt + 3
    print(start_lt, end_lt)

    qref = pd.read_csv('./PYR/HMS/Results/csv/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.csv', parse_dates=["Unnamed: 0"])
    qref.index = qref["Unnamed: 0"]
    qref.index.name = None
    qref = qref.drop(["Unnamed: 0"], axis=1)
    qref_max = qref.resample('Y').max()
    qref_max = qref_max[:-1]
    qref
    
    qcorr = qref.corr(method='pearson')
    np_res = qcorr.to_numpy()
    qcorr_max = qref_max.corr(method='pearson')
    np_res_max = qcorr_max.to_numpy()

    for ens in ensembles:
        qcorr_max[str(ens)][str(ens):] = qcorr[str(ens)][str(ens):]
    np_res = qcorr_max.to_numpy()
    
    fig, ax = plt.subplots(1)
    plt.figure(figsize=(15,15))
    # divnorm=colors.TwoSlopeNorm(vmin=-1.0, vcenter=0, vmax=1.0)
    c = ax.matshow(qcorr_max, cmap='Greens') #, norm=divnorm)
    ax.set_xticks(ensembles)
    ax.set_yticks(ensembles)
    ax.set_title('Correlación entre miembros de caudal, '+str(start_lt)+' a '+str(end_lt)+' días', x=0.625)
    ax.set_xlabel('Miembro, Q diario', va='center')
    ax.set_ylabel('Miembro, Q diario', va='top')
    ax.xaxis.tick_bottom()
    ax2 = ax.secondary_xaxis('top', functions=(ensemble1, ensemble2))
    ax2.set_xticks(ensembles)
    ax2.set_xlabel('Miembro, Q max anual')
    ay2 = ax.secondary_yaxis('right', functions=(ensemble1, ensemble2))
    ay2.set_yticks(ensembles)
    ay2.set_ylabel('Miembro, Q max anual', rotation=270)
    fig.colorbar(c, ax=ax, location='right', fraction=0.15, pad=0.1)
    # fig.colorbar(c, ax=ax, location='bottom', fraction=0.05, pad=0.15)
    for (i, j), z in np.ndenumerate(np_res):
        # print(i,j,z)
        if i==j:
            ax.text(i, j, '{:0.2f}'.format(z), color='w', weight='normal', ha='center', va='center', fontsize=8)
        else:
            ax.text(j, i, '{:0.2f}'.format(z), color='k', weight='normal', ha='center', va='center', fontsize=8)
            # path_effects=[pe.withStroke(linewidth=0.5, foreground='k')])
    #            bbox=dict(boxstyle='round', facecolor='white', edgecolor='0.3'))
    name = 'Qdy_corr_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)
    fig.savefig('./DATOS/VARIOS/Figures/discharge/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'/'+name+'.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    plt.close()


################################################################################
#Q max plot for Q obs and Q refs

os.chdir('D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS')

years = range(2000, 2021, 2)
years_obs = range(1973, 1994, 2)

def years_ref(x):
    return x
def years_obs(x):
    return x

#Datasets discharge
Qobs = pd.read_csv('../PYR/HMS/Results/csv/Q_mean.csv', parse_dates=['Unnamed: 0'])
Qobs.index = Qobs['Unnamed: 0']
Qobs = Qobs.drop(['Unnamed: 0'], axis=1)
Qobs.index.name = None
Qobs.columns = ['Qobs']
Qobs_m = Qobs.resample('Y').max()
Qobs_p = Qobs.resample('Y').mean()
Qobs_m.index.year

for lt in slt:
    # start_lt = 5
    start_lt = lt
    end_lt = start_lt + 3
    print('lt', start_lt, end_lt)

    #Discharge
    Q = pd.read_csv('../PYR/HMS/Results/csv/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.csv', parse_dates=['Unnamed: 0'])
    Q.index = Q['Unnamed: 0']
    Q.index.name = None
    Q = Q.drop(['Unnamed: 0'], axis=1)
    Q_maxlt = Q.resample('Y').max()
    Q_maxlt = Q_maxlt[:-1]
    
    fig, ax = plt.subplots(1)
    ax.plot(Q_maxlt.index.year, Q_maxlt, label=Q_maxlt.columns)
    ax.plot(Qobs_m.index.year+(2000-1973), Qobs_m, label='Qobs', alpha=0.7, color='k', ls='--')
    ax.legend(ncol=3, prop={'size': 8})
    ax.set_title('Gasto máximo anual')
    ax.set_ylabel('Gasto [$m^3$/s]')
    ax.set_xlabel('Año, Q repronósticos, '+str(start_lt)+' a '+str(end_lt)+' días')
    ax.set_xticks(years)
    ax2 = ax.secondary_xaxis('top', functions=(ensemble1, ensemble2))
    ax2.set_xticks(years)
    ax2.set_xticklabels(years_obs)
    ax2.set_xlabel('Año, Q observado')
    # plt.savefig('../SSP/Q_max_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    plt.savefig('../SSP/Q_max_obsLT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    plt.close()


help(ax2.set_xticklabels)


################################################################################
#Q scatter plot for Q daily and Q max

start_lt = 5
# start_lt = lt
end_lt = start_lt + 3
print(start_lt, end_lt)

qref = pd.read_csv('./PYR/HMS/Results/csv/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.csv', parse_dates=["Unnamed: 0"])
qref.index = qref["Unnamed: 0"]
qref.index.name = None
qref = qref.drop(["Unnamed: 0"], axis=1)
qref

ens = 0
x = qref[str(ens+1)]
y = qref[str(ens+2)]
corrp = pearsonr(x,y)
qmax = max([max(x),max(y)])
f = np.polyfit(x,y,1)
z = np.poly1d(f)

plt.scatter(x,y, marker='.', s=2, label='Data')
plt.plot([0,qmax],[0,qmax], ls='--', lw=1, color='k', label='1:1')
# plt.plot(x,z(x), ls='-', lw=1, color='b', label='Reg')
plt.plot(x.sort_values(), y.sort_values(), ls='-', lw=1, color='g', label='Q-Q')
plt.title('Comparison of Q mean daily, Reforecast='+str(ens+1)+' vs '+str(ens+2))
plt.xlabel('Q mean daily, Reforecast='+str(ens+1))
plt.ylabel('Q mean daily, Reforecast='+str(ens+2))
plt.text(0, max([max(x),max(y)]), ha='left', va='top',
         s= 'Cc = '+str(round(corrp[0],3))+'\np-value = '+str(round(corrp[1],3)))
plt.legend(loc='lower right')
plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/Q'+str(start_lt)+str(end_lt)+'/'+id+'/correlation/scatter2/'+id+'_Ref='+str(ens+1)+'vs'+str(enss+1)+'_Q mean daily_wr.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.close()