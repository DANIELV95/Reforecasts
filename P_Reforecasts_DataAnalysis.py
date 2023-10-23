# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 08:56:25 2021

@author: HIDRAULICA-Dani
"""

import os
import numpy as np
import pandas as pd
import netCDF4 as nc
from datetime import datetime, timedelta
from dateutil.rrule import rrule, WEEKLY, MO, TH
# import matplotlib.pyplot as plt

os.chdir("D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/ECMWF/nc")
os.listdir()

ensemble = [0,1,2,3,4,5,6,7,8,9]
refs_list = os.listdir()
ilon = [6,7,8]
ilat = [3,4,5]
start = datetime(1900,1,1)
slt = [*range(2,17)] #1,17
# slt = [2,5,8,11,14] #17
rtf_dates = list(rrule(WEEKLY, byweekday=[MO,TH], dtstart=datetime(2019,12,31), until=datetime(2020,12,31)))
variables = ['169-175'] #'228228', '121-122', '169-175', '165-166']
# variables24 = ['165-166', '169-175', '130']



# Control forecasts

for var in variables:
    # var = '228228'
    for lt in slt:
        # start_lt = 5
        start_lt = lt
        end_lt = start_lt + 3
        print(start_lt, end_lt)
        
        # drop_list = [*range(0,start_lt*4), *range(end_lt*4-1,1620)] #for lt =! 1
        
        # np.set_printoptions(suppress=True) # print w/o scientific notation
        
        for lat in ilat:
            for lon in ilon:
                # lat = 6
                # lon = 3
                
                if var == '228228': #Precipitation
                    df_Plt = pd.DataFrame()
                    keep_list = np.array([])
                    for i in range(0,20):
                        keep = np.arange((start_lt-1)*4+i*(4*20+1), end_lt*4+i*(4*20+1))
                        keep_list = np.append(keep_list, keep)
                    
                    for rtf_day in rtf_dates:
                        date = str(rtf_day)[:10]
                        ref = nc.Dataset('c_param_'+var+'_'+date+'.nc')
                        
                        # start = datetime(int(date[0:4])-year,int(date[5:7]),int(date[8:10]))
                        dates = ref.variables['time'][:].data/24
                        delta = dates*timedelta(days=1)
                        day = start + delta
                        df_day = pd.DataFrame(pd.to_datetime(day), columns=['Date'])
                        df_day = df_day.iloc[keep_list]
                        # df_day_drop = df_day.drop(drop_list, axis=0)
                        df_gid = df_day
                        # df_gid.to_csv('days.csv')
                        print(lat, lon, rtf_day)
                  
                        # for ens in ensemble:
                            # print(ens+1)
                        TP = ref.variables['tp'][:].data[:,lat,lon]
                        TP_1 = np.append([0],TP[:-1])
                        P = np.round(TP, 1) - np.round(TP_1, 1)
                        P[P <= 0] = 0
                        df_P = pd.DataFrame(P, columns=['c']) #put data in dataframe
                        df_P = df_P.iloc[keep_list]
                        # df_P = df_P.drop(drop_list, axis=0)
                        df_gid = pd.concat([df_gid,df_P], axis=1)
                                                
                        df_gid.index = df_gid['Date']
                        df_gid = df_gid.drop(['Date'], axis=1)
                        df_Plt = df_Plt.append(df_gid)
                        
                    df_Plt = df_Plt.sort_index()
                    df_Plt = df_Plt[~df_Plt.index.duplicated(keep='last')]
                    df_Plt.to_csv('./csv/c_P_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'_6h.csv')
                    df_Plt = df_Plt.resample('D').sum()
                    df_Plt.to_csv('./csv/c_P_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'.csv')
                    # df_Plt
                    # plt.plot(df_Plt)
                
                elif var == '121-122': #Temperature
                    df_Tlt = pd.DataFrame()
                    keep_list = np.array([])
                    for i in range(0,20):
                        keep = np.arange((start_lt-1)*4+i*(4*20), end_lt*4+i*(4*20))
                        keep_list = np.append(keep_list, keep)
                    
                    for rtf_day in rtf_dates:
                        date = str(rtf_day)[:10]
                        ref = nc.Dataset('c_param_'+var+'_'+date+'.nc')
                        
                        # start = datetime(int(date[0:4])-year,int(date[5:7]),int(date[8:10]))
                        dates = ref.variables['time'][:].data/24
                        delta = dates*timedelta(days=1)
                        day = start + delta
                        df_day = pd.DataFrame(pd.to_datetime(day), columns=['Date'])
                        df_day = df_day.iloc[keep_list]
                        # df_day_drop = df_day.drop(drop_list, axis=0)
                        df_gid = df_day
                        # df_gid.to_csv('days.csv')
                        print(lat, lon, rtf_day)
                        
                        # for ens in ensemble:
                        # print(ens+1)
                        Tmin = ref.variables['mn2t6'][:].data[:,lat,lon] - 273.15
                        Tmax = ref.variables['mx2t6'][:].data[:,lat,lon] - 273.15
                        Tm = np.round((Tmin+Tmax)/2, 1)
                        
                        df_T = pd.DataFrame(Tm, columns=['c']) #put data in dataframe
                        df_T = df_T.iloc[keep_list]
                        df_gid = pd.concat([df_gid,df_T], axis=1)
                        
                        df_gid.index = df_gid['Date']
                        df_gid = df_gid.drop(['Date'], axis=1)
                        df_Tlt = df_Tlt.append(df_gid)
                        
                    df_Tlt = df_Tlt.sort_index()
                    df_Tlt = df_Tlt[~df_Tlt.index.duplicated(keep='last')]
                    df_Tlt.to_csv('./csv/c_T_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'_6h.csv')
                    df_Tlt = df_Tlt.resample('D').mean()
                    df_Tlt.to_csv('./csv/c_T_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'.csv')
                    #df_Tlt
                    # plt.plot(df_Tlt)
                    
                elif var == '165-166': #Wind
                    df_Wult = pd.DataFrame()
                    df_Wvlt = pd.DataFrame()
                    df_Wwlt = pd.DataFrame()
                    keep_list = np.array([])
                    for i in range(0,20):
                        keep = np.arange((start_lt-1)+i*(20+1), end_lt+i*(20+1))
                        keep_list = np.append(keep_list, keep)
                    
                    for rtf_day in rtf_dates:
                        date = str(rtf_day)[:10]
                        ref = nc.Dataset('c_param_'+var+'_'+date+'.nc')
                        
                        # start = datetime(int(date[0:4])-year,int(date[5:7]),int(date[8:10]))
                        dates = ref.variables['time'][:].data/24
                        delta = dates*timedelta(days=1)
                        day = start + delta
                        df_day = pd.DataFrame(pd.to_datetime(day), columns=['Date'])
                        df_day = df_day.iloc[keep_list]
                        # df_day_drop = df_day.drop(drop_list, axis=0)
                        df_gidu = df_day
                        df_gidv = df_day
                        df_gidw = df_day
                        # df_gid.to_csv('days.csv')
                        print(lat, lon, rtf_day)
                        
                        # for ens in ensemble:
                            # print(ens+1)
                        Wu = ref.variables['u10'][:].data[:,lat,lon]
                        Wv = ref.variables['v10'][:].data[:,lat,lon]
                        Ww = np.round((Wu**2+Wv**2)**0.5, 1)
                        
                        df_Wu = pd.DataFrame(Wu, columns=['c']) #put data in dataframe
                        df_Wu = df_Wu.iloc[keep_list]
                        df_gidu = pd.concat([df_gidu,df_Wu], axis=1)
                        df_Wv = pd.DataFrame(Wv, columns=['c']) #put data in dataframe
                        df_Wv = df_Wv.iloc[keep_list]
                        df_gidv = pd.concat([df_gidv,df_Wv], axis=1)
                        df_Ww = pd.DataFrame(Ww, columns=['c']) #put data in dataframe
                        df_Ww = df_Ww.iloc[keep_list]
                        df_gidw = pd.concat([df_gidw,df_Ww], axis=1)
                        
                        df_gidu.index = df_gidu['Date']
                        df_gidu = df_gidu.drop(['Date'], axis=1)
                        df_Wult = df_Wult.append(df_gidu)
                        df_gidv.index = df_gidv['Date']
                        df_gidv = df_gidv.drop(['Date'], axis=1)
                        df_Wvlt = df_Wvlt.append(df_gidv)
                        df_gidw.index = df_gidw['Date']
                        df_gidw = df_gidw.drop(['Date'], axis=1)
                        df_Wwlt = df_Wwlt.append(df_gidw)
                        
                    df_Wult = df_Wult.sort_index()
                    df_Wult = df_Wult[~df_Wult.index.duplicated(keep='last')]
                    df_Wult.to_csv('./csv/c_Wu_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'.csv')
                    df_Wvlt = df_Wvlt.sort_index()
                    df_Wvlt = df_Wvlt[~df_Wvlt.index.duplicated(keep='last')]
                    df_Wvlt.to_csv('./csv/c_Wv_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'.csv')
                    df_Wwlt = df_Wwlt.sort_index()
                    df_Wwlt = df_Wwlt[~df_Wwlt.index.duplicated(keep='last')]
                    df_Wwlt.to_csv('./csv/c_Ww_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'.csv')
                    # df_Wlt
                    # plt.plot(df_Wlt)
                
                else: # var = '169-175' #Radiation
                    df_Rlts = pd.DataFrame()
                    df_Rltl = pd.DataFrame()
                    keep_list = np.array([])
                    for i in range(0,20):
                        keep = np.arange((start_lt-1)+i*(20+1), end_lt+i*(20+1))
                        keep_list = np.append(keep_list, keep)
                    
                    for rtf_day in rtf_dates:
                        date = str(rtf_day)[:10]
                        ref = nc.Dataset('c_param_'+var+'_'+date+'.nc')
                        
                        # start = datetime(int(date[0:4])-year,int(date[5:7]),int(date[8:10]))
                        dates = ref.variables['time'][:].data/24
                        delta = dates*timedelta(days=1)
                        day = start + delta
                        df_day = pd.DataFrame(pd.to_datetime(day), columns=['Date'])
                        df_day = df_day.iloc[keep_list]
                        # df_day_drop = df_day.drop(drop_list, axis=0)
                        df_gids = df_day
                        df_gidl = df_day
                        # df_gid.to_csv('days.csv')
                        print(lat, lon, rtf_day)
                        
                        # for ens in ensemble:
                            # print(ens+1)
                        Rsw = ref.variables['ssrd'][:].data[:,lat,lon]/86400
                        Rsw_1 = np.append([0],Rsw[:-1])
                        Rsws = np.round(Rsw, 1) - np.round(Rsw_1, 1)
                        Rsws[Rsws <= 0] = 0
                        
                        Rlw = ref.variables['strd'][:].data[:,lat,lon]/86400
                        Rlw_1 = np.append([0],Rlw[:-1])
                        Rlws = np.round(Rlw, 1) - np.round(Rlw_1, 1)
                        Rlws[Rlws <= 0] = 0
                        
                        df_Rs = pd.DataFrame(Rsws, columns=['c']) #put data in dataframe
                        df_Rs = df_Rs.iloc[keep_list]
                        df_gids = pd.concat([df_gids,df_Rs], axis=1)
                        df_Rl = pd.DataFrame(Rlws, columns=['c']) #put data in dataframe
                        df_Rl = df_Rl.iloc[keep_list]
                        df_gidl = pd.concat([df_gidl,df_Rl], axis=1)
                        
                        df_gids.index = df_gids['Date']
                        df_gids = df_gids.drop(['Date'], axis=1)
                        df_Rlts = df_Rlts.append(df_gids)
                        df_gidl.index = df_gidl['Date']
                        df_gidl = df_gidl.drop(['Date'], axis=1)
                        df_Rltl = df_Rltl.append(df_gidl)
                        
                    df_Rlts = df_Rlts.sort_index()
                    df_Rlts = df_Rlts[~df_Rlts.index.duplicated(keep='last')]
                    df_Rlts.to_csv('./csv/c_Rs_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'.csv')
                    df_Rltl = df_Rltl.sort_index()
                    df_Rltl = df_Rltl[~df_Rltl.index.duplicated(keep='last')]
                    df_Rltl.to_csv('./csv/c_Rl_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'.csv')
                    # df_Wlt
                    # plt.plot(df_Wlt)

#################################################################################
# Perturbed forecasts

for var in variables:
    for lt in slt:
        # start_lt = 17
        start_lt = lt
        end_lt = start_lt + 3
        print(start_lt, end_lt)
        
        # drop_list = [*range(0,start_lt*4), *range(end_lt*4-1,1620)] #for lt =! 1
        
        
        
        # np.set_printoptions(suppress=True) # print w/o scientific notation
        
        for lat in ilat:
            for lon in ilon:
                
                if var == '228228': #Precipitation
                    df_Plt = pd.DataFrame()
                    keep_list = np.array([])
                    for i in range(0,20):
                        keep = np.arange((start_lt-1)*4+i*(4*20+1), end_lt*4+i*(4*20+1))
                        keep_list = np.append(keep_list, keep)
                    
                    for rtf_day in rtf_dates:
                        date = str(rtf_day)[:10]
                        ref = nc.Dataset('param_'+var+'_'+date+'.nc')
                        
                        # start = datetime(int(date[0:4])-year,int(date[5:7]),int(date[8:10]))
                        dates = ref.variables['time'][:].data/24
                        delta = dates*timedelta(days=1)
                        day = start + delta
                        df_day = pd.DataFrame(pd.to_datetime(day), columns=['Date'])
                        df_day = df_day.iloc[keep_list]
                        # df_day_drop = df_day.drop(drop_list, axis=0)
                        df_gid = df_day
                        # df_gid.to_csv('days.csv')
                        print(lat, lon, rtf_day)
                        
                        for ens in ensemble:
                            # print(ens+1)
                            TP = ref.variables['tp'][:].data[:,ens,lat,lon]
                            TP_1 = np.append([0],TP[:-1])
                            P = np.round(TP, 1) - np.round(TP_1, 1)
                            P[P <= 0] = 0
                            df_P = pd.DataFrame(P, columns=[str(ens+1)]) #put data in dataframe
                            df_P = df_P.iloc[keep_list]
                            # df_P = df_P.drop(drop_list, axis=0)
                            df_gid = pd.concat([df_gid,df_P], axis=1)
                                                    
                        df_gid.index = df_gid['Date']
                        df_gid = df_gid.drop(['Date'], axis=1)
                        df_Plt = df_Plt.append(df_gid)
                        
                    df_Plt = df_Plt.sort_index()
                    df_Plt = df_Plt[~df_Plt.index.duplicated(keep='last')]
                    df_Plt.to_csv('./csv/P_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'_6h.csv')
                    df_Plt = df_Plt.resample('D').sum()
                    df_Plt.to_csv('./csv/P_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'.csv')
                    # df_Plt
                    # plt.plot(df_Plt)
                
                elif var == '121-122': #Temperature
                    df_Tlt = pd.DataFrame()
                    keep_list = np.array([])
                    for i in range(0,20):
                        keep = np.arange((start_lt-1)*4+i*(4*20), end_lt*4+i*(4*20))
                        keep_list = np.append(keep_list, keep)
                    
                    for rtf_day in rtf_dates:
                        date = str(rtf_day)[:10]
                        ref = nc.Dataset('param_'+var+'_'+date+'.nc')
                        
                        # start = datetime(int(date[0:4])-year,int(date[5:7]),int(date[8:10]))
                        dates = ref.variables['time'][:].data/24
                        delta = dates*timedelta(days=1)
                        day = start + delta
                        df_day = pd.DataFrame(pd.to_datetime(day), columns=['Date'])
                        df_day = df_day.iloc[keep_list]
                        # df_day_drop = df_day.drop(drop_list, axis=0)
                        df_gid = df_day
                        # df_gid.to_csv('days.csv')
                        print(lat, lon, rtf_day)
                        
                        for ens in ensemble:
                            # print(ens+1)
                            Tmin = ref.variables['mn2t6'][:].data[:,ens,lat,lon] - 273.15
                            Tmax = ref.variables['mx2t6'][:].data[:,ens,lat,lon] - 273.15
                            Tm = np.round((Tmin+Tmax)/2, 1)
                            
                            df_T = pd.DataFrame(Tm, columns=[str(ens+1)]) #put data in dataframe
                            df_T = df_T.iloc[keep_list]
                            df_gid = pd.concat([df_gid,df_T], axis=1)
                            
                        df_gid.index = df_gid['Date']
                        df_gid = df_gid.drop(['Date'], axis=1)
                        df_Tlt = df_Tlt.append(df_gid)
                        
                    df_Tlt = df_Tlt.sort_index()
                    df_Tlt = df_Tlt[~df_Tlt.index.duplicated(keep='last')]
                    df_Tlt.to_csv('./csv/T_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'_6h.csv')
                    df_Tlt = df_Tlt.resample('D').mean()
                    df_Tlt.to_csv('./csv/T_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'.csv')
                    #df_Tlt
                    # plt.plot(df_Tlt)
                    
                elif var == '165-166': #Wind
                    df_Wult = pd.DataFrame()
                    df_Wvlt = pd.DataFrame()
                    df_Wwlt = pd.DataFrame()
                    keep_list = np.array([])
                    for i in range(0,20):
                        keep = np.arange((start_lt-1)+i*(20+1), end_lt+i*(20+1))
                        keep_list = np.append(keep_list, keep)
                    
                    for rtf_day in rtf_dates:
                        date = str(rtf_day)[:10]
                        ref = nc.Dataset('param_'+var+'_'+date+'.nc')
                        
                        # start = datetime(int(date[0:4])-year,int(date[5:7]),int(date[8:10]))
                        dates = ref.variables['time'][:].data/24
                        delta = dates*timedelta(days=1)
                        day = start + delta
                        df_day = pd.DataFrame(pd.to_datetime(day), columns=['Date'])
                        df_day = df_day.iloc[keep_list]
                        # df_day_drop = df_day.drop(drop_list, axis=0)
                        df_gidu = df_day
                        df_gidv = df_day
                        df_gidw = df_day
                        # df_gid.to_csv('days.csv')
                        print(lat, lon, rtf_day)
                        
                        for ens in ensemble:
                            # print(ens+1)
                            Wu = ref.variables['u10'][:].data[:,ens,lat,lon]
                            Wv = ref.variables['v10'][:].data[:,ens,lat,lon]
                            Ww = np.round((Wu**2+Wv**2)**0.5, 1)
                            
                            df_Wu = pd.DataFrame(Wu, columns=[str(ens+1)]) #put data in dataframe
                            df_Wu = df_Wu.iloc[keep_list]
                            df_gidu = pd.concat([df_gidu,df_Wu], axis=1)
                            df_Wv = pd.DataFrame(Wv, columns=[str(ens+1)]) #put data in dataframe
                            df_Wv = df_Wv.iloc[keep_list]
                            df_gidv = pd.concat([df_gidv,df_Wv], axis=1)
                            df_Ww = pd.DataFrame(Ww, columns=[str(ens+1)]) #put data in dataframe
                            df_Ww = df_Ww.iloc[keep_list]
                            df_gidw = pd.concat([df_gidw,df_Ww], axis=1)
                        
                        df_gidu.index = df_gidu['Date']
                        df_gidu = df_gidu.drop(['Date'], axis=1)
                        df_Wult = df_Wult.append(df_gidu)
                        df_gidv.index = df_gidv['Date']
                        df_gidv = df_gidv.drop(['Date'], axis=1)
                        df_Wvlt = df_Wvlt.append(df_gidv)
                        df_gidw.index = df_gidw['Date']
                        df_gidw = df_gidw.drop(['Date'], axis=1)
                        df_Wwlt = df_Wwlt.append(df_gidw)
                        
                    df_Wult = df_Wult.sort_index()
                    df_Wult = df_Wult[~df_Wult.index.duplicated(keep='last')]
                    df_Wult.to_csv('./csv/Wu_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'.csv')
                    df_Wvlt = df_Wvlt.sort_index()
                    df_Wvlt = df_Wvlt[~df_Wvlt.index.duplicated(keep='last')]
                    df_Wvlt.to_csv('./csv/Wv_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'.csv')
                    df_Wwlt = df_Wwlt.sort_index()
                    df_Wwlt = df_Wwlt[~df_Wwlt.index.duplicated(keep='last')]
                    df_Wwlt.to_csv('./csv/Ww_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'.csv')
                    # df_Wlt
                    # plt.plot(df_Wlt)
                
                else: # var = '169-175' #Radiation
                    df_Rlts = pd.DataFrame()
                    df_Rltl = pd.DataFrame()
                    keep_list = np.array([])
                    for i in range(0,20):
                        keep = np.arange((start_lt-1)+i*(20+1), end_lt+i*(20+1))
                        keep_list = np.append(keep_list, keep)
                    
                    for rtf_day in rtf_dates:
                        date = str(rtf_day)[:10]
                        ref = nc.Dataset('param_'+var+'_'+date+'.nc')
                        
                        # start = datetime(int(date[0:4])-year,int(date[5:7]),int(date[8:10]))
                        dates = ref.variables['time'][:].data/24
                        delta = dates*timedelta(days=1)
                        day = start + delta
                        df_day = pd.DataFrame(pd.to_datetime(day), columns=['Date'])
                        df_day = df_day.iloc[keep_list]
                        # df_day_drop = df_day.drop(drop_list, axis=0)
                        df_gids = df_day
                        df_gidl = df_day
                        # df_gid.to_csv('days.csv')
                        print(lat, lon, rtf_day)
                        
                        for ens in ensemble:
                            # print(ens+1)
                            Rsw = ref.variables['ssrd'][:].data[:,ens,lat,lon]/86400
                            Rsw_1 = np.append([0],Rsw[:-1])
                            Rsws = np.round(Rsw, 1) - np.round(Rsw_1, 1)
                            Rsws[Rsws <= 0] = 0
                            
                            Rlw = ref.variables['strd'][:].data[:,ens,lat,lon]/86400
                            Rlw_1 = np.append([0],Rlw[:-1])
                            Rlws = np.round(Rlw, 1) - np.round(Rlw_1, 1)
                            Rlws[Rlws <= 0] = 0
                            
                            df_Rs = pd.DataFrame(Rsws, columns=[str(ens+1)]) #put data in dataframe
                            df_Rs = df_Rs.iloc[keep_list]
                            df_gids = pd.concat([df_gids,df_Rs], axis=1)
                            df_Rl = pd.DataFrame(Rlws, columns=[str(ens+1)]) #put data in dataframe
                            df_Rl = df_Rl.iloc[keep_list]
                            df_gidl = pd.concat([df_gidl,df_Rl], axis=1)
                            
                        df_gids.index = df_gids['Date']
                        df_gids = df_gids.drop(['Date'], axis=1)
                        df_Rlts = df_Rlts.append(df_gids)
                        df_gidl.index = df_gidl['Date']
                        df_gidl = df_gidl.drop(['Date'], axis=1)
                        df_Rltl = df_Rltl.append(df_gidl)
                        
                    df_Rlts = df_Rlts.sort_index()
                    df_Rlts = df_Rlts[~df_Rlts.index.duplicated(keep='last')]
                    df_Rlts.to_csv('./csv/Rs_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'.csv')
                    df_Rltl = df_Rltl.sort_index()
                    df_Rltl = df_Rltl[~df_Rltl.index.duplicated(keep='last')]
                    df_Rltl.to_csv('./csv/Rl_'+var+'_ilat'+str(lat)+'_ilon'+str(lon)+'_lt'+str(start_lt)+str(end_lt)+'.csv')
                    # df_Wlt
                    # plt.plot(df_Wlt)
