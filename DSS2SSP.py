# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 13:54:57 2022

@author: HIDRAULICA-Dani
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
# import pcraster as pcr
from osgeo import gdal
import glob
import netCDF4 as nc
from mpl_toolkits.basemap import Basemap
import datetime as dt
from osgeo import gdal
import rasterio
from rasterio.plot import show
import spotpy
import pyodbc
import shutil
import subprocess
from pyextremes import EVA
import pyhecdss #only for environment with Py3.7.11


os.chdir('D:/DANI/2021/TEMA4_PRONOSTICOS/PYR/HMS/')
os.listdir()

#general variables
# years = [*range(1973,1995)] #Years with discharge data
years = [*range(2000,2021)]
today = dt.datetime.today()
nday = today.day
nmonth = today.strftime('%B')
nyear = today.year
ntime = today.strftime('%T')
ndate = today.strftime('%D')
Tr = [2, 5, 10, 25, 50, 100, 250, 500, 1000]

#ensemble = [0,1,2,3,4,5,6,7,8,9]
ensemble = [0,1,2,3,4,5,6,7,8,9,10] #with control
#slt = [*range(2,17)]
slt = [*range(2,17,3)]


for lt in slt:
    # start_lt = 2
    start_lt = lt
    end_lt = start_lt + 3
    print('lt', start_lt, end_lt)
    
    Qa = pd.DataFrame()
    
    Q = pd.read_csv('./Results/csv/Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.csv', parse_dates=['Unnamed: 0'])
    Q.index = Q['Unnamed: 0']
    Q.index.name = None
    Q = Q.drop(['Unnamed: 0'], axis=1)

    for ens in ensemble:
        # ens = 1
        Qa = pd.concat([Qa,Q[str(ens)]])
    
    start_date = dt.datetime(1900, 1, 1) #Qa.index[0]
    delta = dt.timedelta(days=1)
    end_date = start_date + delta*(len(Qa)-1)
    date_index = pd.date_range(start=start_date, end=end_date)
    
    Qa.index = date_index
    Qa.columns = ['Q_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)]
    Qa.index.freq = '1D'
    
    # to DSS
    fname=r'eva.dss'
    d = pyhecdss.DSSFile('./DSS/'+fname, create_new=True)
    Dataset = 'ECMWF-REF-HMS' #input('Dataset, A ')
    Location = 'RIOLASILLA' #input('Location, B ')
    Variable = 'FLOW' #input('Variable, C ')
    Timestep = '1Day' #input('Timestep, E ')
    Units = 'M3/S' #input('Units ')
    Type = 'PER-AVER' #input('Type' )
    Comments = 'LT '+str(start_lt).zfill(2)+'-'+str(end_lt).zfill(2) #input('Comments, F ')
    path = '/'+Dataset+'/'+Location+'/'+Variable+'//'+Timestep+'/'+Comments+'/'
    data_dss = Qa
    d.write_rts(path, data_dss, Units, Type)
    d.close()

################################################################################
#Precip

for lt in slt:
    # start_lt = 2
    start_lt = lt
    end_lt = start_lt + 3
    print('lt', start_lt, end_lt)
    
    Pa = pd.DataFrame()
    
    Pc = pd.read_csv('../../DATOS/ECMWF/nc/csv/wa/c_P_228228_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=['Unnamed: 0'])
    Pc.index = Pc['Unnamed: 0']
    Pc = Pc.drop(['Unnamed: 0'], axis=1)
    Pc.index.name = None
    Pc.columns = ['0']
    Pc.index.freq = '1D'
    
    Pe = pd.read_csv('../../DATOS/ECMWF/nc/csv/wa/P_228228_lt'+str(start_lt)+str(end_lt)+'_wa.csv', parse_dates=['Unnamed: 0'])
    Pe.index = Pe['Unnamed: 0']
    Pe = Pe.drop(['Unnamed: 0'], axis=1)
    Pe.index.name = None
    Pe.index.freq = '1D'
    
    P = pd.concat([Pc, Pe], axis=1)
    
    for ens in ensemble:
        # ens = 1
        Pa = pd.concat([Pa,P[str(ens)]])
    
    start_date = dt.datetime(1900, 1, 1) #Qa.index[0]
    delta = dt.timedelta(days=1)
    end_date = start_date + delta*(len(Pa)-1)
    date_index = pd.date_range(start=start_date, end=end_date)
    
    Pa.index = date_index
    Pa.columns = ['P_LT_'+str(start_lt).zfill(2)+str(end_lt).zfill(2)]
    Pa.index.freq = '1D'
    
    # to DSS
    fname=r'eva.dss'
    d = pyhecdss.DSSFile('./DSS/'+fname, create_new=True)
    Dataset = 'ECMWF-REF' #input('Dataset, A ')
    Location = 'RIOLASILLA' #input('Location, B ')
    Variable = 'PRECIP-INC' #input('Variable, C ')
    Timestep = '1Day' #input('Timestep, E ')
    Units = 'MM' #input('Units ')
    Type = 'PER-CUM' #input('Type' )
    Comments = 'LT '+str(start_lt).zfill(2)+'-'+str(end_lt).zfill(2) #input('Comments, F ')
    path = '/'+Dataset+'/'+Location+'/'+Variable+'//'+Timestep+'/'+Comments+'/'
    data_dss = Pa
    d.write_rts(path, data_dss, Units, Type)
    d.close()
    
    
################################################################################
#Flow HMS with CLICOM and ERA5

