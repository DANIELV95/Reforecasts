# -*- coding: utf-8 -*-
"""
Created on Fri Jun 24 12:40:01 2022

@author: HIDRAULICA-Dani
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import spotpy
import datetime as dt

os.chdir('D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS')

#Variables
ensemble = [0,1,2,3,4,5,6,7,8,9,10] #with control
#slt = [*range(2,17)]
slt = [*range(2,17,3)]
Tr = [2,5,10,25,50,100,300,1000,1250]
years = range(2000, 2021, 2)

Q = pd.read_csv('./VARIOS/Tables/Qhms_CLICopt27.csv', parse_dates=['Unnamed: 0'])
Q.index = Q['Unnamed: 0']
Q = Q.drop(['Unnamed: 0'], axis=1)
Q.index.name = None
Q.columns = ['ERA5', 'CLICOM', 'OBS']
Q = Q.dropna()
Qm = Q.resample('M').mean()



spotpy.objectivefunctions.nashsutcliffe(Q['OBS'], Q['CLICOM'])
spotpy.objectivefunctions.nashsutcliffe(Q['OBS'], Q['ERA5'])
spotpy.objectivefunctions.nashsutcliffe(Qm['OBS'], Qm['CLICOM'])
spotpy.objectivefunctions.nashsutcliffe(Qm['OBS'], Qm['ERA5'])

spotpy.objectivefunctions.kge(Q['OBS'], Q['CLICOM'])
spotpy.objectivefunctions.kge(Q['OBS'], Q['ERA5'])
spotpy.objectivefunctions.kge(Qm['OBS'], Qm['CLICOM'])
spotpy.objectivefunctions.kge(Qm['OBS'], Qm['ERA5'])

# Qclic_mon
# Qm['CLICOM']



Q = pd.read_csv('./VARIOS/Tables/Qhms_CLICopt36.csv', parse_dates=['Unnamed: 0'])
Q.index = Q['Unnamed: 0']
Q = Q.drop(['Unnamed: 0'], axis=1)
Q.index.name = None
Q.columns = ['ERA5', 'CLICOM', 'OBS']
Q = Q.dropna()
Qm = Q.resample('M').mean()



spotpy.objectivefunctions.nashsutcliffe(Q['OBS'], Q['CLICOM'])
spotpy.objectivefunctions.nashsutcliffe(Q['OBS'], Q['ERA5'])

spotpy.objectivefunctions.nashsutcliffe(Qm['OBS'], Qm['CLICOM'])
spotpy.objectivefunctions.nashsutcliffe(Qm['OBS'], Qm['ERA5'])

