# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 16:20:27 2022

@author: HIDRAULICA-Dani
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

os.chdir('D:/DANI/2021/TEMA4_PRONOSTICOS/')

D = 24 #hours
R = 16.6*D**0.475*25.4 #mm
R = 422*D**0.475

df_obs = pd.read_csv('./DATOS/VARIOS/Tables/precip_wa.csv', parse_dates=['Unnamed: 0'])
df_obs.index = df_obs['Unnamed: 0']
df_obs.index.name = None
df_obs = df_obs.drop(['Unnamed: 0'], axis=1)
df_obs.columns = ['P']
df_obs = df_obs[df_obs.index >= '1950-01-01']

data = df_obs.resample('Y').max()
data = data.sort_values(by='P', ascending=False)
data = data.dropna()
xbar_n = data.mean()
s_n = data.std()

data_m = data[1:]
xbar_m = data_m.mean()
s_m = data_m.std()

n = len(data)
ratio_xbar = xbar_m/xbar_n
ratio_s = s_m/s_n

#from figure 4.2, Xn1 factor = 0.97
#from figure 4.4, Xn2 factor = 1.00
Xn1 = 0.97
Xn2 = 1.00
xbar = xbar_n*Xn1*Xn2

#from figure 4.3, Sn1 factor = 0.92
#from figure 4.4, Sn2 factor = 1.00
Sn1 = 0.92
Sn2 = 1.00
s = s_n*Sn1*Sn2

#from figure 4.1, km = 16
km = 16

PMP24 = xbar + km*s
adj_PMP24 = 1.13*PMP24

#Area adjustment, area factor = 1.00
areaf = 1.00
PMP = areaf*adj_PMP24
