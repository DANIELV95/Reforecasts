# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 12:07:49 2022

@author: HIDRAULICA-Dani
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
Pobs_m

Pobs_e = Pobs_m['Pobs'].sort_values(ascending=False)
Pobs_e = Pobs_e.reset_index(drop=True)
Pobs_e.index = Pobs_e.index+1
Pobs_e = pd.DataFrame(Pobs_e.values, Pobs_e.index, columns=['P'])
Trw = (len(Pobs_e)+1)/Pobs_e.index
Pobs_e['Trw'] = Trw

