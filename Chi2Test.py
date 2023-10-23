# -*- coding: utf-8 -*-
"""
Created on Tue Mar 15 15:29:45 2022

@author: HIDRAULICA-Dani
"""

#Chi squared test
#Load Libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import describe, lognorm, chi2, chisquare, skew, kurtosis #mode,
from scipy.stats.mstats import hdmedian

os.chdir("D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/")
os.listdir()

#Read data and transform to Ln
Pobs = pd.read_csv('./VARIOS/Tables/precip_wa.csv', parse_dates=['Unnamed: 0'])
Pobs.index = Pobs['Unnamed: 0']
Pobs = Pobs.drop(['Unnamed: 0'], axis=1)
Pobs.index.name = None
Pobs.columns = ['Pobs']
Pobs_m = Pobs.resample('Y').max()
Pobs_m['Pobs'][:15] = np.nan
Pobs

data = Pobs_m.dropna()[6:].values.flatten()
data.sort()
data_ln = np.log(data)

#Estimate parameters
describe(data_ln)
n = len(data)
vmin = data.min()
vmax = data.max()
mu = data_ln.mean()
sx = data_ln.std(ddof=1)
sk = skew(data_ln)
kr = kurtosis(data_ln)
med = hdmedian(data)
# mode = mode(data)
sc = np.exp(mu)

#Set bins intervals
ps = 5
f = 0.7
lmin = round(vmin/ps)*ps
lmax = 200 #round(vmax/ps*f)*ps #can be changed
inter = 10
step = (lmax-lmin)/inter

bins = [0]
for i in range(inter-1):
    lim_supi = lmin+step*(i+1)
    bins = np.append(bins, lim_supi)
bins = np.append(bins, vmax)

lim_sup = bins[1:]
lim_max = bins[-2]
lim_inf = lim_sup - step
lim_inf[-1] = lim_max

#Generate histogram and compute probabilities for data
hist = np.histogram(data, bins=bins)
ni = hist[0]
fi = ni/n
Fi = np.cumsum(fi)

#Compute frequencies for Ln distribution
dist_ln = lognorm(s=sx, loc=0, scale=sc)

Fd_sup = dist_ln.cdf(lim_sup)
Fd_sup[-1] = 1
Fd_inf = dist_ln.cdf(lim_inf)
Fd_inf[0] = 0
fd = Fd_sup - Fd_inf
nd = fd*n

#Compute X2
xi2 = n*(fi-fd)**2/fd
x2 = sum(xi2)

alpha = 0.05
v = 2
df = inter - 1 - v
Xi2 = chisquare(ni, nd, ddof=df)
X2 = Xi2[0]
pv = Xi2[1]
Xc2 = chi2.ppf(1-alpha, df)

if Xc2 > X2:
    print('X2 =', round(X2,2), '> Xc2 =', round(Xc2,2), '/ Accept H0')
else:
    print('X2 =', round(X2,2), '< Xc2 =', round(Xc2,2), '/ Reject H0')

#results X2
res_val = np.transpose([lim_inf, lim_sup, ni, fi, Fi, Fd_sup, fd, nd, xi2])
col_name = ['Linf','Lsup','ni','fi','Fi','Fd','fd','nd','Xi2']
res_id = np.arange(inter)+1
res = pd.DataFrame(res_val, res_id, columns=col_name)
res.to_csv('results_X2.csv')
