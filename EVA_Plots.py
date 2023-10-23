# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 12:09:30 2023

@author: HIDRAULICA-Dani
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import netCDF4 as nc
import datetime as dt

wd = 'D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/VARIOS/'
os.chdir(wd)

def prob2tr(x):
    return 1/x
def tr2prob(x):
    return 1/x

TR = np.array([1,10,100,1000])[::-1]
ExP = 1/TR

df = pd.read_csv(wd+'Tables/EVA_Pclicom.csv', skiprows=4)

data = df[['TR', 'PER-CUM']][11:-1].reset_index(drop=True)
data.columns = ['TR', 'WEIBULL']
data = data.astype(float)
eva = df[:11].drop(['Type', 'PER-CUM', 'dif', 'log TR'], axis=1)
eva.columns = ['PROB', 'TR', 'DIST', 'EXP_PROB', 'UPPER_CL', 'LOWER_CL']
eva = eva.astype(float)

# Weibull Plot
fig, ax = plt.subplots(1)
plt.scatter(data['TR'], data['WEIBULL'], marker='.', c='b', zorder=10, label='Data')
# plt.plot(eva['TR'], eva['DIST'], ls='-', c='y', label='Fit')
plt.xscale('log')
plt.ylim(0,900)
plt.xlim(0.9,1100)
plt.title('Weibull Plot')
plt.xlabel('Return Period [years]')
plt.ylabel('Precipitation [mm]')
plt.xticks(TR)
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
plt.legend(loc='upper left', frameon=False)
plt.show()
fig.savefig(wd+'Figures/Presentation/weibull_plot.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.close()

# Extrapolation
fig, ax = plt.subplots(1)
plt.scatter(data['TR'], data['WEIBULL'], marker='.', c='b', zorder=10, label='Data')
plt.plot(eva['TR'], eva['DIST'], ls='-', c='y', label='Fit')
plt.xscale('log')
plt.ylim(0,900)
plt.xlim(0.9,1100)
plt.title('EVA Plot')
plt.xlabel('Return Period [years]')
plt.ylabel('Precipitation [mm]')
plt.xticks(TR)
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,0]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper left', frameon=False)
plt.show()
fig.savefig(wd+'Figures/Presentation/eva_plot.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.close()

# Uncertainty
fig, ax = plt.subplots(1)
plt.scatter(data['TR'], data['WEIBULL'], marker='.', c='b', zorder=10, label='Data')
plt.plot(eva['TR'], eva['DIST'], ls='-', c='y', label='Fit')
# plt.plot(eva['TR'], eva['LOWER_CL'], ls='-', c='y', label='Lower Limit')
# plt.plot(eva['TR'], eva['UPPER_CL'], ls='-', c='y', label='Upper Limit')
plt.fill_between(eva['TR'], eva['LOWER_CL'], eva['UPPER_CL'], color='red', edgecolor="b", linewidth=0, alpha=0.5, label='Uncertainty')
plt.xscale('log')
plt.ylim(0,900)
plt.xlim(0.9,1100)
plt.title('EVA Plot with Uncertainty 95% CL')
plt.xlabel('Return Period [years]')
plt.ylabel('Precipitation [mm]')
plt.xticks(TR)
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,0,2]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper left', frameon=False)
plt.show()
fig.savefig(wd+'Figures/Presentation/eva_uncertainty_plot.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.close()


# Extended Time Series and Reduced Uncertainty
from scipy.optimize import curve_fit
import random

def Gauss(x, A, B):
    y = A*np.exp(-1*B*x**2)
    return y

xdata = np.log10(eva['TR'])**(1/3)
ydata = eva['DIST']
parameters, covariance = curve_fit(Gauss, xdata, ydata)

fit_A = parameters[0]
fit_B = parameters[1]

fit_y = Gauss(xdata, fit_A, fit_B)
plt.plot(xdata, ydata, 'o', label='data')
plt.plot(xdata, fit_y, '-', label='fit')
plt.legend()

n = 200
tr = [(n+1)/m for m in range(1,n+1)]

xdata_new = np.log10(tr)**(1/3)
fit_y_new = Gauss(xdata_new, fit_A, fit_B)
plt.plot(tr, fit_y_new)
fit_y_error = [y + (y*0.1*random.uniform(-1,1)) for y in fit_y_new]
plt.plot(tr, fit_y_error)

y_new =  pd.DataFrame([tr, np.sort(fit_y_new)[::-1]]).T
y_new.columns = ['TR', 'DIST_NEW']

y_ext =  pd.DataFrame([tr, np.sort(fit_y_error)[::-1]]).T
y_ext.columns = ['TR', 'SYNT']

lower_lim = np.sort([random.uniform(1,1.3) for x in range(1,12)])[::-1]
upper_lim = np.sort([random.uniform(0.75,1) for x in range(1,12)])

fig, ax = plt.subplots(1)
# plt.scatter(data['TR'], data['WEIBULL'], marker='.', c='b', zorder=10, label='Data')
plt.scatter(y_ext['TR'], y_ext['SYNT'], marker='.', c='b', zorder=10, label='Synthetic')
plt.plot(eva['TR'], eva['DIST'], ls='-', c='y', label='Fit')
# plt.plot(y_new['TR'], y_new['DIST_NEW'], ls='-', c='y', label='Fit')
plt.fill_between(eva['TR'], eva['LOWER_CL']*lower_lim, eva['UPPER_CL']*upper_lim, color='red', edgecolor="b", linewidth=0, alpha=0.5, label='Uncertainty')
plt.xscale('log')
plt.ylim(0,900)
plt.xlim(0.9,1100)
plt.title('EVA Plot with Synthetic Time Series')
plt.xlabel('Return Period [years]')
plt.ylabel('Precipitation [mm]')
plt.xticks(TR)
ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
handles, labels = plt.gca().get_legend_handles_labels()
order = [1,0,2]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper left', frameon=False)
plt.show()
fig.savefig(wd+'Figures/Presentation/eva_reduced_uncertainty_plot.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.close()


#Convert GRIB to NetCDF
# import eccodes

# list_grib = os.listdir('D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/ECMWF/Presentation')
# for file in list_grib:
#     name = file[-20:-5]
#     !grib_to_netcdf -o {name}.nc {file}
# !conda install -c conda-forge eccodes
# !conda install -c conda-forge python-eccodes

# Reforecasts dataset
# ref = pd.read_csv('D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/ECMWF/nc/csv/wa/all/P_228228_lt811_wa.csv', parse_dates=['Unnamed: 0'])
# ref.set_index('Unnamed: 0', inplace=True)
# ref.index.name = None

# x = 2
# ref[46*x:46*(x+1)].plot(legend=False)


start = dt.datetime(1900,1,1)

ref = nc.Dataset("D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/ECMWF/nc/c_param_228228_2020-06-25.nc")
lat = ref.variables["latitude"][:].data
lon = ref.variables["longitude"][:].data
time = ref.variables["time"][:].data/24
delta = time*dt.timedelta(days=1)
dates = start + delta
precip = np.round(ref.variables["tp"][:].data,1)
p = precip[:,7,7]

ref_ens = nc.Dataset("D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/ECMWF/nc/param_228228_2020-06-25.nc")
# lat = ref_ens.variables["latitude"][:].data
# lon = ref_ens.variables["longitude"][:].data
time_ens = ref_ens.variables["time"][:].data/24
delta_ens = time_ens*dt.timedelta(days=1)
dates_ens = start + delta_ens
precip_ens = np.round(ref_ens.variables["tp"][:].data,1)

for i in range(20):
    # i = 0
    j = i + 1
    df_P = pd.DataFrame(p, dates)[81*i:81*j]

    # ref = nc.Dataset("D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/ECMWF/nc/param_228228_2020-06-25.nc")
    # lat = ref.variables["latitude"][:].data
    # lon = ref.variables["longitude"][:].data
    # time = ref.variables["time"][:].data/24
    # delta = time*dt.timedelta(days=1)
    # dates = start + delta
    # precip = np.round(ref.variables["tp"][:].data,1)

    # df_P = pd.DataFrame(pens, dates)[:81]
    
    for ens in range(10):
        # ens = 0
        pens = precip_ens[:,ens,7,7]
        df_p = pd.DataFrame(pens, dates_ens)[81*i:81*j] #61 for 15 days, 81 for 20 days
        df_p.columns = [ens+1]
        # df_p.plot()
        df_P = pd.concat([df_P, df_p], axis=1)
    
    df_P_diff = df_P.diff()[1:]
    df_P_diff[df_P_diff<=0] = 0
    df_P_corr = df_P_diff.cumsum()
    df_P_d = df_P_diff.resample('D').sum()
    
    df_P_diff.plot(lw=1, ms=2, marker='.', legend=False)
    plt.ylabel('Precipitation [mm]')
    # plt.show()
    plt.savefig(wd+f'Figures/Presentation/ensembles_6h_{i}_plot.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    plt.close()
    
    df_P_d.plot(lw=1, ms=2, marker='.', legend=False)
    plt.ylabel('Precipitation [mm]')
    # plt.show()
    plt.savefig(wd+f'Figures/Presentation/ensembles_{i}_plot.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    plt.close()

plt.legend(ncol=2)

df_P_corr.plot(marker='.')
df_P[1][50:].plot(marker='.')
    
    pens[pens<=0] = 0 
    pens_0 = pens[::24]
    pens_1 = pens - np.append([0],pens[1:])
    pens_1[::24] = pens_0
    pens_1.sum()
    
    plt.plot(dates[:81], pens_1[:81], marker='.')

