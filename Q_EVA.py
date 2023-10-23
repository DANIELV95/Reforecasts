# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 17:03:41 2022

@author: HIDRAULICA-Dani
"""

#EVA for precipitation data comparing observed and ERA5

import os
import pandas as pd
import numpy as np
import spotpy
import csv
# import pyextremes
from pyextremes import EVA
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import matplotlib.lines as mlines
import gc

os.chdir("D:/DANI/2021/TEMA4_PRONOSTICOS/DATOS/")
os.listdir()

#Variables
Tr = [2,5,10,20,50,100,200,1000] #25,300,500,10000]
Q_ds = ['Qobs', 'Qpobs', 'ERA5', 'CLICOM']

#Datasets discharge
Qobs = pd.read_csv('../PYR/HMS/Results/csv/Q_mean.csv', parse_dates=['Unnamed: 0'])
Qobs.index = Qobs['Unnamed: 0']
Qobs = Qobs.drop(['Unnamed: 0'], axis=1)
Qobs.index.name = None
Qobs.columns = ['Qobs']
Qobs_m = Qobs.resample('Y').max()
Qobs

Qera = pd.read_csv('../PYR/HMS/Results/csv/Q_ERA5.csv', parse_dates=['Unnamed: 0'])
Qera.index = Qera['Unnamed: 0']
Qera = Qera.drop(['Unnamed: 0'], axis=1)
Qera.index.name = None
Qera_m = Qera.resample('Y').max()
Qera

Qclic = pd.read_csv('../PYR/HMS/Results/csv/Q_CLICOM.csv', parse_dates=['Unnamed: 0'])
Qclic.index = Qclic['Unnamed: 0']
Qclic = Qclic.drop(['Unnamed: 0'], axis=1)
Qclic.index.name = None
Qclic_m = Qclic.resample('Y').max()[:-5]
Qclic

Qpobs = pd.read_csv('../PYR/HMS/Results/csv/Q_Pobs.csv', parse_dates=['Unnamed: 0'])
Qpobs.index = Qpobs['Unnamed: 0']
Qpobs = Qpobs.drop(['Unnamed: 0'], axis=1)
Qpobs.index.name = None
Qpobs_m = Qpobs.resample('Y').max()
Qpobs


Q_datasets = {'Qobs':Qobs_m, 'Qpobs':Qpobs_m, 'ERA5':Qera_m, 'CLICOM':Qclic_m}

#Maximos de gasto anual
plt.plot(Qobs_m.index.year, Qobs_m, label='Qobs', alpha=0.7, color='k')
plt.plot(Qpobs_m.index.year, Qpobs_m, label='Qpobs', alpha=0.7, color='r')
plt.plot(Qclic_m.index.year, Qclic_m, label='CLICOM', alpha=0.7, color='y')
plt.plot(Qera_m.index.year, Qera_m, label='ERA5', alpha=0.7, color='b')
plt.legend()
plt.title('Gasto máximo anual')
plt.ylabel('Gasto [$m^3$/s]')
plt.savefig('../SSP/Discharge_max.jpg', format='jpg', dpi=1000)
plt.close()


#Chi squared test
#Load Libraries
from scipy.stats import describe, lognorm, pearson3, chi2, chisquare, skew, kurtosis #mode,
from scipy.stats.mstats import hdmedian

#Read data and transform to Ln
for ds in Q_ds:
    print(ds)
    # ds = 'Qobs'
    data = Q_datasets[ds].values.flatten()
    data.sort()
    data_ln = np.log(data)
    
    #Estimate parameters
    describe(data)
    n = len(data)
    vmin = data.min()
    vmax = data.max()
    mu = data.mean()
    sx = data.std(ddof=1)
    sk = skew(data)
    kr = kurtosis(data)
    med = hdmedian(data)
    # mode = mode(data)
    mu_ln = data_ln.mean()
    sx_ln = data_ln.std(ddof=1)
    sc_ln = np.exp(mu_ln)
    
    #Set bins intervals
    ps = 5
    f = 0.7
    lmin = round(vmin/ps)*ps
    lmax = round(vmax/ps*f)*ps #can be changed
    # lmax = 120
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
    dist_ln = lognorm(s=sx_ln, loc=0, scale=sc_ln)
    dist_lp3 = pearson3(skew=sk, loc=mu, scale=sx)
    
    distribs = {'LN':dist_ln, 'LP3':dist_lp3}
    
    dst = 'LP3'
    dst = 'LN'
    Fd_sup = distribs[dst].cdf(lim_sup)
    Fd_sup[-1] = 1
    Fd_inf = dist_ln.cdf(lim_inf)
    Fd_inf[0] = 0
    fd = Fd_sup - Fd_inf
    nd = fd*n
    
    x = np.linspace(20,220,100)
    fitted_data = distribs[dst].pdf(x)
    plt.hist(data, density=True)
    plt.plot(x,fitted_data,'r-')
    
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
        conc = 'X2 > Xc2'+'\n'+'Se acepta H0'
        print('X2 =', round(X2,2), '> Xc2 =', round(Xc2,2), '/ Accept H0')
    else:
        conc = 'X2 < Xc2'+'\n'+'Se rechaza H0'
        print('X2 =', round(X2,2), '< Xc2 =', round(Xc2,2), '/ Reject H0')
    
    #results X2
    res_val = np.transpose([lim_inf, lim_sup, ni, fi, Fi, Fd_sup, fd, nd, xi2])
    col_name = ['Linf','Lsup','ni','fi','Fi','Fd','fd','nd','Xi2']
    res_id = np.arange(inter)+1
    res = pd.DataFrame(res_val, res_id, columns=col_name)
    res.to_csv('../SSP/chi2/Q_results_X2_'+ds+'.csv')
    
    
    #histogram and cumulative frequencies plots
    loc_bar = lim_inf + step/2
    b_width = step/3
    t_labels = loc_bar[:-1].astype(int).tolist()
    t_labels.append('>'+str(loc_bar[-1].astype(int)))
    
    plt.bar(loc_bar-b_width/2, ni, width=b_width, label='Muestra')
    plt.bar(loc_bar+b_width/2, nd, width=b_width, label='Ajuste')
    plt.title('Histrograma, '+ds)
    plt.ylabel('Frecuencia')
    plt.xlabel('Precipitación [mm]')
    plt.xticks(loc_bar, labels=t_labels)
    plt.legend()
    plt.text(loc_bar[-3]+b_width, max(max(ni),max(nd))*0.7,
             'X2='+str(round(X2,2))+'\n'
             'Xc2='+str(round(Xc2,2))+'\n'+conc,
             ha='left', va='center')
    plt.savefig('../SSP/chi2/Q_histogram_'+ds+'.jpg', format='jpg', dpi=1000)
    plt.close()
    
    plt.bar(loc_bar, Fi, width=b_width*2, label='Muestra') #, ec='k', fill=False)
    plt.plot(loc_bar, Fd_sup, label='Dist. Ln', c='r')
    plt.title('Distribución y polígono de frecuencia acumulada, '+ds)
    plt.ylabel('Frecuencia acumulada')
    plt.xlabel('Precipitación [mm]')
    plt.xticks(loc_bar, labels=t_labels)
    plt.legend()
    plt.savefig('../SSP/chi2/Q_freqcum_'+ds+'.jpg', format='jpg', dpi=1000)
    plt.close()

################################################################################
from scipy.special import gamma

ds = 'Qobs'
data = Q_datasets[ds].values.flatten()
data.sort()
data_ln = np.log(data)

#Estimate parameters
describe(data)
n = len(data)
vmin = data.min()
vmax = data.max()
mu = data.mean()
sx = data.std(ddof=1)
sc = data.std()
sk = skew(data)
kr = kurtosis(data)
med = hdmedian(data)
# mode = mode(data)
mu_ln = data_ln.mean()
sx_ln = data_ln.std(ddof=1)
sc_ln = np.exp(mu_ln)
sk_ln = skew(data_ln)

# x = np.linspace(10,220,10)
# x_ln = np.log(x)
x = data
x_ln = np.log(data)

miu = 0
s = 1
beta_p3 = (2/sk)**2
lambda_p3 = s/beta_p3**0.5
epsilon_p3 = miu - s*beta_p3**0.5
f_p3 = lambda_p3**beta_p3 * ((x - epsilon_p3)**(beta_p3-1)) * np.exp(-lambda_p3*(x - epsilon_p3)) / gamma(beta_p3)


beta = 2/sk
alpha = beta**2
lamb = -beta
f_p3 = abs(beta) * ((beta*(x-lamb))**(alpha-1)) * np.exp(-beta*(x-lamb)) / gamma(alpha)
pearson3.pdf(x, skew=sk, loc=0, scale=1)


beta = 2/sk_ln
alpha = beta**2
lamb = -beta
f_lp3 = abs(beta) * ((beta*(x_ln-lamb))**(alpha-1)) * np.exp(-beta*(x_ln-lamb)) / gamma(alpha) / x


f_p3 = abs(beta) * ((beta*(x-mu-lamb))**(alpha-1)) * np.exp(-beta*(x-mu-lamb)) / gamma(alpha) / sc

pearson3.pdf(x, skew=sk, loc=mu, scale=sc)
y = (x - mu)/sc
pearson3.pdf(y, skew=sk) / sc


################################################################################

import scipy.stats
import scipy.stats.distributions as dists

data = np.random.normal(0, 0.5, 1000)

dist_lp3.pdf(data)

data = Q_datasets[ds].values.flatten()
data.sort()
data_ln = np.log(data)

s, loc, scale = lognorm.fit(data)
x = np.linspace(0,200,100)
fitted_data = lognorm.pdf(x, s=s, loc=loc, scale=scale) #s=sigma, scale=exp(mean), loc=0
plt.hist(data, density=True)
plt.plot(x,fitted_data,'r-')

np.exp(data.mean())
data_ln.std()

help(plt.hist)

loc, scale = dists.norm.fit(data)
x = np.linspace(0,200,100)
fitted_data = dists.norm.pdf(x, loc, scale) #loc=mean, scale=var
plt.hist(data, density=True)
plt.plot(x,fitted_data,'r-')

data.mean()
data.std()


loc, scale = dists.norm.fit(data_ln)
x = np.linspace(2,6,100)
fitted_data = dists.norm.pdf(x, loc, scale) #loc=mean, scale=var
plt.hist(data_ln, density=True)
plt.plot(x,fitted_data,'r-')

data_ln.mean()
data_ln.std()


skew, loc, scale = pearson3.fit(data)





data = np. random. normal(0, 0.5, 1000)
mean, var = scipy. stats. distributions. norm. fit(data)
x = np. linspace(-5,5,100)
fitted_data = scipy. stats. distributions. norm. pdf(x, mean, var)
plt.hist(data, density=True)
plt.plot(x,fitted_data,'r-')










#Perform EVA

obs = Pobs['P']
sim = Pera['P']
sim2 = Pclic['P']
df = pd.concat([sim2, sim, obs], axis=1)
data = ['CLICOM', 'ERA5', 'Observed']
df.columns = data
# df = df.dropna()

#Observed data eva
eva_obs = EVA(df['Observed'])
eva_obs.get_extremes(method='BM', block_size='365.2425D', errors='ignore')
eva_obs.fit_model() #distribution = 'gumbel_r') #forced to follow gumbel distribution
summary_obs = eva_obs.get_summary(return_period=Tr, alpha=0.95, n_samples=1000)
Tr_res = summary_obs.drop(['upper ci', 'lower ci'], axis=1)
Tr_res.columns = ['Pobs']
#         print(eva_obs)

eva_obs.plot_extremes()
plt.title('Extreme Value Analysis - Block Maxima, Observed')
plt.xlabel('Year')
plt.ylabel('Precipitation [mm/day]')
name_plot='./VARIOS/Figures/EVA/fig_EVA_BM_Observed.jpg'
plt.savefig(name_plot, format='jpg', dpi=1000)
plt.close()

fig, ax = eva_obs.plot_diagnostic(return_period=Tr, alpha=0.95)
name_plot='./VARIOS/Figures/EVA/fig_EVA_diag_Observed.jpg'
plt.savefig(name_plot, format='jpg', dpi=1000)
extent = ax[0].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
name_plot='./VARIOS/Figures/EVA/sp1_fig_EVA_diag_Observed.jpg'
plt.savefig(name_plot, format='jpg', dpi=1000, bbox_inches=extent.expanded(1.05,1.05))
plt.close()

f = open('./VARIOS/Figures/EVA/eva_obs.txt', 'w')
print(eva_obs, file=f)
f.close()

#Simulated data eva

eva = EVA(df['ERA5'])
eva.get_extremes(method='BM', block_size='365.2425D', errors='ignore')
eva.fit_model(distribution = 'gumbel_r')
summary = eva.get_summary(return_period=Tr, alpha=0.95, n_samples=1000)

eva.plot_extremes()
plt.title('Extreme Value Analysis - Block Maxima, ERA5')
plt.xlabel('Year')
plt.ylabel('Precipitation [mm/day]')
name_plot='./VARIOS/Figures/EVA/fig_EVA_BM_ERA5.jpg'
plt.savefig(name_plot, format='jpg', dpi=1000)
plt.close()

fig, ax = eva.plot_diagnostic(return_period=Tr, alpha=0.95)
ax[2].set_ylabel('ERA5')
ax[3].set_ylabel('ERA5')
ax[0].plot(summary_obs.index,summary_obs['return value'].values, color='y', ls='dotted', alpha=0.8)
sim_lg = mlines.Line2D([], [], label='ERA5', color='r')
obs_lg = mlines.Line2D([], [], label='Pobs', color='y', ls='dotted', alpha=0.8)
ax[0].legend(handles=[sim_lg, obs_lg])
name_plot='./VARIOS/Figures/EVA/fig_EVA_diag_ERA5.jpg'
plt.savefig(name_plot, format='jpg', dpi=1000)
extent = ax[0].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
name_plot='./VARIOS/Figures/EVA/sp1_fig_EVA_diag_ERA5.jpg'
plt.savefig(name_plot, format='jpg', dpi=1000, bbox_inches=extent.expanded(1.05,1.05))
plt.close()

f = open('./VARIOS/Figures/EVA/eva_ERA5.txt', 'w')
print(eva, file=f)
f.close()

Tr_res_sim = summary.drop(['upper ci', 'lower ci'], axis=1)
Tr_res_sim.columns = ['ERA5']
Tr_res = pd.concat([Tr_res,Tr_res_sim], axis=1)
f = open('./VARIOS/Figures/EVA/Tr_res_ERA5.txt', 'w')
print(Tr_res, file=f)
f.close()

#Simulated2 data eva

eva = EVA(df['CLICOM'])
eva.get_extremes(method='BM', block_size='365.2425D', errors='ignore')
eva.fit_model() #distribution = 'gumbel_r')
summary = eva.get_summary(return_period=Tr, alpha=0.95, n_samples=1000)

eva.plot_extremes()
plt.title('Extreme Value Analysis - Block Maxima, CLICOM')
plt.xlabel('Year')
plt.ylabel('Precipitation [mm/day]')
name_plot='./VARIOS/Figures/EVA/fig_EVA_BM_ERA5.jpg'
plt.savefig(name_plot, format='jpg', dpi=1000)
plt.close()

fig, ax = eva.plot_diagnostic(return_period=Tr, alpha=0.95)
ax[2].set_ylabel('ERA5')
ax[3].set_ylabel('ERA5')
ax[0].plot(summary_obs.index,summary_obs['return value'].values, color='y', ls='dotted', alpha=0.8)
sim_lg = mlines.Line2D([], [], label='ERA5', color='r')
obs_lg = mlines.Line2D([], [], label='Pobs', color='y', ls='dotted', alpha=0.8)
ax[0].legend(handles=[sim_lg, obs_lg])
name_plot='./VARIOS/Figures/EVA/fig_EVA_diag_ERA5.jpg'
plt.savefig(name_plot, format='jpg', dpi=1000)
extent = ax[0].get_tightbbox(fig.canvas.renderer).transformed(fig.dpi_scale_trans.inverted())
name_plot='./VARIOS/Figures/EVA/sp1_fig_EVA_diag_ERA5.jpg'
plt.savefig(name_plot, format='jpg', dpi=1000, bbox_inches=extent.expanded(1.05,1.05))
plt.close()

f = open('./VARIOS/Figures/EVA/eva_ERA5.txt', 'w')
print(eva, file=f)
f.close()

Tr_res_sim = summary.drop(['upper ci', 'lower ci'], axis=1)
Tr_res_sim.columns = ['ERA5']
Tr_res = pd.concat([Tr_res,Tr_res_sim], axis=1)
f = open('./VARIOS/Figures/EVA/Tr_res_ERA5.txt', 'w')
print(Tr_res, file=f)
f.close()
