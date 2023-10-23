# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 17:34:17 2022

@author: HIDRAULICA-Dani
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

os.chdir('D:/DANI/2021/TEMA4_PRONOSTICOS/SSP')
files = os.listdir('./results')
files = files[:-1]

Tr = [2,5,10,20,50,100,200,500,1000]
Pr = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1]
slt = [*range(2,17,3)]

def prob2tr(x):
    return 1/x
def tr2prob(x):
    return 1/x

#Spanish images
for file in files:
    # file = files[5]
    df = pd.read_csv('./results/'+file, skiprows=5, names=['id', 'prob', 'Data', 'Value', 'Exp Prob', 'lci', 'uci'])
    df['Tr'] = round(1/(df['prob']/100),2)
    data = pd.DataFrame([df['prob']/100, df['Tr'], df['Data'].replace(' ', np.nan).astype(float)]) #, columns=['prob', 'Tr', 'Data'])
    data = data.transpose().dropna().reset_index(drop=True)
    data = data.sort_values(by='Data', ascending=False)
    eva = df.drop(['Data', 'id'], axis=1)
    eva = eva.replace(' ', np.nan).dropna().reset_index(drop=True).astype(float)
    eva['prob'] = eva['prob']/100
    
    if file[0] == 'P':
        df_obs = pd.read_csv('./results/Pclicom.csv', skiprows=5, names=['id', 'prob', 'Data', 'Value', 'Exp Prob', 'lci', 'uci'])
        df_obs['Tr'] = round(1/(df_obs['prob']/100),2)
        data_obs = pd.DataFrame([df_obs['prob']/100, df_obs['Tr'], df_obs['Data'].replace(' ', np.nan).astype(float)]) #, columns=['prob', 'Tr', 'Data'])
        data_obs = data_obs.transpose().dropna().reset_index(drop=True)
        data_obs = data_obs.sort_values(by='Data', ascending=False)
        eva_obs = df_obs.drop(['Data', 'id'], axis=1)
        eva_obs = eva_obs.replace(' ', np.nan).dropna().reset_index(drop=True).astype(float)
        eva_obs['prob'] = eva_obs['prob']/100
        
        if file[:4]=='Pref':
            prep = 'Prep'+(file[-8:-6]+'-'+file[-6:-4]).replace('0', '')
        else:
            prep = file[:-4]
        
        fig, ax = plt.subplots(1)
        plt.scatter(data_obs['prob'], data_obs['Data'], marker='.', c='c', zorder=10, label='Datos Pclicom')
        plt.plot(eva_obs['prob'], eva_obs['Value'], ls='-', c='y', label='Ajuste Pclicom')
        plt.plot(eva_obs['prob'], eva_obs['lci'], ls='--', c='y', alpha=0.3)
        plt.plot(eva_obs['prob'], eva_obs['uci'], ls='--', c='y', alpha=0.3)
        plt.fill(np.append(eva_obs['prob'], eva_obs['prob'][::-1]), np.append(eva_obs['lci'], eva_obs['uci'][::-1]), c='y', alpha=0.2)
        plt.scatter(data['prob'], data['Data'], marker=',', s=3, c='b', zorder=9, label='Datos '+prep)
        plt.plot(eva['prob'], eva['Value'], ls='-', c='g', label='Ajuste '+prep)
        plt.plot(eva['prob'], eva['lci'], ls='--', c='g', alpha=0.3)
        plt.plot(eva['prob'], eva['uci'], ls='--', c='g', alpha=0.3)
        plt.fill(np.append(eva['prob'], eva['prob'][::-1]), np.append(eva['lci'], eva['uci'][::-1]), c='g', alpha=0.2)
        plt.title('Comparación de periodos de retorno, Pclicom vs '+prep)
        plt.xlabel('Probabilidad')
        plt.ylabel('Precipitación [mm]')
        ax.invert_xaxis()
        plt.xscale('log')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        plt.xticks(Pr)
        sax = ax.secondary_xaxis(location='top', functions=(prob2tr, tr2prob))
        sax.set_xlabel('Periodo de retorno [años]')
        sax.get_xaxis().set_major_formatter(FormatStrFormatter('%g'))
        sax.set_xticks(Tr)
        plt.legend()
        fig.savefig('./eva/comparison_Pclicom_vs_'+prep+'.jpg', format='jpg', dpi=1000, bbox_inches='tight')
        
        
    else:
        df_obs = pd.read_csv('./results/Qbandas.csv', skiprows=5, names=['id', 'prob', 'Data', 'Value', 'Exp Prob', 'lci', 'uci'])
        df_obs['Tr'] = round(1/(df_obs['prob']/100),2)
        data_obs = pd.DataFrame([df_obs['prob']/100, df_obs['Tr'], df_obs['Data'].replace(' ', np.nan).astype(float)]) #, columns=['prob', 'Tr', 'Data'])
        data_obs = data_obs.transpose().dropna().reset_index(drop=True)
        data_obs = data_obs.sort_values(by='Data', ascending=False)
        eva_obs = df_obs.drop(['Data', 'id'], axis=1)
        eva_obs = eva_obs.replace(' ', np.nan).dropna().reset_index(drop=True).astype(float)
        eva_obs['prob'] = eva_obs['prob']/100
        
        if file[:4]=='Qref':
            qrep = 'Qrep'+(file[-8:-6]+'-'+file[-6:-4]).replace('0', '')
        else:
            qrep = file[:-4]
        
        fig, ax = plt.subplots(1)
        plt.scatter(data_obs['prob'], data_obs['Data'], marker='.', c='c', zorder=10, label='Datos Qobs')
        plt.plot(eva_obs['prob'], eva_obs['Value'], ls='-', c='y', label='Ajuste Qobs')
        plt.plot(eva_obs['prob'], eva_obs['lci'], ls='--', c='y', alpha=0.3)
        plt.plot(eva_obs['prob'], eva_obs['uci'], ls='--', c='y', alpha=0.3)
        plt.fill(np.append(eva_obs['prob'], eva_obs['prob'][::-1]), np.append(eva_obs['lci'], eva_obs['uci'][::-1]), c='y', alpha=0.2)
        plt.scatter(data['prob'], data['Data'], marker=',', s=3, c='b', zorder=9, label='Datos '+qrep)
        plt.plot(eva['prob'], eva['Value'], ls='-', c='g', label='Ajuste '+qrep)
        plt.plot(eva['prob'], eva['lci'], ls='--', c='g', alpha=0.3)
        plt.plot(eva['prob'], eva['uci'], ls='--', c='g', alpha=0.3)
        plt.fill(np.append(eva['prob'], eva['prob'][::-1]), np.append(eva['lci'], eva['uci'][::-1]), c='g', alpha=0.2)
        plt.title('Comparación de periodos de retorno, Qobs vs '+qrep)
        plt.xlabel('Probabilidad')
        plt.ylabel('Gasto [$m^3$/s]')
        plt.xticks(Pr)
        ax.invert_xaxis()
        plt.xscale('log')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        sax = ax.secondary_xaxis(location='top', functions=(prob2tr, tr2prob))
        sax.set_xlabel('Periodo de retorno [años]')
        sax.get_xaxis().set_major_formatter(FormatStrFormatter('%g'))
        sax.set_xticks(Tr)
        plt.legend()
        fig.savefig('./eva/comparison_QOBS_vs_'+qrep+'.jpg', format='jpg', dpi=1000, bbox_inches='tight')


#English images
for file in files:
    # file = files[5]
    df = pd.read_csv('./results/'+file, skiprows=5, names=['id', 'prob', 'Data', 'Value', 'Exp Prob', 'lci', 'uci'])
    df['Tr'] = round(1/(df['prob']/100),2)
    data = pd.DataFrame([df['prob']/100, df['Tr'], df['Data'].replace(' ', np.nan).astype(float)]) #, columns=['prob', 'Tr', 'Data'])
    data = data.transpose().dropna().reset_index(drop=True)
    data = data.sort_values(by='Data', ascending=False)
    eva = df.drop(['Data', 'id'], axis=1)
    eva = eva.replace(' ', np.nan).dropna().reset_index(drop=True).astype(float)
    eva['prob'] = eva['prob']/100
    
    if file[0] == 'P':
        df_obs = pd.read_csv('./results/Pclicom.csv', skiprows=5, names=['id', 'prob', 'Data', 'Value', 'Exp Prob', 'lci', 'uci'])
        df_obs['Tr'] = round(1/(df_obs['prob']/100),2)
        data_obs = pd.DataFrame([df_obs['prob']/100, df_obs['Tr'], df_obs['Data'].replace(' ', np.nan).astype(float)]) #, columns=['prob', 'Tr', 'Data'])
        data_obs = data_obs.transpose().dropna().reset_index(drop=True)
        data_obs = data_obs.sort_values(by='Data', ascending=False)
        eva_obs = df_obs.drop(['Data', 'id'], axis=1)
        eva_obs = eva_obs.replace(' ', np.nan).dropna().reset_index(drop=True).astype(float)
        eva_obs['prob'] = eva_obs['prob']/100
        
        if file[:4]=='Pref':
            prep = 'Pref'+(file[-8:-6]+'-'+file[-6:-4]).replace('0', '')
        else:
            prep = file[:-4]
        
        fig, ax = plt.subplots(1)
        plt.scatter(data_obs['prob'], data_obs['Data'], marker='.', c='c', zorder=10, label='Data Pclicom')
        plt.plot(eva_obs['prob'], eva_obs['Value'], ls='-', c='y', label='Fit Pclicom')
        plt.plot(eva_obs['prob'], eva_obs['lci'], ls='--', c='y', alpha=0.3)
        plt.plot(eva_obs['prob'], eva_obs['uci'], ls='--', c='y', alpha=0.3)
        plt.fill(np.append(eva_obs['prob'], eva_obs['prob'][::-1]), np.append(eva_obs['lci'], eva_obs['uci'][::-1]), c='y', alpha=0.2)
        plt.scatter(data['prob'], data['Data'], marker=',', s=3, c='b', zorder=9, label='Data '+prep)
        plt.plot(eva['prob'], eva['Value'], ls='-', c='g', label='Fit '+prep)
        plt.plot(eva['prob'], eva['lci'], ls='--', c='g', alpha=0.3)
        plt.plot(eva['prob'], eva['uci'], ls='--', c='g', alpha=0.3)
        plt.fill(np.append(eva['prob'], eva['prob'][::-1]), np.append(eva['lci'], eva['uci'][::-1]), c='g', alpha=0.2)
        plt.title('Return period comparison, Pclicom vs '+prep)
        plt.xlabel('Probability')
        plt.ylabel('Precipitation [mm]')
        ax.invert_xaxis()
        plt.xscale('log')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        plt.xticks(Pr)
        sax = ax.secondary_xaxis(location='top', functions=(prob2tr, tr2prob))
        sax.set_xlabel('Return period [years]')
        sax.get_xaxis().set_major_formatter(FormatStrFormatter('%g'))
        sax.set_xticks(Tr)
        plt.legend()
        fig.savefig('./eva/GOOD_ENG/comparison_Pclicom_vs_'+prep+'_eng.png', format='png', dpi=300, bbox_inches='tight')
        
        
    else:
        df_obs = pd.read_csv('./results/Qbandas.csv', skiprows=5, names=['id', 'prob', 'Data', 'Value', 'Exp Prob', 'lci', 'uci'])
        df_obs['Tr'] = round(1/(df_obs['prob']/100),2)
        data_obs = pd.DataFrame([df_obs['prob']/100, df_obs['Tr'], df_obs['Data'].replace(' ', np.nan).astype(float)]) #, columns=['prob', 'Tr', 'Data'])
        data_obs = data_obs.transpose().dropna().reset_index(drop=True)
        data_obs = data_obs.sort_values(by='Data', ascending=False)
        eva_obs = df_obs.drop(['Data', 'id'], axis=1)
        eva_obs = eva_obs.replace(' ', np.nan).dropna().reset_index(drop=True).astype(float)
        eva_obs['prob'] = eva_obs['prob']/100
        
        if file[:4]=='Qref':
            qrep = 'Qref'+(file[-8:-6]+'-'+file[-6:-4]).replace('0', '')
        else:
            qrep = file[:-4]
        
        fig, ax = plt.subplots(1)
        plt.scatter(data_obs['prob'], data_obs['Data'], marker='.', c='c', zorder=10, label='Data Qobs')
        plt.plot(eva_obs['prob'], eva_obs['Value'], ls='-', c='y', label='Fit Qobs')
        plt.plot(eva_obs['prob'], eva_obs['lci'], ls='--', c='y', alpha=0.3)
        plt.plot(eva_obs['prob'], eva_obs['uci'], ls='--', c='y', alpha=0.3)
        plt.fill(np.append(eva_obs['prob'], eva_obs['prob'][::-1]), np.append(eva_obs['lci'], eva_obs['uci'][::-1]), c='y', alpha=0.2)
        plt.scatter(data['prob'], data['Data'], marker=',', s=3, c='b', zorder=9, label='Data '+qrep)
        plt.plot(eva['prob'], eva['Value'], ls='-', c='g', label='Fit '+qrep)
        plt.plot(eva['prob'], eva['lci'], ls='--', c='g', alpha=0.3)
        plt.plot(eva['prob'], eva['uci'], ls='--', c='g', alpha=0.3)
        plt.fill(np.append(eva['prob'], eva['prob'][::-1]), np.append(eva['lci'], eva['uci'][::-1]), c='g', alpha=0.2)
        plt.title('Return period comparison, Qobs vs '+qrep)
        plt.xlabel('Probability')
        plt.ylabel('Discharge [$m^3$/s]')
        plt.xticks(Pr)
        ax.invert_xaxis()
        plt.xscale('log')
        ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
        sax = ax.secondary_xaxis(location='top', functions=(prob2tr, tr2prob))
        sax.set_xlabel('Return period [years]')
        sax.get_xaxis().set_major_formatter(FormatStrFormatter('%g'))
        sax.set_xticks(Tr)
        plt.legend()
        fig.savefig('./eva/GOOD_ENG/comparison_QOBS_vs_'+qrep+'_eng.png', format='png', dpi=300, bbox_inches='tight')


#English image All in 1 (2,5)
# plt.rcParams['axes.grid'] = True

i = -1
fig, ax = plt.subplots(5,2)
fig.set_size_inches(10.5, 12.5, forward=True)
# fig.suptitle('Return period comparison for different lead times', fontsize=14)
fig.subplots_adjust(top=0.91)
for file in files:
    # file = files[3]
    if i>=4:
        i = -1
    df = pd.read_csv('./results/'+file, skiprows=5, names=['id', 'prob', 'Data', 'Value', 'Exp Prob', 'lci', 'uci'])
    df['Tr'] = round(1/(df['prob']/100),2)
    data = pd.DataFrame([df['prob']/100, df['Tr'], df['Data'].replace(' ', np.nan).astype(float)]) #, columns=['prob', 'Tr', 'Data'])
    data = data.transpose().dropna().reset_index(drop=True)
    data = data.sort_values(by='Data', ascending=False)
    eva = df.drop(['Data', 'id'], axis=1)
    eva = eva.replace(' ', np.nan).dropna().reset_index(drop=True).astype(float)
    eva['prob'] = eva['prob']/100
    
    if file[0] == 'P':
        df_obs = pd.read_csv('./results/Pobs.csv', skiprows=5, names=['id', 'prob', 'Data', 'Value', 'Exp Prob', 'lci', 'uci'])
        # df_obs = pd.read_csv('./results/Pclicom.csv', skiprows=5, names=['id', 'prob', 'Data', 'Value', 'Exp Prob', 'lci', 'uci'])
        df_obs['Tr'] = round(1/(df_obs['prob']/100),2)
        data_obs = pd.DataFrame([df_obs['prob']/100, df_obs['Tr'], df_obs['Data'].replace(' ', np.nan).astype(float)]) #, columns=['prob', 'Tr', 'Data'])
        data_obs = data_obs.transpose().dropna().reset_index(drop=True)
        data_obs = data_obs.sort_values(by='Data', ascending=False)
        eva_obs = df_obs.drop(['Data', 'id'], axis=1)
        eva_obs = eva_obs.replace(' ', np.nan).dropna().reset_index(drop=True).astype(float)
        eva_obs['prob'] = eva_obs['prob']/100
    else:
        df_obs = pd.read_csv('./results/Qbandas.csv', skiprows=5, names=['id', 'prob', 'Data', 'Value', 'Exp Prob', 'lci', 'uci'])
        df_obs['Tr'] = round(1/(df_obs['prob']/100),2)
        data_obs = pd.DataFrame([df_obs['prob']/100, df_obs['Tr'], df_obs['Data'].replace(' ', np.nan).astype(float)]) #, columns=['prob', 'Tr', 'Data'])
        data_obs = data_obs.transpose().dropna().reset_index(drop=True)
        data_obs = data_obs.sort_values(by='Data', ascending=False)
        eva_obs = df_obs.drop(['Data', 'id'], axis=1)
        eva_obs = eva_obs.replace(' ', np.nan).dropna().reset_index(drop=True).astype(float)
        eva_obs['prob'] = eva_obs['prob']/100
        
    if file[:4]=='Pref':
        i += 1
        j = 0
        prep = 'Pref'+(file[-8:-6]+'-'+file[-6:-4]).replace('0', '')
        print(i, prep)
    elif file[:4]=='Qref':
        i += 1
        j = 1
        qrep = 'Qref'+(file[-8:-6]+'-'+file[-6:-4]).replace('0', '')
        print(i, qrep)
    else:
        continue
    
    print(i, j)
    if j == 0:
        obs = 'Pclicom'
        rep = prep
        yticks1 = [*np.arange(0,1200,200)]
        ax[i,j].tick_params(direction='in', which='both', left=True, right=True, labelleft=True, labelright=False, labelbottom=False)
    else:
        ax[i,j].tick_params(direction='in', which='both', left=True, right=True, labelleft=False, labelright=True, labelbottom=False)
        obs = 'Qbandas'
        rep = qrep
        yticks1 = [*np.arange(0,1750,250)]
    
    ax[i,j].scatter(data_obs['prob'], data_obs['Data'], marker=',', s=3, c='c', zorder=6, label='Data '+obs)
    ax[i,j].plot(eva_obs['prob'], eva_obs['Value'], ls='-', c='y', label='Fit '+obs, zorder=4)
    ax[i,j].plot(eva_obs['prob'], eva_obs['lci'], ls='--', c='y', alpha=0.3, zorder=2)
    ax[i,j].plot(eva_obs['prob'], eva_obs['uci'], ls='--', c='y', alpha=0.3, zorder=2)
    ax[i,j].fill(np.append(eva_obs['prob'], eva_obs['prob'][::-1]), np.append(eva_obs['lci'], eva_obs['uci'][::-1]), c='y', alpha=0.2, zorder=2)
    ax[i,j].scatter(data['prob'], data['Data'], marker=',', s=3, c='b', zorder=7, label='Data '+rep)
    ax[i,j].plot(eva['prob'], eva['Value'], ls='-', c='g', label='Fit '+rep, zorder=5)
    ax[i,j].plot(eva['prob'], eva['lci'], ls='--', c='g', alpha=0.3, zorder=3)
    ax[i,j].plot(eva['prob'], eva['uci'], ls='--', c='g', alpha=0.3, zorder=3)
    ax[i,j].fill(np.append(eva['prob'], eva['prob'][::-1]), np.append(eva['lci'], eva['uci'][::-1]), c='g', alpha=0.2, zorder=3)
    # ax[i,j].set_ylabel('Precipitation [mm]')
    sax = ax[i,j].secondary_xaxis(location='top', functions=(prob2tr, tr2prob), zorder=5)
    # sax.set_xlabel('Return period [years]')
    # sax.get_xaxis().set_major_formatter(FormatStrFormatter('%g'))
    # sax.set_xticks(Tr)
    sax.tick_params(direction='in', which='both', labeltop=False)
    # sax.set_visible(False)
    ax[i,j].invert_xaxis()
    ax[i,j].set_xscale('log')
    ax[i,j].xaxis.set_major_formatter(FormatStrFormatter('%g'))
    ax[i,j].set_xticks(Pr)
    ax[i,j].set_ylim(yticks1[0],yticks1[-1])
    ax[i,j].legend(fontsize=8)
    ax[i,j].grid(visible=True, axis='both', zorder=1)

ax[0,0].set_title('Precipitation [mm]')
ax[0,1].set_title('Discharge [$m^3$/s]')
ax[4,0].set_xlabel('Probability')
ax[4,1].set_xlabel('Probability')
ax[4,0].tick_params(direction='in', which='both', labelbottom=True)
ax[4,1].tick_params(direction='in', which='both', labelbottom=True)
ax[4,0].set_xticks(Pr)
ax[4,1].set_xticks(Pr)
sax = ax[0,0].secondary_xaxis(location='top', functions=(prob2tr, tr2prob), zorder=5)
sax.tick_params(direction='in', which='both')
sax.set_xlabel('Return period [years]')
sax.get_xaxis().set_major_formatter(FormatStrFormatter('%g'))
sax.set_xticks(Tr)
sax = ax[0,1].secondary_xaxis(location='top', functions=(prob2tr, tr2prob), zorder=5)
sax.tick_params(direction='in', which='both')
sax.set_xlabel('Return period [years]')
sax.get_xaxis().set_major_formatter(FormatStrFormatter('%g'))
sax.set_xticks(Tr)
plt.subplots_adjust(wspace=0.02, hspace=0.07)
# fig.savefig('./eva/GOOD_ENG/00_AllinOne_comparison_PObs-Q-Obs_vs_P-Q-Ref_eng300.jpg', format='jpg', dpi=300, bbox_inches='tight')
# fig.savefig('./eva/GOOD_ENG/00_AllinOne_comparison_P-Q-Obs_vs_P-Q-Ref_eng300.jpg', format='jpg', dpi=300, bbox_inches='tight')
# fig.savefig('./eva/GOOD_ENG/00_AllinOne_comparison_P-Q-Obs_vs_P-Q-Ref_eng.png', format='png', dpi=300, bbox_inches='tight')

help(sax.tick_params)
help(sax.set_xticks)
help(sax.set_ylim)
help(plt.legend)

    # fig, ax = plt.subplots(1)
    # plt.scatter(data_obs['prob'], data_obs['Data'], marker='.', c='c', zorder=10, label='Data Qobs')
    # plt.plot(eva_obs['prob'], eva_obs['Value'], ls='-', c='y', label='Fit Qobs')
    # plt.plot(eva_obs['prob'], eva_obs['lci'], ls='--', c='y', alpha=0.3)
    # plt.plot(eva_obs['prob'], eva_obs['uci'], ls='--', c='y', alpha=0.3)
    # plt.fill(np.append(eva_obs['prob'], eva_obs['prob'][::-1]), np.append(eva_obs['lci'], eva_obs['uci'][::-1]), c='y', alpha=0.2)
    # plt.scatter(data['prob'], data['Data'], marker=',', s=3, c='b', zorder=9, label='Data '+qrep)
    # plt.plot(eva['prob'], eva['Value'], ls='-', c='g', label='Fit '+qrep)
    # plt.plot(eva['prob'], eva['lci'], ls='--', c='g', alpha=0.3)
    # plt.plot(eva['prob'], eva['uci'], ls='--', c='g', alpha=0.3)
    # plt.fill(np.append(eva['prob'], eva['prob'][::-1]), np.append(eva['lci'], eva['uci'][::-1]), c='g', alpha=0.2)
    # plt.title('Return period comparison, Qobs vs '+qrep)
    # plt.xlabel('Probability')
    # plt.ylabel('Discharge [$m^3$/s]')
    # plt.xticks(Pr)
    # ax.invert_xaxis()
    # plt.xscale('log')
    # ax.xaxis.set_major_formatter(FormatStrFormatter('%g'))
    # sax = ax.secondary_xaxis(location='top', functions=(prob2tr, tr2prob))
    # sax.set_xlabel('Return period [years]')
    # sax.get_xaxis().set_major_formatter(FormatStrFormatter('%g'))
    # sax.set_xticks(Tr)
    # plt.legend()
    # fig.savefig('./eva/GOOD_ENG/comparison_QOBS_vs_'+qrep+'_eng.png', format='png', dpi=300, bbox_inches='tight')



###################################################################################
#Uncertainty reduction comparison precipitation

Peva = pd.read_csv('./P_eva_results.csv')
Peva.index = Peva['Unnamed: 0']
Peva = Peva.drop(['Unnamed: 0'], axis=1)
Peva.index.name = None
Peva

Pleva = pd.read_csv('./Pl_eva_results.csv')
Pleva.index = Pleva['Unnamed: 0']
Pleva = Pleva.drop(['Unnamed: 0'], axis=1)
Pleva.index.name = None
Pleva

Pueva = pd.read_csv('./Pu_eva_results.csv')
Pueva.index = Pueva['Unnamed: 0']
Pueva = Pueva.drop(['Unnamed: 0'], axis=1)
Pueva.index.name = None
Pueva

Perr = Pueva - Pleva

# diff = pd.DataFrame()
# for col in Peva.columns:
#     diff[col] = Perr['Pobs'] - Perr[col]

Pdiff = pd.DataFrame([Perr['Pobs'] - Perr[col] for col in Peva.columns], index=Perr.columns, columns=Perr.index).T
Pratio = pd.DataFrame([Perr['Pobs']/Perr[col] for col in Peva.columns], index=Perr.columns, columns=Perr.index).T
Pperc = pd.DataFrame([Pdiff[col]/Perr['Pobs']*100 for col in Peva.columns], index=Perr.columns, columns=Perr.index).T

# Pratio.to_csv('Pratio.csv')
# Pperc.to_csv('Pperc.csv')

plt.plot(Peva, label=Peva.columns)
plt.plot(Pleva, label=Pleva.columns)
plt.plot(Pueva, label=Pueva.columns)
plt.legend()

############################################
#Uncertainty reduction comparison discahrge

Qeva = pd.read_csv('./Q_eva_results.csv')
Qeva.index = Qeva['Unnamed: 0']
Qeva = Qeva.drop(['Unnamed: 0'], axis=1)
Qeva.index.name = None
Qeva

Qleva = pd.read_csv('./Ql_eva_results.csv')
Qleva.index = Qleva['Unnamed: 0']
Qleva = Qleva.drop(['Unnamed: 0'], axis=1)
Qleva.index.name = None
Qleva

Queva = pd.read_csv('./Qu_eva_results.csv')
Queva.index = Queva['Unnamed: 0']
Queva = Queva.drop(['Unnamed: 0'], axis=1)
Queva.index.name = None
Queva

Qerr = Queva - Qleva

# diff = pd.DataFrame()
# for col in Peva.columns:
#     diff[col] = Perr['Pobs'] - Perr[col]
    
Qdiff = pd.DataFrame([Qerr['Qobs'] - Qerr[col] for col in Qeva.columns], index=Qerr.columns, columns=Qerr.index).T
Qratio = pd.DataFrame([Qerr['Qobs']/Qerr[col] for col in Qeva.columns], index=Qerr.columns, columns=Qerr.index).T
Qperc = pd.DataFrame([Qdiff[col]/Qerr['Qobs']*100 for col in Qeva.columns], index=Qerr.columns, columns=Qerr.index).T

# Qratio.to_csv('Qratio.csv')
# Qperc.to_csv('Qperc.csv')

plt.plot(Qeva, label=Qeva.columns)
# plt.plot(Qleva, label=Qleva.columns)
# plt.plot(Queva, label=Queva.columns)
plt.legend()



###################################################################################
#Add new time series to HecDSS
import pyhecdss

ensembles = [0,1,2,3,4,5,6,7,8,9,10]

Qr58 = pd.read_csv('D:/DANI/2021/TEMA4_PRONOSTICOS/PYR/HMS/Results/csv/Q_LT_0508.csv', parse_dates=['Unnamed: 0'])
Qr58.index = Qr58['Unnamed: 0']
Qr58.index.name = None
Qr58 = Qr58.drop(['Unnamed: 0'], axis=1)
Qr58_m = Qr58.resample('Y').max()[:-1]
Qr58_m

Qr811 = pd.read_csv('D:/DANI/2021/TEMA4_PRONOSTICOS/PYR/HMS/Results/csv/Q_LT_0811.csv', parse_dates=['Unnamed: 0'])
Qr811.index = Qr811['Unnamed: 0']
Qr811.index.name = None
Qr811 = Qr811.drop(['Unnamed: 0'], axis=1)
Qr811_m = Qr811.resample('Y').max()[:-1]
Qr811_m

Qr1114 = pd.read_csv('D:/DANI/2021/TEMA4_PRONOSTICOS/PYR/HMS/Results/csv/Q_LT_1114.csv', parse_dates=['Unnamed: 0'])
Qr1114.index = Qr1114['Unnamed: 0']
Qr1114.index.name = None
Qr1114 = Qr1114.drop(['Unnamed: 0'], axis=1)
Qr1114_m = Qr1114.resample('Y').max()[:-1]
Qr1114_m

#Q05-11
Qr_m = pd.DataFrame()
Qr = Qr58_m.append(Qr811_m)
for ens in ensembles:
    Qr_m = pd.concat([Qr_m, Qr[str(ens)]], ignore_index=True)
Qr_m.columns = ['Qrep511']
dates = pd.date_range(start='1800-01-01', periods=len(Qr_m), freq='Y')
Qr_m.index = dates

# Qr_m.to_csv('D:/DANI/2021/TEMA4_PRONOSTICOS/PYR/HMS/Results/csv/Q_LTmax_050811.csv')

# to DSS
fname=r'eva1.dss'
d = pyhecdss.DSSFile('./'+fname, create_new=True)
Dataset = 'ECMWF-REF' #input('Dataset, A ')
Location = 'RIOLASILLA' #input('Location, B ')
Variable = 'FLOW' #input('Variable, C ')
Timestep = '1Year' #input('Timestep, E ')
Units = 'M3/S' #input('Units ')
Type = 'INST-VAL' #input('Type' )
Comments = 'LT 05-11' #input('Comments, F ')
path = '/'+Dataset+'/'+Location+'/'+Variable+'//'+Timestep+'/'+Comments+'/'
data_dss = Qr_m
d.write_rts(path, data_dss, Units, Type)
d.close()

###########################

#Q05-14
Qr_m = pd.DataFrame()
Qr = Qr58_m.append(Qr811_m)
Qr = Qr.append(Qr1114_m)
for ens in ensembles:
    Qr_m = pd.concat([Qr_m, Qr[str(ens)]], ignore_index=True)
Qr_m.columns = ['Qrep514']
dates = pd.date_range(start='1678-01-01', periods=584, freq='Y')

remove_n = len(Qr_m) - 584
drop_indices = np.random.choice(Qr_m.index, remove_n, replace=False)
Qr_m = Qr_m.drop(drop_indices)

Qr_m.index = dates

# Qr_m.to_csv('D:/DANI/2021/TEMA4_PRONOSTICOS/PYR/HMS/Results/csv/Q_LTmax_05081114rand.csv')

# to DSS
fname=r'eva1.dss'
d = pyhecdss.DSSFile('./'+fname, create_new=False)
Dataset = 'ECMWF-REF' #input('Dataset, A ')
Location = 'RIOLASILLA' #input('Location, B ')
Variable = 'FLOW' #input('Variable, C ')
Timestep = '1Year' #input('Timestep, E ')
Units = 'M3/S' #input('Units ')
Type = 'INST-VAL' #input('Type' )
Comments = 'LT 05-14' #input('Comments, F ')
path = '/'+Dataset+'/'+Location+'/'+Variable+'//'+Timestep+'/'+Comments+'/'
data_dss = Qr_m
d.write_rts(path, data_dss, Units, Type)
d.close()


################################################################################
#Comparison of extreme values for different lead times discharge

idx=['Qobs','Qrep2-5','Qrep5-8','Qrep8-11','Qrep11-14','Qrep14-17']

df = pd.read_csv('./results/Qbandas.csv', skiprows=5, names=['id', 'prob', 'Data', 'Value', 'Exp Prob', 'uci', 'lci'])
df['Tr'] = round(1/(df['prob']/100),2)
data = pd.DataFrame([df['prob']/100, df['Tr'], df['Data'].replace(' ', np.nan).astype(float)]) #, columns=['prob', 'Tr', 'Data'])
data = data.transpose().dropna().reset_index(drop=True)
data = data.sort_values(by='Data', ascending=False)
eva = df.drop(['Data', 'id'], axis=1)
eva = eva.replace(' ', np.nan).dropna().reset_index(drop=True).astype(float)
eva['prob'] = eva['prob']/100
eva.index = eva['Tr']
eva.index.name = None
Tr_res = eva.drop(['prob', 'Tr', 'Exp Prob'], axis=1)

Q = pd.DataFrame(Tr_res['Value'])
Ql = pd.DataFrame(Tr_res['lci']) #upper and lower confidence intervals are inverted
Qu = pd.DataFrame(Tr_res['uci'])

for lt in slt:
    # start_lt = 5
    start_lt = lt
    end_lt = start_lt + 3
    print(start_lt, end_lt)
    
    df = pd.read_csv('./results/Qref'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.csv', skiprows=5, names=['id', 'prob', 'Data', 'Value', 'Exp Prob', 'uci', 'lci'])
    df['Tr'] = round(1/(df['prob']/100),2)
    data = pd.DataFrame([df['prob']/100, df['Tr'], df['Data'].replace(' ', np.nan).astype(float)]) #, columns=['prob', 'Tr', 'Data'])
    data = data.transpose().dropna().reset_index(drop=True)
    data = data.sort_values(by='Data', ascending=False)
    eva = df.drop(['Data', 'id'], axis=1)
    eva = eva.replace(' ', np.nan).dropna().reset_index(drop=True).astype(float)
    eva['prob'] = eva['prob']/100
    eva.index = eva['Tr']
    eva.index.name = None
    Tr_res = eva.drop(['prob', 'Tr', 'Exp Prob'], axis=1)


    Q = pd.concat([Q, Tr_res['Value']], axis=1)
    Ql = pd.concat([Ql, Tr_res['lci']], axis=1)
    Qu = pd.concat([Qu, Tr_res['uci']], axis=1)
    
Q.columns = idx
Ql.columns = idx
Qu.columns = idx
# Qerr = Qu - Ql

# Q = Q.transpose()
# Ql = Ql.transpose()
# Qu = Qu.transpose()
# Qerr = Qerr.transpose()

Q = Q.sort_index()
Ql = Ql.sort_index()
Qu = Qu.sort_index()
# Qerr = Qerr.sort_index()

Q = Q[Q.index>=2]
Ql = Ql[Ql.index>=2]
Qu = Qu[Qu.index>=2]
# Qerr = Qerr[Qerr.index>=2]

Q.index = Q.index.astype(int)
Ql.index = Ql.index.astype(int)
Qu.index = Qu.index.astype(int)
# Qerr.index = Qerr.index.astype(int)

# Q.to_csv('Q_eva_results.csv')
# Ql.to_csv('Ql_eva_results.csv')
# Qu.to_csv('Qu_eva_results.csv')
# Qerr.to_csv('Qerr_eva_results.csv')

Trx = [[2,20,200], [5,50,500], [10,100,1000]]
colors = {2:'b', 5:'b', 10:'b', 20:'y', 50:'y', 100:'y', 200:'g', 500:'g', 1000:'g'}
# colors = {'Qobs':'b', 'Qref2-5':'y','Qref5-8':'g','Qref8-11':'o','Qref11-14':'r','Qref14-17':'c'}
axvl = [0.5,1.5,2.5,3.5,4.5,5.5]

############
# FINAL IMAGE COMPARISON OF RETURN PERIODS AND UNCERTAINTY

Qerr1 = np.concatenate((np.array(Q[Q.index<11] - Ql[Q.index<11]).T, np.array(Qu[Q.index<11] - Q[Q.index<11]).T), axis=1).reshape(6,2,3)
Qerr2 = np.concatenate((np.array(Q[((Q.index<101) & (Q.index>11))] - Ql[((Q.index<101) & (Q.index>11))]).T, np.array(Qu[((Q.index<101) & (Q.index>11))] - Q[((Q.index<101) & (Q.index>11))]).T), axis=1).reshape(6,2,3)
Qerr3 = np.concatenate((np.array(Q[Q.index>101] - Ql[Q.index>101]).T, np.array(Qu[Q.index>101] - Q[Q.index>101]).T), axis=1).reshape(6,2,3)

fig, ax = plt.subplots(1,3, figsize=(15,7))
# ax2 = ax.twinx()
width = 0.9
yticks1 = [*np.arange(15,230,15)]
yticks2 = [*np.arange(50,700,40)]
yticks3 = [*np.arange(90,1620,90)]
Q[Q.index<11].plot(kind='bar', ax=ax[0], width=width, yerr=Qerr1, ylim=[yticks1[0],yticks1[-1]], yticks=yticks1, rot=0, zorder=10)
Q[((Q.index<101) & (Q.index>11))].plot(kind='bar', ax=ax[1], width=width, yerr=Qerr2, ylim=[yticks2[0],yticks2[-1]], yticks=yticks2, rot=0, zorder=10)
Q[Q.index>101].plot(kind='bar', ax=ax[2], width=width, yerr=Qerr3, ylim=[yticks3[0],yticks3[-1]], yticks=yticks3, rot=0, zorder=10)
# ax[2].yaxis.tick_right()
# ax[1].yaxis.set_visible(False)
ax[0].set_ylabel('Gasto [$m^3$/s]', fontsize=12)
# ax[2].set_ylabel('Gasto [$m^3$/s]')
ax[0].set_ylim(bottom=15, top=225) # 0-225
ax[1].set_ylim(bottom=50, top=650)
ax[2].set_ylim(bottom=90, top=1530)
# ax[2].yaxis.set_label_position("right")
ax[0].legend(ncol=1, loc='upper left', frameon=False)
ax[1].get_legend().remove()
ax[2].get_legend().remove()
plt.subplots_adjust(wspace=0.12, hspace=0)
ax[0].tick_params(axis="both", direction="out", pad=1, labelsize='medium')
ax[1].tick_params(axis="y", direction="out", pad=1, labelsize='medium') #-62
ax[1].tick_params(axis="x", direction="out", pad=1, labelsize='medium')
ax[2].tick_params(axis="both", direction="out", pad=1, labelsize='medium')
# ax[1].yaxis.set_ticks_position('none') 
ax[1].set_xlabel('Periodo de retorno [años]', loc='center', fontsize=12)
# ax[2].xaxis.set_label_coords(0, -.1)
ax[1].set_title('Comparación de periodos de retorno e incertidumbre estimados para caudales observados y series de tiempo sintéticas', fontsize=14, pad=10) #, x=0, y=1)
ax[0].grid(True, zorder=0)
ax[1].grid(True, zorder=0)
ax[2].grid(True, zorder=0)
plt.savefig('./comparison3_Tr2-1000.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.close()

############
# FINAL IMAGE COMPARISON OF RETURN PERIODS AND UNCERTAINTY
#ENGLISH

idx=['Qobs','Qref2-5','Qref5-8','Qref8-11','Qref11-14','Qref14-17']
Q.columns = idx
Ql.columns = idx
Qu.columns = idx

fig, ax = plt.subplots(1,3, figsize=(15,7))
# ax2 = ax.twinx()
width = 0.9
yticks1 = [*np.arange(15,230,15)]
yticks2 = [*np.arange(50,700,40)]
yticks3 = [*np.arange(90,1620,90)]
Q[Q.index<11].plot(kind='bar', ax=ax[0], width=width, yerr=Qerr1, ylim=[yticks1[0],yticks1[-1]], yticks=yticks1, rot=0, zorder=10)
Q[((Q.index<101) & (Q.index>11))].plot(kind='bar', ax=ax[1], width=width, yerr=Qerr2, ylim=[yticks2[0],yticks2[-1]], yticks=yticks2, rot=0, zorder=10)
Q[Q.index>101].plot(kind='bar', ax=ax[2], width=width, yerr=Qerr3, ylim=[yticks3[0],yticks3[-1]], yticks=yticks3, rot=0, zorder=10)
# ax[2].yaxis.tick_right()
# ax[1].yaxis.set_visible(False)
ax[0].set_ylabel('Discharge [$m^3$/s]', fontsize=12)
# ax[2].set_ylabel('Gasto [$m^3$/s]')
ax[0].set_ylim(bottom=15, top=225) # 0-225
ax[1].set_ylim(bottom=50, top=650)
ax[2].set_ylim(bottom=90, top=1530)
# ax[2].yaxis.set_label_position("right")
ax[0].legend(ncol=1, loc='upper left', frameon=False)
ax[1].get_legend().remove()
ax[2].get_legend().remove()
plt.subplots_adjust(wspace=0.12, hspace=0)
ax[0].tick_params(axis="both", direction="out", pad=1, labelsize='medium')
ax[1].tick_params(axis="y", direction="out", pad=1, labelsize='medium') #-62
ax[1].tick_params(axis="x", direction="out", pad=1, labelsize='medium')
ax[2].tick_params(axis="both", direction="out", pad=1, labelsize='medium')
# ax[1].yaxis.set_ticks_position('none') 
ax[1].set_xlabel('Return period [years]', loc='center', fontsize=12)
# ax[2].xaxis.set_label_coords(0, -.1)
ax[1].set_title('Return period and uncertainty comparison Qobs vs Qref with differet lead times', fontsize=14, pad=10) #, x=0, y=1)
ax[0].grid(True, zorder=0)
ax[1].grid(True, zorder=0)
ax[2].grid(True, zorder=0)
plt.savefig('./comparison3_Tr2-1000_ENG.jpg', format='jpg', dpi=300, bbox_inches='tight')
plt.close()


################################################################################
#Comparison of extreme values for different lead times precipitation

idx=['Pobs','Prep2-5','Prep5-8','Prep8-11','Prep11-14','Prep14-17']

df = pd.read_csv('./results/Pclicom.csv', skiprows=5, names=['id', 'prob', 'Data', 'Value', 'Exp Prob', 'uci', 'lci'])
df['Tr'] = round(1/(df['prob']/100),2)
data = pd.DataFrame([df['prob']/100, df['Tr'], df['Data'].replace(' ', np.nan).astype(float)]) #, columns=['prob', 'Tr', 'Data'])
data = data.transpose().dropna().reset_index(drop=True)
data = data.sort_values(by='Data', ascending=False)
eva = df.drop(['Data', 'id'], axis=1)
eva = eva.replace(' ', np.nan).dropna().reset_index(drop=True).astype(float)
eva['prob'] = eva['prob']/100
eva.index = eva['Tr']
eva.index.name = None
Tr_res = eva.drop(['prob', 'Tr', 'Exp Prob'], axis=1)

P = pd.DataFrame(Tr_res['Value'])
Pl = pd.DataFrame(Tr_res['lci']) #upper and lower confidence intervals are inverted
Pu = pd.DataFrame(Tr_res['uci'])

for lt in slt:
    # start_lt = 5
    start_lt = lt
    end_lt = start_lt + 3
    print(start_lt, end_lt)
    
    df = pd.read_csv('./results/Pref'+str(start_lt).zfill(2)+str(end_lt).zfill(2)+'.csv', skiprows=5, names=['id', 'prob', 'Data', 'Value', 'Exp Prob', 'uci', 'lci'])
    df['Tr'] = round(1/(df['prob']/100),2)
    data = pd.DataFrame([df['prob']/100, df['Tr'], df['Data'].replace(' ', np.nan).astype(float)]) #, columns=['prob', 'Tr', 'Data'])
    data = data.transpose().dropna().reset_index(drop=True)
    data = data.sort_values(by='Data', ascending=False)
    eva = df.drop(['Data', 'id'], axis=1)
    eva = eva.replace(' ', np.nan).dropna().reset_index(drop=True).astype(float)
    eva['prob'] = eva['prob']/100
    eva.index = eva['Tr']
    eva.index.name = None
    Tr_res = eva.drop(['prob', 'Tr', 'Exp Prob'], axis=1)

    P = pd.concat([P, Tr_res['Value']], axis=1)
    Pl = pd.concat([Pl, Tr_res['lci']], axis=1)
    Pu = pd.concat([Pu, Tr_res['uci']], axis=1)
    
P.columns = idx
Pl.columns = idx
Pu.columns = idx
# Qerr = Qu - Ql

# Q = Q.transpose()
# Ql = Ql.transpose()
# Qu = Qu.transpose()
# Qerr = Qerr.transpose()

P = P.sort_index()
Pl = Pl.sort_index()
Pu = Pu.sort_index()
# Qerr = Qerr.sort_index()

P = P[P.index>=2]
Pl = Pl[Pl.index>=2]
Pu = Pu[Pu.index>=2]
# Qerr = Qerr[Qerr.index>=2]

P.index = P.index.astype(int)
Pl.index = Pl.index.astype(int)
Pu.index = Pu.index.astype(int)
# Qerr.index = Qerr.index.astype(int)

# P.to_csv('P_eva_results.csv')
# Pl.to_csv('Pl_eva_results.csv')
# Pu.to_csv('Pu_eva_results.csv')
# Qerr.to_csv('Qerr_eva_results.csv')

Trx = [[2,20,200], [5,50,500], [10,100,1000]]
colors = {2:'b', 5:'b', 10:'b', 20:'y', 50:'y', 100:'y', 200:'g', 500:'g', 1000:'g'}
# colors = {'Qobs':'b', 'Qref2-5':'y','Qref5-8':'g','Qref8-11':'o','Qref11-14':'r','Qref14-17':'c'}
axvl = [0.5,1.5,2.5,3.5,4.5,5.5]

############
# FINAL IMAGE COMPARISON OF RETURN PERIODS AND UNCERTAINTY

Perr1 = np.concatenate((np.array(P[P.index<11] - Pl[P.index<11]).T, np.array(Pu[P.index<11] - P[P.index<11]).T), axis=1).reshape(6,2,3)
Perr2 = np.concatenate((np.array(P[((P.index<101) & (P.index>11))] - Pl[((P.index<101) & (P.index>11))]).T, np.array(Pu[((P.index<101) & (P.index>11))] - P[((P.index<101) & (P.index>11))]).T), axis=1).reshape(6,2,3)
Perr3 = np.concatenate((np.array(P[P.index>101] - Pl[P.index>101]).T, np.array(Pu[P.index>101] - P[P.index>101]).T), axis=1).reshape(6,2,3)

fig, ax = plt.subplots(1,3, figsize=(15,7))
# ax2 = ax.twinx()
width = 0.9
yticks1 = [*np.arange(20,150,10)]
yticks2 = [*np.arange(70,400,30)]
yticks3 = [*np.arange(100,1060,80)]
P[P.index<11].plot(kind='bar', ax=ax[0], width=width, yerr=Perr1, ylim=[yticks1[0],yticks1[-1]], yticks=yticks1, rot=0, zorder=10)
P[((P.index<101) & (P.index>11))].plot(kind='bar', ax=ax[1], width=width, yerr=Perr2, ylim=[yticks2[0],yticks2[-1]], yticks=yticks2, rot=0, zorder=10)
P[P.index>101].plot(kind='bar', ax=ax[2], width=width, yerr=Perr3, ylim=[yticks3[0],yticks3[-1]], yticks=yticks3, rot=0, zorder=10)
# ax[2].yaxis.tick_right()
# ax[1].yaxis.set_visible(False)
ax[0].set_ylabel('Precipitación [mm]', fontsize=12)
# ax[2].set_ylabel('Gasto [$m^3$/s]')
ax[0].set_ylim(bottom=20, top=140) # 0-225
ax[1].set_ylim(bottom=70, top=370)
ax[2].set_ylim(bottom=100, top=980)
# ax[2].yaxis.set_label_position("right")
ax[0].legend(ncol=1, loc='upper left', frameon=False)
ax[1].get_legend().remove()
ax[2].get_legend().remove()
plt.subplots_adjust(wspace=0.12, hspace=0)
ax[0].tick_params(axis="both", direction="out", pad=1, labelsize='medium')
ax[1].tick_params(axis="y", direction="out", pad=1, labelsize='medium') #-62
ax[1].tick_params(axis="x", direction="out", pad=1, labelsize='medium')
ax[2].tick_params(axis="both", direction="out", pad=1, labelsize='medium')
# ax[1].yaxis.set_ticks_position('none') 
ax[1].set_xlabel('Periodo de retorno [años]', loc='center', fontsize=12)
# ax[2].xaxis.set_label_coords(0, -.1)
ax[1].set_title('Comparación de periodos de retorno e incertidumbre estimados para precipitación observada y series de tiempo sintéticas', fontsize=14, pad=10) #, x=0, y=1)
ax[0].grid(True, zorder=0)
ax[1].grid(True, zorder=0)
ax[2].grid(True, zorder=0)
plt.savefig('./Pcomparison3_Tr2-1000.jpg', format='jpg', dpi=1000, bbox_inches='tight')
plt.close()

############
# FINAL IMAGE COMPARISON OF RETURN PERIODS AND UNCERTAINTY
#ENGLISH

idx=['Pobs','Pref2-5','Pref5-8','Pref8-11','Pref11-14','Pref14-17']
P.columns = idx
Pl.columns = idx
Pu.columns = idx

fig, ax = plt.subplots(1,3, figsize=(15,7))
# ax2 = ax.twinx()
width = 0.9
yticks1 = [*np.arange(20,150,10)]
yticks2 = [*np.arange(70,400,30)]
yticks3 = [*np.arange(100,1060,80)]
P[P.index<11].plot(kind='bar', ax=ax[0], width=width, yerr=Perr1, ylim=[yticks1[0],yticks1[-1]], yticks=yticks1, rot=0, zorder=10)
P[((P.index<101) & (P.index>11))].plot(kind='bar', ax=ax[1], width=width, yerr=Perr2, ylim=[yticks2[0],yticks2[-1]], yticks=yticks2, rot=0, zorder=10)
P[P.index>101].plot(kind='bar', ax=ax[2], width=width, yerr=Perr3, ylim=[yticks3[0],yticks3[-1]], yticks=yticks3, rot=0, zorder=10)
# ax[2].yaxis.tick_right()
# ax[1].yaxis.set_visible(False)
ax[0].set_ylabel('Precipitation [mm]', fontsize=12)
# ax[2].set_ylabel('Gasto [$m^3$/s]')
ax[0].set_ylim(bottom=20, top=140) # 0-225
ax[1].set_ylim(bottom=70, top=370)
ax[2].set_ylim(bottom=100, top=980)
# ax[2].yaxis.set_label_position("right")
ax[0].legend(ncol=1, loc='upper left', frameon=False)
ax[1].get_legend().remove()
ax[2].get_legend().remove()
plt.subplots_adjust(wspace=0.12, hspace=0)
ax[0].tick_params(axis="both", direction="out", pad=1, labelsize='medium')
ax[1].tick_params(axis="y", direction="out", pad=1, labelsize='medium') #-62
ax[1].tick_params(axis="x", direction="out", pad=1, labelsize='medium')
ax[2].tick_params(axis="both", direction="out", pad=1, labelsize='medium')
# ax[1].yaxis.set_ticks_position('none') 
ax[1].set_xlabel('Return period [years]', loc='center', fontsize=12)
# ax[2].xaxis.set_label_coords(0, -.1)
ax[1].set_title('Return period and uncertainty comparison Pobs vs Pref with differet lead times', fontsize=14, pad=10) #, x=0, y=1)
ax[0].grid(True, zorder=0)
ax[1].grid(True, zorder=0)
ax[2].grid(True, zorder=0)
plt.savefig('./Pcomparison3_Tr2-1000_ENG.jpg', format='jpg', dpi=300, bbox_inches='tight')
plt.close()

#################

import pylab as pl

pl.errorbar(Q.index, Q['Qobs'], yerr=Qerr['Qobs'], fmt='bo')
pl.errorbar(Q.index, Q['Qrep2-5'], yerr=Qerr['Qrep2-5'], fmt='bo')

x = np.arange(0,9)
fig, ax = plt.subplots(figsize=(15,7))
plt.fill_between(x, Qu['Qobs'], Ql['Qobs'], where=(x>2)&(x<=50))
ax.scatter(np.arange(0,9), Q['Qobs'], marker='_')

# fig.set_size_inches(6, 4, forward=True)
ax.errorbar(np.arange(0,9)-0.25, Q['Qobs'], yerr=Qerr['Qobs'], fmt='.', linewidth=3, capsize=3)
ax.errorbar(np.arange(0,9)-0.15, Q['Qrep2-5'], yerr=Qerr['Qrep2-5'], fmt='.', linewidth=3, capsize=3)
ax.errorbar(np.arange(0,9)-0.05, Q['Qrep5-8'], yerr=Qerr['Qrep5-8'], fmt='.', linewidth=3, capsize=3)
ax.errorbar(np.arange(0,9)+0.05, Q['Qrep8-11'], yerr=Qerr['Qrep8-11'], fmt='.', linewidth=3, capsize=3)
ax.errorbar(np.arange(0,9)+0.15, Q['Qrep11-14'], yerr=Qerr['Qrep11-14'], fmt='.', linewidth=3, capsize=3)
ax.errorbar(np.arange(0,9)+0.25, Q['Qrep14-17'], yerr=Qerr['Qrep14-17'], fmt='.', linewidth=3, capsize=3)
plt.xticks(np.arange(0,9), Q.index)



help(plt.xticklabel)
#################

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax2 = fig.add_subplot(121)
ax1.plot(Q[Q.index<51], kind='bar', ax=ax[0], width=width, position=0)
Q[Q.index>51].plot(kind='bar', ax=ax[1], width=width, position=1)
ax2.yaxis.tick_right()
ax1.set_ylabel('Gasto [$m^3$/s]')
ax2.set_ylabel('Gasto [$m^3$/s]')
ax2.yaxis.set_label_position("right")
ax.set_xlabel('Periodo de retorno [años]')

plt.show()


help(plt.title)

Q[Q.index<51].plot.bar(yerr=Qerr[Q.index<51], rot=0) #y=[2,5]
Q[Q.index>51].plot.bar(yerr=Qerr[Q.index>51], rot=0) #y=[2,5]
plt.xlabel('Periodo de retorno')

plt.bar(Q.index, Q['Qobs'].values)

help(Q.plot.bar)

# for ids in idx:
#     # ids = 'Qobs'
#     plt.errorbar(Q.index, Q[ids], yerr=Qerr[ids], marker='_', ms=10, ls='', color=colors[ids], elinewidth=10, alpha=0.5)

# plt.scatter(Q.transpose().index, Q.transpose()[2])

# plt.plot(Q, ls='', marker='.', label=Q.columns)
# plt.legend()


# Q[Q.index>101].plot.bar(yerr=Qerr[Q.index>101], rot=0, width=0.9) #y=[2,5]
# plt.xlabel('Periodo de retorno [años]')
# plt.ylabel('Gasto [$m^3$/s]')
# plt.title('Comparación de periodos de retorno, 200 a 1000 años')
# plt.legend(ncol=2, fontsize=10, loc='upper left')
# plt.savefig('./comparison_Tr200-1000.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.close()

# Q[((Q.index<101) & (Q.index>11))].plot.bar(yerr=Qerr[((Q.index<101) & (Q.index>11))], rot=0, width=0.9) #y=[2,5]
# plt.xlabel('Periodo de retorno [años]')
# plt.ylabel('Gasto [$m^3$/s]')
# plt.title('Comparación de periodos de retorno, 20 a 100 años')
# plt.legend(ncol=2, fontsize=10, loc='upper left')
# plt.savefig('./comparison_Tr20-100.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.close()

# Q[Q.index<11].plot.bar(yerr=Qerr[Q.index<11], rot=0, width=0.9) #y=[2,5]
# plt.xlabel('Periodo de retorno [años]')
# plt.ylabel('Gasto [$m^3$/s]')
# plt.title('Comparación de periodos de retorno, 2 a 10 años')
# plt.legend(ncol=2, fontsize=10, loc='upper left')
# plt.savefig('./comparison_Tr2-10.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.close()

# ###########

# Q[Q.index>51].plot.bar(yerr=Qerr[Q.index>51], rot=0, width=0.9) #y=[2,5]
# plt.xlabel('Periodo de retorno [años]')
# plt.ylabel('Gasto [$m^3$/s]')
# plt.title('Comparación de periodos de retorno, 100 a 1000 años')
# plt.legend(ncol=2, fontsize=9, loc='upper left')
# plt.savefig('./comparison_Tr100-1000.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.close()

# Q[Q.index<51].plot.bar(yerr=Qerr[Q.index<51], rot=0, width=0.9) #y=[2,5]
# plt.xlabel('Periodo de retorno [años]')
# plt.ylabel('Gasto [$m^3$/s]')
# plt.title('Comparación de periodos de retorno, 2 a 50 años')
# plt.legend(ncol=2, fontsize=9, loc='upper left')
# plt.savefig('./comparison_Tr2-50.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.close()


# ###########

# Q.plot.bar(yerr=Qerr, rot=0, width=0.9)
# # plt.axvline(4.5, ls='--', lw=1, color='k', alpha=0.5)
# plt.xlabel('Periodo de retorno [años]')
# plt.ylabel('Gasto [$m^3$/s]')
# plt.title('Comparación de periodos de retorno, 2 a 1000 años')
# plt.legend(ncol=2, fontsize=10, loc='upper left')
# plt.savefig('./comparison_Tr2-1000.jpg', format='jpg', dpi=1000, bbox_inches='tight')
# plt.close()

# ###########

# fig, ax = plt.subplots(1,2)
# # ax2 = ax.twinx()
# width = 0.9
# Q[Q.index<101].plot(kind='bar', ax=ax[0], width=width, yerr=Qerr[Q.index<101], rot=0)
# Q[Q.index>101].plot(kind='bar', ax=ax[1], width=width, yerr=Qerr[Q.index>101], rot=0)
# ax[1].yaxis.tick_right()
# ax[0].set_ylabel('Gasto [$m^3$/s]')
# ax[1].set_ylabel('Gasto [$m^3$/s]')
# ax[1].yaxis.set_label_position("right")
# ax[0].legend(ncol=1, fontsize=8, loc='upper left')
# ax[1].get_legend().remove()
# plt.subplots_adjust(wspace=0, hspace=0)
# plt.xlabel('Periodo de retorno [años]', loc='center')
# ax[1].xaxis.set_label_coords(0, -.1)
# plt.title('Comparación de periodos de retorno', x=0, y=1)


# for trx in Trx:
#     # trx = Trx[0]
#     print(trx[0])
#     for tr in trx:
#         plt.errorbar(Q.index, Q['Qobs'], yerr=Qerr['Qobs'], marker='_', ms=10, ls='', color=colors, elinewidth=10, alpha=0.5)
#         # plt.axhline(Q[tr]['Qobs'], ls='--', color=colors[tr], alpha=0.5)
#         # plt.scatter(Q.index, Q[tr], marker='_', s=200, color=colors[tr] , label=str(tr)+' years')
#     [plt.axvline(_axvl, ls='--', lw=1, color='k', alpha=0.5) for _axvl in axvl]
#     plt.xlim(-0.5,3.5)
#     plt.xlabel('Lead time (days)')
#     plt.ylabel('Discharge [$m^3$/s]')
#     plt.title('Comparison of discharge return periods')
#     plt.legend(fontsize=8, loc='best', ncol=3)
#     # plt.show()
#     # plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/EVAcomparison/'+id+'_Qobs_vs_Qref_'+str(trx[0])+'.jpg', format='jpg', dpi=1000, bbox_inches='tight')
#     # plt.close()

# Q.transpose()
# Qerr.transpose()

for trx in Trx:
    # trx = Trx[1]
    print(trx[0])
    for tr in trx:
        plt.errorbar(Q.transpose().index, Q.transpose()[tr], yerr=Qerr.transpose()[tr], marker='_', ms=50, ls='', color=colors[tr], elinewidth=50, alpha=0.5)
        # plt.axhline(Q[tr]['Qobs'], ls='--', color=colors[tr], alpha=0.5)
        # plt.scatter(Q.index, Q[tr], marker='_', s=200, color=colors[tr] , label=str(tr)+' years')
    [plt.axvline(_axvl, ls='--', lw=1, color='k', alpha=0.5) for _axvl in axvl]
    plt.xlim(-0.5,5.5)
    plt.xlabel('Tiempo de espera [días]')
    plt.ylabel('Gasto [$m^3$/s]')
    plt.title('Comparación de periodos de retorno de caudal')
    plt.legend(fontsize=8, loc='best', ncol=3)
    # plt.show()
    # plt.savefig('C:/Users/villarre/DataAnalysis/Reforecasts/figures/'+str(model)+'/EVAcomparison/'+id+'_Qobs_vs_Qref_'+str(trx[0])+'.jpg', format='jpg', dpi=1000, bbox_inches='tight')
    # plt.close()

i = 0
Trx = Tr[:]
fig, ax = plt.subplots(len(Trx),1)
for tr in Trx:
    ax[i].errorbar(Q.transpose().index, Q.transpose()[tr], yerr=Qerr.transpose()[tr], marker='_', ms=50, ls='', color=colors[tr], elinewidth=50, alpha=0.5)
    if tr != Trx[-1]:
        ax[i].xaxis.set_visible(False)
    else:
        ax[i].set_xlabel('Serie de tiempo')
    # ax[len(Trx)].xaxis.set_visible(False)
    i = i + 1
plt.ylabel('Gasto [$m^3$/s]')



ax[1]

help(plt.errorbar)
