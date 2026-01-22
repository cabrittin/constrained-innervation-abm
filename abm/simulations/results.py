"""                            
@name: 
@description:                  

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""
import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy import stats
from tqdm import tqdm
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from scipy.stats import ttest_ind_from_stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportions_ztest    

from pycsvparser import read,write
from abm import analyze
from .utils import _format_df_cols,_classify_sdm,_ecdf
from .utils import _ecdf_range

DOM_MAP = {0:'#9a00cb',1:'#29c000',2:'#cbcbcb',3:'#000000'}

def sdm_kde(args):
    _format_df_cols(args) 
    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    df = df.groupby(['parameter_group']).mean()
    bandwidth = float(args.bandwidth)
    x = np.linspace(-0.5,4.5,500)
    sdm = df['sdm'].to_numpy()
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(sdm[:,np.newaxis])
    sdm_density = np.exp(kde.score_samples(x[:,np.newaxis]))
    
    sdmthresh = None
    locmax = analyze.get_local_extreme(sdm_density,type_extreme='max')
    locmin = analyze.get_local_extreme(sdm_density,type_extreme='min')
    sdmthresh = analyze.get_sdm_thresh(df,bandwidth=bandwidth)
    
    print(f"SDM threshold: {sdmthresh}")
    fig,ax = plt.subplots(1,1,figsize=(1.5,1.5))
    ax.hist(sdm,bins=60,range=(0,4),density=True,facecolor='#d9d9d9')
    ax.plot(x,sdm_density,color='k')
    if sdmthresh is not None: 
        ax.axvline(sdmthresh,linestyle='--',linewidth=1,color='k') 
    ax.set_ylim([0,1.0])
    ax.set_xlabel('SDM',fontsize=7)
    ax.set_ylabel('Density',fontsize=7)
    ax.set_yticks([0,0.5,1.0])
    ax.set_xticks([0,2,4])
    ax.tick_params(axis='y',labelsize=6)
    ax.tick_params(axis='x',labelsize=6)
    plt.tight_layout()

def sdm_kde_pi(args):
    _format_df_cols(args) 
    
    #df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    #x = np.linspace(-0.5,4.5,500)
    x = np.linspace(-0.5,8.5,1000)
    bandwidth = float(args.bandwidth)
    
    def get_density(df,x,bandwidth):
        df = df.groupby(['parameter_group']).mean()
        sdm = df['sdm'].to_numpy()
        kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
        kde.fit(sdm[:,np.newaxis])
        sdm_density = np.exp(kde.score_samples(x[:,np.newaxis]))
        return sdm_density 
    
    ptrs = [args.df_sim_sig1,args.df_sim_sig2,args.df_sim_sig3]
    sigs = []
    for p in ptrs:
        df = analyze.format_sdm_from_path(p,args.df_emp,args.fit_cols) 
        sigs.append(get_density(df,x,bandwidth))

    fig,ax = plt.subplots(1,1,figsize=(1.8,1.2))
    #ax.hist(sdm,bins=60,range=(0,4),density=True,facecolor='#d9d9d9')
    #ax.hist(sdm,bins=120,range=(0,8),density=True,facecolor='#d9d9d9')
    ax.plot(x,sigs[0],color='#1f77b4',label='σ=1')
    ax.plot(x,sigs[1],color='#ff7f0e',label='σ=2')
    ax.plot(x,sigs[2],color='#d62728',label='σ=3')
   

    ax.set_ylim([0,1.0])
    ax.set_xlim([0,8])
    ax.set_xlabel('SDM',fontsize=7)
    ax.set_ylabel('Density',fontsize=7)
    ax.set_yticks([0,0.5,1.0])
    ax.set_xticks([0,2,4,6,8])
    ax.tick_params(axis='y',labelsize=6)
    ax.tick_params(axis='x',labelsize=6)
    plt.tight_layout()


def sdm_density(args):
    _format_df_cols(args) 
    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    df = df.groupby(['parameter_group']).mean()
    bandwidth = float(args.bandwidth)
    sdmthresh = analyze.get_sdm_thresh(df,bandwidth=bandwidth)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
    
    emp = pd.read_csv(args.df_emp)
    emp = emp.mean(axis=0)
    
    fig,ax = plt.subplots(1,1,figsize=(1.4,1.4))
    sns.kdeplot(ax=ax,data=df,
                x='num_domains',y='vsr',hue='sdm_group',
                bw_method=0.5,palette=DOM_MAP,
                common_norm=False,legend=False,linewidth=0.1)
    ax.plot([emp['num_domains']],[emp['vsr']],marker="*",ms=6,color='k') 
    ax.tick_params(axis='x',labelsize=6)
    ax.tick_params(axis='y',labelsize=6)
    ax.set_yticks([0,0.5,1.0])
    ax.set_xticks([2,3,4,5])
    #ax.set_xlabel('# domains',fontsize=8)
    #ax.set_ylabel('vsr',fontsize=8)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.tight_layout()

def affinity_vs_scaffold_hue_sdm(args):
    _format_df_cols(args) 
    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    df = df.groupby(['parameter_group']).mean()
    bandwidth = float(args.bandwidth)
    sdmthresh = analyze.get_sdm_thresh(df,bandwidth=bandwidth)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
    
    fig,ax = plt.subplots(1,1,figsize=(1.3,1.5))
    sns.boxplot(ax=ax,data=df,
                x='average_response_rate',y='num_pioneers',
                hue='sdm_group',palette=DOM_MAP,hue_order=[1,0],
                width=0.5,linewidth=0.5)
    ax.set_yticks([0,5,10,15,20,25,30,35])
    ax.set_xticklabels(['off','on'])
    ax.tick_params(axis='x',labelsize=6)
    ax.tick_params(axis='y',labelsize=6) 
    ax.set_ylabel('# pioneers',fontsize=7)
    ax.set_xlabel('Pioneer-follower affinity',fontsize=6)
    plt.legend([],[],frameon=False)
    plt.tight_layout()
 
def sdm_robust_specificity(args):
    _format_df_cols(args) 
    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    sdmthresh = float(args.sdmthresh)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
    
    DF = [df[df['run'] == i] for i in range(3)] 
    n_high_sdm = DF[0].shape[0] - DF[0]['sdm_group'].sum()
    print(f"Total samples: {DF[0].shape[0]}, #High SDM sims: {n_high_sdm}")
    
    DF[0] = DF[0][DF[0]['sdm_group'] == 1]
    for i in range(1,3):
        DF[i] = DF[i][DF[i]['parameter_group'].isin(DF[0]['parameter_group'].tolist())]
    
    scolumn = args.specificity_column 
    run_number = int(args.run_number)
    fig,ax = plt.subplots(1,1,figsize=(2,2))
    Hfp,Hfn,xbins = _classify_specificity(DF[run_number],scolumn)
    _plot_switch(ax,Hfp,Hfn,xbins,color=args.bar_color) 
    ax.set_xlabel(args.xlabel,fontsize=8)
    plt.tight_layout()
    _save_plot(args)
    if args.show_plot == "True": plt.show()

def sdm_robust_locality_neigh_cont_2d(args):
    _format_df_cols(args) 
    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    sdmthresh = float(args.sdmthresh)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
    
    nd = np.load(args.data_sim)
    nd = nd.mean(axis=1)
    df['mean_degree'] = nd[:-7]
    df['nc_group'] = df['mean_degree'].apply(_classify_nc,args=(0.5,)) 

    lcolumn = args.locality_column 
    if lcolumn == 'fd': df[lcolumn] = df[lcolumn] - 1
    scolumn = args.specificity_column 
    
    # Classify specificity  and filter runs
    df['loc_group'] = df.apply(_classify_locality_2d,axis=1,args=(scolumn,))
    
    Z = np.zeros((4,10))
    for i in range(Z.shape[0]):
        yvals,xbins = _neigh_cont_by_locality(df[df['loc_group']==i],lcolumn)
        Z[i,:] = yvals
   

    if scolumn == 'fs': Z = Z[:-1,:]
    fig,ax = plt.subplots(1,1,figsize=(3,1.5))
    im = ax.imshow(Z,vmin=0,vmax=1)
    
    if scolumn == 'fs':
        ax.set_yticks([-0.5,0.5,1.5,2.5])
        ax.set_yticklabels([1.0,0.9,0.8,0.7])
    else: 
        ax.set_yticks([-0.5,0.5,1.5,2.5,3.5])
        ax.set_yticklabels([1.0,0.9,0.8,0.7,0.6])
     
    ax.tick_params(axis='y',labelsize=6)
    #ax.set_xticks([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5])
    #ax.set_xticklabels([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    
    ax.set_xticks([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5])
    ax.set_xticklabels([0,1,2,3,4,5,6,7,8,9,10])
    
    ax.tick_params(axis='x',labelsize=6)
    ax.set_ylabel(args.ylabel,fontsize=8)
    ax.set_xlabel(args.xlabel,fontsize=8)
    cbar = fig.colorbar(im,shrink=0.65)
    cbar.set_label('Neigh contacts',fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    plt.tight_layout()

def sdm_robust_locality_2d(args):
    _format_df_cols(args) 
    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    
    sdmthresh = float(args.sdmthresh)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 

    lcolumn = args.locality_column 
    if lcolumn == 'fd': df[lcolumn] = df[lcolumn] - 1
    scolumn = args.specificity_column 
    df['loc_group'] = df.apply(_classify_locality_2d,axis=1,args=(scolumn,))
    
    Z = np.zeros((4,10))
    for i in range(Z.shape[0]):
        Hfp,Hfn,xbins = _classify_locality(df[df['loc_group']==i],lcolumn)
        Z[i,:] = _collapse_locality(Hfp,Hfn)
    
    if scolumn == 'fs': Z = Z[:-1,:]
    fig,ax = plt.subplots(1,1,figsize=(3,1.5))
    im = ax.imshow(Z,vmin=0,vmax=1)
    
    if scolumn == 'fs':
        ax.set_yticks([-0.5,0.5,1.5,2.5])
        ax.set_yticklabels([1.0,0.9,0.8,0.7])
    else: 
        ax.set_yticks([-0.5,0.5,1.5,2.5,3.5])
        ax.set_yticklabels([1.0,0.9,0.8,0.7,0.6])
     
    ax.tick_params(axis='y',labelsize=6)
    #ax.set_xticks([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5])
    #ax.set_xticklabels([0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0])
    
    ax.set_xticks([-0.5,0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5])
    ax.set_xticklabels([0,1,2,3,4,5,6,7,8,9,10])
    
    ax.tick_params(axis='x',labelsize=6)
    ax.set_ylabel(args.ylabel,fontsize=8)
    ax.set_xlabel(args.xlabel,fontsize=8)
    cbar = fig.colorbar(im,shrink=0.65)
    cbar.set_label('robustness',fontsize=8)
    cbar.ax.tick_params(labelsize=6)
    plt.tight_layout()


def specificity_num_pioneers_per_follower(args):
    _format_df_cols(args)
    efile = args.data_emp
    edat = np.load(efile)

    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    sdmthresh = float(args.sdmthresh)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
    nd = np.load(args.data_sim)
    
    scolumn = args.specificity_column 
    run_process = list(map(int,args.run_process.split(',')))
    scaffold = int(args.scaffold_size)
    min_specificity = int(args.min_specificity) / 10.

    #print('nd',nd.shape,run_process,df.shape)
    x = np.arange(15)
    B = np.zeros((len(df),len(x))) 
    bdx = 0
    for (idx,row) in df.iterrows():
        if row['run'] not in [1,2,3]: continue
        if row['num_pioneers'] < 14: continue
        if row['num_pioneers'] > 20: continue
        if row['sdm_group'] == 0: continue
        spec = row['fs'] 
        loc = row['fd'] 
        if loc >= 5: continue
        if spec > 0.9: continue
        if spec < 0.8: continue

        
        #if row['run'] not in run_process: continue
        #if row['sdm_group'] == 0: continue
        #npio = int(row['num_pioneers'])
        #spec = int(10*row[scolumn])
        #spec = row[scolumn] 
        #if npio in list(range(5,25)) and spec == min_specificity: 
        #if npio not in list(range(13,20)): continue 
        #if npio != 14: continue 
        #if spec <= min_specificity:# or spec >= 0.9: 
        #if spec >= 0.9 or spec<0.8: continue
        z = nd[idx,:] 
        #print(idx,row['fs'],row['fd'],np.mean(z[z>-1])/npio,np.mean(z[z>-1]),row['average_response_rate'])
        y = _ecdf(z[z>-1],x)
        B[bdx,:] = y
        bdx += 1
    #print(bdx) 
    #B = _ecdf_compile(np.array(B),dsort=np.arange(15))
    B = B[:bdx,:]
    x = np.arange(15)
    fig,ax = plt.subplots(1,1,figsize=(1.5,1.5))
    emin,emax = _ecdf_range(B,num_std=int(args.num_std))
    ax.fill_between(x,emin,emax,color='g',alpha=0.3,label=f'Sim. n = {B.shape[0]} ')
    ax.fill_between(x,edat[1,:],edat[2,:], color='k', alpha=0.3,label='Emp. n = 3')

    ax.set_ylim([0,1])
    ax.set_xlim([0,6])
    ax.set_xlabel('# pioneer contacts per follower',fontsize=8)
    ax.set_ylabel('ECDF',fontsize=8)
    ax.tick_params(axis='y',labelsize=6)
    ax.tick_params(axis='x',labelsize=6)
    ax.legend(loc='lower right',fontsize=6) 
    plt.tight_layout()

def specificity_num_pioneers_per_follower_scaffold(args):
    _format_df_cols(args)
    efile = args.data_emp
    edat = np.load(efile)

    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    sdmthresh = float(args.sdmthresh)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
    nd = np.load(args.data_sim)
    
    scolumn = args.specificity_column 
    run_process = list(map(int,args.run_process.split(',')))
    scaffold = int(args.scaffold_size)
    min_specificity = int(args.min_specificity)
    
    scaff = [14,19]

    x = np.arange(15)
    A = np.zeros((len(df),len(x))) 
    B = np.zeros((len(df),len(x))) 
    adx = 0
    bdx = 0
    for (idx,row) in df.iterrows():
        if row['run'] not in run_process: continue
        if row['sdm_group'] == 0: continue
        npio = int(row['num_pioneers'])
        spec = int(10*row[scolumn])
        #if npio in list(range(5,25)) and spec == min_specificity: 
        if spec != min_specificity: continue 
        z = nd[idx,:] 
        y = _ecdf(z[z>-1],x)
        
        if npio == scaff[0]:
            A[adx,:] = y
            adx += 1
        elif npio == scaff[1]: 
            B[bdx,:] = y
            bdx += 1

    #B = _ecdf_compile(np.array(B),dsort=np.arange(15))
    A = A[:adx,:]
    B = B[:bdx,:]
    x = np.arange(15)
    fig,ax = plt.subplots(1,1,figsize=(2,2))
    emin,emax = _ecdf_range(A,num_std=int(args.num_std))
    ax.fill_between(x,emin,emax,color='g',alpha=0.6,label=f'Scaff: {scaff[0]} (n={A.shape[0]})')
    emin,emax = _ecdf_range(B,num_std=int(args.num_std))
    ax.fill_between(x,emin,emax,color='g',alpha=0.3,label=f'Scaff: {scaff[1]} (n={B.shape[0]})')
    ax.fill_between(x,edat[1,:],edat[2,:], color='k', alpha=0.3,label='Emp. n = 3')

    ax.set_ylim([0,1])
    ax.set_xlim([0,6])
    ax.set_xlabel('# pioneer contacts per follower',fontsize=8)
    ax.set_ylabel('ECDF',fontsize=8)
    ax.tick_params(axis='y',labelsize=6)
    ax.tick_params(axis='x',labelsize=6)
    ax.legend(loc='lower right',fontsize=6) 
    plt.tight_layout()
    _save_plot(args)

    if args.show_plot == "True": plt.show()


def specificity(args):
    from matplotlib.ticker import AutoMinorLocator
    _format_df_cols(args)
    efile = args.data_emp
    edat = np.load(efile)

    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    sdmthresh = float(args.sdmthresh)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
    nd = np.load(args.data_sim)
    
    scolumn = args.specificity_column 
    run_process = list(map(int,args.run_process.split(',')))
    scaffold = int(args.scaffold_size)
    min_specificity = int(args.min_specificity)

    x = np.linspace(0,1,101)
    B = np.zeros((len(df),len(x))) 
    bdx = 0
    srec = []
    for (idx,row) in df.iterrows():
        if row['run'] not in run_process: continue
        if row['sdm_group'] == 0: continue 
        if row['num_pioneers'] < 10: continue
        if row['num_pioneers'] > 25: continue 
        spec = row[scolumn]
        if spec < 0.8 or spec >= 0.9: continue
        z = nd[idx,:] 
        y = 1 - _ecdf(-1*z[z<0],x) 
        B[bdx,:] = y 
        srec.append(row['ps'])
        bdx += 1

    B = B[:bdx,:]
    #print(min(srec),max(srec)) 
    fig,ax = plt.subplots(1,1,figsize=(1.5,1.5))
    emin,emax = _ecdf_range(B,num_std=int(args.num_std))
    ax.fill_between(x,emin,emax,color='g',alpha=0.3,label=f'Sim. n = {B.shape[0]}')
    #ax.fill_between(x,edat[1,:],edat[2,:], color='k', alpha=0.3,label='Emp. n = 3')
    ax.plot(x,1-edat,'k',label='Emp.')

    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    #ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('Pioneer uniqueness',fontsize=8)
    ax.set_ylabel('1 - ECDF',fontsize=8)
    #ax.set_title(f'Scaffold size: {scaffold}',fontsize=6)
    ax.tick_params(axis='y',labelsize=6)
    ax.tick_params(axis='x',labelsize=6)
    ax.legend(loc='lower left',fontsize=6) 
    plt.tight_layout()

def similarity(args):
    from matplotlib.ticker import AutoMinorLocator
    _format_df_cols(args)
    efile = args.data_emp
    edat = pd.read_csv(efile)
 
    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    sdmthresh = float(args.sdmthresh)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
    nd = np.load(args.data_sim)
    print(nd) 

    scolumn = args.specificity_column 
    run_process = list(map(int,args.run_process.split(',')))
    scaffold = int(args.scaffold_size)
    min_specificity = int(args.min_specificity)

    x = np.linspace(0,1,101)
    B = np.zeros((len(df),len(x))) 
    bdx = 0
    srec = []
    for (idx,row) in df.iterrows():
        if row['run'] not in run_process: continue
        npio = int(row['num_pioneers'])
        #if npio != scaffold: continue 
        #if npio not in list(range(14,20)): continue 
        spec = row['ps']
        #if spec < 0.7: continue
        if spec > 0.8: continue
        y = nd[idx,:] 
        if row['sdm_group'] == 1: 
            B[bdx,:] = y 
            srec.append(row['ps'])
            bdx += 1
    
    B = B[:bdx,:]
    print(min(srec),max(srec)) 
    fig,ax = plt.subplots(1,1,figsize=(1.5,1.5))
    emin,emax = _ecdf_range(B,num_std=int(args.num_std))
    ax.fill_between(x,emin,emax,color='g',alpha=0.3,label=f'Sim. n = {B.shape[0]}')
    #ax.fill_between(x,edat[1,:],edat[2,:], color='k', alpha=0.3,label='Emp. n = 3')
    #ax.plot(x,1-edat,'k',label='Emp.')

    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    #ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlabel('Pioneer uniqueness',fontsize=8)
    ax.set_ylabel('1 - ECDF',fontsize=8)
    #ax.set_title(f'Scaffold size: {scaffold}',fontsize=6)
    ax.tick_params(axis='y',labelsize=6)
    ax.tick_params(axis='x',labelsize=6)
    ax.legend(loc='lower left',fontsize=6) 
    plt.tight_layout()
    _save_plot(args)

    if args.show_plot == "True": plt.show()

def sdm_scaffold_elbow(args):
    import statsmodels.api as sm
    from kneed import KneeLocator

    _format_df_cols(args) 
    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    df = df.groupby(['parameter_group']).mean()
    sdmthresh = float(args.sdmthresh)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
    df = df[df['sdm_group']==1]
    
    print(len(df))
    #isplit = int(np.where(df['num_pioneers'] == int(args.scaffold_size))[0])
    #print(np.where(df['num_pioneers'] == int(args.scaffold_size)))
    
    x = df['num_pioneers'].to_numpy()
    y = df['sdm'].to_numpy()
    
    smoothed = sm.nonparametric.lowess(endog=y, exog=x, frac=0.7)
    
    kn = KneeLocator(smoothed[:,0], smoothed[:,1], S=1.0, curve='convex', direction='decreasing') 
    elbow_x = kn.knee
    elbow_y = kn.knee_y

    fig,ax = plt.subplots(1,1,figsize=(2,2))
    sns.scatterplot(ax=ax,data=df,x='num_pioneers',y='sdm',s=20,color=args.bar_color)
    ax.plot(smoothed[:,0],smoothed[:,1],color='r',linestyle='--')
    ax.axvline(elbow_x,color='k',linestyle='--',alpha=0.5) #Number of empirical pioneering processes
    
    ax.set_xticks([0,5,10,15,20,25,30,35])
    ax.set_xlabel('# pioneers',fontsize=8)
    ax.set_ylabel('SDM',fontsize=8)
    ax.set_xlim([0,35])
    ax.set_ylim([0,sdmthresh])
    ax.tick_params(axis='x',labelsize=6)
    ax.tick_params(axis='y',labelsize=6)
    plt.tight_layout()
    
    _save_plot(args)
    if args.show_plot == "True": plt.show()

def scaffold_pf_conserved(args):
    _format_df_cols(args) 
    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    df = df.groupby(['parameter_group']).mean()
    sdmthresh = float(args.sdmthresh)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
    #df = df[df['sdm_group']==1]
    
    fig,ax = plt.subplots(1,2,figsize=(4,2))
    sns.lineplot(ax=ax[0],data=df,x='num_pioneers',y='conserved_contact_frac',
                 hue='average_response_rate',palette={0:'k',-1:'r'})
    ax[0].set_xticks([0,5,10,15,20,25,30,35]) 
    ax[0].set_ylim([0,1])
    ax[0].set_yticks([0,0.25,0.5,0.75,1.0])
    ax[0].tick_params(axis='x',labelsize=6)
    ax[0].tick_params(axis='y',labelsize=6)
    ax[0].set_ylabel('Fraction conserved contacts',fontsize=8)
    ax[0].set_xlabel('# pioneers',fontsize=8)

    sns.scatterplot(ax=ax[1],data=df,x='conserved_contact_frac',y='num_domains',
                    hue='sdm_group',palette={1:'#29c000',0:'#9a00cb'},s=10)
    ax[1].set_yticks([0,1,2,3,4,5]) 
    ax[1].set_xlim([0,1])
    ax[1].set_xticks([0,0.25,0.5,0.75,1.0])
    ax[1].tick_params(axis='x',labelsize=6)
    ax[1].tick_params(axis='y',labelsize=6)
    ax[1].set_xlabel('Fraction conserved contacts',fontsize=8)
    ax[1].set_ylabel('# domains',fontsize=8)

    plt.tight_layout()
    
    _save_plot(args)
    if args.show_plot == "True": plt.show()

def scaffold_size_vs_num_domains_hue_sdm(args):
    _format_df_cols(args) 
    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    df = df.groupby(['parameter_group']).mean()
    sdmthresh = float(args.sdmthresh)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
    df = df[df['num_pioneers'] > 4]
    

    fig,ax = plt.subplots(1,1,figsize=(1.5,1.5))
    scatter = sns.scatterplot(ax=ax,data=df,
                    x='num_pioneers',y='num_domains',
                    hue='sdm_group',palette={1:'#29c000',0:'#9a00cb'},
                    s=20,legend=False)
    
    ax.set_xticks([0,5,10,15,20,25,30,35])
    ax.set_ylim([3,5]) 
    ax.set_yticks([3,4,5]) 
    ax.tick_params(axis='x',labelsize=6)
    ax.tick_params(axis='y',labelsize=6)
    ax.set_xlabel('# fixed axons',fontsize=8)
    ax.set_ylabel('# domains',fontsize=8)
    plt.tight_layout() 

def scaffold_size_vs_bimodal_hue_sdm(args):
    _format_df_cols(args) 
    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    df = df.groupby(['parameter_group']).mean()
    sdmthresh = float(args.sdmthresh)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
    df = df[df['num_pioneers'] > 4]
    

    fig,_ax = plt.subplots(2,1,figsize=(1.5,2.5),sharex=True)
    
    ax = _ax[0] 
    scatter = sns.scatterplot(ax=ax,data=df,
                    x='num_pioneers',y='num_domains',
                    hue='sdm_group',palette={1:'#29c000',0:'#9a00cb'},
                    s=20,legend=False)
    
    ax.set_ylim([3,5]) 
    ax.set_yticks([3,4,5]) 
    ax.tick_params(axis='x',labelsize=6)
    ax.tick_params(axis='y',labelsize=6)
    ax.set_xlabel('# fixed axons',fontsize=8)
    ax.set_ylabel('# domains',fontsize=8)

    ax = _ax[1]
    scatter = sns.scatterplot(ax=ax,data=df,
                    x='num_pioneers',y='vsr',
                    hue='sdm_group',palette={1:'#29c000',0:'#9a00cb'},
                    s=20,legend=False)
    
    ax.set_xticks([5,15,25,35])
    ax.set_ylim([0,1]) 
    ax.set_yticks([0.0,0.25,0.5,0.75,1.0]) 
    ax.tick_params(axis='x',labelsize=6)
    ax.tick_params(axis='y',labelsize=6)
    ax.set_xlabel('# fixed axons',fontsize=8)
    ax.set_ylabel('Repro shape',fontsize=8)
    plt.tight_layout() 

def total_contacts(args):
    _format_df_cols(args) 
    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    df = df.groupby(['parameter_group']).mean()
    sdmthresh = float(args.sdmthresh)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
    #df = df[df['num_pioneers'] > 4]

    nd = np.load(args.data_sim) * 69
    nd = nd.sum(axis=1)
    ndr = nd.reshape(-1, 100) 
    mean_ndr = ndr.mean(axis=1)
    
    df['mean_contact'] = mean_ndr[:-1]
    df = df[df['num_pioneers'] > 4]

    fig,ax = plt.subplots(1,1,figsize=(1.5,1.5))
    scatter = sns.boxplot(ax=ax,data=df,
                    y='mean_contact',x='sdm_group', legend=False)
    
    ax.set_xticks([0,1])
    ax.set_xticklabels(['No','Yes'])
    ax.tick_params(axis='x',labelsize=6)
    ax.tick_params(axis='y',labelsize=6)
    ax.set_ylabel('# contacts',fontsize=8)
    ax.set_xlabel("Affinity",fontsize=8)
    plt.tight_layout() 

def num_fixed_vs_total_contacts(args):
    _format_df_cols(args) 
    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    df = df.groupby(['parameter_group']).mean()
    sdmthresh = float(args.sdmthresh)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
    #df = df[df['num_pioneers'] > 4]

    nd = np.load(args.data_sim) * 69
    nd = nd.sum(axis=1)
    #for n in nd: print(int(n)) 
    ndr = nd.reshape(-1, 100) 
    mean_ndr = ndr.mean(axis=1)
    
    df['mean_contact'] = mean_ndr[:-1]
    df = df[df['num_pioneers'] > 4]

    fig,ax = plt.subplots(1,1,figsize=(1.5,1.5))
    scatter = sns.scatterplot(ax=ax,data=df,
                    x='num_pioneers',y='mean_contact',
                    hue='sdm_group',palette={1:'#29c000',0:'#9a00cb'},
                    s=20,legend=False)
    
    ax.set_xticks([5,15,25,35])
    #ax.set_ylim([0,1]) 
    #ax.set_yticks([0.0,0.25,0.5,0.75,1.0]) 
    ax.tick_params(axis='x',labelsize=6)
    ax.tick_params(axis='y',labelsize=6)
    ax.set_xlabel('# fixed axons',fontsize=8)
    ax.set_ylabel('# contacts',fontsize=8)
    plt.tight_layout() 


def frac_neighbors_across_metrics(args):
    _format_df_cols(args) 
    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    df = df.groupby(['parameter_group']).mean()
    sdmthresh = float(args.sdmthresh)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
    #df = df[df['num_pioneers'] > 4]

    nd = np.load(args.data_sim)
    nd = nd.mean(axis=1)
    ndr = nd.reshape(-1, 100) 
    mean_ndr = ndr.mean(axis=1)
    
    df['mean_degree'] = mean_ndr[:-1]
    df = df[df['num_pioneers'] > 4]

    fig,_ax = plt.subplots(2,2,figsize=(2.5,2.5),sharex=True)
    _ax = _ax.ravel() 
    ax = _ax[0]
    scatter = sns.scatterplot(ax=ax,data=df,
                    y='num_domains',x='mean_degree',
                    s=20,
                    hue='average_response_rate', palette={0:'#29c000',-1:'#9a00cb'},
                    legend=False)
    
    ax.set_ylim([3,5]) 
    ax.set_yticks([3,4,5]) 
    ax.tick_params(axis='x',labelsize=6)
    ax.tick_params(axis='y',labelsize=6)
    ax.set_ylabel('# domains',fontsize=8)
   
    ax = _ax[1]
    scatter = sns.scatterplot(ax=ax,data=df,
                    y='vsr',x='mean_degree',
                    s=20,
                    hue='average_response_rate', palette={0:'#29c000',-1:'#9a00cb'},
                    legend=False)
    
    ax.set_ylim([0,1]) 
    ax.set_yticks([0,0.5,1.0]) 
    ax.tick_params(axis='x',labelsize=6)
    ax.tick_params(axis='y',labelsize=6)
    ax.set_ylabel('# domains',fontsize=8)
 
    ax = _ax[2]
    scatter = sns.scatterplot(ax=ax,data=df,
                    y='conserved_contact_frac',x='mean_degree',
                    s=20,
                    hue='average_response_rate', palette={0:'#29c000',-1:'#9a00cb'},
                    legend=False)
    
    ax.set_ylim([0,1]) 
    ax.set_yticks([0,0.5,1.0]) 
    ax.tick_params(axis='x',labelsize=6)
    ax.tick_params(axis='y',labelsize=6)
    ax.set_ylabel('Cons. contacts',fontsize=8)
    
    ax.set_xlim([0,1])
    ax.set_xticks([0,0.5,1.0])
    ax.set_xlabel('Neigh contacts',fontsize=8) 

    ax = _ax[3]
    scatter = sns.scatterplot(ax=ax,data=df,
                    y='variable_contact_frac',x='mean_degree',
                    s=20,
                    hue='average_response_rate', palette={0:'#29c000',-1:'#9a00cb'},
                    legend=False)
    
    ax.set_ylim([0,0.2]) 
    ax.set_yticks([0,0.1,0.2]) 
    ax.tick_params(axis='x',labelsize=6)
    ax.tick_params(axis='y',labelsize=6)
    ax.set_ylabel('Indv. contacts',fontsize=8)
    ax.set_xlim([0,1])
    ax.set_xticks([0,0.5,1.0])
    ax.set_xlabel('Neigh contacts',fontsize=8) 

    plt.tight_layout()

def _frac_neighbors_across_metrics_kde(func):
    def inner(args,**kwargs): 
        _format_df_cols(args) 
        df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
        sdmthresh = float(args.sdmthresh)
        df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 

        nd = np.load(args.data_sim)
        df = func(args,df=df,nd=nd)
        
        fig,_ax = plt.subplots(1,4,figsize=(6,1.5))
        
        ax = _ax[0]
        sns.kdeplot(ax=ax,data=df, 
                    x='mean_degree',y='num_domains', 
                    fill=True, cmap="viridis",hue_norm=(0,100))
        ax.set_ylim([3,5]) 
        ax.set_yticks([3,4,5])
        ax.set_ylabel('# domains',fontsize=8)
        ax.set_xlim([0,1])
        ax.set_xticks([0,0.5,1.0]) 
        ax.tick_params(axis='both', labelsize=6)
        ax.set_xlabel('Neigh contacts',fontsize=8)

        ax = _ax[1]
        sns.kdeplot(ax=ax,data=df, 
                    x='mean_degree',y='vsr', 
                    fill=True, cmap="viridis",hue_norm=(0,100))
        ax.set_ylim([0,1]) 
        ax.set_yticks([0,0.5,1.0])
        ax.set_ylabel('Repro shape ',fontsize=8)
        ax.set_xlim([0,1])
        ax.set_xticks([0,0.5,1.0]) 
        ax.tick_params(axis='both', labelsize=6)
        ax.set_xlabel('Neigh contacts',fontsize=8)
       
        ax = _ax[2]
        sns.kdeplot(ax=ax,data=df, 
                    x='mean_degree',y='conserved_contact_frac', 
                    fill=True, cmap="viridis",hue_norm=(0,100))
        ax.set_ylim([0,1]) 
        ax.set_yticks([0,0.5,1.0])
        ax.set_ylabel('Cons contacts',fontsize=8)
        ax.set_xlim([0,1])
        ax.set_xticks([0,0.5,1.0]) 
        ax.tick_params(axis='both', labelsize=6)
        ax.set_xlabel('Neigh contacts',fontsize=8)
        
        ax = _ax[3]
        sns.kdeplot(ax=ax,data=df, 
                    x='mean_degree',y='variable_contact_frac', 
                    fill=True, cmap="viridis",hue_norm=(0,100))
        ax.set_ylim([0,0.2]) 
        ax.set_yticks([0,0.1,0.2])
        ax.set_ylabel('Indiv. contacts',fontsize=8)
        ax.set_xlim([0,1])
        ax.set_xticks([0,0.5,1.0]) 
        ax.tick_params(axis='both', labelsize=6)
        ax.set_xlabel('Neigh contacts',fontsize=8)
        
        """
        cmap = plt.get_cmap('viridis')
        norm = Normalize(vmin=0, vmax=100)
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=_ax, orientation='vertical')
        cbar.set_label('# simulation runs')
        """ 

        plt.tight_layout()
    
    return inner

@_frac_neighbors_across_metrics_kde
def kde_metrics_high_specificity(*args,df=None,nd=None,**kwargs):
    nd = nd.mean(axis=1)
    df['mean_degree'] = nd[:-100]
    df = df[df['num_pioneers'] > 4]
    return df[df['average_response_rate']==0]

@_frac_neighbors_across_metrics_kde
def kde_metrics_no_affinity(*args,df=None,nd=None,**kwargs):
    nd = nd.mean(axis=1)
    df['mean_degree'] = nd[:-100]
    df = df[df['num_pioneers'] > 4]
    return df[df['average_response_rate']==-1]

@_frac_neighbors_across_metrics_kde
def kde_metrics_moderate_specificity(args,df=None,nd=None,**kwargs):
    nd = nd.mean(axis=1)
    df['mean_degree'] = nd[:-7]
    
    pg = np.load(args.data_sim_2)[:-7,:,:]
    #pg = np.load(args.data_sim)[:-7,:,:]
    print(pg.shape) 
    pg = pg.sum(axis=1)
    print(pg.shape)
    pgsum = pg.sum(axis=1)
    pg = pg / pgsum[:,np.newaxis]
    
    df['conserved_contact_frac'] = pg[:,-1]
    df['variable_contact_frac'] = pg[:,0]

    df = df[df['num_pioneers'] > 4]
    return df[(df['sdm_group'] == 1) 
              & (df['fs'] < 0.9) 
              & (df['fs'] >= 0.8)
              & (df['fd'] < 4)]


def emp_frac_axons_contacted(args):
    _format_df_cols(args)
    efile = args.data_emp
    edat = np.load(efile)

    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    nd = np.load(args.data_sim)
    
    x = np.linspace(0,1,101)
    A = np.zeros((len(df),len(x))) 
    B = np.zeros((len(df),len(x))) 
    adx = 0
    bdx = 0
    for (idx,row) in df.iterrows():
        if row['num_pioneers'] < 5: continue
        y = _ecdf(nd[idx,:],x)
        if row['average_response_rate'] < 0:
            A[adx,:] = y
            adx += 1
        else:
            B[bdx,:] = y
            bdx += 1
    
    A = A[:adx,:]
    B = B[:bdx,:]
    fig,ax = plt.subplots(1,1,figsize=(1.5,1.5))
    emin,emax = _ecdf_range(B,num_std=int(args.num_std))
    emin[emin<0] = 0 
    ax.fill_between(x,emin,emax,color='#29c000',alpha=0.6,
                    label=f'Low SDM: n={B.shape[0]}')
    
    emin,emax = _ecdf_range(A,num_std=int(args.num_std))
    emin[emin<0] = 0 
    ax.fill_between(x,emin,emax,color='#9a00cb',alpha=0.6,
                    label=f'High SDM: n={A.shape[0]}')

    ax.fill_between(x,edat[1,:],edat[2,:], color='k', 
                    alpha=0.5,label='Emp. n = 3')
    
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.set_xlabel('Axons contacted (Frac.)',fontsize=8)
    ax.set_ylabel('ECDF',fontsize=8)
    ax.tick_params(axis='y',labelsize=6)
    ax.tick_params(axis='x',labelsize=6)
    ax.legend(loc='lower right',fontsize=6) 
    #plt.tight_layout()

def pg_synapse_fraction_follower(args):
    Z = np.load(args.data_emp)
    N = Z['N']
    D = Z['D']
    
    print(N)
    def sum_follower_pg(y,idx):
        ysum = y.sum(axis=1, keepdims=True)
        y = y / ysum
        return y[:,idx]
    
    def bar_values(N,D,idx):
        y0 = sum_follower_pg(D[:,:,-1],idx)
        y1 = sum_follower_pg(N[:,:,1:].sum(axis=2),idx)
        y0mu = y0.mean()
        y0std = y0.std()
        y1mu = y1.mean()
        y1std = y1.std()
        bar0 = np.array([y0mu,y1mu])
        yerr0 = [y0std,y1std]
        return bar0,yerr0

    bar0,yerr0 = bar_values(N,D,0)
    bar1,yerr1 = bar_values(N,D,1)
    bar2,yerr2 = bar_values(N,D,4)
    bar3,yerr3 = bar_values(N,D,2)
    bar4,yerr4 = bar_values(N,D,3)

    width = 0.35 
    labels = ['Cons.','Synaptic']
    fig, ax = plt.subplots(1,1,figsize=(1.5,1.5))
    ax.bar(labels, bar0, width, label='p-f')
    ax.bar(labels, bar1, width, bottom=bar0 ,label='f-f')
    ax.bar(labels, bar2, width, bottom=bar0 + bar1 ,label="f-f'")
    ax.bar(labels, bar3, width, bottom=bar0 + bar1 + bar2,label="p-p'")
    ax.bar(labels, bar4, width, bottom=bar0 + bar1 + bar2 + bar3,label="p-f'")
    ax.set_ylim([0,1.15])
    ax.set_yticks([0,0.25,0.5,0.75,1.0])
    ax.tick_params(axis='both',labelsize=6) 
    ax.set_ylabel('Fraction of contacts',fontsize=8)
    ax.set_title('Synapse dist.',fontsize=8) 
    ax.legend(fontsize=6)
    plt.tight_layout()
 

def _dep_pg_composition_distilled(args):
    Z = np.load(args.data_emp)
    N = Z['N']
    D = Z['D']
    print(D) 
    distill_idx = [0,1,4]
    
    # Conserved contact fraction
    y0 = D[:,:,-1]
    y0sum = y0.sum(axis=1, keepdims=True)
    y0 = y0 / y0sum
    y0 = y0[:,distill_idx]
    y0mean = y0.mean(axis=0)

    # Synapse fraction
    y1 =  N[:,:,1:].sum(axis=2)
    y1sum = y1.sum(axis=1, keepdims=True)
    y1 = y1 / y1sum
    y1 = y1[:,distill_idx]
    y1mean = y1.mean(axis=0) 

    fig,ax = plt.subplots(1,1,figsize=(1.5,1.5))
    x = np.arange(y1.shape[1])
    width = 0.35
    colors = ['#1f77b4','#ff7f0e'] 
    ax.bar(x-width/2, y0mean, width=width,
           color=colors[0],alpha=0.5,zorder=1,
           label='Conserved adjacency')
    ax.bar(x+width/2, y1mean, width=width,
           color=colors[1],alpha=0.5,zorder=1,
           label='Synaptic')
    
    print(y0)
    for j,data in enumerate([y0,y1]):
        for i, points in enumerate(data.T):
            # Add a small jitter in x so points don't overlap exactly
            jitter = np.random.uniform(-width/6, width/6, size=len(points))
            dx = [-width/2, width/2][j] 
            ax.scatter(np.full_like(points, x[i]+dx) + jitter, points,
                       color=colors[j], s=10, zorder=2)
    
    ax.legend(fontsize=6)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(["p-f","f-f","f-f'"],fontsize=8)
    ax.tick_params(axis='y', labelsize=6) 
    ax.set_ylabel('Fraction of contacts',fontsize=8)    

    plt.tight_layout()

def pg_cam_differential(args):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    df0 = pd.read_csv(args.data_0)
    df0[['x_mean','y_mean']] = df0[['x_mean','y_mean']] + 1
    df0['label'] = 'Cons:Var'
    
    """
    df1 = pd.read_csv(args.data_1)
    df1[['x_mean','y_mean']] = df1[['x_mean','y_mean']] + 1
    df1['label'] = 'p-f:f-f'
    """ 
    
    df1 = pd.read_csv(args.data_2)
    df1[['x_mean','y_mean']] = df1[['x_mean','y_mean']] + 1
    df1['label'] = "p-f:f-f'"
    #df1['label'] = "p-f:p-f'"

    df = pd.concat([df0,df1])
    df['ratio'] = np.log2(df['x_mean'] / df['y_mean'])

    fig,ax = plt.subplots(1,1,figsize=(1.5,1.5))
    ax_main = sns.ecdfplot(ax=ax,data=df,x='ratio',hue='label')
    xticks = [-1,0,1,2,3,4,5]  # locations (log2 scale)
    ax.set_xticks(xticks)
    ax.set_xticklabels([f'{int(2**v)}' for v in xticks],fontsize=6)
    #ax.set_xlim([-1,1])
    ax.set_ylabel('ECDF',fontsize=8)
    ax.set_xlabel('CAM similarity ratio (log2 scale)',fontsize=8)
    ax.tick_params(axis='both',labelsize=6) 
    #ax.set_xscale('log', base=2)
    #ax.set_xticks([1,2,4,8,16,32])
    #ax.set_xticklabels([1,2,4,8,16,32])
    #ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower right')
    #sns.ecdfplot(data=df, x='ratio', hue='label', ax=ax_inset, legend=False)
    #ax_inset.set_ylabel('')
    #ax_inset.tick_params(axis='both', which='major', labelsize=6)
    #ax_inset.set_yticks([0,0.5,1.0]) 
    #ax_inset.set_xlabel('')
    #plt.tight_layout()

def pg_cam_shared_scaled(args):
    df = pd.read_csv(args.data)
    df = df[df['is_pio']==0]
    #df = df[df['pg_id'].isin([1,3])]

    fig,_ax = plt.subplots(2,2,figsize=(6,6))
    _ax = _ax.ravel() 
    
    iters = [('var','cons'),('cons','pf'),('var','pf'),('fnf','pf')]

    for i,(_x,_y) in enumerate(iters):
        ax = _ax[i] 
        sns.scatterplot(ax=ax,data=df,x=_x,y=_y)
        ax.plot([-2,0,2],[-2,0,2],'--',color='r') 
        ax.set_ylim([-2,2])
        ax.set_xlim([-2,2])  
        ax.tick_params(axis='both',labelsize=6) 
     
    plt.tight_layout()

def pg_cam_shared(args):
    df = pd.read_csv(args.data)
    df = df[df['is_pio']==0]
    #df = df[df['pg_id'].isin([1,3])]

    fig,ax = plt.subplots(1,1,figsize=(3,3))
    sns.boxplot(ax=ax,data=df,x='pg_id',y='zscore')
    ax.set_ylim([-2,2])
    ax.tick_params(axis='both',labelsize=6) 
 
    plt.tight_layout()



def assigned_pioneer_groups(args):
    from kmodes.kmodes import KModes
    df = pd.read_csv(args.df_emp)
    numpio = int(df['is_pioneer'].sum())
    pio = df[df['is_pioneer']==1]['cell'].tolist()
    fol = df[df['is_pioneer']==0]['cell'].tolist()
    
    X = df[df['is_pioneer']==0].iloc[:,2:].to_numpy()
    
    def cluster_reorder_binary_index(arr, k_clusters):
        # Cluster rows using K-modes
        km = KModes(n_clusters=k_clusters, init='Huang', n_init=20, verbose=0)
        clusters = km.fit_predict(arr)
        
        # Reorder the array based on cluster assignments
        return np.argsort(clusters)

    k_clusters = 4
    idx = cluster_reorder_binary_index(X.T, k_clusters)
    X = X[:,idx]
    pio = [pio[i] for i in idx]
    
    k_clusters = 10
    idx = cluster_reorder_binary_index(X, k_clusters)
    X = X[idx,:]
    fol = [fol[i] for i in idx]


    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.imshow(X,cmap='binary', interpolation='nearest')
    ax.set_xticks(np.arange(len(pio))) 
    ax.set_xticklabels(pio,fontsize=8,rotation=90)
    ax.set_yticks(np.arange(len(fol))) 
    ax.set_yticklabels(fol,fontsize=8)
    plt.tight_layout()

    _save_plot(args)
    if args.show_plot == "True": plt.show()


def _pg_breakdown(func):
    def inner(args): 
        emp,z = func(args) 
        vmax = float(args.vmax) 
        
        Z = np.zeros((70,z.shape[-2],z.shape[-1]))
        T = np.zeros((70,z.shape[-1])) 
        for i in range(70):
            idx = slice(i*100,(i+1)*100) 
            x = z[idx,:,:].mean(axis=0)
            T[i,:] = x.sum(axis=0)
            xsum = x.sum(axis=1)
            xsum[xsum==0] = 1
            x = x / xsum[:,np.newaxis]
            Z[i,:,:] = x
        
        _format_df_cols(args) 
        df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
        #df = analyze.format_sdm_from_path(args.dataframe,args.df_empirical,args.fit_cols) 
        df = df.groupby(['parameter_group']).mean()
        sdmthresh = float(args.sdmthresh)
        df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
        
        #pmin = np.where(df['num_pioneers']>=14)[0]
        #df = df[df['num_pioneers']>=14]
        #Z = Z[pmin,:,:]

        low = np.where((df['sdm_group']==1) & (df['num_pioneers'] >= 10))[0]
        high = np.where((df['sdm_group'] == 0) & (df['num_pioneers'] >= 10))[0]
        
        Z0 = Z[high,:,:].mean(axis=0)
        Z1 = Z[low,:,:].mean(axis=0)
        
        T0 = T[high,:].mean(axis=0)
        T1 = T[low,:].mean(axis=0)
        
        T0 = T0 / T0.sum()
        T1 = T1 / T1.sum()
        
        cmap = 'PRGn'
        xlabels = np.arange(Z0.shape[1]) + 1
        fig,_ax = plt.subplots(2,3,figsize=(8,4))
        ax = _ax.flatten() 
        img = ax[0].imshow(Z0,cmap=cmap,vmin=0,vmax=vmax)
        ax[0].set_yticks([0,1,2,3,4]) 
        ax[0].set_yticklabels(["p-f","f-f","p-p'","p-f'","f-f'"])
        ax[0].set_xticks(np.arange(Z0.shape[1])) 
        ax[0].set_xticklabels(xlabels)

        ax[1].imshow(Z1,cmap=cmap,vmin=0,vmax=vmax)
        ax[1].set_yticks([0,1,2,3,4]) 
        ax[1].set_yticklabels(["p-f","f-f","p-p'","p-f'","f-f'"])
        ax[1].set_xticks(np.arange(Z1.shape[1])) 
        ax[1].set_xticklabels(xlabels)
        
        temp = emp.sum(axis=0)
        esum = emp.sum(axis=1)
        esum[esum==0] = 1
        emp = emp / esum[:,np.newaxis]

        ax[2].imshow(emp,cmap=cmap,vmin=0,vmax=vmax)
        ax[2].set_yticks([0,1,2,3,4]) 
        ax[2].set_yticklabels(["p-f","f-f","p-p'","p-f'","f-f'"])
        ax[2].set_xticks(np.arange(emp.shape[1])) 
        ax[2].set_xticklabels(xlabels)

        ax[3].imshow([T0],cmap=cmap,vmin=0,vmax=vmax)
        ax[3].set_yticks([0]) 
        #ax[0].set_yticklabels(["total"])
        ax[3].set_xticks(np.arange(len(temp))) 
        ax[3].set_xticklabels(xlabels)

        ax[4].imshow([T1],cmap=cmap,vmin=0,vmax=vmax)
        ax[4].set_yticks([0]) 
        #ax[1].set_yticklabels(["total"])
        ax[4].set_xticks(np.arange(len(temp))) 
        ax[4].set_xticklabels(xlabels)

        temp = temp / temp.sum()
        ax[5].imshow([temp],cmap=cmap,vmin=0,vmax=vmax)
        ax[5].set_yticks([0]) 
        #ax[2].set_yticklabels(["total"])
        ax[5].set_xticks(np.arange(len(temp))) 
        ax[5].set_xticklabels(xlabels)

        fig.colorbar(img, ax=ax, orientation='horizontal', fraction=.1)

        #if args.show_plot == "True": plt.show()

    return inner

@_pg_breakdown
def pg_breakdown_bimodal(args):
    emp = np.load(args.data_emp)
    z = np.load(args.data_sim)
    return emp[:,1:],z[:,:,1:]

@_pg_breakdown
def pg_breakdown_domain(args):
    emp = np.load(args.data_emp)
    z = np.load(args.data_sim)
    return emp,z

def _pg_composition(func):
    def inner(args): 
        E = np.load(args.data_emp)
        R = None
        try:
            R = np.load(args.data_emp_rand)
        except:
            pass

        z = np.load(args.data_sim)
        print(args.data_sim,z) 
        A = np.zeros(z.shape)
        B = np.zeros(z.shape)
        adx = 0
        bdx = 0
        _format_df_cols(args) 
        df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
        sdmthresh = float(args.sdmthresh)
        df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
     
        for (idx,row) in df.iterrows():
            if row['run'] not in [1,2,3]: continue
            if row['num_pioneers'] < 14: continue
            if row['num_pioneers'] > 20: continue
            if row['sdm_group'] == 0: continue
            spec = row['fs'] 
            loc = row['fd'] 
            if loc > 4: continue 
            
            #Hack to handle situations where only parts of z were computed
            _z = z[(idx, *[slice(None)] * (z.ndim - 1))] 
            if _z.min() == _z.max(): continue

            if spec > 0.9:
                slices = (adx, *[slice(None)] * (A.ndim - 1))
                A[slices] = _z 
                adx += 1
            elif spec < 0.9 and spec >= 0.8:
                slices = (bdx, *[slice(None)] * (B.ndim - 1))
                B[slices] = _z
                bdx += 1
        slices = (slice(0, adx), *[slice(None)] * (A.ndim - 1)) 
        A = A[slices]
        slices = (slice(0, bdx), *[slice(None)] * (B.ndim - 1)) 
        B = B[slices]
        
        func(args,A=A,B=B,E=E,R=R)

    return inner

@_pg_composition
def estimate_specificity_contact_increase(*args,A=None,B=None,E=None,**kwargs):
    print(A.sum(axis=2).sum(axis=1).mean())
    print(B.sum(axis=2).sum(axis=1).mean())
    
    print(A.sum(axis=2).mean(axis=0))
    print(B.sum(axis=2).mean(axis=0))

    A = A[:,:,-1].sum(axis=1)
    B = B[:,:,-1].sum(axis=1)

    medA = np.median(A)
    medB = np.median(B)

    increase = medB/medA - 1
    print(increase)
    print(increase*0.85)

    x = np.linspace(500,1000,501)
    y0 = np.array([np.sum(A <= _x) / len(A) for _x in x ])
    y1 = np.array([np.sum(B <= _x) / len(B) for _x in x ])
    
    fig,ax = plt.subplots(1,1,figsize=(1.25,1.25))
    ax.plot(x,y0,color='blue',label=f'High spec.: n={A.shape[0]}')
    ax.plot(x,y1,color='orange',label=f'Moderate spec.: n={B.shape[0]}')
    ax.set_xlabel('# conserved contacts',fontsize=8)
    ax.set_ylabel('ECDF',fontsize=8)
    ax.legend(fontsize=6)
    ax.set_xlim([600,1000])
    ax.set_ylim([0,1]) 
    ax.tick_params(axis='both',labelsize=6)
    #plt.tight_layout()

@_pg_composition
def compare_specificity_neigh_contacts(args,A=None,B=None,E=None,**kwargs):
    x = np.linspace(0,1,101)
    
    A = np.apply_along_axis(_ecdf,1,A,x)
    B = np.apply_along_axis(_ecdf,1,B,x)

    fig,ax = plt.subplots(1,1,figsize=(1.5,1.5))
    emin,emax = _ecdf_range(A,num_std=int(args.num_std))
    emin[emin<0] = 0 
    ax.fill_between(x,emin,emax,color='blue',alpha=0.4,
                    label=f'High spec.: n={A.shape[0]}')
 
    emin,emax = _ecdf_range(B,num_std=int(args.num_std))
    emin[emin<0] = 0 
    ax.fill_between(x,emin,emax,color='orange',alpha=0.6,
                    label=f'Moderate spec.: n={B.shape[0]}')
    
    #ax.fill_between(x,edat[1,:],edat[2,:], color='k', 
    #                alpha=0.5,label='Emp. n = 3')
    ax.plot(x,E[0,:],color='k',label='Emp. n=3')

    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.set_xlabel('Neigh contacts',fontsize=8)
    ax.set_ylabel('ECDF',fontsize=8)
    ax.tick_params(axis='y',labelsize=6)
    ax.tick_params(axis='x',labelsize=6)
    ax.legend(loc='lower right',fontsize=6) 

@_pg_composition
def pg_composition_distilled(*args,A=None,B=None,E=None,R=None,**kwargs):
    def scaled_intra(D):
        dsum = D.sum(axis=2)
        dscale = dsum.sum(axis=1)
        dsum = dsum / dscale[:,np.newaxis] 
        mu = dsum[:,:2].sum(axis=1).mean()
        std = dsum[:,:2].sum(axis=1).std()
        return mu,std
    
    print(A.shape)
    
    adx = A.shape[0]
    bdx = B.shape[0]
    edx = E.shape[0]
    rdx = R.shape[0]
    
    amu,astd = scaled_intra(A)
    bmu,bstd = scaled_intra(B)
    emu,estd = scaled_intra(E)
    rmu,rstd = scaled_intra(R)
    
    print(amu)
    print(bmu)
    print(emu)
    print(rmu)
    
    print(adx,bdx,edx,rdx)

    bar0 = np.array([amu,bmu,emu])
    bar1 = 1 - bar0
    yerr = [astd,bstd,estd]
    width = 0.35 
    labels = ['High', 'Moderate', 'Emp']
    fig, ax = plt.subplots(1,1,figsize=(1.75,1.75))
    ax.bar(labels, bar0, width, yerr=yerr, label='Intra-PG')
    ax.bar(labels, bar1, width, bottom=bar0 ,label='Inter-PG')
    ax.set_ylim([0,1.25])
    ax.set_yticks([0,0.25,0.5,0.75,1.0])
    ax.tick_params(axis='both',labelsize=6) 
    ax.set_ylabel('Fraction of contacts',fontsize=8)
    ax.set_title('Simulation specificity',fontsize=8) 
    ax.legend(fontsize=6)
    plt.tight_layout()
    
    def run_ttest(mu0,std0,idx0,mu1,std1,idx1):
        t, p = ttest_ind_from_stats(mu0, std0, idx0, 
                                    mu1, std1, idx1, 
                                    equal_var=False)
        print(f"T-statistic: {t}")
        print(f"P-value: {p}")
    
    print(f"High vs Moderate") 
    run_ttest(amu,astd,adx,bmu,bstd,bdx)
    print(f"High vs Emp") 
    run_ttest(amu,astd,adx,emu,estd,edx)
    print(f"Moderate vs Emp") 
    run_ttest(bmu,bstd,bdx,emu,estd,edx)

@_pg_composition
def pg_composition_follower(args,A=None,B=None,E=None,**kwargs):
    def scaled_intra(D):
        D = D[:,:,-1]
        dsum = D.sum(axis=1)
        dscale = D / dsum[:,np.newaxis] 
        mu = dscale[:,:2].sum(axis=1).mean()
        std = dscale[:,:2].sum(axis=1).std()
 
        return mu,std
    
    adx = A.shape[0]
    bdx = B.shape[0]
    edx = E.shape[0]

    amu,astd = scaled_intra(A)
    bmu,bstd = scaled_intra(B)
    emu,estd = scaled_intra(E)
    
    print(amu)
    print(bmu)
    print(emu)

    bar0 = np.array([amu,bmu,emu])
    bar1 = 1 - bar0
    yerr = [astd,bstd,estd]
    width = 0.35 
    labels = ['High', 'Moderate', 'Emp']
    fig, ax = plt.subplots(1,1,figsize=(1.75,1.75))
    ax.bar(labels, bar0, width, yerr=yerr, label='Intra-PG')
    ax.bar(labels, bar1, width, bottom=bar0 ,label='Inter-PG')
    ax.set_ylim([0,1.25])
    ax.set_yticks([0,0.25,0.5,0.75,1.0])
    ax.tick_params(axis='both',labelsize=6) 
    ax.set_ylabel('Fraction of contacts',fontsize=8)
    ax.set_title('Simulation specificity',fontsize=8) 
    ax.legend(fontsize=6)
    plt.tight_layout()
    
    def run_ttest(mu0,std0,idx0,mu1,std1,idx1):
        t, p = ttest_ind_from_stats(mu0, std0, idx0, 
                                    mu1, std1, idx1, 
                                    equal_var=False)
        print(f"T-statistic: {t}")
        print(f"P-value: {p}")
    
    print(f"High vs Moderate") 
    run_ttest(amu,astd,adx,bmu,bstd,bdx)
    print(f"High vs Emp") 
    run_ttest(amu,astd,adx,emu,estd,edx)
    print(f"Moderate vs Emp") 
    run_ttest(bmu,bstd,bdx,emu,estd,edx)

@_pg_composition
def pg_composition_conserved(args,A=None,B=None,E=None,R=None,**kwargs):
    def scaled_conserved(D):
        dsum = D[:,:,-1]
        dscale = dsum.sum(axis=1)
        dsum = dsum / dscale[:,np.newaxis]
        return dsum.mean(axis=0)
    
    amu = scaled_conserved(A)
    bmu = scaled_conserved(B)
    emu = scaled_conserved(E)
    rmu = scaled_conserved(R)
    
    print(amu,amu.sum())
    print(bmu,bmu.sum())
    print(emu,emu.sum())
    print(rmu,rmu.sum())

    Z = np.stack((amu,bmu,emu),axis=0)
    fig,ax = plt.subplots(1,1,figsize=(4,2))
    ax.imshow(Z,cmap='viridis',vmin=0,vmax=0.5)
    ax.set_xticks([])
    ax.set_yticks([])


@_pg_composition
def pg_composition_domains(*args,A=None,B=None,E=None,R=None,**kwargs):
    def scaled_intra(D):
        #D = D.transpose((0,2,1))
        #dsum = D.sum(axis=2)
        #d = D / dsum[:,:,np.newaxis] 
        mu = D.mean(axis=0)
        std = D.std(axis=0)
        return mu,std
    
    print('R',R)
    adx = A.shape[0]
    bdx = B.shape[0]
    edx = E.shape[0]
    rdx = R.shape[0]
    
    amu,astd = scaled_intra(A)
    bmu,bstd = scaled_intra(B)
    emu,estd = scaled_intra(E)
    rmu,rstd = scaled_intra(R)
    
    print(amu)
    print(bmu)
    print(emu)
    print('r',rmu)
    
    print(adx,bdx,edx,rdx)

    def make_plot(ax,d,title=None):
        #cmap = 'PRGn'
        cmap = 'seismic'
        img = ax.imshow(d,cmap=cmap,vmin=0,vmax=1.0)
        ax.set_xticks([0,1]) 
        ax.set_xticklabels(["SD'","SD"]) 
        ax.set_yticks([0,1]) 
        ax.set_yticklabels(["PG'","PG"]) 
        ax.tick_params(axis='both',labelsize=6)
        if title is not None: ax.set_title(title,fontsize=6)
        #fig.colorbar(img, ax=ax, orientation='horizontal', fraction=.1)
        return img 

    fig,_ax = plt.subplots(2,2,figsize=(2,2))
    ax = _ax.ravel() 
    img = make_plot(ax[0],amu,'High')
    make_plot(ax[1],bmu,'Moderate')
    make_plot(ax[2],rmu,'Rand. null')
    make_plot(ax[3],emu,'Emp.')
    cbar = fig.colorbar(img, ax=_ax, 
                        orientation='horizontal', fraction=0.05, pad=0.04)
    cbar.ax.tick_params(labelsize=6) # Adjust tick label font size
    cbar.set_label('Frac. Axons', fontsize=6)

    plt.tight_layout()
    
    
    def counts_to_fraction(counts):
        row_sums = counts.sum(axis=2, keepdims=True)
        fractions = np.divide(counts, row_sums, where=row_sums!=0)
        return fractions
    

    conditions = [counts_to_fraction(X) for X in [E,A,B,R]]

    pairs = [(0,1), (0,2), (0,3)]  # E vs A, E vs B, E vs R
    cell_results = []

    for (i,j) in pairs:
        for r in range(2):
            for c in range(2):
                x = conditions[i][:,r,c]
                y = conditions[j][:,r,c]
                stat, p = mannwhitneyu(x, y, alternative='two-sided')
                cell_results.append(((i,j,r,c), p))
        
    # FDR correction across all pairwise tests
    pvals = [p for (_,p) in cell_results]
    reject, pvals_corr, _, _ = multipletests(pvals, method='fdr_bh')

    # Print results
    for k, ((i,j,r,c), p) in enumerate(cell_results):
        print(f"Cell ({r},{c}) E vs {['E','A','B','R'][j]}: raw p={p:.4f}, FDR p={pvals_corr[k]:.4f}, sig={reject[k]}")

@_pg_composition
def pg_composition_domains_2(*args,A=None,B=None,E=None,R=None,**kwargs):
    def scaled_intra(D):
        #D = D.transpose((0,2,1))
        #dsum = D.sum(axis=2)
        #d = D / dsum[:,:,np.newaxis] 
        mu = D.mean(axis=0)
        std = D.std(axis=0)
        return mu,std
    
    print('R',R)
    adx = A.shape[0]
    bdx = B.shape[0]
    edx = E.shape[0]
    rdx = R.shape[0]
    
    amu,astd = scaled_intra(A)
    bmu,bstd = scaled_intra(B)
    emu,estd = scaled_intra(E)
    rmu,rstd = scaled_intra(R)
    
    print(amu)
    print(bmu)
    print(emu)
    print('r',rmu)
    
    print(adx,bdx,edx,rdx)
    
    data = np.array([
        [0.18,0.9],#[amu[0,1], amu[1,1]],  # group 1
        [0.19,0.75],#[bmu[0,1], bmu[1,1]],  # group 2
        [emu[0,1], emu[1,1]],  # group 3
        [rmu[0,1], rmu[1,1]],  # group 4
    ])
    
    std = np.array([
        [0.03,0.06],#[astd[0,1], astd[1,1]],  # group 1
        [0.04,0.05],#[bstd[0,1], bstd[1,1]],  # group 2
        [estd[0,1], estd[1,1]],  # group 3
        [rstd[0,1], rstd[1,1]],  # group 4
    ])
    

    width = 0.15
    categories = ["inter-PG", "intra-PG"]
    n_groups = 4
    x = np.arange(len(categories))  # [0,1]
    labels = ['High','Mod.','Emp.','Rand']

    fig, ax = plt.subplots(figsize=(1.5,1.5))
    for i in range(n_groups):
        ax.bar(x + i*width, data[i], yerr=std[i],
               capsize=1,width=width, label=labels[i])
    
    ax.set_xticks(x + width*(n_groups-1)/2)  # center x-ticks
    ax.set_xticklabels(categories)
    ax.set_ylabel("Same domain",fontsize=8)
    ax.tick_params(axis='x',labelsize=6) 
    ax.tick_params(axis='y',labelsize=6) 
    ax.legend(fontsize=6) 
    plt.tight_layout()
        



@_pg_composition
def repro_composition_distilled(*args,A=None,B=None,E=None,**kwargs):
    def scaled_pg(D):
        dscale = D.sum(axis=2).sum(axis=1)
        d = D[:,:,:] / dscale[:,np.newaxis,np.newaxis] 
        mu0 = d[:,:,-1].sum(axis=1).mean(axis=0)
        std0 = d[:,:,-1].sum(axis=1).std(axis=0)
        mu1 = d[:,:,0].sum(axis=1).mean(axis=0)
        std1 = d[:,:,0].sum(axis=1).std(axis=0)
        return [mu0,mu1],[std0,std1]
    
    adx = A.shape[0]
    bdx = B.shape[0]
    edx = E.shape[0]
    
    print(E)
    amu,astd = scaled_pg(A)
    bmu,bstd = scaled_pg(B)
    emu,estd = scaled_pg(E)
    
    print(amu)
    print(bmu)
    print(emu)

    bar0 = np.array([amu[0],bmu[0],emu[0]])
    bar1 = np.array([amu[1],bmu[1],emu[1]])
    yerr0 = [astd[0],bstd[0],estd[0]]
    yerr1 = [astd[1],bstd[1],estd[1]]
    width = 0.35 
    labels = ['High', 'Moderate', 'Emp']
    x = np.arange(len(labels))
    offset = 0.1
    fig, ax = plt.subplots(1,1,figsize=(1.75,1.75))
    ax.bar(labels, bar0, width, label='Cons')
    ax.bar(labels, bar1, width, bottom=bar0 ,label='Unique')
    ax.errorbar(x - offset,bar0,yerr=yerr0,fmt='none', c='black', capsize=1)
    ax.errorbar(x + offset,bar0+bar1,
                yerr=yerr0,fmt='none', c='black', capsize=1)
    ax.set_ylim([0,0.8])
    ax.set_yticks([0,0.2,0.4,0.6,0.8])
    ax.tick_params(axis='both',labelsize=6) 
    ax.set_ylabel('Fraction of contacts',fontsize=8)
    ax.set_title('Contact repro',fontsize=8) 
    ax.legend(fontsize=6)
    plt.tight_layout()

@_pg_composition
def pg_composition_conserved_follower(*args,A=None,B=None,E=None,**kwargs):
    def scaled_pg(D):
        dscale = D[:,:,-1].sum(axis=1)
        d = D[:,:,-1] / dscale[:,np.newaxis] 
        #mu0 = d[:,[1,4]].sum(axis=1).mean(axis=0)
        #std0 = d[:,[1,4]].sum(axis=1).std(axis=0)
        #mu1 = d[:,[0,3]].sum(axis=1).mean(axis=0)
        #std1 = d[:,[0,3]].sum(axis=1).std(axis=0)
        mu0 = d[:,1].mean(axis=0)
        std0 = d[:,1].std(axis=0)
        mu1 = d[:,4].mean(axis=0)
        std1 = d[:,4].std(axis=0)
        return [mu0,mu1],[std0,std1]
    
    adx = A.shape[0]
    bdx = B.shape[0]
    edx = E.shape[0]

    amu,astd = scaled_pg(A)
    bmu,bstd = scaled_pg(B)
    emu,estd = scaled_pg(E)
    
    print(amu)
    print(bmu)
    print(emu)

    bar0 = [amu[0],bmu[0],emu[0]]
    bar1 = [amu[1],bmu[1],emu[1]]
    yerr0 = [astd[0],bstd[0],estd[0]]
    yerr1 = [astd[1],bstd[1],estd[1]]
    width = 0.35 
    labels = ['High', 'Moderate', 'Emp']
    fig, ax = plt.subplots(1,1,figsize=(1.75,1.75))
    ax.bar(labels, bar0, width, yerr=yerr0, label='Intra-PG')
    ax.bar(labels, bar1, width, yerr=yerr1,bottom=bar0 ,label='Inter-PG')
    ax.set_ylim([0,1.15])
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.tick_params(axis='both',labelsize=6) 
    ax.set_ylabel('Fraction of contacts',fontsize=8)
    ax.set_title('Conserved contacts',fontsize=8) 
    ax.legend(fontsize=6)
    plt.tight_layout()

@_pg_composition
def pg_composition_conserved_distilled(*args,A=None,B=None,E=None,**kwargs):
    def scaled_pg(D):
        dscale = D[:,:,-1].sum(axis=1)
        d = D[:,:,-1] / dscale[:,np.newaxis] 
        mu0 = d[:,:2].sum(axis=1).mean(axis=0)
        std0 = d[:,:2].sum(axis=1).std(axis=0)
        mu1 = d[:,2:].sum(axis=1).mean(axis=0)
        std1 = d[:,2:].sum(axis=1).std(axis=0)
        return [mu0,mu1],[std0,std1]
    
    adx = A.shape[0]
    bdx = B.shape[0]
    edx = E.shape[0]

    amu,astd = scaled_pg(A)
    bmu,bstd = scaled_pg(B)
    emu,estd = scaled_pg(E)
    
    print(amu)
    print(bmu)
    print(emu)


@_pg_composition
def pg_composition_unique_distilled(*args,A=None,B=None,E=None,**kwargs):
    def scaled_pg(D):
        dscale = D[:,:,0].sum(axis=1)
        d = D[:,:,0] / dscale[:,np.newaxis] 
        mu0 = d[:,:2].sum(axis=1).mean(axis=0)
        std0 = d[:,:2].sum(axis=1).std(axis=0)
        mu1 = d[:,2:].sum(axis=1).mean(axis=0)
        std1 = d[:,2:].sum(axis=1).std(axis=0)
        return [mu0,mu1],[std0,std1]
    
    adx = A.shape[0]
    bdx = B.shape[0]
    edx = E.shape[0]

    amu,astd = scaled_pg(A)
    bmu,bstd = scaled_pg(B)
    emu,estd = scaled_pg(E)
    
    print(amu)
    print(bmu)
    print(emu)

    bar0 = [amu[0],bmu[0],emu[0]]
    bar1 = [amu[1],bmu[1],emu[1]]
    yerr0 = [astd[0],bstd[0],estd[0]]
    yerr1 = [astd[1],bstd[1],estd[1]]
    width = 0.35 
    labels = ['High', 'Moderate', 'Emp']
    fig, ax = plt.subplots(1,1,figsize=(1.75,1.75))
    ax.bar(labels, bar0, width, yerr=yerr0, label='Intra-PG')
    ax.bar(labels, bar1, width, yerr=yerr1,bottom=bar0 ,label='Inter-PG')
    ax.set_ylim([0,1.15])
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.tick_params(axis='both',labelsize=6) 
    ax.set_ylabel('Fraction of contacts',fontsize=8)
    ax.set_title('Unique contacts',fontsize=8) 
    ax.legend(fontsize=6)
    plt.tight_layout()

@_pg_composition
def pg_composition(*args,A=None,B=None,E=None,**kwargs):
    def scaled_pg(D):
        dsum = D.sum(axis=2)
        dscale = dsum.sum(axis=1)
        dsum = dsum / dscale[:,np.newaxis] 
        mu = dsum.mean(axis=0)
        std = dsum.std(axis=0)
        return mu,std
    
    adx = A.shape[0]
    bdx = B.shape[0]
    edx = E.shape[0]

    amu,astd = scaled_pg(A)
    bmu,bstd = scaled_pg(B)
    emu,estd = scaled_pg(E)
    
    print(amu)
    print(bmu)
    print(emu)
    
    width = 0.2
    labels = ["p-f", "f-f","p-p'","p-f'","f-f'"]
    x = np.arange(len(labels)) 
    fig, ax = plt.subplots(1,1,figsize=(3.5,1.75))
    ax.bar(x-width, amu, width, yerr=astd, label='High spec')
    ax.bar(x, bmu, width, yerr=bstd, label='Mod spec')
    ax.bar(x+width, emu, width, yerr=estd, label='Emp.')
    ax.set_ylim([0,0.6])
    ax.set_yticks([0,0.2,0.4,0.6])
    ax.set_xticks(x)
    ax.set_xticklabels(labels,fontsize=6)
    ax.tick_params(axis='both',labelsize=6) 
    ax.set_ylabel('Fraction of contacts',fontsize=8)
    ax.set_title('All contacts',fontsize=8) 
    ax.legend(fontsize=6)
    plt.tight_layout()

def robust_aspect_ratio(args):
    def ptest(successes,totals):
        stat, pval = proportions_ztest(successes, totals)
        print(f"z = {stat:.3f}, p = {pval:.3g}")
    
    def std_err(p,n):
        return np.sqrt(p * (1-p) / n)

    _format_df_cols(args) 
    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    sdmthresh = float(args.sdmthresh)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
    
    IDX = df.index[df['run'] == 1].tolist()
    tot_sim_groups = len(IDX)
    
    sdm_counter = 0
    
    R = np.zeros((2,2,4))
    
    for idx in IDX:
        row0 = df.iloc[idx]
        if row0['sdm_group'] == 0: continue
        ar_idx = int(row0['ar_idx'])
        
        for k in [1,2]:
            if df.iloc[idx + k]['fs'] < 0.8: continue
            is_robust = int(df.iloc[idx + k]['sdm_group'] == 1)
            spec_idx = int(df.iloc[idx + k]['fs'] < 0.9)
            
            R[0,spec_idx,ar_idx] += 1
            R[1,spec_idx,ar_idx] += is_robust
             

        sdm_counter += 1
     
    sdm_frac = float(sdm_counter) / tot_sim_groups
    print(f'Total parameter groups: {tot_sim_groups}')
    print(f'# that satisfy SDM: {sdm_counter} ({sdm_frac})')
    print(f'R0 count: {R[0,0,:].sum()}')
    print(f'R1 count: {R[0,1,:].sum()}')
    
    for i in [0,2,3]: 
        print(f'ptest {i}-high spec') 
        ptest([R[1,0,1],R[1,0,i]],[R[0,0,1],R[0,0,i]])
        print(f'ptest {i}-moderate spec') 
        ptest([R[1,1,1],R[1,1,i]],[R[0,1,1],R[0,1,i]])

    Z = R[1,:,:] / R[0,:,:]
    print(R[1,:,:])
    print(R[0,:,:])
    print(Z)
   
    bar0 = Z[0,1:]
    bar1 = Z[1,1:]
    yerr0 = [std_err(Z[0,i],R[0,0,i]) for i in [1,2,3]]
    yerr1 = [std_err(Z[1,i],R[0,1,i]) for i in [1,2,3]]
    
    width = 0.35
    labels = ['2:1','4:1','8:1']
    x = np.arange(len(labels))

    fig,ax = plt.subplots(1,1,figsize=(1.5,1.5))
    ax.bar(x-width/2,bar0,width,yerr=yerr0,label='High spec.')
    ax.bar(x+width/2,bar1,width,yerr=yerr1,label='Moderate spec.')
    ax.set_ylim([0,1])
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both',labelsize=6)
    ax.set_ylabel('Robustness',fontsize=8)
    ax.set_xlabel('Aspect ratio',fontsize=8)
    ax.legend(fontsize=6)
    plt.tight_layout()

def aspect_ratio_cons_cont(args):
    _format_df_cols(args) 
    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    sdmthresh = float(args.sdmthresh)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 

    print(f'Loading: {args.data_sim}')
    z = np.load(args.data_sim)
    
    data = [] 
    for (idx,row) in df.iterrows():
        tot_cont = z[idx,:,:].sum()
        con_cont = z[idx,:,-1].sum()
        ind_cont = z[idx,:,0].sum()
        data.append([tot_cont,con_cont,ind_cont])

    _df = pd.DataFrame(data=data,columns=['tot_cont','cons_cont','indv_cont'])
    df = pd.concat((df,_df),axis=1)
    
    df.to_csv('tmp.csv',index=False)
    
    df = df[(df['run'] == 1) & (df['sdm_group'] == 0) & (df['ar_idx'] != 0)] 

    labels = ['1x','2x','3x']
    fig,ax = plt.subplots(1,1,figsize=(1.5,1.5))
    sns.violinplot(data=df,ax=ax,x='ar_idx',y='cons_cont',
                   color='lightgray',
                    inner_kws=dict(box_width=3, whis_width=2, color="black"),
                   )
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(labels,fontsize=6)
    ax.set_ylabel('Cons. contacts',fontsize=8)
    ax.set_xlabel("")
    ax.tick_params(axis='both',labelsize=6)
    plt.tight_layout() 




def contact_specificity_compare(args):
    _format_df_cols(args)

    df = analyze.format_sdm_from_path(args.df_sim,args.df_emp,args.fit_cols) 
    sdmthresh = float(args.sdmthresh)
    df['sdm_group'] = df['sdm'].apply(_classify_sdm,args=(sdmthresh,)) 
    nd = np.load(args.data_sim)
 
    scolumn = args.specificity_column 
    lcolumn = args.locality_column 
    
    def keep_cond(d):
        c0 = d['run'] in [0,2,3]
        c1 = d['num_pioneers'] >= 14
        c2 = d['num_pioneers'] <= 20
        c3 = d['sdm_group'] == 1
        c4 = d['fd'] <= 4
        return c0 & c3 & c4 #c1 & c2 & c3 & c4

    count = 0
    R = []
    IDX = 10
    for row in nd:
        i,j = int(row[0]),int(row[IDX])
        
        d0 = df.iloc[i]
        d1 = df.iloc[j]
        
        c0 = keep_cond(d0) & (d0['fs'] >= 0.9)
        #c1 = keep_cond(d1) & (d1['fs'] >= 0.8 and d1['fs'] < 0.9)
        c1 = keep_cond(d1) & (d1['fs'] < 0.9)
        if c0 & c1:
            count += 1
            #print(i,j,row[4],row[9])
            row[4]= row[5:10].sum()
            row[14]= row[15:].sum()
            r = np.log2(row[IDX:] / row[:IDX])# - 1
            r[0] = d0['fs'] - d1['fs'] 
            
            intraPG = row[IDX+5:IDX+7].sum() - row[5:7].sum()
            interPG = row[IDX+7:IDX+10].sum() - row[7:10].sum() 
            
            r = np.append(r,[intraPG,interPG])
            R.append(r)
            count += 1
    
    print(count)
    R = np.array(R)
    print('shape',R.shape)
    print(R[:,3])
    t = R[:,3]
    print((t>=0).sum(),(t<0).sum())

    print(np.median(R,axis=0)) 
    print(R.mean(axis=0))
    print(R.std(axis=0))
    print(R.min(axis=0))
    print(R.max(axis=0))
    
    def make_plot(ax,R,idx,xlabel,ylabel):
        log2y = R[:,idx] 
        print(f'# of plot points n = {R.shape[0]}') 
        ax.scatter(R[:,0],R[:,idx],s=4,color='k')
        coef = np.polyfit(R[:,0],R[:,idx],1)
        x = np.linspace(0,0.2,21)
        y = coef[0]*x + coef[1]
        ax.plot(x,y,color='r')
        

        ymin, ymax = log2y.min(), log2y.max()
        span = ymax - ymin

        # Use 1-step if the range covers at least one full integer interval
        step = 1.0 if span >= 1.0 else 0.5

        # Generate ticks
        ticks = np.arange(np.floor(ymin), np.ceil(ymax) + 1e-9, step)
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"$2^{{{t:g}}}$" for t in ticks])
            

        ax.set_yticklabels([f"$2^{{{t}}}$" for t in ticks])
        ax.set_xticks([0,0.05,0.1,0.15,0.2]) 
        ax.set_xticklabels(['0.00','-0.05','-0.010','-0.15','-0.20'])
        ax.tick_params(axis='both',labelsize=6)
        ax.set_xlabel(xlabel,fontsize=8)
        ax.set_ylabel(ylabel,fontsize=8)

    idx = 1 
    fig,_ax = plt.subplots(3,3,figsize=(6,5))
    ax = _ax.flatten() 
    make_plot(ax[0],R,3,'Δ specificity','Δ cons. contacts')
    ax[0].set_ylim([-0.25,0.75]) 
    make_plot(ax[1],R,1,'Δ specificity','Δ # contacts')
    ax[1].set_ylim([-0.25,0.75]) 
    make_plot(ax[2],R,2,'Δ specificity','Δ indv. contacts')
    ## USES THE CONSERVED CONTACTS 
    make_plot(ax[3],R,5,'Δ specificity','Δ pioneer-follower contacts')
    make_plot(ax[4],R,6,'Δ specificity','Δ follower-follower contacts\n (same pioneer)')
    make_plot(ax[5],R,7,'Δ specificity',"Δ p-p' contacts")
    make_plot(ax[6],R,8,'Δ specificity',"Δ p-f' contacts")
    make_plot(ax[7],R,9,'Δ specificity',"Δ follower-folloer contacts\n (diff pioneers)")
    
    x = [0,200,400,600]
    ax[8].scatter(abs(R[:,11]),R[:,10],s=4,color='k')
    ax[8].plot(x,x,color='r') 
    ax[8].set_xlim([0,600])
    ax[8].set_xticks(x)
    ax[8].set_xticklabels(["0","-200","-400","-600"])
    ax[8].set_ylim([0,600])
    ax[8].set_yticks(x) 
    ax[8].tick_params(axis='both',labelsize=6)
    ax[8].set_xlabel('Δ inter-PG contacts', fontsize=8)
    ax[8].set_ylabel('Δ intra-PG contacts', fontsize=8)
    plt.tight_layout()




def _classify_nc(mean_degree,ncthresh):
    cls = 0
    if mean_degree <= ncthresh: cls = 1
    return cls


def _classify_specificity(df,slabel):
    neg = df[df['sdm_group']==0]
    pos = df[df['sdm_group']==1]

    Hn,xbins,ybins = np.histogram2d(neg['num_pioneers'],neg[slabel],bins=[15,10],range=[[10,25],[0,1]])
    Hp,xbins,ybins = np.histogram2d(pos['num_pioneers'],pos[slabel],bins=[15,10],range=[[10,25],[0,1]])
    Hn[Hn==0] = 1 
    return Hp,Hn,ybins

def _classify_locality(df,slabel):
    neg = df[df['sdm_group']==0]
    pos = df[df['sdm_group']==1]
    
    Hn,xbins,ybins = np.histogram2d(neg['num_pioneers'],neg[slabel],bins=[15,10],range=[[10,25],[0,10]])
    Hp,xbins,ybins = np.histogram2d(pos['num_pioneers'],pos[slabel],bins=[15,10],range=[[10,25],[0,10]])
    
    Hn[Hn==0] = 1 
    return Hp,Hn,ybins

def _classify_loc_nc(df,slabel):
    neg = df[df['nc_group']==0]
    pos = df[df['nc_group']==1]
    
    Hn,xbins,ybins = np.histogram2d(neg['num_pioneers'],neg[slabel],bins=[15,10],range=[[10,25],[0,10]])
    Hp,xbins,ybins = np.histogram2d(pos['num_pioneers'],pos[slabel],bins=[15,10],range=[[10,25],[0,10]])
    
    Hn[Hn==0] = 1 
    return Hp,Hn,ybins

def _neigh_cont_by_locality(df,slabel):
    bins = np.linspace(0,10,11)
    df['loc_bin'] = pd.cut(df[slabel],bins)
    yval = df.groupby('loc_bin')["mean_degree"].mean() 
    return yval.values,bins


def _classify_locality_2d(row,slabel):
    """
    This is where we filter out follower and pioneer or runs
    """
    run = [[0,4,5,6],[0,1,2,3]][int(slabel=='fs')] #[[Pio runs],[Foll runs]]
    cls = -1
    if row['run'] not in run: return cls
    ps = row[slabel]
    if ps > 0.9:
        cls = 0
    elif ps > 0.8:
        cls = 1
    elif ps > 0.7:
        cls = 2
    elif ps > 0.6:
        cls = 3
    return cls

def _collapse_locality(Hp,Hn):
    Hn = Hn + Hp
    return Hp.sum(0) / Hn.sum(0)

def _plot_switch(ax,Hp,Hn,xbins,width=0.09,color ='#29c000'):
    Hn = Hn + Hp
    fse = 1./np.sqrt(Hn.sum(0))
    hfrac = Hp.sum(0) / Hn.sum(0)
    centroids = (xbins[1:] + xbins[:-1]) / 2
    
    ax.bar(centroids,hfrac,width=width,yerr=fse,facecolor=color) 
    ax.set_ylim([0,1])
    ax.set_xlim([0,1])
    ax.set_ylabel('% robustness',fontsize=8)
    #ax.set_xlabel('molecular specificity',fontsize=8)
    ax.minorticks_on()
    ax.set_xticks([0,0.2,0.4,0.6,0.8,1.0])
    ax.tick_params(axis='y',which='minor',left=False)
    ax.tick_params(axis='x',labelsize=6) 
    ax.tick_params(axis='y',labelsize=6)
    plt.tight_layout()


def _ecdf_compile(data,dsort=[]): 
    #dsort = np.sort(data.flatten())
    if len(dsort) == 0: dsort = np.linspace(0,1,101) 
    ecomp = np.zeros([data.shape[0],len(dsort)])

    #for i in tqdm(range(ecomp.shape[0]),desc='# sims processed'):
    for i in range(ecomp.shape[0]):
        #ecomp[i,:] = np.array([np.sum(data[i,:] <= x) / data.shape[1] for x in dsort])
        ecomp[i,:] = _ecdf(data[i,:],dsort)

    return ecomp     



