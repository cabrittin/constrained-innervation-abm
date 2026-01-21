"""
@name: analysis.py                        
@description:                  
    
    Module for analysis of simulation data

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity


def format_sdm_from_path(sim_df,emp_df,fit_cols):
    """
    Format simulation distance metric from dataframe paths.
    Syntactic sugar. 

    Args:
    -----
    sim_df: str
        Path to simulation dataframe
    emp_df: str
        Path to empirical dataframe
    fit_cols: list of str
        Columns of dataframe usde to compute SDM

    """
    emp = pd.read_csv(emp_df) 
    sim = pd.read_csv(sim_df)
    return format_sdm(sim,emp,fit_cols) 

def format_sdm(sim,emp,fit_cols):
    """
    Format simulation distance metric

    Args:
    -----
    sim_df: Simulation dataframe
    emp_df: Empirical dataframe
    fit_cols: list of str
        Columns of dataframe usde to compute SDM

    """
    sim['sdm'] = sim_dist_metric(sim[fit_cols],emp[fit_cols].mean(axis=0)) 
    #sim['num_targets'] = sim['target_split_low'] + sim['target_split_high']
    return sim 


def sim_dist_metric(sim,emp):
    """
    Computes the simulation distance metric between simulated and empirical data
    """ 
    xsim = sim.to_numpy() 
    xemp = emp.to_numpy() 
    stdsim = xsim.std(0) 
    zemp = np.tile(xemp,(xsim.shape[0],1))
    sdm = np.linalg.norm((xsim-zemp)/stdsim[None,:],axis=1)
    #sdm = sdm / np.sqrt(zsim.shape[1])
    return sdm


def get_local_extreme(data, type_extreme = 'max'):
    """
    This method returns the indeces where there is a change in the trend of the input series.
    type_extreme = None returns all extreme points, max only maximum values and min
    only min,
    """
    a = np.diff(data)
    asign = np.sign(a)
    signchange = ((np.roll(asign, 1) - asign) != 0).astype(int)
    idx = np.where(signchange ==1)[0]

    if type_extreme == 'max' and data[idx[0]] < data[idx[1]]:
        idx = idx[1:][::2]
    elif type_extreme == 'min' and data[idx[0]] > data[idx[1]]:
        idx = idx[1:][::2]
    elif type_extreme is not None:
        idx = idx[::2]
    
    # sort ids by min value
    if 0 in idx:
        idx = np.delete(idx, 0)
    if (len(data)-1) in idx:
        idx = np.delete(idx, len(data)-1)
    idx = idx[np.argsort(data[idx])]
    return idx

def get_sdm_thresh(df,bandwidth=0.1):
    x = np.linspace(-0.5,4.5,500)
    sdm = df['sdm'].to_numpy()
    kde = KernelDensity(bandwidth=bandwidth, kernel='gaussian')
    kde.fit(sdm[:,np.newaxis])
    sdm_density = np.exp(kde.score_samples(x[:,np.newaxis]))
    locmin = get_local_extreme(sdm_density,type_extreme='min')
    idx = np.sort(locmin)[0] 
    return x[idx]
