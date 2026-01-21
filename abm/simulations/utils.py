"""                            
@name: abm.simulations.utils.py
@description:                  
    Utility functions for simulations module

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""
import numpy as np
from pycsvparser import read

def _format_df_cols(args):
    args.fit_cols = [r for r in read.into_list(args.fit_cols) 
                     if not r[0] == '#']

def _classify_sdm(sdm,sdmthresh):
    cls = 0
    if sdm <= sdmthresh: cls = 1
    return cls

def _ecdf(data,dsort):
    return np.array([np.sum(data <= x) / len(data) for x in dsort])

def _ecdf_range(ecdf,num_std=2):
    emean = ecdf.mean(axis=0)
    estd = ecdf.std(axis=0)
    emin = emean - num_std*estd
    emax = emean + num_std*estd
    
    return emin,emax



