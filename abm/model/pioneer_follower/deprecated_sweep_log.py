"""
@name: sweep_log.py                     
@description:                  
    Module for building chemotropic models sweep logs

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""
import os
from configparser import ConfigParser,ExtendedInterpolation
import random
import csv
from collections import defaultdict,namedtuple
import itertools
import numpy as np

def build_sweep(config):
    header,combos = builder(config)
    seeds = get_seeds(config)
    sweep = []
    pg = 0
    for c in combos:
        for (run,seed) in enumerate(seeds):
            fout = f'run_{pg}_{run}.pkl'
            sweep.append([pg,run] + c + seed.tolist() + [fout])
        pg += 1
    
    return [header] + sweep

def get_seeds(config):
    max_seed_val = config.getint('sweep','max_seed_val')
    sweep_depth = config.getint('sweep','depth')
    seed = np.random.randint(0,max_seed_val,size=(sweep_depth,2))
    return seed

def builder(config):
    header = ['parameter_group','run','num_pioneers','precision','locality','num_attractors','seed_pioneer','seed_agent','file']
    min_num_attractors = config.getint('sweep','min_num_attractors')
    max_num_attractors = config.getint('sweep','max_num_attractors')
    min_pioneers = config.getint('sweep','min_pioneers')
    max_pioneers = config.getint('sweep','max_pioneers')
    vary_precision = config.getboolean('sweep','vary_precision')
    precision = [1.]
    if vary_precision:
        precision = np.around(np.linspace(0,1.,6),decimals=2)
    combos = []
    min_locality = config.getint('sweep','min_locality') 
    max_locality = config.getint('sweep','max_locality') 

    for k in range(min_num_attractors,max_num_attractors+1):
        for p in range(min_pioneers,max_pioneers+1):
            if k > p: continue
            #No need for precision if num_attractors or num_pioneers is 0, set default to 1
            #No need for locality if num_attractors or num_pioneers is 0, set default to 0
            if k == 0 or p == 0:
                combos.append([p,1.0,0,k])
            else:
                for pr in precision:
                    _max_loc = min(p-1,max_locality)
                    for l in range(min_locality,_max_loc+1):
                        combos.append([p,pr,l,k])


    return header,combos

def modify_config(cfg,sweep):
    cfg['model']['num_pioneers'] = str(sweep[2])
    cfg['model']['agent_precision'] = str(sweep[3])
    cfg['model']['agent_locality'] = str(sweep[4])
    cfg['agent']['num_attractors'] = str(sweep[5]) 
    cfg['grid']['pioneer_seed'] = str(sweep[6])
    cfg['agent']['pos_seed']=str(sweep[7])

def _builder(config):
    header = ['parameter_group','run','num_pioneers','num_attractors','seed_pioneer','seed_agent','file']
    max_num_attractors = config.getint('sweep','max_num_attractors')
    max_pioneers = config.getint('sweep','max_pioneers')
    combos = []
    for k in range(max_num_attractors):
        for p in range(max_pioneers):
            if k > p: continue
            combos.append([p,k])
    return header,combos
    
def _modify_config(cfg,sweep):
    cfg['model']['num_pioneers'] = str(sweep[2])
    cfg['grid']['pioneer_seed'] = str(sweep[4])
    cfg['agent']['pos_seed']=str(sweep[5])
    cfg['agent']['num_attractors'] = str(sweep[3]) 

