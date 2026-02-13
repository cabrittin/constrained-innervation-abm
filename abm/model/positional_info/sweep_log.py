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
from tqdm import tqdm

NUMSEEDS = 6
HEADER = [
        'parameter_group',
        'run',
        'num_pioneers',
        'average_response_rate',
        'locality_mean',
        'locality_std',
        'pioneer_flips',
        'pioneer_locality_mean',
        'pioneer_locality_std',
        'seed_pioneer',
        'seed_agent',
        'response_seed',
        'locality_seed',
        'pioneer_flips_seed',
        'pioneer_locality_seed',
        'file'
        ]



def build_sweep(config):
    combos = builder(config)
    seeds = get_seeds(config)
    sweep = []
    pg = 0
    for c in tqdm(combos,total=len(combos),desc="Combos:"):
        for (run,seed) in enumerate(seeds):
            fout = f'run_{pg}_{run}.pkl'
            sweep.append([pg,run] + c + seed.tolist() + [fout])
        pg += 1
    
    return [HEADER] + sweep


def build_sampler(config):
    max_seed_val = config.getint('sweep','max_seed_val')
    sweep_depth = config.getint('sweep','depth')
    rng = np.random.default_rng()
    seed = rng.integers(0,max_seed_val,size=(sweep_depth,NUMSEEDS))
    
    rp = list(map(int,config['sweep']['range_pioneers'].split(',')))
    rr = list(map(int,config['sweep']['range_response'].split(',')))
    rlm = list(map(int,config['sweep']['range_locality_mean'].split(',')))
    rls = list(map(int,config['sweep']['range_locality_std'].split(',')))
    
    pf = list(map(int,config['sweep']['range_pioneers_flips'].split(',')))
    plm = list(map(int,config['sweep']['range_pioneers_locality_mean'].split(',')))
    pls = list(map(int,config['sweep']['range_pioneers_locality_std'].split(',')))
    
    sampler = config['sweep']['sampler']
    S = globals()[sampler](sweep_depth,rng,seed,rp,rr,rlm,rls,pf,plm,pls)

    return [HEADER] + S


def sampler_general(sweep_depth,rng,seed,rp,rr,rlm,rls,pf,plm,pls):
    S = []
    for i in tqdm(range(sweep_depth),desc='Samples'):
        p = rng.integers(rp[0],rp[1]+1)
        s = rng.integers(rr[0],p)
        lm = rng.integers(rlm[0],p)
        ls = rng.integers(rls[0],rls[1]+1)
        _pf = rng.integers(pf[0],p)
        _plm = rng.integers(plm[0],max(p-1,1))
        _pls = rng.integers(pls[0],pls[1]+1)

        fout = f'run_{i}_0.pkl'
    
        S.append([i,0,p,s,lm,ls,_pf,_plm,_pls] + seed[i,:].tolist() + [fout])
    
    return S

def sampler_locality(sweep_depth,rng,seed,rp,rr,rlm,rls,pf,plm,pls):
    """
    Cases:
    ------
    0: High specificity, high locality (control)
    1: High specificity, variable locality
    2: 90% specificity, variable locality
    3: 80% specificity, variable locality 
    4: 90% pioneer specificity, variable locality
    5: 80% pioneer specificity, variable locality 

    Follower and pioneer specificity draws will be identical
    Follower and pionner locality draws will be identical
    """

    S = []
    for i in tqdm(range(sweep_depth),desc='Samples'):
        p = rng.integers(rp[0],rp[1]+1)
        #s = rng.integers(rr[0],p)
        lm = rng.integers(rlm[0],p)
        #ls = rng.integers(rls[0],rls[1]+1)
        #_pf = rng.integers(pf[0],p)
        #_plm = rng.integers(plm[0],max(p-1,1))
        #_pls = rng.integers(pls[0],pls[1]+1)
        
        s9 = int(np.round(0.1*p))
        s8 = int(np.round(0.2*p))

        fout = f'run_{i}_0.pkl'
        S.append([i,0,p,0,0,1,0,0,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_1.pkl'
        S.append([i,1,p,0,lm,1,0,0,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_2.pkl'
        S.append([i,2,p,s9,lm,1,0,0,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_3.pkl'
        S.append([i,3,p,s8,lm,1,0,0,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_4.pkl'
        S.append([i,4,p,0,0,1,0,lm,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_5.pkl'
        S.append([i,5,p,0,0,1,s9,lm,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_6.pkl'
        S.append([i,6,p,0,0,1,s8,lm,1] + seed[i,:].tolist() + [fout])
        
    
    return S

def sampler_locality_2(sweep_depth,rng,seed,rp,rr,rlm,rls,pf,plm,pls):
    """
    Cases:
    ------
    0: High specificity, high locality (control)
    1: High specificity, variable locality
    2: 70% specificity, variable locality
    3: 60% specificity, variable locality 
    4: 70% pioneer specificity, variable locality
    5: 60% pioneer specificity, variable locality 

    Follower and pioneer specificity draws will be identical
    Follower and pionner locality draws will be identical
    """

    S = []
    for i in tqdm(range(sweep_depth),desc='Samples'):
        p = rng.integers(rp[0],rp[1]+1)
        #s = rng.integers(rr[0],p)
        lm = rng.integers(rlm[0],p)
        #ls = rng.integers(rls[0],rls[1]+1)
        #_pf = rng.integers(pf[0],p)
        #_plm = rng.integers(plm[0],max(p-1,1))
        #_pls = rng.integers(pls[0],pls[1]+1)
        
        s7 = int(np.round(0.3*p))
        s6 = int(np.round(0.4*p))

        fout = f'run_{i}_0.pkl'
        S.append([i,0,p,0,0,1,0,0,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_1.pkl'
        S.append([i,1,p,0,lm,1,0,0,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_2.pkl'
        S.append([i,2,p,s7,lm,1,0,0,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_3.pkl'
        S.append([i,3,p,s6,lm,1,0,0,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_4.pkl'
        S.append([i,4,p,0,0,1,0,lm,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_5.pkl'
        S.append([i,5,p,0,0,1,s7,lm,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_6.pkl'
        S.append([i,6,p,0,0,1,s6,lm,1] + seed[i,:].tolist() + [fout])
        
    
    return S


def sampler_locality_pioneer(sweep_depth,rng,seed,rp,rr,rlm,rls,pf,plm,pls):
    """
    Focuses on pioneer locality with more dramatic changes in pioneer specificity

    Cases:
    ------
    0: High specificity, high locality (control)
    1: Repeat control
    2: High specificity, variable locality
    3: 80% pioneer specificity, variable locality 
    4: 60% pioneer specificity, variable locality
    5: 40% pioneer specificity, variable locality 
    6: 20% pioneer specificity, variable locality 
    7: 0% pioneer specificity, variable locality 

    Follower and pioneer specificity draws will be identical
    Follower and pionner locality draws will be identical
    """

    S = []
    for i in tqdm(range(sweep_depth),desc='Samples'):
        p = rng.integers(rp[0],rp[1]+1)
        #s = rng.integers(rr[0],p)
        lm = rng.integers(rlm[0],p)
        #ls = rng.integers(rls[0],rls[1]+1)
        #_pf = rng.integers(pf[0],p)
        #_plm = rng.integers(plm[0],max(p-1,1))
        #_pls = rng.integers(pls[0],pls[1]+1)
        
        sp = [int(np.round(f*p)) for f in [0.2,0.4,0.6,0.8,1.0]]

        fout = f'run_{i}_0.pkl'
        S.append([i,0,p,0,0,1,0,0,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_1.pkl'
        S.append([i,1,p,0,0,1,0,0,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_2.pkl'
        S.append([i,2,p,0,0,1,0,lm,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_3.pkl'
        S.append([i,3,p,0,0,1,sp[0],lm,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_4.pkl'
        S.append([i,4,p,0,0,1,sp[1],lm,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_5.pkl'
        S.append([i,5,p,0,0,1,sp[2],lm,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_6.pkl'
        S.append([i,6,p,0,0,1,sp[3],lm,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_7.pkl'
        S.append([i,7,p,0,0,1,sp[4],lm,1] + seed[i,:].tolist() + [fout])
    
    return S


def sampler_specificity(sweep_depth,rng,seed,rp,rr,rlm,rls,pf,plm,pls):
    """
    Cases:
    ------
    0: High specificity, high locality
    1: Variable specificity, high locality
    2: Variable pioneer specificity, high locality
    
    Follower and pioneer specificity draws will be identical
    """

    S = []
    for i in tqdm(range(sweep_depth),desc='Samples'):
        p = rng.integers(rp[0],rp[1]+1)
        s = rng.integers(rr[0],p)

        fout = f'run_{i}_0.pkl'
        S.append([i,0,p,0,0,1,0,0,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_1.pkl'
        S.append([i,1,p,s,0,1,0,0,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_2.pkl'
        S.append([i,2,p,0,0,1,s,0,1] + seed[i,:].tolist() + [fout])
        
    return S

def sampler_specificity_bimodal_locality(sweep_depth,rng,seed,rp,rr,rlm,rls,pf,plm,pls):
    """
    Cases:
    ------
    0: High specificity, high locality
    1: Variable specificity, high locality
    2: Variable specificity, bimodal locality
    
    Follower and pioneer specificity draws will be identical
    """

    S = []
    for i in tqdm(range(sweep_depth),desc='Samples'):
        p = rng.integers(rp[0],rp[1]+1)
        s = rng.integers(rr[0],p)

        fout = f'run_{i}_0.pkl'
        S.append([i,0,p,0,0,1,0,0,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_1.pkl'
        S.append([i,1,p,s,0,1,0,0,1] + seed[i,:].tolist() + [fout])
        
        fout = f'run_{i}_2.pkl'
        S.append([i,2,p,s,0,99,0,0,1] + seed[i,:].tolist() + [fout])
        
    return S


def get_seeds(config):
    max_seed_val = config.getint('sweep','max_seed_val')
    sweep_depth = config.getint('sweep','depth')
    seed = np.random.randint(0,max_seed_val,size=(sweep_depth,NUMSEEDS))
    return seed

def builder(config):
    _rp = list(map(int,config['sweep']['range_pioneers'].split(',')))
    rp = range(_rp[0],_rp[1]+1) if len(_rp) == 2 else _rp
    rr = list(map(int,config['sweep']['range_response'].split(',')))
    rlm = list(map(int,config['sweep']['range_locality_mean'].split(',')))
    rls = list(map(int,config['sweep']['range_locality_std'].split(',')))
    
    pf = list(map(int,config['sweep']['range_pioneers_flips'].split(',')))
    plm = list(map(int,config['sweep']['range_pioneers_locality_mean'].split(',')))
    pls = list(map(int,config['sweep']['range_pioneers_locality_std'].split(',')))

    combos = []
    for _pls in range(pls[0],pls[1]+1):
        for _plm in range(plm[0],plm[1]+1):
            for _pf in range(pf[0],pf[1]+1):
                for ls in range(rls[0],rls[1]+1):
                    for lm in range(rlm[0],rlm[1]+1):
                        for r in range(rr[0],rr[1]+1):
                            for p in rp:
                                if r>p: continue
                                combos.append([p,r,lm,ls,_pf,_plm,_pls])
    return combos

def modify_config(cfg,sweep):
    cfg['model']['num_pioneers'] = str(sweep[2])
    cfg['model']['average_response_rate'] = str(sweep[3])
    cfg['model']['locality_mean'] = str(sweep[4])
    cfg['model']['locality_std'] = str(sweep[5])
    cfg['model']['pioneer_flips'] = str(sweep[6])
    cfg['model']['pioneer_locality_mean'] = str(sweep[7])
    cfg['model']['pioneer_locality_std'] = str(sweep[8])
    cfg['grid']['pioneer_seed'] = str(sweep[9])
    cfg['grid']['agent_seed']=str(sweep[10])
    cfg['model']['response_seed'] = str(sweep[11])
    cfg['model']['locality_seed'] = str(sweep[12])
    cfg['model']['pioneer_flips_seed'] = str(sweep[13])
    cfg['model']['pioneer_locality_seed'] = str(sweep[14])

