"""                            
@name: abm.simulations.format.py
@description:                  
Formatting simulation data for downstream analysis

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""
import os
from configparser import ConfigParser,ExtendedInterpolation
import numpy as np
from tqdm import tqdm

import abm.sweep_logs as sl
import abm.proc_data 
from abm.proc_data import PipeObject

def format_args(args):
    args.config = os.path.join(args.dir,'model.ini')
    args.sweep_log = os.path.join(args.dir,'sweep_log.csv')
    args.sweep_range = None 
    dout = os.path.dirname(args.sweep_log)
    dout += os.path.sep + 'sims' + os.path.sep
    if not os.path.exists(dout): os.mkdir(dout)
    args.dout = dout

def _analyze_node_dist(func):
    def inner(args):
        format_args(args) 
        cfg = ConfigParser(interpolation=ExtendedInterpolation())
        cfg.read(args.config)
        
        sn = sl.sweep_length(args.sweep_log,args.sweep_range)
        din = args.dout
        Z = np.zeros((sn,70))
        idx = 0 
        for s in tqdm(sl.iter_sweep(args.sweep_log,args.sweep_range),total=sn):
            fin = din + s[-1]
            if not os.path.exists(fin): continue
            P = PipeObject(fin)
            Z[idx,:] = func(P) 
            idx += 1

        if args.fout is not None: 
            np.save(args.fout,Z) 
            print(f'Saved to: {args.fout}')

    return inner

@_analyze_node_dist
def analyze_node_dist_fraction_conserved(P):
    return abm.proc_data.get_fraction_conserved(P,max_deg=6)

@_analyze_node_dist
def analyze_node_dist_total_degree(P):
    return abm.proc_data.get_total_degree(P,max_deg=6)

@_analyze_node_dist
def analyze_node_dist_fraction_poles(P):
    return abm.proc_data.get_fraction_poles(P,max_deg=6)

@_analyze_node_dist
def analyze_node_dist_num_pioneers_per_follower(P):
    return abm.proc_data.get_num_pioneers_per_follower(P,max_deg=6)

@_analyze_node_dist
def analyze_node_dist_num_pioneer_contacts_per_follower(P):
    return abm.proc_data.get_num_pioneer_contacts_per_follower(P,
                                                                max_deg=6,
                                                                pct_thresh=0.3)
@_analyze_node_dist
def analyze_node_dist_num_pioneers_per_follower(P):
    return abm.proc_data.get_num_pioneers_per_follower(P,max_deg=6)

@_analyze_node_dist
def analyze_node_dist_specificity(P):
    return abm.proc_data.get_specificity(P,max_deg=6)


def analyze_pioneer_groups_with_specificity(args):
    format_args(args) 
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    max_deg = 6

    sn = sl.sweep_length(args.sweep_log,args.sweep_range)
    din = args.dout
    Z = np.zeros((sn,5,max_deg))
    
    idx = -1
    sn = sl.sweep_length(args.sweep_log,args.sweep_range)
    for s in tqdm(sl.iter_sweep(args.sweep_log),total=sn):
        idx += 1 
        if not os.path.exists(din + s[-1]): continue
        P = PipeObject(din + s[-1])
        Z[idx,:,:] = abm.proc_data.pg_breakdown_reproducibility(P,
                                                              max_deg=max_deg)
        
    np.save(args.fout,Z) 
    print(f'Saved to: {args.fout}')

def analyze_pioneer_groups_domains_with_specificity(args):
    format_args(args) 
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    max_deg = 6

    sn = sl.sweep_length(args.sweep_log,args.sweep_range)
    din = args.dout
    #Z = np.zeros((sn,5,2))
    Z = np.zeros((sn,2,2))
    
    idx = -1
    sn = sl.sweep_length(args.sweep_log,args.sweep_range)
    for s in tqdm(sl.iter_sweep(args.sweep_log),total=sn):
        idx += 1 
        if not os.path.exists(din + s[-1]): continue
        P = PipeObject(din + s[-1])
        P.communities() 
        if P.comms is None: continue
        #Z[idx,:,:] = abm.proc_data.pg_breakdown_domains(P,max_deg=max_deg)
        #Z[idx,:,:] = abm.proc_data.pg_domain_similarity(P,max_deg=max_deg)
        d = abm.proc_data.pg_domain_similarity(P,max_deg=max_deg)
        print(idx,d.min(),d.max())
        Z[idx,:,:] = array_axis_sum_rescale(d,axis=1)

    np.save(args.fout,Z) 
    print(f'Saved to: {args.fout}')

def array_axis_sum_rescale(arr,axis=0):
    asum = arr.sum(axis=axis, keepdims=True)
    asum[asum==0] = 1
    return arr / asum

def analyze_specificity_change(args):
    format_args(args) 
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    max_deg = 6

    sn = sl.sweep_length(args.sweep_log,args.sweep_range)
    din = args.dout
    #Z = np.zeros((sn,5,2))
    Z = np.zeros((sn,2,2))
    
    
    lstA = np.arange(999)*7 
    lstB = np.concatenate((lstA + 2,lstA + 3)) 
    lstA = np.concatenate((lstA,lstA))
    

    sweep_pairs = zip(lstA,lstB)
    
    Z = np.zeros((len(lstA),20))
    idx = 0
    for (i,j,s0,s1) in tqdm(
            sl.iter_sweep_by_pairs(args.sweep_log,sweep_pairs),
            total=len(lstA),desc='Pairs processed'):
        P0 = PipeObject(din+s0[-1])
        P1 = PipeObject(din+s1[-1])
        
        t0 = P0.number_of_edges()
        t1 = P1.number_of_edges()
        
        v0 = abm.proc_data.variance_dist(P0,norm=False)
        v1 = abm.proc_data.variance_dist(P1,norm=False)
        
        #d0 = abm.proc_data.get_population_degree(P0,max_deg=max_deg).mean()
        #d1 = abm.proc_data.get_population_degree(P1,max_deg=max_deg).mean()
        d0 = np.mean([d for n,d in P0.degree()])
        d1 = np.mean([d for n,d in P1.degree()])
        
        R0 = abm.proc_data.pg_breakdown_reproducibility(P0,max_deg=max_deg)
        R1 = abm.proc_data.pg_breakdown_reproducibility(P1,max_deg=max_deg)
        
        R0 = R0[:,-1]
        R1 = R1[:,-1]

        tmp = [i,t0,v0[0],v0[-1],d0,R0[0],R0[1],R0[2],R0[3],R0[4],
               j,t1,v1[0],v1[-1],d1,R1[0],R1[1],R1[2],R1[3],R1[4]]
        
        Z[idx,:] = tmp
        
        idx += 1
    
    np.save(args.fout,Z) 
    print(f'Saved to: {args.fout}')


