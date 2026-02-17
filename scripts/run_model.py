"""
@name:                         
@description:                  


@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""
import os
import sys
from configparser import ConfigParser,ExtendedInterpolation
import argparse
from inspect import getmembers,isfunction
from tqdm import tqdm
import importlib
import networkx as nx
import numpy as np
import multiprocessing_on_dill as mp
import pandas as pd
import pickle

from pycsvparser import read,write
import abm.sweep_logs as sl
from abm.batchrunner import batch_mp_vol,batch_mp_pipe
from abm.hooks import GraphHook as HK
from abm.proc_data import process_graphs
import abm.proc_data 
from abm.proc_data import PipeObject
from abm.model.model_base import get_position_encoding

def format_args(args):
    min_args = (args.dir is not None) or (args.config is not None and args.sweep_log is not None)
    assert min_args, "Must specify model directory or the config and sweep logs" 
    
    if args.config is None: args.config = os.path.join(args.dir,'model.ini')
    if args.sweep_log is None: args.sweep_log = os.path.join(args.dir,'sweep_log.csv')
    
    dout = args.dout
    if dout is None: 
        dout = os.path.dirname(args.sweep_log)
        dout += os.path.sep + 'sims' + os.path.sep
        if not os.path.exists(dout): os.mkdir(dout)
    args.dout = dout
 

def split_sweep_log(args):
    def iter_splits(N, nb):
        step = N / nb
        for i in range(nb):
            yield "{}:{}".format(round(step*i), round(step*(i+1)))

    format_args(args) 
    num_pg = sl.num_param_groups(args.sweep_log) + 1
    split = [s for s in iter_splits(num_pg,args.num_splits)]
    print(' '.join(split))

def build_sweep(args):
    format_args(args) 
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    model_class = cfg['sweep']['model_class']    
    sl = importlib.import_module(f"abm.model.{model_class}.sweep_log")    
    sweep = sl.build_sweep(cfg)
    write.from_list(args.sweep_log,sweep)
    print(f'Wrote to {args.sweep_log}')

def build_sampler(args):
    format_args(args) 
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    model_class = cfg['sweep']['model_class']    
    sl = importlib.import_module(f"abm.model.{model_class}.sweep_log")    
    sweep = sl.build_sampler(cfg)
    write.from_list(args.sweep_log,sweep)
    print(f'Wrote to {args.sweep_log}')


def test_load(args):
    format_args(args) 
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    model_class = cfg['sweep']['model_class']    
    msl = importlib.import_module(f"abm.model.{model_class}.sweep_log")    
    module = importlib.import_module(f"abm.model.{model_class}.model")    
    #M = getattr(module,cfg['model']['class'])
    M = getattr(module,'Model')
    num_models = cfg.getint('model','num_models')

    for s in sl.iter_sweep(args.sweep_log):
        msl.modify_config(cfg,s)
        model = M(cfg)
        model.init()
        model.print_sweep_params()

def analyze_agent_placement(args):
    format_args(args) 
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    if args.fout is None: args.fout = os.path.join(args.dir,'position_encoding.npz')

    model_class = cfg['sweep']['model_class']    
    msl = importlib.import_module(f"abm.model.{model_class}.sweep_log")    
    module = importlib.import_module(f"abm.model.{model_class}.model")    
    M = getattr(module,'Model')
    
    nrows = cfg.getint('grid','dim_rows') 
    ncols = cfg.getint('grid','dim_cols') 
    
    sn = sl.sweep_length(args.sweep_log,args.sweep_range)
    idx = np.zeros(sn,dtype=np.uint8)
    E = np.zeros((sn,nrows*ncols),dtype=np.uint8)
    
    for (i,s) in tqdm(enumerate(sl.iter_sweep(args.sweep_log)),total=sn):
        msl.modify_config(cfg,s)
        model = M(cfg)
        model.init()
        idx[i] = model.num_pioneers 
        get_position_encoding(model,E[i,:])    

    np.savez(args.fout,pioneers=idx,encoding=E)
    print(f"Wrote to {args.fout}")

def _analyze(func):
    def inner(args,**kwargs):
        format_args(args) 
        cfg = ConfigParser(interpolation=ExtendedInterpolation())
        cfg.read(args.config)
        if args.fout is None: 
            args.fout = os.path.join(args.dir,'dataframe.csv')
        
        print(f"Initiating analysis for {args.fout}")
        
        header,data = func(args,cfg)
        df = pd.DataFrame(data,columns=header)
        df.to_csv(args.fout,index=False)
        print(f"Wrote to {args.fout}")
        
    return inner

@_analyze
def analyze_encoding(args,cfg=None,**kwargs):
    model_class = cfg['sweep']['model_class']    
    msl = importlib.import_module(f"abm.model.{model_class}.sweep_log")    
    module = importlib.import_module(f"abm.model.{model_class}.model")    
    M = getattr(module,'Model')
    
    df_header = sl.get_header(args.sweep_log)
    param_keep = slice(0,9)
    df_header = df_header[param_keep]

    data = []
    sn = sl.sweep_length(args.sweep_log,args.sweep_range)
    for s in tqdm(sl.iter_sweep(args.sweep_log),total=sn):
        msl.modify_config(cfg,s)
        model = M(cfg)
        model.init()
        encoder = model.encoding() 
        rec = s[param_keep] + encoder
        data.append(rec)
    
    pipe_cols = ['fs','ps','fl','pl','fl0','pl0','fd','pd']
    header = df_header + list(pipe_cols) 
    
    return header,data

@_analyze
def analyze_pg(args,cfg=None):
    pipe_cols,pipe_str = zip(*read.into_list(cfg['analyze']['pipe_file'],multi_dim=True))
    pipe = [getattr(abm.proc_data,p) for p in list(pipe_str)]
    num_models = cfg.getint('model','num_models')
    max_d = cfg.getint('analyze','max_dendrogram_distance')  
    df_header = sl.get_header(args.sweep_log)
    param_keep = slice(2,-1)
    del df_header[1]
    del df_header[-1]
    din = args.dout
   
    tree = sl.sweep_tree(args.sweep_log,args.sweep_range)
    data = []
    for (k,v) in tqdm(tree.items(),desc='Param group iter'):
        result = batch_mp_pipe(v,pipe,din,deg=num_models,max_d=max_d,weight='wnorm')
        se = result.std(0) / np.sqrt(result.shape[0])
        rec = [v[0][0]] + v[0][param_keep] + result.mean(0).tolist() + se.tolist()
        data.append(rec)
    
    # add var cols
    pipe_cols_se = [f"{p}_se" for p in pipe_cols]
    header = df_header + list(pipe_cols) + list(pipe_cols_se)
    
    return header,data

@_analyze
def analyze(args,cfg=None):
    pipe_cols,pipe_str = zip(*read.into_list(cfg['analyze']['pipe_file'],multi_dim=True))
    pipe = [getattr(abm.proc_data,p) for p in list(pipe_str)]
    num_models = cfg.getint('model','num_models')
    max_d = cfg.getint('analyze','max_dendrogram_distance')  
    num_fields = 9
    try: #backward compatibility
        num_fields = cfg.getint('sweep','num_fields') + 1
    except:
        pass

    df_header = sl.get_header(args.sweep_log)
    param_keep = slice(0,num_fields)
    df_header = df_header[param_keep]
    din = args.dout
    
    data = []
    sn = sl.sweep_length(args.sweep_log,args.sweep_range)
    for s in tqdm(sl.iter_sweep(args.sweep_log,args.sweep_range),total=sn):
        fin = din + s[-1]
        P = PipeObject(fin)
        tmp = [f(P,num_pioneers=s[2],deg=num_models,max_d=max_d,weight='wnorm') for f in pipe]
        data.append(s[param_keep] + tmp)
    
    header = df_header + list(pipe_cols)
    return header,data

def _analyze_node_dist_model(func):
    def inner(args,**kwargs):
        format_args(args) 
        cfg = ConfigParser(interpolation=ExtendedInterpolation())
        cfg.read(args.config)
        if args.fout is None: 
            args.fout = os.path.join(args.dir,'dataframe.csv')
        
        print(f"Initiating analysis for {args.fout}")
        
        Z = func(args,cfg)
 
        if args.fout is not None: 
            np.save(args.fout,Z) 
            print(f'Saved to: {args.fout}')

       
    return inner

@_analyze_node_dist_model
def analyze_pioneer_similarity(args,cfg=None,**kwargs):
    model_class = cfg['sweep']['model_class']    
    msl = importlib.import_module(f"abm.model.{model_class}.sweep_log")    
    module = importlib.import_module(f"abm.model.{model_class}.model")    
    M = getattr(module,'Model')
    
    sn = sl.sweep_length(args.sweep_log,args.sweep_range)
    Z = np.zeros((sn,101))
    x = np.linspace(0,1,101)
    idx = 0 

    for s in tqdm(sl.iter_sweep(args.sweep_log),total=sn):
        msl.modify_config(cfg,s)
        model = M(cfg)
        model.init()
        z = abm.proc_data.similarity(model.E)     
        Z[idx,:] = np.array([np.sum(z <= _x) / len(z) for _x in x ])
        idx += 1

    return Z


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


def analyze_pioneer_groups(args):
    format_args(args) 
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    max_deg = 6

    sn = sl.sweep_length(args.sweep_log,args.sweep_range)
    din = args.dout
    Z = np.zeros((sn,5,max_deg))
    
    split = 3500
    sweep_pairs = zip(range(split),range(split,2*split))
     
    idx = 0
    for (i,j,s0,s1) in tqdm(sl.iter_sweep_by_pairs(args.sweep_log,sweep_pairs),total=split,desc='Pairs processed'):
        P0 = PipeObject(din+s0[-1])
        P1 = PipeObject(din+s1[-1])

        P0.nodes = P1.nodes
        
        Z[i,:,:] = abm.proc_data.pg_breakdown_reproducibility(P0,max_deg=max_deg)    
        Z[j,:,:] = abm.proc_data.pg_breakdown_reproducibility(P1,max_deg=max_deg)    
        
        idx += 1

    if args.fout is not None: 
        np.save(args.fout,Z) 
        print(f'Saved to: {args.fout}')

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
        
    if args.fout is not None: 
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
        if P.comms is None: continue
        #Z[idx,:,:] = abm.proc_data.pg_breakdown_domains(P,max_deg=max_deg)
        Z[idx,:,:] = abm.proc_data.pg_domain_similarity(P,max_deg=max_deg)
        
    if args.fout is not None: 
        np.save(args.fout,Z) 
        print(f'Saved to: {args.fout}')

def model58_targetted_domain_run(args):
    format_args(args) 
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    max_deg = 6
    
    IDX = read.into_list('mat/model58_graphs_domain_index.txt')
    IDX = list(map(int,IDX)) 

    sn = sl.sweep_length(args.sweep_log,args.sweep_range)
    din = args.dout
    Z = np.zeros((sn,5,2))
    
    idx = -1
    sn = sl.sweep_length(args.sweep_log,args.sweep_range)
    for s in tqdm(sl.iter_sweep(args.sweep_log),total=sn):
        idx += 1 
        if idx not in IDX: continue
        if not os.path.exists(din + s[-1]): continue
        fout = din + s[-1] 
        P = PipeObject(fout)
        P.index_communities(deg=6)
        with open(fout, 'wb') as f:
            pickle.dump(P, f, pickle.HIGHEST_PROTOCOL)
            

def analyze_pioneer_groups_domains(args):
    format_args(args) 
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    max_deg = 6

    sn = sl.sweep_length(args.sweep_log,args.sweep_range)
    din = args.dout
    Z = np.zeros((sn,5,2))
    
    split = 3500
    sweep_pairs = zip(range(split),range(split,2*split))
     
    idx = 0
    for (i,j,s0,s1) in tqdm(sl.iter_sweep_by_pairs(args.sweep_log,sweep_pairs),total=split,desc='Pairs processed'):
        if int(s0[2]) < 10: continue
        if int(s1[2]) < 10: continue
        P0 = PipeObject(din+s0[-1])
        P1 = PipeObject(din+s1[-1])

        P0.nodes = P1.nodes
        
        P0.index_communities(deg=max_deg)
        P1.index_communities(deg=max_deg)

        Z[i,:,:] = abm.proc_data.pg_breakdown_domains(P0,max_deg=max_deg)    
        Z[j,:,:] = abm.proc_data.pg_breakdown_domains(P1,max_deg=max_deg)    
        
        idx += 1
                
        
    if args.fout is not None: 
        np.save(args.fout,Z) 
        print(f'Saved to: {args.fout}')


def run(args):
    format_args(args) 
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    model_class = cfg['sweep']['model_class']    
    msl = importlib.import_module(f"abm.model.{model_class}.sweep_log")    
    module = importlib.import_module(f"abm.model.{model_class}.model")    
    #M = getattr(module,cfg['model']['class'])
    M = getattr(module,'Model')
    num_models = cfg.getint('model','num_models')
    dout = args.dout
    print(f'Data written to {dout}')
    
    sn = sl.sweep_length(args.sweep_log,args.sweep_range)
    for s in tqdm(sl.iter_sweep(args.sweep_log,args.sweep_range),total=sn):
        msl.modify_config(cfg,s)
        models = [M(cfg,model_id=i) for i in range(num_models)]
        hooks = [[HK()] for i in range(num_models)]
        hooks = batch_mp_vol(models,hooks)
        G = [h[0]() for h in hooks]
        H = process_graphs(G)
        fout = dout + s[-1]
        #nx.write_gpickle(H,fout)
        with open(fout, 'wb') as f:
            pickle.dump(H, f, pickle.HIGHEST_PROTOCOL)


def concat_dataframes(args):
    df = [pd.read_csv(f) for f in args.merge.split(',')]
    df1 = pd.concat(df,axis=0,ignore_index=True)
    if args.fout: df1.to_csv(args.fout,index=False)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mode',
                        action = 'store',
                        help = 'Function call')
    
    parser.add_argument('--dir',
                action = 'store',
                dest = 'dir',
                default = None,
                required = False,
                help = 'Model directory')

    parser.add_argument('--config',
                action = 'store',
                dest = 'config',
                default = None,
                required = False,
                help = 'Model config file')

    parser.add_argument('--sweep_log',
                action = 'store',
                dest = 'sweep_log',
                default = None,
                required = False,
                help = 'Path to sweep log file')
    
    parser.add_argument('--sweep_range',
                action = 'store',
                dest = 'sweep_range',
                default = None,
                required = False,
                help = 'Index range for sweep file. If single value the format is int1 if a range the format is int1,int2')
    
    parser.add_argument('--dout',
                action = 'store',
                dest = 'dout',
                default = None,
                required = False,
                help = 'Path to output directory for pkl files')
    
    parser.add_argument('--fout',
                action = 'store',
                dest = 'fout',
                default = None,
                required = False,
                help = 'Path to output file')
    
    parser.add_argument('--num_splits',
                action = 'store',
                dest = 'num_splits',
                type = int,
                default = 1,
                required = False,
                help = 'Number of splits for file')
    
    parser.add_argument('--merge',
                action = 'store',
                dest = 'merge',
                default = None,
                required = False,
                help = 'Comma separted path to files to merge')
    
    args = parser.parse_args()
    eval(args.mode + '(args)')
