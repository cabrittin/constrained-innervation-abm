"""                            
@name: abm.empirical.format.py 
@description:                  
Module for formatting empirical data

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""
from configparser import ConfigParser,ExtendedInterpolation
from itertools import combinations
import numpy as np
import networkx as nx
import pandas as pd
from tqdm import tqdm
import copy
from collections import defaultdict

from pycsvparser import read
import toolbox.graphs.modify as gm
import toolbox.graphs as graph
from abm.proc_data import PipeObject
import abm.proc_data 


from .measures import assign_pioneer_groups,iter_graphs_mem
from .measures import follower_groups_all,initialize_graph_pioneers
from .measures import pg_graph_build,array_axis_sum_rescale
from .measures import prune_nodes_wo_target,iter_graphs_chem
from .measures import pg_synapse_tally

def witvliet_source_to_graphml(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    tag = args.tag
    fin = cfg[tag]['fin']
    sheets = cfg[tag]['sheets'].split(',')
    fout = cfg[tag]['fout'].split(',') 
    ref_nodes = read.into_list(cfg['mat']['ref_nodes'])

    for (sh,_f) in zip(sheets,fout):
        print(sh,_f) 
        f = cfg['files'][_f] 
        
        df = pd.read_excel(fin, sheet_name=sh, header=2) 
        df = df.dropna(how='all')
        df = df.iloc[:, 2:]
        df.index = df.iloc[:, 0]  
        df = df.iloc[:, 1:]  

        G = nx.from_pandas_adjacency(df)
        #nodes_rm = [n for n in G.nodes() if n not in ref_nodes]
        #G.remove_nodes_from(nodes_rm) 

        #nx.write_graphml(G,f,prettyprint=True)
        #print(f'Wrote to: {f}')

def build_consensus(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)

    remove = read.into_list(cfg['mat']['remove'])
    papillary = read.into_list(cfg['mat']['papillary']) 
    misc_rm = read.into_list(cfg['mat']['misc']) 
    left = read.into_list(cfg['mat']['left_nodes'])
    right = read.into_list(cfg['mat']['right_nodes'])
    lrmap = read.into_lr_dict(cfg['mat']['lrmap'])
    nodes = read.into_list(cfg['mat']['ref_nodes'])
    pioneers = read.into_list(cfg['mat']['pioneers'])
    
    glabel = args.group_label
    filter_graph = bool(args.group_label) 

    for (gname,grp) in cfg[glabel].items():
        graphs = [] 
        for d in grp.split(','):
            print(f'Loading {cfg["files"][d]}') 
            G = nx.read_graphml(cfg['files'][d])
            print(G.number_of_edges())
            G.remove_nodes_from(remove)
            G.remove_nodes_from(papillary)
            G.remove_nodes_from(misc_rm)
            print(G.number_of_edges())
            if filter_graph: 
                G = gm.filter_graph_edge(G,
                                     cfg.getint('params',
                                                'lower_weight_threshold'))
            print(G.number_of_edges())
            GL = gm.split_graph(G,left)
            GR = gm.split_graph(G,right)
            GR = gm.map_graph_nodes(GR,lrmap)
            GL.remove_nodes_from(right)
            GR.remove_nodes_from(right)
            print(GL.number_of_edges(),GR.number_of_edges()) 
            graphs += [GL,GR]

        for g in graphs: gm.normalize_edge_weight(g)
        max_delta = len(graphs)
        C = [
                graph.consensus(graphs,i+1,
                             nodes=nodes,
                             weight=['weight','wnorm']) 
                for i in range(max_delta)
             ]
        #for c in C: print('\t',c.number_of_edges()) 
        M = graph.index_merge(C)
        graph.zip_index_consensus_graph_edges(M,graphs)
        print(M.number_of_edges())
        pmap = dict([(n,0) for n in M.nodes()])
        for p in pioneers: pmap[p] = 1
        nx.set_node_attributes(M,pmap,'is_pioneer')
        
        #print(M.number_of_edges())
        nx.write_graphml(M,cfg['files'][gname])
        print(f"Wrote to {cfg['files'][gname]}")


def to_supplemental_table(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    writer = pd.ExcelWriter(cfg['files']['supp_table'], engine='xlsxwriter')
    
    pio = sorted(read.into_list(cfg['mat']['pioneers']))
    nmap = read.into_dict(cfg['mat']['node_map'])

    col_0_2 = [f'd{i}_weight' for i in range(6)]
    cols_0 = ['cell_1','cell_2','weight_norm','reproducibility','datasets_present'] + col_0_2
    cols_1 = ['cell','is_pioneer','domain'] 
    glabel = 'groups'
    for (gname,grp) in cfg[glabel].items():
        G = PipeObject(fin=cfg['files'][gname])
        _df = [] 
        for (u,v,data) in tqdm(G.edges(data=True),total=G.number_of_edges(),desc=f"{gname} edges"):
            row_1 = [nmap[u],nmap[v],data['wnorm'],data['id'],data['g_index']]
            row_2 = [0 for i in range(6)]
            gweight = list(map(float,data['g_index_weight'].split('-'))) 
            gindex = list(map(int,data['g_index'].split('-'))) 
            for gdx,gw in zip(gindex,gweight):
                row_2[gdx] = gw 
            
            _df.append(row_1 + row_2)
        
        df = pd.DataFrame(data=_df,columns = cols_0)
        sheet_name = gname + '_contact'
        df.to_excel(writer, sheet_name=sheet_name, index=False)

        initialize_graph_pioneers(G,pio)
        G.index_communities(deg=6)
        _df = [] 
        for u in tqdm(sorted(G.nodes()),total=G.number_of_nodes(),desc=f"{gname} nodes"):
            _df.append([nmap[u],G.nodes[u]['is_pioneer'],G.comm_index[u]])
        
        df = pd.DataFrame(data=_df,columns = cols_1)
        sheet_name = gname + '_domains'
        df.to_excel(writer, sheet_name=sheet_name, index=False)

    glabel = 'groups_chem'
    for (gname,grp) in cfg[glabel].items():
        G = PipeObject(fin=cfg['files'][gname])
        _df = [] 
        for (u,v,data) in tqdm(G.edges(data=True),total=G.number_of_edges(),desc=f"{gname} synapse"):
            row_1 = [nmap[u],nmap[v],data['wnorm'],data['id'],data['g_index']]
            row_2 = [0 for i in range(6)]
            gweight = list(map(float,data['g_index_weight'].split('-'))) 
            gindex = list(map(int,data['g_index'].split('-'))) 
            for gdx,gw in zip(gindex,gweight):
                row_2[gdx] = gw 
            
            _df.append(row_1 + row_2)
        
        df = pd.DataFrame(data=_df,columns = cols_0)
        sheet_name = gname + '_synapse'
        df.to_excel(writer, sheet_name=sheet_name, index=False)


    writer.close()

def build_dataframe(args):
    pipe_cols,pipe_str = zip(*read.into_list(args.pipe,multi_dim=True))
    pipe = [getattr(abm.proc_data,p) for p in list(pipe_str)]
    cols = ['param_group','num_followers','num_pioneers','num_time_groups'] + list(pipe_cols)
    
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    data = [] 
    for (gname,grp) in cfg['groups'].items():
        print(f"Loading {cfg['files'][gname]}")
        G = PipeObject(fin=cfg['files'][gname])
        param_group = -1 
        num_pio = len([n for n in G.nodes() if G.nodes[n]['is_pioneer'] == 1])
        num_fol = len([n for n in G.nodes() if G.nodes[n]['is_pioneer'] == 0])
        
        num_time_groups = -1 
     
        r = [f(G,deg=G.max_deg,label=0,num_pioneers=num_pio,label_pio=False,weight='wnorm') for f in pipe]
        data.append([param_group,num_fol,num_pio,num_time_groups] + r)
        
    df = pd.DataFrame(data,columns=cols)
    df.to_csv(cfg['files']['dataframe'],index=False)  
    print(f"Wrote to {cfg['files']['dataframe']}")



def build_num_pioneers_per_follower(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    max_deg = cfg.getint('params','max_deg')
    pct_thresh = cfg.getfloat('params','pg_pct_thresh')
    pio = sorted(read.into_list(cfg['mat']['pioneers']))
    x = np.arange(15)
    ecomp = np.zeros((3,len(x)))
    df = []
    for idx,G in iter_graphs_mem(cfg):
        assign_pioneer_groups(G,pio,pct_thresh=pct_thresh)
        num_targets=[]
        for u in G.nodes():
            if G.nodes[u]['is_pioneer'] == 1: continue
            if len(G.nodes[u]['target']) == 0: continue
            num_targets.append(len(G.nodes[u]['target']))
            df.append([idx,len(G.nodes[u]['target'])])

        z = np.array(num_targets) 
        unique_values, counts = np.unique(z, return_counts=True)
        print("Unique Values:", unique_values)
        print("Counts:", counts)

        y = np.array([np.sum(z <= _x) / len(z) for _x in x])
        ecomp[idx,:] = y 
 
    edat = np.zeros((3,ecomp.shape[1]))
    edat[0,:] = ecomp.mean(axis=0)
    edat[1,:] = ecomp.min(axis=0)
    edat[2,:] = ecomp.max(axis=0)
 
    np.save(args.fout,edat)
    print(f'Wrote to {args.fout}') 

def build_pioneer_groups(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
 
    pio = sorted(read.into_list(cfg['mat']['pioneers']))
    max_deg = cfg.getint('params','max_deg')
    pct_thresh = cfg.getfloat('params','pg_pct_thresh')
    P = np.zeros((68,len(pio)),dtype=int)
    W = np.zeros((68,len(pio)))
    for idx,G in iter_graphs_mem(cfg):
        assign_pioneer_groups(G,pio,pct_thresh=pct_thresh)
        if idx == 0:
            nrec = sorted(list(G.nodes()))
            nmap = dict([(n,i) for (i,n) in enumerate(nrec)]) 
        
        fg = follower_groups_all(G,pioneers=pio)
        for (jdx,(p,F)) in enumerate(zip(pio,fg)):
            for v in [p] + F: 
                P[nmap[v],jdx] += 1
                if v != pio[jdx]: 
                    W[nmap[v],jdx] += G[v][pio[jdx]]['z-weight']
    
    W[P>0] = W[P>0] / P[P>0]
    P = P / 3.0
    P[P < 0.5] = 0
    P[P > 0] = 1
    P = P.astype(int)

    W = W * P
    
    ikeep = np.where(P.sum(axis=1) > 0)[0]
    A = []
    B = []
    for i in ikeep:
        A.append([nrec[i],G.nodes[nrec[i]]['is_pioneer']] + P[i,:].tolist())
        B.append([nrec[i],G.nodes[nrec[i]]['is_pioneer']] + W[i,:].tolist())

    cols = ['cell','is_pioneer'] + [f'pg_{i}' for i in range(len(pio))]
    df = pd.DataFrame(data=A,columns=cols)
    df2 = pd.DataFrame(data=B,columns=cols)
    
    psum = P[ikeep,:].sum(axis=1)
    print('# PG range',min(psum),max(psum))
    psum = P[ikeep,:].sum(axis=0)
    print('# follower range',min(psum),max(psum))

    df.to_csv(args.fout_1,index=False)
    print(f'Wrote to {args.fout_1}') 
    
    df2.to_csv(args.fout_2,index=False)
    print(f'Wrote to {args.fout_2}') 
 

def pioneer_follower_distance(args):
    df = pd.read_csv(args.df_emp)
    df = df[df['is_pioneer'] == 0]
    cells = df['cell'].tolist()
    pio = df.columns[2:-1].tolist()
    
    df = df.iloc[:,2:-1].to_numpy()
    idx = np.argmax(df,axis=1)
    imax = np.max(df,axis=1)
    imax[imax>10] = -1

    for i,cell in enumerate(cells):
        print(f'{cells[i]},{pio[idx[i]]},{imax[i]}')

def degree_dist(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    x = np.linspace(0,1,101)
    ecomp = np.zeros((3,len(x)))
    
    for (idx,(gname,grp)) in enumerate(cfg['groups'].items()):
        print(f"Loading {cfg['files'][gname]}")
        G = PipeObject(fin=cfg['files'][gname])
        z = abm.proc_data.get_total_degree(G,max_deg=6)
        y = np.array([np.sum(z <= _x) / len(z) for _x in x])
        ecomp[idx,:] = y 

    
    edat = np.zeros((3,ecomp.shape[1]))
    edat[0,:] = ecomp.mean(axis=0)
    edat[1,:] = ecomp.min(axis=0)
    edat[2,:] = ecomp.max(axis=0)
    
    np.save(args.fout,edat)
    print(f'Wrote to {args.fout}')

def degree_dist_pg(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    pio = sorted(read.into_list(cfg['mat']['pioneers']))
    max_deg = cfg.getint('params','max_deg')
    pct_thresh = cfg.getfloat('params','pg_pct_thresh')
 
    x = np.linspace(0,1,101)
    ecomp = np.zeros((3,len(x)))
    
    D = defaultdict(lambda: [-1,-1,-1])
    for (idx,(gname,grp)) in enumerate(cfg['groups'].items()):
        print(f"Loading {cfg['files'][gname]}")
        G = PipeObject(fin=cfg['files'][gname])
        assign_pioneer_groups(G,pio,pct_thresh=pct_thresh)
        fg = follower_groups_all(G,pioneers=pio)
        prune_nodes_wo_target(G)
        print(G.number_of_nodes())
        z = abm.proc_data.get_total_degree(G,max_deg=6)
        for (udx,u) in enumerate(sorted(G.nodes())): D[u][idx] = z[udx]

        y = np.array([np.sum(z <= _x) / len(z) for _x in x])
        ecomp[idx,:] = y 
    
    for (k,v) in sorted(D.items()):
        print(f"{k},{v[0]},{v[1]},{v[2]}")
    
    edat = np.zeros((3,ecomp.shape[1]))
    edat[0,:] = ecomp.mean(axis=0)
    edat[1,:] = ecomp.min(axis=0)
    edat[2,:] = ecomp.max(axis=0)
    
    np.save(args.fout,edat)
    print(f'Wrote to {args.fout}')

def build_pg_breakdown_reproducibility(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
 
    build_func = abm.proc_data.pg_breakdown_reproducibility
    
    pio = sorted(read.into_list(cfg['mat']['pioneers']))
    max_deg = cfg.getint('params','max_deg')
    pct_thresh = cfg.getfloat('params','pg_pct_thresh')
    Z = np.zeros((3,5,max_deg))

    rsize = 100
    R = np.zeros((3*rsize,5,max_deg))
    rdx = 0

    for idx,G in iter_graphs_mem(cfg):
        G.index_communities(deg=max_deg)
        H = copy.deepcopy(G)
        Z[idx,:,:] = pg_graph_build(H,pio,build_func,pct_thresh,max_deg) 
        
        for j in tqdm(range(rsize),desc='Rand iter'):
            H = copy.deepcopy(G)
            R[rdx,:,:] = pg_graph_build(H,pio,build_func,pct_thresh,max_deg,rand=True) 
            rdx += 1 

    np.save(args.fout,Z)
    print(f'Wrote to: {args.fout}') 

    np.save(args.fout_rand,R)
    print(f'Wrote to: {args.fout_rand}') 


def build_pg_breakdown_domains(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    build_func = abm.proc_data.pg_domain_similarity

    pio = sorted(read.into_list(cfg['mat']['pioneers']))
    max_deg = cfg.getint('params','max_deg')
    pct_thresh = cfg.getfloat('params','pg_pct_thresh')
    S = np.zeros((3,2,2))
    
    rsize = 100
    R = np.zeros((3*rsize,2,2))
    rdx = 0
     
    P = defaultdict(lambda: [[-1,-1],[-1,-1],[-1,-1]])
    for idx,G in iter_graphs_mem(cfg):
        G.index_communities(deg=max_deg)
        H = copy.deepcopy(G)
        p = defaultdict(lambda:[-1,-1])
        d = pg_graph_build(H,pio,build_func,pct_thresh,max_deg,
                           container=p,flag=True) 
        S[idx,:,:] = array_axis_sum_rescale(d,axis=1)

        ## Record data
        for (k,v) in p.items(): P[k][idx] = v

        for j in tqdm(range(rsize),desc='Rand iter'):
            H = copy.deepcopy(G)
            d = pg_graph_build(H,pio,build_func,pct_thresh,max_deg,rand=True) 
            R[rdx,:,:] = array_axis_sum_rescale(d,axis=1)
            rdx += 1 
    
    for k,v in P.items():
        _k = ",".join(k)
        _v = ",".join([",".join(list(map(str,vv))) for vv in v])
        print(f"{_k},{_v}")
    

    np.save(args.fout,S)
    print(f'Wrote to: {args.fout}') 

    np.save(args.fout_rand,R)
    print(f'Wrote to: {args.fout_rand}') 



def build_pg_synapse(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    build_func = pg_synapse_tally

    pio = sorted(read.into_list(cfg['mat']['pioneers']))
    max_deg = cfg.getint('params','max_deg')
    pct_thresh = cfg.getfloat('params','pg_pct_thresh')
    N = np.zeros((3,5,7)) 
    D = np.zeros((3,5,7)) 
    
    rsize = 200
    R = np.zeros((3*rsize,5,7))
    rdx = 0

    P = defaultdict(lambda: [[-1,-1,-1],[-1,-1,-1],[-1,-1,-1]])
    for ((mdx,G),(cdx,C)) in zip(iter_graphs_mem(cfg),iter_graphs_chem(cfg)):
        H = copy.deepcopy(G)
        p = defaultdict(lambda:[-1,-1,-1])
        D[mdx,:,:],N[mdx,:,:] = pg_graph_build(H,pio,
                                               build_func,
                                               pct_thresh,max_deg,
                                               container=p,C=C) 

        for (k,v) in p.items(): P[k][mdx] = v
        for j in tqdm(range(rsize),desc='Rand iter'):
            H = copy.deepcopy(G)
            _,R[rdx,:,:] = pg_graph_build(H,pio,
                                        build_func,
                                        pct_thresh,max_deg,
                                        C=C,rand=True) 
            rdx += 1 

    for k,v in P.items():
        _k = ",".join(k)
        _v = ",".join([",".join(list(map(str,vv))) for vv in v])
        print(f"{_k},{_v}")

    np.savez(args.fout,N=N,D=D,R=R)
    print(f'Wrote to {args.fout}')




