"""                            
@name: abm.empirical.measures.py 
@description:                  
Shared functions for assessing empirical data

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

from random import sample
import numpy as np
import networkx as nx

from abm.proc_data import PipeObject
import abm.proc_data

def iter_graphs_mem(cfg):
    for (idx,(gname,grp)) in enumerate(cfg['groups'].items()):
        print(f"Loading {cfg['files'][gname]}")
        G = PipeObject(fin=cfg['files'][gname])
        yield idx,G 

def iter_graphs_chem(cfg):
    for (idx,(gname,grp)) in enumerate(cfg['groups_chem'].items()):
        print(f"Loading {cfg['files'][gname]}")
        G = nx.read_graphml(cfg['files'][gname])
        yield idx,G 

def assign_pioneer_groups(G,pio,max_deg=6,pct_thresh=0):
    initialize_graph_pioneers(G,pio) 
    
    weight = tally_pioneer_contact_weights(G,pio,max_deg=max_deg)    
    weight = sorted(weight)
    wdx = int(len(weight) * pct_thresh)
    wmin = weight[wdx]
    
    weight = np.log(weight)
    mu = weight.mean()
    std = weight.std()
    
    for u in G.nodes():
        if G.nodes[u]['is_pioneer'] == 1: continue
        G.nodes[u]['target'] = [] 
        trank = [] 
        for v in pio: 
            if not G.has_edge(u,v): continue 
            if G[u][v]['id'] < max_deg: continue
            contact = sum(map(float,G[u][v]['g_index_weight'].split('-')))
            #contact = (np.log(contact) - mu) / std
            if contact <= wmin: continue
            G[u][v]['z-weight'] = (np.log(contact) - mu) / std
            trank.append((v,contact))
        if len(trank) == 0: continue
        trank = sorted(trank, key=lambda x: x[1], reverse=True)
        G.nodes[u]['target'] = [t[0] for t in trank]

def tally_pioneer_contact_weights(G,pio,max_deg=6):
    weight = []
    for u in G.nodes():
        if G.nodes[u]['is_pioneer'] == 1: continue
        for v in pio:
            if not G.has_edge(u,v): continue
            if G[u][v]['id'] < max_deg: continue 
            tot_contact = sum(map(float,G[u][v]['g_index_weight'].split('-')))
            weight.append(tot_contact) 
    return weight 

def follower_groups_all(G,pioneers=None,**kwargs):
    fg = dict([(p,[]) for p in pioneers]) 
    for u in G.nodes():
        if G.nodes[u]['is_pioneer'] == 1: continue
        if len(G.nodes[u]['target']) == 0: continue 
        for t in G.nodes[u]['target']: fg[t].append(u)
    #print(fg) 
    return [fg[p] for p in pioneers]

def initialize_graph_pioneers(G,pio):
    for n in G.nodes(): G.nodes[n]['is_pioneer'] = 0 
    for n in pio: 
        G.nodes[n]['is_pioneer'] = 1
        G.nodes[n]['target'] = [n]

def pg_graph_build(G,_pio,build_func,pct_thresh,max_deg,rand=False,**kwargs):
    pio = _pio
    if rand: 
        rpio = [n for n in G.nodes() if n not in _pio]
        pio = sample(rpio,len(_pio))
    assign_pioneer_groups(G,pio,pct_thresh=pct_thresh)
    fg = follower_groups_all(G,pioneers=pio)
    prune_nodes_wo_target(G)
    return build_func(G,pio=pio,fg=fg,max_deg=max_deg,**kwargs)

def pg_synapse_tally(G,C=None,pio=None,fg=None,
                     max_deg=None,container=None,**kwargs):
    rmnodes = [n for n in C.nodes() if not G.has_node(n)]
    C.remove_nodes_from(rmnodes)
    Z = abm.proc_data.pg_edge_list(G,pio=pio,fg=fg,max_deg=max_deg)
    D = np.zeros((5,max_deg+1))
    N = np.zeros((5,max_deg+1))
    record = False
    if container is not None: record = True
    for (pdx,rdx,u,v) in Z:
        D[pdx,rdx] += 1 
        w0,w1 = 0,0 
        if C.has_edge(u,v): 
            w0 = C[u][v]['id']
        if C.has_edge(v,u): 
            w1 = C[v][u]['id']
        w = max([w0,w1]) 
        N[pdx,w] += 1
        if record: container[tuple(sorted([u,v]))] = [pdx,rdx,w] 
    return D,N

def prune_nodes_wo_target(G):
    rm_nodes = [u for u in G.nodes() if len(G.nodes[u]['target']) == 0]
    G.remove_nodes_from(rm_nodes)
    G.reset_node_index()

def array_axis_sum_rescale(arr,axis=0):
    asum = arr.sum(axis=axis, keepdims=True)
    asum[asum==0] = 1
    return arr / asum


