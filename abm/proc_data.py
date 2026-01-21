"""
@name: proc_data.py
@description:
    Module for processing data
    
    All pipe functions should start with 'get_' and accept **kwargs

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import numpy as np
import networkx as nx
import networkx.algorithms.community as nx_comm
from igraph import Graph
import warnings
import logging
from collections import defaultdict
from scipy.cluster.hierarchy import linkage,fcluster,dendrogram
from itertools import combinations,product
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
from typing import Union

import tb_popgraphs.graphs.modify as gm
import tb_popgraphs.graphs as graph
from tb_popgraphs.generators.population import PopulationGenerator
from tb_popgraphs.graphs.pop_communities import pop_comm,pop_assign_comm,pop_comm_generate


#warnings.simplefilter("error","RuntimeWarning")

class PipeObject(nx.Graph):
    """
    PipeObject class use to facilitate analysis pipe.

    Used to cache repreated computation in the pipe
    """
    def __init__(self,fin=None,graph=None,label=None):
        """
        Args:
        -----
        fin : str
            Path to pickle networkx file
        """
        if (fin is not None) and  (graph is not None):
            logging.warning('Both fin and graph provided at input, graph will take precedent')

        if fin is not None: 
            if '.graphml' in fin:
                super().__init__(nx.read_graphml(fin))
            elif '.pkl':
                #super().__init__(nx.read_gpickle(fin))
                with open(fin, 'rb') as f:
                    G = pickle.load(f)
                    super().__init__(G)
                    if hasattr(G,"comms"): self.comms = G.comms
                    if hasattr(G,"comm_index"): self.comm_index = G.comm_index

            else:
                loggin.error("Graph file is neither pkl nor graphml")
        if graph is not None: super().__init__(graph)
        
        self.label = label
        self.deg = {}
        if not hasattr(self,"comms"): self.comms = None
        if not hasattr(self,"comm_index"): self.comm_index = defaultdict(int)
        self.max_deg = max([w for (u,v,w) in self.edges(data='id')])
        self.vdist = None
        self.graphs = None
        self.node_ref = sorted(self.nodes())
        self.cont_length_index = defaultdict(float)
        self.neigh_repro = None
        self.PG = None #Pioneer groups
        
        self.reset_node_index()

    def reset_node_index(self):
        self.node_index = defaultdict(int)
        for (i,n) in enumerate(sorted(self.nodes())):
            self.node_index[n] = i

    def unzip_graphs(self):
        self.graphs = graph.unzip_index_consensus_graph_edges(self)

    def degree_graph(self,deg,**kwargs):
        try:
            return self.deg[deg]
        except:
            self.deg[deg] = degree_graph(self,deg)
            return self.deg[deg]
    
    def communities(self,deg=None,overwrite=False,keep_pop=False,**kwargs):
        self.unzip_graphs() 
        if self.comms is None or overwrite:
            if deg is None: deg = self.max_deg
            self.comms = list(self.nodes())
            if self.degree_graph(deg).number_of_edges() > 0: #Is this still needed???
                comms = get_communities(self.graphs,keep_pop=keep_pop,**kwargs)
                if keep_pop:
                    self.comms = comms[0]
                    self.pop_comms = comms[1]
                else:
                    self.comms = comms

    def index_communities(self,**kwargs):
        if self.comms is None: self.communities(**kwargs)
        for (i,comm) in enumerate(self.comms):
            for c in comm: self.comm_index[self.node_ref[c]] = i 
    
    def index_pioneer_groups(self,**kwargs):
        if self.PG is None:
            F,fmap = build_pioneer_groups(self)
            self.PG = {'F':F,'fmap':fmap}

    def variance_dist(self,overwrite=False,**kwargs): 
        if self.vdist is None or overwrite:
            self.vdist = variance_dist(self)
    
    def index_contact_length(self,**kwargs):
        wtot = 0
        for (u,v) in self.degree_graph(deg=self.max_deg).edges():
            w = self.edges[u,v]['wnorm']
            wtot += w
            #self.cont_length_index[u] += 
        print(wtot)
    
    def get_neighborhood_reproduciblity(self,**kwargs):
        self.neigh_repro = get_neighborhood_reproducibility(self)

    def pioneers(self,pioneer=True):
        pval = int(pioneer)
        for (k,v) in nx.get_node_attributes(self,'is_pioneer').items():
            if v == pval:
                yield k
    
    def clear_pioneers(self):
        """
        Set is_pioneer attribute to 0 for all nodes
        """
        for n in self.nodes(): self.nodes[n]['is_pioneer'] = 0
    
    def set_pioneers(self,nodes):
        for n in nodes: self.nodes[n]['is_pioneer'] = 1


    def pioneer_test_split_0(self):
        """
        Returns 3 set of nodes:
        i) pioneeers
        ii) followers 
        """
        self.degree_graph(self.max_deg) 
        pio = list(self.pioneers(pioneer=True))
        fol = list(self.pioneers(pioneer=False))
        return sorted(pio),sorted(fol)
     
    def pioneer_test_split(self):
        """
        Returns 3 set of nodes:
        i) pioneeers
        ii) followers with the highest degree (same number of nodes as pioneer)
        iii) the remaining nodes
        """
        self.degree_graph(self.max_deg) 
        pio = list(self.pioneers(pioneer=True))
        fol = list(self.pioneers(pioneer=False))
        deg = sorted([(n,self.deg[self.max_deg].degree(n)) for n in fol],key=lambda x: x[1],reverse=True)
        fol = [n[0] for n in deg] 
        N = len(pio)
        return sorted(pio),sorted(fol[:N]),sorted(fol[N:])

def get_follower_specificity(G,**kwargs):
    return np.mean([G.nodes[n]['specificity'] for n in G.nodes() if G.nodes[n]['is_pioneer'] == 0])

def get_pioneer_specificity(G,**kwargs):
    return np.mean([G.nodes[n]['specificity'] for n in G.nodes() if G.nodes[n]['is_pioneer'] == 1])

def get_follower_locality(G,**kwargs):
    return np.mean([G.nodes[n]['locality'] for n in G.nodes() if G.nodes[n]['is_pioneer'] == 0])

def get_pioneer_locality(G,**kwargs):
    return np.mean([G.nodes[n]['locality'] for n in G.nodes() if G.nodes[n]['is_pioneer'] == 1])

def get_follower_locality0(G,**kwargs):
    return np.mean([G.nodes[n]['locality0'] for n in G.nodes() if G.nodes[n]['is_pioneer'] == 0])

def get_pioneer_locality0(G,**kwargs):
    return np.mean([G.nodes[n]['locality0'] for n in G.nodes() if G.nodes[n]['is_pioneer'] == 1])

def get_follower_locality_distance(G,**kwargs):
    return np.mean([G.nodes[n]['locality_distance'] for n in G.nodes() if G.nodes[n]['is_pioneer'] == 0])

def get_pioneer_locality_distance(G,**kwargs):
    return np.mean([G.nodes[n]['locality_distance'] for n in G.nodes() if G.nodes[n]['is_pioneer'] == 1])


def get_communities(H,iters=100,max_d=10,sigma=0.23,tqdm_disable=True,keep_pop=False,**kwargs):
    PG = PopulationGenerator(H,weight='weight',lower_log_thresh=None,noise=sigma)
    nodes = sorted(H[0].nodes())
    num_nodes = H[0].number_of_nodes()
    ndict = dict([(n,i) for (i,n) in enumerate(nodes)])
    delta = len(H)
    C = pop_comm_generate(PG,iters,nodes=nodes,delta=delta,edge_thresh=None,tqdm_disable=tqdm_disable) 

    clusters = pop_assign_comm(C,max_d)
    comms = defaultdict(list)
    for (i,c) in enumerate(clusters): comms[c].append(i)
    if keep_pop:
        return list(comms.values()),C
    else:     
        return list(comms.values())

def get_number_domains(G,**kwargs):
    G.communities(**kwargs)
    return len(G.comms)

def get_number_domain_isolates(G,iso_thresh=3,**kwargs):
    G.communities(**kwargs)
    #print([c for c in G.comms if len(c) < iso_thresh])
    iso = []
    for c in G.comms:
        if len(c) < iso_thresh:
            iso += c
    return len(iso) / float(G.number_of_nodes())

def get_domain_size_mean(G,remove_isolates=True,iso_thresh=3,**kwargs):
    G.communities(**kwargs)
    comms = G.comms[:]
    if remove_isolates: comms = [c for c in comms if len(c) >= iso_thresh]
    csize = [len(c) for c in comms]
    mu = 0 
    if len(csize) > 0: mu = np.mean(csize)
    return float(mu) / G.number_of_nodes()

def get_domain_size_dispersion(G,remove_isolates=True,iso_thresh=3,**kwargs):
    G.communities(**kwargs)
    comms = G.comms[:]
    if remove_isolates: comms = [c for c in comms if len(c) >= iso_thresh]
    csize = [len(c) for c in comms]
    disp = 0
    if len(csize) > 0: disp = np.divide(np.var(csize),np.mean(csize))
    return disp

def characterize_edges(func):
    def inner(G,**kwargs):
        G.index_communities(**kwargs)
        G.index_pioneer_groups(**kwargs)
        num = 0
        den = 0
        for (u,v) in G.edges():
            num,den = func(G,edge=(u,v),num=num,den=den,**kwargs)
        
        frac = 0
        if den > 0: frac = num / float(den) 
        return frac
    return inner

def is_same_pioneer_group(G,u,v):
    return bool(G.PG['F'][G.PG['fmap'][u],G.PG['fmap'][v]])

@ characterize_edges
def get_frac_pg_contact_intradomain(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge 
    if G.nodes[u]['is_pioneer'] or G.nodes[v]['is_pioneer']: return num,den
    if G[u][v]['id'] < G.max_deg: return num,den 
    same_group = is_same_pioneer_group(G,u,v)
    intradomain = G.comm_index[u] == G.comm_index[v]
    if same_group: den += 1
    if same_group and intradomain: num += 1
    return num,den

@ characterize_edges
def get_frac_pioneer_contact_intradomain(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge 
    if G[u][v]['id'] < G.max_deg: return num,den 
    is_pio = G.nodes[u]['is_pioneer'] or G.nodes[v]['is_pioneer']
    intradomain = G.comm_index[u] == G.comm_index[v]
    if is_pio: den += 1
    if is_pio and intradomain: num += 1
    return num,den

@ characterize_edges
def get_frac_pf_contact_intradomain(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge  
    if G[u][v]['id'] < G.max_deg: return num,den 
    is_pf = (G.nodes[u]['is_pioneer'] != G.nodes[v]['is_pioneer'])
    intradomain = G.comm_index[u] == G.comm_index[v]
    if is_pf: den += 1
    if is_pf and intradomain: num += 1
    return num,den

@ characterize_edges
def get_frac_intradomain_contact_pg(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge 
    if G.nodes[u]['is_pioneer'] or G.nodes[v]['is_pioneer']: return num,den
    if G[u][v]['id'] < G.max_deg: return num,den 
    same_group = is_same_pioneer_group(G,u,v)
    intradomain = G.comm_index[u] == G.comm_index[v]
    if intradomain: den += 1
    if intradomain and same_group: num += 1
    return num,den

@ characterize_edges
def get_frac_interdomain_contact_pg(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge 
    if G.nodes[u]['is_pioneer'] or G.nodes[v]['is_pioneer']: return num,den
    if G[u][v]['id'] < G.max_deg: return num,den 
    same_group = is_same_pioneer_group(G,u,v)
    interdomain = G.comm_index[u] != G.comm_index[v]
    if interdomain: den += 1
    if interdomain and same_group: num += 1
    return num,den

@ characterize_edges
def get_frac_intradomain_contact_pf(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge 
    if G[u][v]['id'] < G.max_deg: return num,den 
    is_pf = (G.nodes[u]['is_pioneer'] != G.nodes[v]['is_pioneer'])
    intradomain = G.comm_index[u] == G.comm_index[v]
    if intradomain: den += 1
    if intradomain and is_pf: num += 1
    return num,den

@ characterize_edges
def get_frac_interdomain_contact_pf(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge 
    if G[u][v]['id'] < G.max_deg: return num,den 
    is_pf = (G.nodes[u]['is_pioneer'] != G.nodes[v]['is_pioneer'])
    interdomain = G.comm_index[u] != G.comm_index[v]
    if interdomain: den += 1
    if interdomain and is_pf: num += 1
    return num,den

@ characterize_edges
def get_frac_intradomain_contact_var(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge 
    intradomain = G.comm_index[u] == G.comm_index[v]
    is_var = G[u][v]['id'] < G.max_deg  
    if intradomain: den += 1
    if intradomain and is_var: num += 1
    return num,den

@ characterize_edges
def get_frac_interdomain_contact_var(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge 
    interdomain = G.comm_index[u] != G.comm_index[v]
    is_var = G[u][v]['id'] < G.max_deg 
    if interdomain: den += 1
    if interdomain and is_var: num += 1
    return num,den

@ characterize_edges
def get_frac_var_contact_intradomain(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge 
    intradomain = G.comm_index[u] == G.comm_index[v]
    is_var = G[u][v]['id'] < G.max_deg
    if is_var: den += 1
    if intradomain and is_var: num += 1
    return num,den

@ characterize_edges
def get_frac_var_contact_interdomain(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge 
    interdomain = G.comm_index[u] != G.comm_index[v]
    is_var = G[u][v]['id'] < G.max_deg  
    if is_var: den += 1
    if interdomain and is_var: num += 1
    return num,den

@ characterize_edges
def get_frac_cons_contact_intradomain(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge 
    interdomain = G.comm_index[u] == G.comm_index[v]
    is_cons = G[u][v]['id'] == G.max_deg 
    if is_cons: den += 1
    if interdomain and is_cons: num += 1
    return num,den

@ characterize_edges
def get_frac_cons_contact_intradomain_weight(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge 
    intradomain = G.comm_index[u] == G.comm_index[v]
    is_cons = G[u][v]['id'] == G.max_deg 
    w = G[u][v]['wnorm'] 
    if is_cons: den += w
    if intradomain and is_cons: num += w
    return num,den

@ characterize_edges
def get_freq_conserved_contact_intradomain(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge
    freq = int(G[u][v]['id'])
    intradomain = G.comm_index[u] == G.comm_index[v]
    if intradomain: 
        den += 1
        num += freq
    return num,den

@ characterize_edges
def get_freq_conserved_contact_interdomain(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge
    freq = int(G[u][v]['id'])
    intradomain = G.comm_index[u] == G.comm_index[v]
    if intradomain: 
        den += 1
        num += freq
    return num,den

@ characterize_edges
def get_freq_conserved_contact_interdomain(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge
    freq = int(G[u][v]['id'])
    interdomain = G.comm_index[u] != G.comm_index[v]
    if interdomain: 
        den += 1
        num += freq
    return num,den

def is_intradomain_freq(G,freq,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge
    is_freq = freq == int(G[u][v]['id'])
    intradomain = G.comm_index[u] == G.comm_index[v]
    if intradomain: den += 1
    if intradomain and is_freq: num += 1 
    return num,den
 
def is_interdomain_freq(G,freq,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge
    is_freq = freq == int(G[u][v]['id'])
    interdomain = G.comm_index[u] != G.comm_index[v]
    if interdomain: den += 1
    if interdomain and is_freq: num += 1 
    return num,den
       
@ characterize_edges
def get_intradomain_f1(G,edge=None,num=None,den=None,**kwargs):
    return is_intradomain_freq(G,1,edge=edge,num=num,den=den,**kwargs)

@ characterize_edges
def get_intradomain_f2(G,edge=None,num=None,den=None,**kwargs):
    return is_intradomain_freq(G,2,edge=edge,num=num,den=den,**kwargs)

@ characterize_edges
def get_intradomain_f3(G,edge=None,num=None,den=None,**kwargs):
    return is_intradomain_freq(G,3,edge=edge,num=num,den=den,**kwargs)

@ characterize_edges
def get_intradomain_f4(G,edge=None,num=None,den=None,**kwargs):
    return is_intradomain_freq(G,4,edge=edge,num=num,den=den,**kwargs)

@ characterize_edges
def get_intradomain_f5(G,edge=None,num=None,den=None,**kwargs):
    return is_intradomain_freq(G,5,edge=edge,num=num,den=den,**kwargs)

@ characterize_edges
def get_intradomain_f6(G,edge=None,num=None,den=None,**kwargs):
    return is_intradomain_freq(G,6,edge=edge,num=num,den=den,**kwargs)

@ characterize_edges
def get_interdomain_f1(G,edge=None,num=None,den=None,**kwargs):
    return is_interdomain_freq(G,1,edge=edge,num=num,den=den,**kwargs)

@ characterize_edges
def get_interdomain_f2(G,edge=None,num=None,den=None,**kwargs):
    return is_interdomain_freq(G,2,edge=edge,num=num,den=den,**kwargs)

@ characterize_edges
def get_interdomain_f3(G,edge=None,num=None,den=None,**kwargs):
    return is_interdomain_freq(G,3,edge=edge,num=num,den=den,**kwargs)

@ characterize_edges
def get_interdomain_f4(G,edge=None,num=None,den=None,**kwargs):
    return is_interdomain_freq(G,4,edge=edge,num=num,den=den,**kwargs)

@ characterize_edges
def get_interdomain_f5(G,edge=None,num=None,den=None,**kwargs):
    return is_interdomain_freq(G,5,edge=edge,num=num,den=den,**kwargs)

@ characterize_edges
def get_interdomain_f6(G,edge=None,num=None,den=None,**kwargs):
    return is_interdomain_freq(G,6,edge=edge,num=num,den=den,**kwargs)


def characterize_conserved_connectivity(func):
    def inner(G,**kwargs):
        G.index_communities(**kwargs)
        num = 0
        den = 0
        for (u,v) in G.degree_graph(G.max_deg).edges():
            num,den = func(G,edge=(u,v),num=num,den=den)
        
        return num / float(den)
    return inner

@characterize_conserved_connectivity
def get_frac_conserved_contacts_are_intradomain(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge
    den += 1
    if G.comm_index[u] == G.comm_index[v]: num += 1
    return num,den

@characterize_conserved_connectivity
def get_frac_pioneer_contacts_are_intradomain(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge
    u_is_pio = G.nodes[u]['is_pioneer'] == 1
    v_is_pio = G.nodes[v]['is_pioneer'] == 1
    is_pio_edge = u_is_pio or v_is_pio 
    intradomain = G.comm_index[u] == G.comm_index[v]
    if is_pio_edge: den += 1
    if is_pio_edge and intradomain: num += 1
    return num,den

@characterize_conserved_connectivity
def get_frac_follower_contacts_are_intradomain(G,edge=None,num=None,den=None,**kwargs):
    (u,v) = edge
    u_is_follower = G.nodes[u]['is_pioneer'] == 0
    v_is_follower = G.nodes[v]['is_pioneer'] == 0
    is_follower_edge = u_is_follower or v_is_follower
    intradomain = G.comm_index[u] == G.comm_index[v]
    if is_follower_edge: den += 1
    if is_follower_edge and intradomain: num += 1
    return num,den



def get_frac_pioneer_group_contacts_are_intradomain(G,**kwargs):
    """
    NOT SURE HOW TO PUT THIS IN THE characterize_intradomain_connectivity DECORATOR
    WITHOUT UNECESSARILY CALLING build_pioneer_groups
    """
    G.index_communities(**kwargs)
    F,fmap = build_pioneer_groups(G)
    num = 0
    den = 0
    for (u,v) in G.degree_graph(G.max_deg).edges():
        if G.nodes[u]['is_pioneer']: continue
        if G.nodes[v]['is_pioneer']: continue
        same_group = F[fmap[u],fmap[v]]
        intradomain = G.comm_index[u] == G.comm_index[v]
        if same_group: den += 1
        if same_group and intradomain: num += 1
    
    return num / float(den)


def get_intradomain_connectivity(G,**kwargs):
    G.index_communities(**kwargs)
    count = 0
    for (u,v) in G.degree_graph(G.max_deg).edges():
        if G.comm_index[u] == G.comm_index[v]: count += 1
    return float(count) / G.deg[G.max_deg].number_of_edges()

def get_vsr(G,**kwargs):
    G.variance_dist(**kwargs)
    return variance_spread_ratio(G.vdist)

def get_fraction_conserved_edges(G,deg=18,**kwargs):
    G.variance_dist(**kwargs)
    return G.vdist[-1]

def get_fraction_variable_edges(G,deg=18,**kwargs):
    G.variance_dist(**kwargs)
    return G.vdist[0]

def get_agent_density(G,max_agents=0,**kwargs):
    assert max_agents > 0
    return G.number_of_nodes() / float(max_agents) 

def get_pioneer_density(G,**kwargs):
    N = float(G.number_of_nodes())
    pio = [n for n in G.nodes() if G.nodes[n]['is_pioneer'] == 1] 
    return len(pio) / N

def get_degree_mean(G,**kwargs):
    deg = [d[1] for d in list(G.degree_graph(**kwargs).degree())]
    return np.mean(deg) / float(G.number_of_nodes())

def get_general_run_degree_mean(G,**kwargs):
    """ Assumes all graphs have the same number of nodes """ 
    if G.graphs is None: G.unzip_graphs()
    deg = []
    for g in G.graphs:
        deg += [d[1] for d in list(g.degree())]
    return np.mean(deg) / float(G.number_of_nodes())

def get_pooled_contact_specificity(G,**kwargs):
    if G.graphs is None: G.unzip_graphs()
    z = [get_contact_specificity(g) for g in G.graphs]
    return np.mean(z)

def get_contact_specificity(G,**kwargs):
    N = G.number_of_nodes()
    M = len([g for g in G.nodes() if G.nodes[g]['is_pioneer'] == 1])
    z = np.zeros(N-M)
    idx = 0
    for u in G.nodes(): 
        if G.nodes[u]['is_pioneer'] == 1: continue
        s = 0. 
        for v in G.neighbors(u):
            s += int(G.nodes[v]['is_pioneer'] == 1)
        z[idx] = s / M

        idx += 1
    print(z)
    return 1 - z.mean()

def get_pooled_pioneer_contact_probability(G,**kwargs):
    if G.graphs is None: G.unzip_graphs()
    z = [get_pioneer_contact_probability(g) for g in G.graphs]
    return np.mean(z)

def get_pioneer_contact_probability(G,**kwargs):
    N = G.number_of_nodes()
    M = len([g for g in G.nodes() if G.nodes[g]['is_pioneer'] == 1])
    z = np.zeros(N-M)
    idx = 0
    for u in G.nodes(): 
        if G.nodes[u]['is_pioneer'] == 1: continue
        for v in G.neighbors(u):
            if G.nodes[v]['is_pioneer'] == 1:
                z[idx] = 1
                break
        idx += 1

    return np.mean(z)

def get_clustering(G,attr=None,**kwargs):
    z = np.zeros(G.number_of_nodes()) 
    C = nx.clustering(G.degree_graph(**kwargs),weight=attr)
    for k,v, in C.items(): z[G.node_index[k]] = v
    return z

def get_neighborhood_reproducibility(G,**kwargs):
    z = np.zeros((G.number_of_nodes(),G.max_deg))
    for u in G.nodes():
        for v in G.neighbors(u):
            z[G.node_index[u],G[u][v]['id']-1] += 1
    return z / z.sum(axis=1)[:,None]

def get_neighborhood_conserved(G,**kwargs):
    if G.neigh_repro is None: G.get_neighborhood_reproduciblity()
    return G.neigh_repro[:,-2:].sum(1) 

def get_neighborhood_variance(G,**kwargs):
    if G.neigh_repro is None: G.get_neighborhood_reproduciblity()
    return G.neigh_repro[:,:2].sum(1) 
 
def get_target_array(G,**kwargs):
    tidmap = dict([(G.nodes[n]['target_id'],n) for n in G.nodes()])
    tarray = np.zeros([G.number_of_nodes(),2],dtype=np.uint8)
    for u in G.nodes():
        for t in G.nodes[u]['targets']:
            tarray[G.node_index[u],0] += 1
            #print(t,tidmap[t],G.node_index[aidmap[t]])
            tarray[G.node_index[tidmap[t]],1] += 1
    return tarray 

def get_contact_length(G,**kwargs):
    cons_edges = [e for e in G.degree_graph(**kwargs).edges()]
    z = np.zeros(G.number_of_nodes()) 
    for (u,v) in cons_edges:
        w = G[u][v]['wnorm']
        z[G.node_index[u]] += w
        z[G.node_index[v]] += w
    return z

def get_attractee_contact_length(G,**kwargs):
    cons_edges = [e for e in G.degree_graph(**kwargs).edges()]
    z = np.zeros(G.number_of_nodes()) 
    for (u,v) in cons_edges:
        w = G[u][v]['wnorm']
        uid = G.nodes[u]['agent_id']
        vid = G.nodes[v]['agent_id']
        c1 = uid in G.nodes[v]['targets']
        c2 = vid in G.nodes[u]['targets']
        if c1 or c2:
            z[G.node_index[u]] += w
            z[G.node_index[v]] += w
        #if uid in G.nodes[v]['targets']: z[G.node_index[u]] += w
        #if vid in G.nodes[u]['targets']: z[G.node_index[v]] += w

    return z

def get_non_attractee_contact_length(G,**kwargs):
    cons_edges = [e for e in G.degree_graph(**kwargs).edges()]
    z = np.zeros(G.number_of_nodes()) 
    for (u,v) in cons_edges:
        w = G[u][v]['wnorm']
        uid = G.nodes[u]['agent_id']
        vid = G.nodes[v]['agent_id']
        c1 = uid in G.nodes[v]['targets']
        c2 = vid in G.nodes[u]['targets']
        if not c1 and not c2:
            z[G.node_index[u]] += w
            z[G.node_index[v]] += w
        #if uid not in G.nodes[v]['targets']: z[G.node_index[u]] += w
        #if vid not in G.nodes[u]['targets']: z[G.node_index[v]] += w

    return z

def get_attractee_degree(G,**kwargs):
    cons_edges = [e for e in G.degree_graph(**kwargs).edges()]
    z = np.zeros(G.number_of_nodes()) 
    for (u,v) in cons_edges:
        uid = G.nodes[u]['agent_id']
        vid = G.nodes[v]['agent_id'] 
        c1 = uid in G.nodes[v]['targets']
        c2 = vid in G.nodes[u]['targets']
        if c1 or c2:
            z[G.node_index[u]] += 1
            z[G.node_index[v]] += 1
        #if uid in G.nodes[v]['targets']: z[G.node_index[u]] += 1
        #if vid in G.nodes[u]['targets']: z[G.node_index[v]] += 1

    return z

def get_non_attractee_degree(G,**kwargs):
    cons_edges = [e for e in G.degree_graph(**kwargs).edges()]
    z = np.zeros(G.number_of_nodes()) 
    for (u,v) in cons_edges:
        uid = G.nodes[u]['agent_id']
        vid = G.nodes[v]['agent_id']
        c1 = uid in G.nodes[v]['targets']
        c2 = vid in G.nodes[u]['targets']
        if not c1 and not c2:
            z[G.node_index[u]] += 1
            z[G.node_index[v]] += 1
        #if uid not in G.nodes[v]['targets']: z[G.node_index[u]] += 1
        #if vid not in G.nodes[u]['targets']: z[G.node_index[v]] += 1

    return z

def get_sorted_contact_length(G,weight='weight',**kwargs):
    H = G.degree_graph(**kwargs) 
    cl = {}
    for u in H.nodes():
        cl[u] = sorted([H[u][v][weight] for v in H.neighbors(u)],reverse=True)
    return cl


def get_degree(G,**kwargs):
    z = np.zeros(G.number_of_nodes()) 
    for (n,d) in list(G.degree_graph(**kwargs).degree()):
        z[G.node_index[n]] = d
    return z

def get_fraction_conserved(G,max_deg=None,**kwargs):
    '''fraction conserved'''
    Z = np.zeros((G.number_of_nodes(),max_deg))
    for i in range(1,max_deg+1):
        for (n,d) in list(G.degree_graph(deg=i).degree()):
            Z[G.node_index[n],i-1] = d
    return Z[:,-1] / Z.sum(1)

def get_fraction_poles(G,max_deg=None,**kwargs):
    '''fraction conserved'''
    Z = np.zeros((G.number_of_nodes(),max_deg))
    for i in range(1,max_deg+1):
        for (n,d) in list(G.degree_graph(deg=i).degree()):
            Z[G.node_index[n],i-1] = d
    return (Z[:,0] + Z[:,-1]) / Z.sum(1)

def get_total_degree(G,max_deg=None,**kwargs):
    Z = np.zeros(G.number_of_nodes())
    nmap = dict([(n,i) for (i,n) in enumerate(sorted(G.nodes()))]) 
    for i in range(1,max_deg+1):
        for (n,d) in list(G.degree()):
            Z[nmap[n]] = d
    return Z / (G.number_of_nodes()-1)

def get_population_degree(G,max_deg=None,**kwargs):
    Z = np.zeros((G.number_of_nodes(),max_deg),dtype=int)
    for i in range(1,max_deg+1):
        for (n,d) in list(G.degree_graph(deg=i).degree()):
            Z[G.node_index[n],i-1] = d
    return Z.sum(1)

def get_specificity(G,**kwargs):
    Z = np.zeros(G.number_of_nodes())
    for (idx,u) in enumerate(sorted(G.nodes())):
        f_or_p = [1,-1][G.nodes[u]['is_pioneer']]
        Z[idx] = f_or_p * G.nodes[u]['specificity']
    return Z

def get_num_pioneers_per_follower(G,**kwargs):
    Z = -1 * np.ones(G.number_of_nodes(),dtype=int)
    for (idx,u) in enumerate(sorted(G.nodes())):
        if G.nodes[u]['is_pioneer'] == 1: continue
        Z[idx] = len(G.nodes[u]['target'])
    return Z

def get_num_pioneer_contacts_per_follower(G,max_deg=None,
                                          pct_thresh=0,**kwargs):
    Z = -1 * np.ones(G.number_of_nodes(),dtype=int)
    pio = [u for u in G.nodes() if G.nodes[u]['is_pioneer']]

    sorted_nodes = sorted(G.nodes())
    weight = []
    for u in sorted_nodes:
        if G.nodes[u]['is_pioneer'] == 1: continue
        for v in pio:
            if not G.has_edge(u,v): continue
            if G[u][v]['id'] < max_deg: continue
            tot_contact = sum(map(int,G[u][v]['g_index_weight'].split('-')))
            weight.append(tot_contact)
    
    weight = sorted(weight)
    wdx = int(len(weight) * pct_thresh)
    wmin = weight[wdx]
    
    for (idx,u) in enumerate(sorted_nodes):
        if G.nodes[u]['is_pioneer'] == 1: continue
        count = 0 
        for v in pio: 
            if not G.has_edge(u,v): continue 
            if G[u][v]['id'] < max_deg: continue
            contact = sum(map(int,G[u][v]['g_index_weight'].split('-')))
            if contact <= wmin: continue
            count += 1
        Z[idx] = count
    
    return Z

def similarity(E):
    n,m = E.shape
    Z = np.zeros((n*(n-1))//2)
    for idx,(i,j) in enumerate(combinations(list(range(n)),2)):
        psum = E[i,:] + E[j,:]
        psum[psum>0] = 1
        psum = psum.sum()
        Z[idx] = np.dot(E[i,:],E[j,:]) / psum
    return Z 

def _pioneers(G,**kwargs):
    return np.sort([int(u) for u in G.nodes() if G.nodes[u]['is_pioneer'] == 1])

def _follower_groups(G,pioneers=None,**kwargs):
    if pioneers is None: pioneers = _pioneers(G)
    pioneers = sorted(pioneers)
    fg = [[] for i in range(len(pioneers))]
    for u in G.nodes():
        if G.nodes[u]['is_pioneer'] == 1: continue
        for v in G.nodes[u]['target']: fg[v].append(u)
    return fg

def _pg_breakdown_reproducibility(G,max_deg,pio=None,fg=None,**kwargs):
    Z = np.zeros((5,max_deg+1))
    if pio is None: pio = _pioneers(G) 
    if fg is None: fg = _follower_groups(G,pioneers=pio)
    increment_pg_reproducibility(G,Z[0,:],iter_pg_p_f,pioneers=pio,followers=fg)
    increment_pg_reproducibility(G,Z[1,:],iter_pg_f_f,followers=fg)
    increment_pg_reproducibility(G,Z[2,:],iter_pg_p_np,pioneers=pio,followers=fg)
    increment_pg_reproducibility(G,Z[3,:],iter_pg_p_nf,pioneers=pio,followers=fg)
    increment_pg_reproducibility(G,Z[4,:],iter_pg_f_nf,followers=fg)
    return Z

def pg_breakdown_reproducibility(G,max_deg,pio=None,fg=None,**kwargs):
    elist = pg_edge_list(G,max_deg,pio=pio,fg=fg) 
    Z = np.zeros((5,max_deg))
    for (label,idx,u,v) in elist:
        if label < 0: print(label,idx,u,v)
        Z[label,idx-1] += 1
    return Z

def pg_breakdown_domains(G,max_deg,pio=None,fg=None,**kwargs):
    if G.comms is None: G.index_communities(deg=max_deg)
    elist = pg_edge_list(G,max_deg,pio=pio,fg=fg) 
    Z = np.zeros((5,2))
    for (label,idx,u,v) in elist:
        if label < 0: print(label,idx,u,v)
        if idx < max_deg: continue 
        ddx = 1
        if G.comm_index[u] == G.comm_index[v]: ddx = 0
        Z[label,ddx] += 1
    return Z

def pg_domain_similarity(G,max_deg,pio=None,fg=None,
                         flag=False,container=None,**kwargs):
    from collections import defaultdict  
    if G.comms is None: G.index_communities(deg=max_deg)
    Z = np.zeros((2,2))
    fdx = 0 
    frec = defaultdict(float)
    if flag == True: print(G.number_of_nodes())
    
    record = False
    if container is not None: record = True
    for (u,v) in combinations(G.nodes(),2):
        utarget = set(G.nodes[u]['target'])
        vtarget = set(G.nodes[v]['target'])
        same_pg = int(len(utarget & vtarget) > 0)   
        same_dom = int(G.comm_index[u] == G.comm_index[v])
        Z[same_pg,same_dom] += 1
        
        if record: container[tuple(sorted((u,v)))] = [same_pg,same_dom]
        if flag and  same_pg == 1 and same_dom == 0: 
            frec[u] += 1
            frec[v] += 1
            fdx += 1
            gdx = 0
            #if G.has_edge(u,v): gdx = G[u][v]['id']
            #print(u,v,G.has_edge(u,v),gdx)
        #if same_pg and not same_dom: print(u,v,same_pg,same_dom)
    #if flag:
    #    for (k,v) in frec.items():
    #        print(k,v,v/fdx)

    return Z


def jaccard_index(list1, list2):
    """
    Calculates the Jaccard index (similarity) between two lists.

    Args:
        list1: The first list.
        list2: The second list.

    Returns:
        The Jaccard index as a float, or 0.0 if both lists are empty.
    """
    set1 = set(list1)
    set2 = set(list2)

    intersection = set1.intersection(set2)
    union = set1.union(set2)

    if not union:  # Handle the case where both lists are empty
        return 0.0
    else:
        return len(intersection) / len(union)

def pg_edge_list(G,max_deg,pio=None,fg=None,**kwargs):
    Z = []
    if pio is None: pio = _pioneers(G) 
    if fg is None: fg = _follower_groups(G,pioneers=pio)
    nmap = dict([(u,i) for (i,u) in enumerate(sorted(G.nodes()))])
    P = np.zeros(G.number_of_nodes())
    F = np.zeros((G.number_of_nodes(),len(pio)))
    for (pdx,p) in enumerate(pio): 
        P[nmap[p]] = 1
        F[nmap[p],pdx] = 1
    for pdx,_fg in enumerate(fg):
        for f in _fg: F[nmap[f],pdx] = 1
    
    Z = []
    for (u,v) in G.edges():
        #label = -1
        idx = G[u][v]['id']
        if has_2_p(u,v,P,nmap):
            label = 2
        elif has_p(u,v,P,nmap) and in_same_pg(u,v,F,nmap):
            label = 0
        elif has_p(u,v,P,nmap) and not in_same_pg(u,v,F,nmap):
            label = 3
        elif not has_p(u,v,P,nmap) and in_same_pg(u,v,F,nmap):
            label = 1
        elif not has_p(u,v,P,nmap) and not in_same_pg(u,v,F,nmap):
            label = 4 
        else:
            print(label)
        Z.append((label,idx,u,v))
    
    return Z

def has_p(u,v,P,nmap):
    return P[nmap[u]] + P[nmap[v]] > 0

def has_2_p(u,v,P,nmap):
    return P[nmap[u]] + P[nmap[v]] == 2

def in_same_pg(u,v,F,nmap):
    return np.dot(F[nmap[u]],F[nmap[v]]) > 0

def _pg_breakdown_domains(G,max_deg,pio=None,fg=None,**kwargs):
    Z = np.zeros((5,2))
    if pio is None: pio = _pioneers(G) 
    if fg is None: fg = _follower_groups(G,pioneers=pio)
    if G.comms is None: G.index_communities(deg=max_deg)
    increment_pg_domain(G,Z[0,:],iter_pg_p_f,pioneers=pio,followers=fg)
    increment_pg_domain(G,Z[1,:],iter_pg_f_f,followers=fg)
    increment_pg_domain(G,Z[2,:],iter_pg_p_np,pioneers=pio,followers=fg)
    increment_pg_domain(G,Z[3,:],iter_pg_p_nf,pioneers=pio,followers=fg)
    increment_pg_domain(G,Z[4,:],iter_pg_f_nf,followers=fg)
    return Z    

def pg_breakdown_index(G,pio=None,fg=None,**kwargs):
    Z = []
    if pio is None: pio = _pioneers(G) 
    if fg is None: fg = _follower_groups(G,pioneers=pio)
    index_pg_contacts(G,Z,0,iter_pg_p_f,pioneers=pio,followers=fg,**kwargs)
    index_pg_contacts(G,Z,1,iter_pg_f_f,pioneers=pio,followers=fg,**kwargs)
    index_pg_contacts(G,Z,2,iter_pg_p_np,pioneers=pio,followers=fg,**kwargs)
    index_pg_contacts(G,Z,3,iter_pg_p_nf,pioneers=pio,followers=fg,**kwargs)
    index_pg_contacts(G,Z,4,iter_pg_f_nf,pioneers=pio,followers=fg,**kwargs)
    return Z    

def iter_pg_p_f(pioneers: list | tuple =[] ,followers: list | tuple = [],**kwargs):
    for (pdx,u) in enumerate(pioneers):
        for v in followers[pdx]:
            yield u,v
 
def iter_pg_f_f(followers: list | tuple = [], **kwargs):
    for F in followers:
        for (u,v) in combinations(F,2):
            if u == v: continue
            yield u,v

def iter_pg_p_np(pioneers: list | tuple = [], **kwargs):
    for (u,v) in combinations(pioneers,2):
        yield u,v

def iter_pg_p_nf(pioneers: list | tuple = [], followers: list | tuple = [], **kwargs):
    for (pdx,u) in enumerate(pioneers):
        for (fdx,F) in enumerate(followers):
            if pdx == fdx: continue
            for v in F:
                yield u,v

def iter_pg_f_nf(followers: list | tuple = [],**kwargs):
    for (F0,F1) in combinations(followers,2):
        for (u,v) in product(F0,F1):
            if u == v: continue
            yield u,v

def increment_pg_reproducibility(G,container,iter_ptr,**kwargs):
    for u,v in iter_ptr(**kwargs):
        idx = 0
        if G.has_edge(u,v): idx = G[u][v]['id']
        container[idx] += 1 

def retrieve_pg_edges(G,container,iter_ptr,label=0,**kwargs):
    for u,v in iter_ptr(**kwargs):
        if G.has_edge(u,v): 
            idx = G[u][v]['id']
            container.append((label,idx,u,v))
    print(len(container))

def increment_pg_domain(G,container,iter_ptr,**kwargs):
    for u,v in iter_ptr(**kwargs):
        if G.has_edge(u,v) and G[u][v]['id'] == G.max_deg:
            if G.comm_index[u] == G.comm_index[v]:
                container[0] += 1
            else:
                container[1] += 1

def _index_func(G,u,v):
    return []

def index_pg_contacts(G,container,idx,iter_ptr,index_func=_index_func,**kwargs):
    for u,v in iter_ptr(**kwargs):
        rid = 0
        if G.has_edge(u,v): rid = G[u][v]['id']
        container.append([u,v,G.nodes[u]['is_pioneer'],G.nodes[v]['is_pioneer'],rid,idx]+index_func(G,u,v))

def get_node_ic(G,**kwargs):
    G.index_communities(nodes=sorted(G.nodes()),**kwargs)
    deg = get_degree(G,**kwargs)
    deg[deg==0] = 1 
    ic = np.zeros(deg.shape)
    for (u,v) in G.degree_graph(G.max_deg).edges():
        if G.comm_index[u] == G.comm_index[v]: 
            ic[G.node_index[u]] += 1
            ic[G.node_index[v]] += 1
    return np.divide(ic,deg)

def _get_contact_length(func):
    def inner(G,**kwargs):
        nodes = func(G)
        cons_edges = [e for e in G.degree_graph(**kwargs).edges()]
        z = np.zeros(len(nodes))
        for (u,v) in cons_edges:
            w = G[u][v]['wnorm']
            udx = nodes.get(u,-1)
            if udx > -1: z[udx] += w
            vdx = nodes.get(v,-1)
            if vdx > -1: z[vdx] += w
        return np.mean(z)
    
    return inner

@ _get_contact_length
def get_contact_length_pioneers(G,label_pio=True,**kwargs):
    nodes = [n for n in G.nodes() if G.nodes[n]['is_pioneer'] == 1]
    return dict([(n,idx) for (idx,n) in enumerate(nodes)])

@ _get_contact_length
def get_contact_length_followers(G,label_pio=True,**kwargs):
    nodes = [n for n in G.nodes() if G.nodes[n]['is_pioneer'] == 0]
    return dict([(n,idx) for (idx,n) in enumerate(nodes)])


def _get_betweenness(func):
    def inner(G,**kwargs):
        nodes = func(G)
        z = np.zeros(len(nodes))
        C = nx.betweenness_centrality(G.degree_graph(**kwargs),normalized=True,weight=kwargs['weight'])
        for (idx,p) in enumerate(nodes): z[idx] = C[p]
        return np.mean(z)
    
    return inner

@ _get_betweenness
def get_betweenness(G,label_pio=True,**kwargs):
    return list(G.nodes())

@ _get_betweenness
def get_betweenness_pioneers(G,label_pio=True,**kwargs):
    return [n for n in G.nodes() if G.nodes[n]['is_pioneer'] == 1]

@ _get_betweenness
def get_betweenness_followers(G,label_pio=True,**kwargs):
    return [n for n in G.nodes() if G.nodes[n]['is_pioneer'] == 0]



def get_precision(G,**kwargs):
    z = np.zeros(G.number_of_nodes())
    idx = []
    for u in G.nodes():
        if G.nodes[u]['pioneer_id'] > -1: continue
        num_targs = len(G.nodes[u]['targets'])
        for t in G.nodes[u]['targets']:
            z[G.node_index[u]] += 1 
        
        if num_targs > 0:
            z[G.node_index[u]] /= float(num_targs)
            idx.append(G.node_index[u])
    return np.mean(z[idx])

def get_time_on_target(G,tnorm=200,**kwargs):
    tot = []
    if G.graphs is None: G.unzip_graphs()
    
    for g in G.graphs:
        for u in g.nodes():
            if G.nodes[u]['target_id'] > -1: continue
            for t in G.nodes[u]['targets']:
                t = t + 1
                _tot = 0
                if g.has_edge(u,t): _tot = g[u][t]['weight']
                tot.append(_tot)
    return np.mean(tot) / tnorm

def get_target_hit_rate(G,**kwargs):
    tot = []
    if G.graphs is None: G.unzip_graphs()
    
    for g in G.graphs:
        for u in g.nodes():
            if G.nodes[u]['target_id'] > -1: continue
            for t in G.nodes[u]['targets']:
                t = t + 1
                _tot = 0
                if g.has_edge(u,t): _tot = g[u][t]['weight']
                tot.append(float(int(_tot>0)))
    print(tot,len(tot))
    return np.mean(tot)

def build_pioneer_groups(G):
    H = G.degree_graph(G.max_deg)
    pios = [n for n in G.nodes() if G.nodes[n]['is_pioneer']==1]
    pmap = dict([(p,idx) for (idx,p) in enumerate(pios)])
    foll = [n for n in G.nodes() if G.nodes[n]['is_pioneer']==0]
    fmap = dict([(f,idx) for (idx,f) in enumerate(foll)])
    
    P = np.zeros((len(foll),len(pios)),dtype=np.uint8)
    for (f,i) in fmap.items():
        for (p,j) in pmap.items():
            P[i,j] = H.has_edge(f,p)
    F = P.dot(P.T)
    F[F>0] = 1
    
    return F,fmap


def _get_pioneer_group_reproducibility(func):
    def inner(G,**kwargs):
        F,fmap = build_pioneer_groups(G)
        freq = []
        for (u,v,r) in G.edges(data='id'):
            if G.nodes[u]['is_pioneer']: continue
            if G.nodes[v]['is_pioneer']: continue
            same_group = F[fmap[u],fmap[v]]
            func(same_group=same_group,reproducibility=r,container=freq)
        return freq
    
    return inner

def _get_pioneer_group_mean_reproducibility(func):
    def inner(G,**kwargs):
        freq = func(G,**kwargs)
        return np.mean(freq) / G.max_deg
    return inner

def _get_pioneer_group_high_variance(func):
    def inner(G,label_pio=True,**kwargs):
        freq = func(G,**kwargs)
        var = 0
        if len(freq) > 0: 
            vals,counts = np.unique(np.array(freq),return_counts=True)
            if vals[0] == 1:
                var = float(counts[0]) / len(freq)
        return var
    
    return inner

@ _get_pioneer_group_mean_reproducibility
@ _get_pioneer_group_reproducibility
def inter_pioneer_group_reproducibility(*args,same_group=None,reproducibility=None,container=None):
    if not same_group: container.append(reproducibility)

@ _get_pioneer_group_mean_reproducibility
@ _get_pioneer_group_reproducibility
def intra_pioneer_group_reproducibility(*args,same_group=None,reproducibility=None,container=None):
    if same_group: container.append(reproducibility)

@ _get_pioneer_group_high_variance
@ _get_pioneer_group_reproducibility
def inter_pioneer_group_high_variance(*args,same_group=None,reproducibility=None,container=None):
    if not same_group: container.append(reproducibility)

@ _get_pioneer_group_high_variance
@ _get_pioneer_group_reproducibility
def intra_pioneer_group_high_variance(*args,same_group=None,reproducibility=None,container=None):
    if same_group: container.append(reproducibility)


def get_inter_target_reproducibility(G,**kwargs):
    """
    Note the target comparison needs to be fixed for multiple targets
    ** This assessess the pioneer group ('domain') where followers have different pioneer targets
    CANNOT BE USED FOR EMPIRICAL DATA
    """
    freq = []
    for (u,v,f) in G.edges(data='id'):
        if G.nodes[u]['is_pioneer']: continue
        if G.nodes[v]['is_pioneer']: continue
        utarget = set(G.nodes[u]['target'])
        vtarget = set(G.nodes[v]['target'])
        if len(utarget & vtarget) == 0: freq.append(f)
    return np.mean(freq)
 
def get_intra_target_reproducibility(G,**kwargs):
    """
    Note the target comparison needs to be fixed for multiple targets
    ** This assessess the pioneer group ('domain') where followers have the same pioneer targets
    CANNOT BE USED FOR EMPIRICAL DATA
    """
    freq = []
    for (u,v,f) in G.edges(data='id'):
        if G.nodes[u]['is_pioneer']: continue
        if G.nodes[v]['is_pioneer']: continue
        utarget = set(G.nodes[u]['target'])
        vtarget = set(G.nodes[v]['target'])
        if len(utarget & vtarget) > 0: freq.append(f)
    return np.mean(freq)
 
def get_inter_target_variance(G,**kwargs):
    """
    Note the target comparison needs to be fixed for multiple targets
    """
    freq = []
    for (u,v,f) in G.edges(data='id'):
        if G.nodes[u]['is_pioneer']: continue
        if G.nodes[v]['is_pioneer']: continue
        utarget = set(G.nodes[u]['target'])
        vtarget = set(G.nodes[v]['target'])
        if len(utarget & vtarget) == 0: freq.append(f)
    var = 0
    if len(freq) > 0: 
        vals,counts = np.unique(np.array(freq),return_counts=True)
        if vals[0] == 1:
            var = float(counts[0]) / len(freq)
    return var
 
def get_intra_target_variance(G,**kwargs):
    """
    Note the target comparison needs to be fixed for multiple targets
    """
    freq = []
    for (u,v,f) in G.edges(data='id'):
        if G.nodes[u]['is_pioneer']: continue
        if G.nodes[v]['is_pioneer']: continue
        utarget = set(G.nodes[u]['target'])
        vtarget = set(G.nodes[v]['target'])
        if len(utarget & vtarget) > 0: freq.append(f)
    var = 0
    if len(freq) > 0: 
        vals,counts = np.unique(np.array(freq),return_counts=True)
        if vals[0] == 1:
            var = float(counts[0]) / len(freq)
    return var
             

def get_gini_contact_length(G,**kwargs):
    x = get_contact_length(G,**kwargs)
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (2*len(x)**2 * np.mean(x))

def get_gini_degree(G,**kwargs):
    x = np.array([d[1] for d in list(G.degree_graph(**kwargs).degree())])
    total = 0
    for i, xi in enumerate(x[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (2*len(x)**2 * np.mean(x))

def get_degree_dispersion(G,**kwargs):
    deg = [d[1] for d in list(G.degree_graph(**kwargs).degree())]
    disp = 0 
    mu = np.mean(deg) 
    if mu > 0: disp = np.var(deg) / mu
    return disp

def get_mean_followers_per_pioneer(G,**kwargs):
    deg = []
    H = G.degree_graph(G.max_deg)
    for n in H.nodes():
        if G.nodes[n]['is_pioneer'] == 0: continue
        _deg = 0 
        for m in H.neighbors(n):
            if G.nodes[m]['is_pioneer'] == 0: _deg += 1
        deg.append(_deg)
    mu = 0
    if len(deg) > 0: mu = np.mean(deg)
    num_followers = len([n for n in G.nodes() if G.nodes[n]['is_pioneer'] == 0])
    return mu 

def get_mean_pioneers_per_follower(G,**kwargs):
    deg = []
    H = G.degree_graph(G.max_deg)
    #H = G 
    for n in H.nodes():
        if G.nodes[n]['is_pioneer'] == 1: continue
        _deg = 0 
        for m in H.neighbors(n):
            if G.nodes[m]['is_pioneer'] == 1: _deg += 1
        deg.append(_deg)
    mu = 0
    #print(deg,np.sum(deg),len(deg),np.sum(deg)/float(len(deg)),np.mean(deg))
    if len(deg) > 0: mu = np.mean(deg)
    num_pioneers = len([n for n in G.nodes() if G.nodes[n]['is_pioneer'] == 1])
    return mu 

def get_degree_mean_pioneer(G,**kwargs):
    deg = [d[1] for d in list(G.degree_graph(**kwargs).degree()) if G.nodes[d[0]]['is_pioneer'] == 1]
    mu = 0
    if len(deg) > 0: mu = np.mean(deg)
    return mu / float(G.number_of_nodes())

def get_degree_mean_follower(G,label_pio=True,**kwargs):
    if label_pio: label_pioneers(G,**kwargs) 
    deg = [d[1] for d in list(G.degree_graph(**kwargs).degree()) if G.nodes[d[0]]['is_pioneer'] == 0]
    mu = 0
    if len(deg) > 0: mu = np.mean(deg)
    return mu / float(G.number_of_nodes())

def get_average_clustering(G,attr=None,**kwargs):
    C = nx.clustering(G,weight=attr)
    return np.mean([v for k,v in C.items()])

def get_degree_assortivity(G,attr=None,**kwargs):
    val = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore","RuntimeWarning")
        try: 
            val = nx.degree_assortativity_coefficient(G)
        except:
            pass
    return val

def get_pioneer_domain_predict(G,label_pio=True,**kwargs):
    def secondary_matrix(G,fmap,pmap):
        X = np.zeros((len(fmap),len(pmap)))
        for (f,idx) in fmap.items():
            for (p,jdx) in pmap.items():
                X[idx,jdx] = int(G.deg[G.max_deg].has_edge(f,p))
        
        _X = X @ X.T
        _X[_X>0] = 1
        np.fill_diagonal(_X,0)
        return _X

    if label_pio: label_pioneers(G,**kwargs) 
    #pio,fhd,fol = G.pioneer_test_split()
    pio,fol = G.pioneer_test_split_0()
    pmap = dict([(n,i) for (i,n) in enumerate(pio)])
    #cmap = dict([(n,i) for (i,n) in enumerate(fhd)])
    fmap = dict([(n,i) for (i,n) in enumerate(fol)])
    
    X1 = secondary_matrix(G,fmap,pmap)
    #X2 = secondary_matrix(G,fmap,cmap)
    
    G.index_communities(**kwargs) 
    Y = np.zeros((len(fmap),len(fmap)))
    for (f1,idx) in fmap.items():
        for (f2,jdx) in fmap.items():
            if f1 == f2: continue
            Y[idx,jdx] = int(G.comm_index[f1] == G.comm_index[f2])
    
    P = X1 * Y
    #C = X2 * Y
    
    #return np.log2(P.sum()/float(C.sum()))
    return P.sum()/float(Y.sum())


def label_pioneers(G,num_pioneers=-1,**kwargs):
    if num_pioneers > -1:
        nx.set_node_attributes(G,0,'is_pioneer') 
        for i in range(num_pioneers): 
            if G.has_node(i+1): G.nodes[i+1]['is_pioneer']= 1

def variance_dist(G,norm=True):
    weights = [w for (u,v,w) in G.edges(data='id')]
    max_weight = max(weights)
    data = np.zeros(max_weight)
    for w in weights: data[w-1] += 1
    if norm: data = data / data.sum() 
    return data

def variance_spread_ratio(data):
    drange = len(data) - 1
    dsort = np.argsort(data)[::-1]
    delta = abs(dsort[0] - dsort[1])
    vsr = delta / float(drange)
    return vsr 

def process_graphs(G): 
    """
    For each graph g in list of graphs G, removes the smallest 35% edges by weight,
    normalizes the edges weights into attribute 'wnorm', and builds a consensus graph
    where edges have the 'id' attribute which give the frequency  of reproducibility of that
    edge across the all graphs in G

    Note: This function has not been unit tested because it simply pipes the graphs through
    tb_popgraphs functions which have already been unit tested
    
    Input:
    G : list
        List of networkx graphs
    
    Returns:
    --------
    A consensus graph
    """
    H = [] 
    for g in G:
        gm.filter_graph_edge(g,35)
        gm.normalize_edge_weight(g)
        H.append(g)
    
    C = [graph.consensus(H,i+1,weight=['weight','wnorm']) for i in range(len(G))]
    M = graph.index_merge(C)
    keys = G[0].nodes[1].keys() 
    for key in keys: 
        attr = nx.get_node_attributes(G[0],key) 
        nx.set_node_attributes(M,attr,key)
    graph.zip_index_consensus_graph_edges(M,H) 
    return M

def degree_graph(G,degree):
    H = nx.Graph()
    H.add_nodes_from(list(G.nodes())) ##Important, H and G should have identical nodes
    na = nx.get_node_attributes(G,'is_pioneer')
    nx.set_node_attributes(H,na,'is_pioneer')
    for (u,v,attr) in G.edges(data=True):
        deg = G[u][v]['id']
        if deg != degree: continue
        H.add_edge(u,v)#,weight=G[u][v]['weight'])
        for (key,val) in attr.items(): H[u][v][key] = val
    return H 


