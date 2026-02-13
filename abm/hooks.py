"""
@name:
@description:
    Module for keeping data hooks
     
    Wed 05 Oct 2022 05:27:45 PM EDT
    GraphHook1 is now deprecated  

    Wed 05 Oct 2022 05:27:52 PM EDT
    GraphVolHook and GraphVolHook2 both recover identical graphs.
    See test/test_hooks.py and test/test_graphhooks.py

    However, GrahpVolHook2 runs is half the time.
    See test/test_graphhooks.py

    Moving forward, GraphVolHook2 will be the standard hook.

    GraphVolHook will be removed at a later time.

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import networkx as nx
import numpy as np

import abm.model.model_base as mb

class GraphHook:
    def __init__(self):
        self.G = nx.Graph()

    def __call__(self):
        return self.G

    def pre_action(self,model):
        self.G.add_nodes_from(model.get_agent_attributes())
        self.num_rows = model.num_rows
        self.num_cols = model.num_cols

    def post_action(self,model):
        for V in mb.iter_volume(model):
            for (i,j) in self.sweep():
                if V[i,j] == 0: continue
                for (u,v) in self.iter_neighborhood(i,j):
                    if V[u,v] == 0: continue
                    a = V[i,j]
                    b = V[u,v]
                    if not self.G.has_edge(a,b): self.G.add_edge(a,b,weight=0)
                    self.G[a][b]['weight'] += 1
         
    def sweep(self):
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                yield(i,j)

    def iter_neighborhood(self,i,j):
        neigh = [(i,j+1),(i+1,j-1),(i+1,j),(i+1,j+1)]
        for (_i,_j) in neigh:
            if _i >= self.num_rows: continue
            if _j < 0: continue
            if _j >= self.num_cols: continue
            yield (_i,_j)
 

class GraphVolHook:
    def __init__(self):
        self.G = nx.Graph()

    def __call__(self):
        return self.G

    def pre_action(self,model):
        pioneers = [ (n,{'is_pioneer':1}) for n in model.get_pioneers()]
        followers = [(n,{'is_pioneer':0}) for n in model.get_followers()]
        self.G.add_nodes_from(pioneers + followers)
        self.num_rows = model.num_rows
        self.num_cols = model.num_cols
   
    def post_action(self,model):
        #for k in range(model.vol_analysis_start,model.vol_analysis_end):
        for V in mb.iter_volume(model): 
            #for i,j in zip(*np.where(V[:,:,k] > 0)):
            for i,j in zip(*np.where(V > 0)):
                #u = model.V[i,j,k] 
                u = V[i,j]
                for ni,nj in model.grid.neighborhood(i,j):
                    #v = model.V[ni,nj,k]
                    v = V[ni,nj]
                    if v == 0: continue
                    if u == v: continue
                    if not self.G.has_edge(u,v): self.G.add_edge(u,v,weight=0)
                    self.G[u][v]['weight'] += 0.5

class GraphVolHook2:
    def __init__(self):
        self.G = nx.Graph()

    def __call__(self):
        return self.G

    def pre_action(self,model):
        #pioneers = [ (n,{'is_pioneer':1}) for n in model.get_pioneers()]
        #followers = [(n,{'is_pioneer':0}) for n in model.get_followers()]
        nodes = [] 
        for agent,tlevel in model.traverse_agents():
            #print('hook',agent.targets,agent.tag.target_id)
            attr = dict(agent.tag._asdict())
            attr['tlevel'] = tlevel
            attr['targets'] = agent.targets[:]
            attr['is_pioneer'] = int(agent.tag.pioneer_id != -1)
            nodes.append((agent.tid,attr))
        
        self.G.add_nodes_from(nodes)
        #self.G.add_nodes_from(pioneers + followers)
        #nx.set_node_attributes(self.G,dict(model.get_agent_targets()),'targets')
        self.num_rows = model.num_rows
        self.num_cols = model.num_cols

    def post_action(self,model):
        for V in mb.iter_volume(model):
            for (i,j) in self.sweep():
                if V[i,j] == 0: continue
                for (u,v) in self.iter_neighborhood(i,j):
                    if V[u,v] == 0: continue
                    a = V[i,j]
                    b = V[u,v]
                    if not self.G.has_edge(a,b): self.G.add_edge(a,b,weight=0)
                    self.G[a][b]['weight'] += 1
         
    def sweep(self):
        for i in range(self.num_rows):
            for j in range(self.num_cols):
                yield(i,j)

    def iter_neighborhood(self,i,j):
        neigh = [(i,j+1),(i+1,j-1),(i+1,j),(i+1,j+1)]
        for (_i,_j) in neigh:
            if _i >= self.num_rows: continue
            if _j < 0: continue
            if _j >= self.num_cols: continue
            yield (_i,_j)
    
