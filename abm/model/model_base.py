"""
@name: model_base.py
@description:
    Base functions commonly used by models. 

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import random
import numpy as np
import logging

from abm import grid

def iter_volume(model,full=False):
    v0 = [model.vol_analysis_start,0][int(full)]
    v1 = [model.vol_analysis_end,model.V.shape[2]][int(full)]
    for k in range(v0,v1):
        yield model.V[:,:,k]

def iter_agent_positions(model):
    for i in range(model.num_agents):
        aid = model.M[i,0]
        [x,y] = model.P[i,:]
        yield i,aid,x,y

def load_positions(model,step):
    for idx,aid,x,y in iter_agent_positions(model):
        model.V[x,y,step] = aid

def get_position_encoding(model,dest):
    pid = 0 
    if model.num_pioneers > 0: pid = model.M[model.num_pioneers-1,0]
    dest[:] = model.V[:,:,0].copy().flatten()
    dest[np.logical_and(dest <= pid,dest > 0)] = 1 
    dest[dest > pid] = 2

def update_pioneer(self,adx):
    """
    Update pioneer position.

    Because pioneers grow first, they should just grow straight, i.e. no change in positon

    Parameters:
    ----------
    self: Model class or a named tuple
        Needs to have attributes for position (P), meta data (M) and current 2D grid (X).
    adx: int
        Agent index

    """
    [px,py] = self.P[adx,:]
    aid = self.get_agent_id(adx)
    self.X[px,py] = aid

def update_pioneer_stochastic(self,adx):
    """
    Update pioneer position.

    Allows pioneer trajectory to vary within the pioneer zone

    Parameters:
    ----------
    self: Model class or a named tuple
        Needs to have attributes for position (P), meta data (M) and current 2D grid (X).
    adx: int
        Agent index

    """
    [px,py] = self.P[adx,:]
    aid = self.get_agent_id(adx)
    pmove = self.M[adx,4]
    if pmove == 1 or self.X[px,py] > 0:
        #Random move
        possible_moves = []
        for (i,j) in grid.empty_neighborhood(self.X,px,py):
            if inside_pioneer_zone(self,i,j): possible_moves.append([i,j,0])
        
        if possible_moves:
            possible_moves = np.array(possible_moves)
            move = sample_possible_moves(possible_moves)
            self.X[move[0],move[1]] = aid
            self.P[adx,:] = move
        else:
            direction = pick_direction_shift(self.X,px,py)
            globals()[f'shift_{direction}'](self.X,px,py,aid,self.P)
            if aid not in self.X: 
                logging.warning(self.stamp_log_msg(f'Agent: {aid.tid} not placed'))
    else:
        self.X[px,py] = aid

def inside_pioneer_zone(self,x,y):
    """
    Checks that coordinates x,y is in the pioneer zone
    """
    
    i0,i1,j0,j1 = self.pioneer_zone
    inside_x = (x>=i0) and (x<=i1)
    inside_y = (y>=j0) and (y<=j1)
    return inside_x and inside_y

def update_follower(self,adx):
    """
    Update follower position.

    Parameters:
    ----------
    self: Model class or a named tuple
        Needs to have attributes for position (P), meta data (M), current 2D grid (X) and force (F).
    adx: int
        Agent index

    """
    [px,py] = self.P[adx,:]
    aid = self.get_agent_id(adx)
    possible_moves = []
    
    for (i,j) in grid.empty_neighborhood(self.X,px,py):
        force = 0
        if self.use_force: force = self.get_force(i,j,adx)
        possible_moves.append([i,j,force])
    
    if possible_moves:
        possible_moves = np.array(possible_moves)
        move = sample_possible_moves(possible_moves)
        self.X[move[0],move[1]] = aid
        self.P[adx,:] = move
    else:
        direction = pick_direction_shift(self.X,px,py)
        globals()[f'shift_{direction}'](self.X,px,py,aid,self.P)
        if aid not in self.X: 
            logging.warning(self.stamp_log_msg(f'Agent: {aid.tid} not placed'))

def update_time_on_target(self,adx):
    """
    Update follower position.

    Parameters:
    ----------
    self: Model class or a named tuple
        Needs to have attributes for position (P), meta data (M), current 2D grid (X) and target trackers (T,Tt).
    adx: int
        Agent index

    """
    [px,py] = self.P[adx,:]
    aid = self.M[adx,0] 
    fdx = self.M[adx,1]
    for (i,j) in grid.occupied_neighborhood(self.X,px,py):
        if self.X[i,j] == fdx + 1: ##Need to add 1 to convert fdx to aid value
            self.Tt[adx,self.Tc[adx]] += 1
            break

def shuffle_list(lst,seed=None):
    """
    Shuffles a list

    Parameters:
    -----------
    seed : int (optional, default:None) 
        seed for the random shuffle

    """
    if seed is not None:
        random.Random(seed).shuffle(lst) 
    else:
        random.shuffle(lst)

def sample_possible_moves(field,sample_size=100,eps=1e-2):
    """
    Need to fix empty elements
    """
    
    fsum = field[:,2].sum()
    elements = []
    
    #If field sums to small value (<eps) then randomly choose position
    if fsum < eps:
        for i in range(field.shape[0]):
            elements.append([int(field[i,0]),int(field[i,1])])
    else:
        field[:,2] /= fsum
        num_elements = (sample_size*field[:,2]).astype(int)
        for i in range(field.shape[0]):
            for _ in range(num_elements[i]):
                elements.append([int(field[i,0]),int(field[i,1])])
    random.shuffle(elements)
    e = random.choice(elements)
    return e

def pick_direction_shift(X,i,j):
    """ Methods for shifting agents on the grid """
    (num_rows,num_cols) = X.shape
    possible_directions = []
    if j > 0 and 0 in X[i,:j]: possible_directions.append('left')
    if j < num_cols-1 and 0 in X[i,j:]: possible_directions.append('right')
    if i > 0 and 0 in X[:i,j]: possible_directions.append('up')
    if i < num_rows-1 and 0 in X[i:,j]: possible_directions.append('down')
    if len(possible_directions) > 0:
        direction = random.choice(possible_directions)
    else:
        direction = random.choice(['left','right','up','down'])
    return direction


def _shift_agents(func):
    def inner(X,i,j,xnew,P):
        for (idx,jdx) in func(X,i,j,xnew,P):
            xcur = X[idx,jdx]
            X[idx,jdx] = xnew
            P[xnew-1] = np.array([idx,jdx])
            xnew = xcur
            if xnew == 0: break
        
        ## Warn that an agent was moved out of the volume
        if xnew > 0:
            print(f'Agent: {xnew} shifted out of volume')
        #    logging.warning(self.stamp_log_msg(f'Agent: {xnew} shifted out of volume'))
    return inner
    
@_shift_agents
def shift_right(X,i,j,xnew,P):
    for jdx in range(j,X.shape[1]):
        yield (i,jdx)

@_shift_agents
def shift_left(X,i,j,xnew,P):
    for jdx in reversed(range(j+1)):
        yield (i,jdx)

@_shift_agents
def shift_down(X,i,j,xnew,P):
    for idx in range(i,X.shape[0]):
        yield (idx,j)

@_shift_agents
def shift_up(X,i,j,xnew,P):
    for idx in reversed(range(i+1)):
        yield (idx,j)

