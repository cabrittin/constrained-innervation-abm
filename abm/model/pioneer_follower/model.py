"""
@name: pioneer_follower.model.py                       
@description:                  
    Model class for pioneer_follower. Provides standard pioneer-follower functionality

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""
import numpy as np
import random
import time
import logging
from collections import namedtuple
from matplotlib.colors import ListedColormap
from matplotlib import cm
from scipy.spatial.distance import cdist
from collections import Counter

from abm import grid
from abm import signals
#from abm.model.pioneer_follower import attractor
from abm import signals
from abm.model import model_base as mb
from abm.agent_encoder import AgentEnoder

def iter_signals(dims,pts,kernel):
    for (idx,p) in enumerate(pts):
        s = signals.kernel_point_signal(dims,p,kernel)
        yield (idx,s,p)

def iter_ring_signals(dims,pts,kernel):
    for (idx,p) in enumerate(pts):
        s = signals.kernel_ring_signal(dims,p,kernel)
        yield (idx,s,p)


class Model(AgentEnoder):
    def __init__(self,cfg,model_id=0):
        """
        Load all of the parameters
        
        Attributes:
        -----------
        model_id: int, Used for parallel processing indexing
        cfg: ConfigParser, a loaded config file
       
        
        ## Model parameters set in config file
        num_agents: int, Number of agents
        num_steps: int, Number of growth steps
        vol_analysis_start = int, The growth step to start analyzing simulation
        vol_analysis_end = int, The growth step where to end analyzing simulation
        num_pioneers = int, Number of pioneers
        agent_precision = int,Number of 
        locality = int, Minimal distance to nearest attractor
       

        ## Grid parameters set in config file
        num_rows: int, Number of grid rows
        num_cols: int, Number of grid columns


        ## Convenience attributes
        num_followers: int, Number of followers: num_agents - num_pioneers
        grid: tuple, Grid dimensions


        M: Meta data format
            Col #   Desc
            0       Agent id, the int used to identify agents on the grid, adds 1 to agent index
            1       Current pioneer target index
            2       Is agent a pioneer (1/0)
            3       Is agent targeted (1/0)
        
        F: Numpy array (num_rows,num_cols,num_targets)

        """
        super().__init__(cfg) 
        self.model_id = model_id #Specify model index for parallel processing
        self.cfg = cfg
        
        ## Model parameter 
        self.num_steps = cfg.getint('model','num_steps')
        self.vol_analysis_start = cfg.getint('model','vol_analysis_start')
        self.vol_analysis_end = cfg.getint('model','vol_analysis_end')
 

        ## Initialize instance attributes
        """ Tracks current step """ 
        self.cur_step = 0
        
        """ Order of agent innervation at each step. Shuffled at each step """
        self.tlevels = []
        self.tlevels.append(list(range(self.num_pioneers)))
        self.tlevels.append(list(range(self.num_pioneers,self.num_agents)))
        self.tlevel = 0
        self.agent_order = list(range(self.num_pioneers,self.num_agents))


        """ Attractor fields """
        self.F = np.zeros((self.num_rows,self.num_cols,self.num_agents))

        """ Volume of agents positions """
        self.V = np.zeros((self.num_rows,self.num_cols,self.num_steps),dtype=int)

        """ Array storage agent positons in current slice """
        self.X = np.zeros((self.num_rows,self.num_cols),dtype=int)
        
        """ User force flag """
        """
        2025-12-03: Note this was added to maintain downstream compatibility with model_base which had to be 
        modified for the positional_info class. I have not yet tested this flag in the class.

        """
        self.use_force = False 
        if self.num_pioneers > 0: self.use_force = True

        logging.debug('Model: Loaded')


    def print_sweep_params(self):
        """
        Prints the sweep params. Used for debugging.
        """
        txt = f"{self.num_pioneers},{self.avg_response_rate},{self.locality_mean},{self.locality_std},{self.pioneer_seed},{self.agent_pos_seed},{self.response_seed},{self.locality_seed}"
        print(txt)

    def get_agent_attributes(self):
        meta = ['aid','target','is_pioneer','is_targetted']
        agents = []
        for i in range(self.num_agents):
            target = []
            if self.num_pioneers > 0 and self.avg_response >= 0:
                target = np.where(self.R[i,:] > 0)[0].tolist()
            attr = {'idx':i,'aid':self.M[i,0],
                    'target':target,'is_pioneer':self.M[i,1],
                    'specificity':self.Mf[i,0],'locality0':self.Mf[i,1],
                    'locality':self.Mf[i,2],
                    'locality_distance':self.Mf[i,3]}
            agents.append((attr['aid'],attr))
        return agents

    def get_number_of_innervation_level(self):
        """
        Returns the number of intervation levels.
        """
        return len(self.tlevels)

    def get_meta(self):
        return np.copy(self.M)
    

    def add_meta(self,meta):
        self.M = np.hstack((self.M,meta)) 

    def is_pioneer(self,aid):
        return int(self.M[aid,1])

    def init(self):
        logging.debug('Model: Initialize')
        self.init_agent_pos()
        self.init_encoding()
        self.pioneer_correlate()
        self.init_specificity()
        if self.num_pioneers > 0: self.init_response_field()
        mb.load_positions(self,0)
        self.cur_step += 1

    def init_response_field(self):
        logging.debug('Model: Initialize:::Response field')
        pts = self.P[:self.num_pioneers,:].tolist()
        kernel = signals.gaussian_kernel(10,1) ## These needs to be defined externally
        Pf = np.zeros((self.num_rows,self.num_cols,self.num_pioneers))
        for (idx,f,cnt) in iter_signals(self.grid,pts,kernel):
            Pf[:,:,idx] = f / f.max()

        for i in range(self.num_agents):
            for j in range(self.num_pioneers):
                self.F[:,:,i] += self.R[i,j] * Pf[:,:,j]
            fmax = self.F[:,:,i].max()
            if fmax > 0: self.F[:,:,i] /= fmax

        ## Compute locality 
        pts = self.P.tolist()
        kernelo = signals.ring_kernel(10,2,1) ## These needs to be defined externally
        kerneli = signals.tophat_kernel(10,2) ## These needs to be defined externally
        #kerneli = signals.gaussian_kernel(10,1) ## These needs to be defined externally
        ksumo = np.sum(kernelo)
        ksumi = np.sum(kerneli)
        Pfi = np.zeros((self.num_rows,self.num_cols,self.num_agents))
        Pfo = np.zeros((self.num_rows,self.num_cols,self.num_agents))
        for (idx,f,cnt) in iter_signals(self.grid,pts,kerneli):
            Pfi[:,:,idx] = f / f.max()

        for (idx,f,cnt) in iter_ring_signals(self.grid,pts,kernelo):
            Pfo[:,:,idx] = f / f.max()

        eps = 1e-9
        for i in range(self.num_agents):
            num = np.sum(np.multiply(Pfi[:,:,i],self.F[:,:,i]))
            den = np.sum(np.multiply(Pfo[:,:,i],self.F[:,:,i]))
            self.Mf[i,1] = num /ksumi
            self.Mf[i,2] = np.log(num+eps) - np.log(den+eps)


    def catch_pioneers_not_being_targetted(self):
        not_targetted = self.R[self.num_pioneers:,:].sum(0)
        jnt = np.where(not_targetted == 0)[0]
        if len(jnt) > 0:
            pass

    def catch_targets_not_being_targetted(self):
        no_targets = [i for i in range(self.num_pioneers) if self.M[i,3] == 0]
        flag = 0
        if len(no_targets) > 0:
            flag = 1
            D = cdist(self.P[:self.num_pioneers,:],self.P[self.num_pioneers:,:],metric='euclidean')
            for pdx in no_targets:
                fdx = self.num_pioneers + np.argmin(D[pdx,:])
                self.M[fdx,1] = pdx
                self.M[pdx,3] = 1
                self.T[fdx,0] = pdx 
        
        return len(no_targets) 
    
    
    def run_stages(self):
        num_tlevel = len(self.tlevels)
        for t in range(num_tlevel):
            if len(self.tlevels[self.tlevel]) == 0: 
                self.next_tlevel()
                continue
            for i in range(1,self.num_steps):
                self.step()
            self.next_tlevel()
 
    def next_tlevel(self):
        self.tlevel += 1
        self.cur_step = 0

    def step(self):
        if self.cur_step >= self.V.shape[2]: 
            logging.info(f'Reached end of volume of current temporal stage')
            return 0
        self.step_start() 
        random.shuffle(self.tlevels[self.tlevel])
        self.X = np.copy(self.V[:,:,self.cur_step])
        for adx in self.tlevels[self.tlevel]:
            if self.is_pioneer(adx):
                self.step_pioneer(adx)
            else:
                self.step_follower(adx)     
            
        self.step_end()
        self.V[:,:,self.cur_step] = np.copy(self.X)
        self.cur_step += 1
    
    """ The below methods are intended as easy handles for class inheritance """
    
    def step_start(self):
        """ Perform any necessary updates at beginning of a step """
        pass

    def step_end(self):
        """ Perform any necessary update at the end of a step """
        pass

    def step_pioneer(self,adx):
        mb.update_pioneer(self,adx)

    def step_follower(self,adx):
        mb.update_follower(self,adx)
    
    def get_force(self,i,j,idx):
        """
        Returns force at position (i,j) for agent index idx

        Parameters:
        -----------
        i : int, grid row index
        j : int, grid column index
        idx: int, agent index

        Returns:
        --------
        force: float, Force for agent idx at (i,j)
        """
        return self.F[i,j,idx]
    
    def get_agent_id(self,adx):
        """
        Returns agent id for agent index adx

        Parameters:
        -----------
        adx: int, agent index

        """
        return self.M[adx,0]
    

