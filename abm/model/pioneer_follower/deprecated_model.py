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

def iter_signals(dims,pts,kernel):
    for (idx,p) in enumerate(pts):
        s = signals.kernel_point_signal(dims,p,kernel)
        yield (idx,s,p)

class Model:
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
        self.model_id = model_id #Specify model index for parallel processing
        self.cfg = cfg
        #self.neighborhood = list(range(9))
        
        ######### Load parameters #############
        ## Model parameter 
        self.num_agents = cfg.getint('model','num_agents') 
        self.num_steps = cfg.getint('model','num_steps')
        self.vol_analysis_start = cfg.getint('model','vol_analysis_start')
        self.vol_analysis_end = cfg.getint('model','vol_analysis_end')
        self.num_pioneers = cfg.getint('model','num_pioneers')
        self.num_followers = self.num_agents - self.num_pioneers
        self.agent_precision = cfg.getfloat('model','agent_precision')
        self.agent_locality = cfg.getint('model','agent_locality')

        ## Grid parameters
        self.num_rows = cfg.getint('grid','dim_rows') 
        self.num_cols = cfg.getint('grid','dim_cols') 
        self.grid = grid.grid_tuple(self.num_rows,self.num_cols)

        ## Agent parameters
        self.agent_num = cfg.getint('model','num_agents')
        self.num_targets = self.num_pioneers
        self.num_attractors = cfg.getint('agent','num_attractors') 
        self.agent_pos_seed = cfg.getint('agent','pos_seed') 
        self.agent_viz_shift = 10 # This is a hack for viz purposes 
        self.max_val = self.agent_num + self.agent_viz_shift
        self.pioneer_zone = list(map(int,self.cfg['grid']['pioneer_zone'].split(',')))
        self.pioneer_seed = self.cfg.getint('grid','pioneer_seed')

        self.set_max_time_on_target()

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
        self.F = np.zeros((self.num_rows,self.num_cols,self.num_targets))
        
        """ Attractor field centroids """
        self.Fc = np.zeros((self.num_targets,2))
    
        """ Agent precision """
        self.APr = np.ones(self.num_agents)*self.agent_precision

        """Array to map attractor targets to follower"""
        self.T = np.zeros((self.num_agents,self.num_attractors),dtype=int)

        """Array to maps time spent on target"""
        self.Tt = np.zeros((self.num_agents,self.num_attractors),dtype=int)
        
        """Array tracks the current attractor being targetted"""
        self.Tc = np.zeros((self.num_agents,1),dtype=int)

        """ Array storage of agent meta data """
        self.M = np.zeros((self.num_agents,4),dtype=int)

        """ Array storage of current agent postions """
        self.P = np.zeros((self.num_agents,2))

        """ Volume of agents positions """
        self.V = np.zeros((self.num_rows,self.num_cols,self.num_steps),dtype=int)
        
        """ Array storage agent positons in current slice """
        self.X = np.zeros((self.num_rows,self.num_cols),dtype=int)
        
        logging.debug('Model: Loaded')

    
    def get_agent_attributes(self):
        meta = ['aid','target','is_pioneer','is_targetted']
        agents = []
        for i in range(self.num_agents):
            attr = {'idx':i,'aid':self.M[i,0],'target':self.T[i,:].tolist(),#self.M[i,1],
                    'is_pioneer':self.M[i,2],'is_targetted':self.M[i,3]}
            agents.append((attr['aid'],attr))
        return agents

    def get_number_of_innervation_level(self):
        """
        Returns the number of intervation levels.
        """
        return len(self.tlevels)
    
    def get_meta(self):
        return np.copy(self.M)

    def is_pioneer(self,aid):
        return int(self.M[aid,2])
    
    def set_max_time_on_target(self):
        self.max_time_on_target = 0
        if self.num_attractors == 1:
            self.max_time_on_target = self.num_steps
        elif self.num_attractors > 1:
            self.max_time_on_target = self.num_steps // (2*self.num_attractors)

    def init(self):
        logging.debug('Model: Initialize')
        self._init_agent_pos() 
        self._init_attractors()
        self._init_assign_attractor() 
        mb.load_positions(self,0)
        self.cur_step += 1

    def _init_agent_pos(self):
        logging.debug('Model: Initialize:::Agent positions')
        if self.num_pioneers == 0:
            logging.debug('Model: Initialize:::Agent positions no pioneer')
            mb.init_agent_positions(self) 
            self.M[:,0] = np.arange(1,self.num_agents+1)
            
        else:
            logging.debug('Model: Initialize:::Agent positions with pioneer')
            mb.init_agent_zone_positions(self)  
            self.M[:,0] = np.arange(1,self.num_agents+1)
            self.M[:self.num_pioneers,1] = -1
            self.M[:self.num_pioneers,2] = 1
    
    def _init_attractors(self):
        logging.debug('Model: Initialize:::Attractor')
        pts = self.P[:self.num_pioneers,:].tolist()
        kernel = signals.gaussian_kernel(10,1) ## These needs to be defined externally
        for (idx,f,cnt) in iter_signals(self.grid,pts,kernel): 
            self.F[:,:,idx] = f / f.max()
            self.Fc[idx,:] = cnt
    
    def _init_assign_attractor(self):
        logging.debug('Model: Initialize:::Assign attractor')
        if self.num_targets == 0 or self.num_attractors == 0: return 0 
        D = cdist(self.P,self.Fc,metric='euclidean')
        target_range = [self.agent_locality,self.num_attractors+self.agent_locality] 
        for i in range(self.num_pioneers,self.num_agents):
            #self.T[i,:] = np.argsort(D[i,:])[:self.num_attractors]
            self.T[i,:] = np.argsort(D[i,:])[target_range[0]:target_range[1]]
            pioneer_idx = self.T[i,0]
            self.M[i,1] = pioneer_idx
            self.M[pioneer_idx,3] = 1

        
        flag = self.catch_targets_not_being_targetted()
        no_targets = len([i for i in range(self.num_pioneers) if self.M[i,3] == 0])
        #flag = len([i for i in range(self.num_pioneers) if self.M[i,3] == 0])
        #if flag > 0: 
        #    logging.warning(f'Model {self.model_id}:::Caught {flag} pioneers without a targeter, fixed to {no_targets} without a targeter::: # pio: {self.num_pioneers}')
    

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
            
            if self.num_attractors > 0:
                mb.update_time_on_target(self,adx)
                if self.Tt[adx,self.Tc[adx]] > self.max_time_on_target and self.Tc[adx] < self.num_attractors-1:
                    self.Tc[adx] += 1
                    self.M[adx,1] = self.T[adx,self.Tc[adx]]
        
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
    
    def get_target_index(self,adx):
        """
        Returns target index for agent index adx

        Parameters:
        -----------
        adx: int, agent index

        """
        return self.M[adx,1]
    

