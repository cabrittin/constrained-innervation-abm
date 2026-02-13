"""
@name: abm.agent_encoder.py                       
@description:                  
    Functions for encoding agent specificity for simulations

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""
import logging
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import norm
from collections import defaultdict

from abm import grid

class AgentEnoder:
    def __init__(self,cfg):
        """
        Agent encoder class.
        
        Molecular specificity:
        ----------------------
        If sensitive to no molecules (average_response_rate=-1), then molecular specificity is defined as 0. 


        M: Meta int data format
            Col #   Desc
            0       Agent id, the int used to identify agents on the grid, adds 1 to agent index
            1       Is agent a pioneer (1/0)
            2       Spatial id, gives nodes an id based on ranked distance from coordinate (0,0),
                    Useful for visual tracking
            3       Is pioneer being responded to (1/0)
        

        Mf: Meta float data format
            Col #   Desc
            0       Molecular specificity 
            1       Locality measure
            2       Locality measure 0
            3       Average draw distance

        """

        ######### Load parameters #############
        ## Model parameter 
        self.num_agents = cfg.getint('model','num_agents') 
        self.num_pioneers = cfg.getint('model','num_pioneers')
        self.num_followers = self.num_agents - self.num_pioneers
        
        self.avg_response_rate = cfg.getfloat('model','average_response_rate')
        self.avg_response = float(self.avg_response_rate) #Keeping the above for debuggin 
        if self.avg_response > 0: self.avg_response = self.avg_response/float(self.num_pioneers) 
        self.locality_mean = list(map(float,cfg['model']['locality_mean'].split(',')))
        self.locality_std = list(map(float,cfg['model']['locality_std'].split(',')))
        
        ## Hack to pass bimodal locality without messing with backward compatibility
        ## We only consider one possible bimodal profile 
        if self.locality_std[0] == 99:
            self.locality_mean = [0.,float(self.num_pioneers-1)]
            self.locality_std = [1.,1.]

        self.response_seed = cfg.getint('model','response_seed')
        self.locality_seed = cfg.getint('model','locality_seed')

        ## For backward compatibility use try 
        self.pioneer_flips = cfg.getint('model','pioneer_flips')
        self.pioneer_locality_mean = list(map(float,cfg['model']['pioneer_locality_mean'].split(',')))
        self.pioneer_locality_std = list(map(float,cfg['model']['pioneer_locality_std'].split(',')))
        self.pioneer_flips_seed = cfg.getint('model','pioneer_flips_seed') 
        self.pioneer_locality_seed = cfg.getint('model','pioneer_locality_seed') 

        ## Grid parameters
        self.num_rows = cfg.getint('grid','dim_rows') 
        self.num_cols = cfg.getint('grid','dim_cols') 
        self.grid = grid.grid_tuple(self.num_rows,self.num_cols)

        self.pioneer_zone = list(map(int,cfg['grid']['pioneer_zone'].split(',')))
        self.pioneer_seed = cfg.getint('grid','pioneer_seed')
        self.agent_pos_seed = cfg.getint('grid','agent_seed') 
        
        """ Array storage of agent meta data """
        self.M = np.zeros((self.num_agents,4),dtype=int)
        self.M[:,0] = np.arange(1,self.num_agents+1)
        self.M[:self.num_pioneers,1] = 1
        
        """ Array storage fo agent specificity and locality """ 
        self.Mf = np.zeros((self.num_agents,4))

        """ Array storage of current agent postions """
        self.P = np.zeros((self.num_agents,2))
        
        """ Grid mainly for compatibility with older viz modules """
        self.V = np.zeros((self.num_rows,self.num_cols,1),dtype=int)
        
        """ Molecular code """
        self.num_codes = np.zeros(self.num_agents) 
        self.Ep = init_pioneers(self.num_pioneers) if self.num_pioneers > 0 else 0
        self.E = np.zeros((self.num_agents,self.num_pioneers)) if self.num_pioneers > 0 else 0
    
        """ Ranked Distance matrix """
        self.RD = np.zeros((self.num_agents,self.num_pioneers)) if self.num_pioneers > 0 else 0
        
        """ Responsiveness """
        self.R = np.zeros((self.num_agents,self.num_pioneers)) if self.num_pioneers > 0 else 0
   


    def get_meta(self):
        return np.copy(self.M)
    
    def average_molecular_specificity(self):
        return self.Mf[self.num_pioneers:,0].mean()
    
    def molecular_specificity(self):
        return self.Mf[self.num_pioneers:,0]
    
    def average_pioneer_specificity(self):
        return self.Mf[:self.num_pioneers,0].mean()
    
    def pioneer_specificity(self):
        return self.Mf[:self.num_pioneers,0]
    
    def average_locality_0(self):
        return self.Mf[self.num_pioneers:,1].mean()
    
    def average_pioneer_locality_0(self):
        return self.Mf[:self.num_pioneers,1].mean()
    
    def average_locality(self):
        return self.Mf[self.num_pioneers:,2].mean()
    
    def locality(self):
        return self.Mf[self.num_pioneers:,2]
    
    def average_pioneer_locality(self):
        return self.Mf[:self.num_pioneers,2].mean()
    
    def pioneer_locality(self):
        return self.Mf[:self.num_pioneers,2]
    
    def locality_distance(self):
        return self.Mf[self.num_pioneers:,3]
    
    def pioneer_locality_distance(self):
        return self.Mf[:self.num_pioneers,3]


    def encoding(self,as_dict=False):
        if as_dict:
            return { 
                    'fs':self.average_molecular_specificity(),
                    'ps':self.average_pioneer_specificity(),
                    'fl':self.average_locality(),
                    'pl':self.average_pioneer_locality(),
                    'fl0':self.average_locality_0(),
                    'pl0':self.average_pioneer_locality_0(),
                    'fd':self.locality_distance().mean(),
                    'pd':self.pioneer_locality_distance().mean()
                    }
        else:         
            return [
                    self.average_molecular_specificity(),
                    self.average_pioneer_specificity(),
                    self.average_locality(),
                    self.average_pioneer_locality(),
                    self.average_locality_0(),
                    self.average_pioneer_locality_0(),
                    self.locality_distance().mean(),
                    self.pioneer_locality_distance().mean()
                    ]
         

    def init_agent_pos(self):
        logging.debug('AgentEnoder: Initialize:::Agent positions')
        if self.num_pioneers == 0:
            logging.debug('AgentEnoder: Initialize:::Agent positions no pioneer')
            init_agent_positions(self) 
            
        else:
            logging.debug('AgentEnoder: Initialize:::Agent positions with pioneer')
            init_agent_zone_positions(self)  
        rd = ranked_distance_matrix(np.array([[0,0]]),self.P)
        for (i,j) in enumerate(rd[0]): self.M[j,2] = i 
        self.RD = ranked_distance_matrix(self.P,self.P[:self.num_pioneers])
    
    def init_encoding(self):
        logging.debug('AgentEnoder: Initialize:::Agent Encoding')
        if self.avg_response < 0: self.Mf[:,0] = 1 
        if self.num_pioneers == 0 or self.avg_response < 0: return 0 
        rng = np.random.default_rng(seed=self.response_seed)
        self.num_codes = specificity_draw(self.num_pioneers,self.avg_response,self.num_agents,rng=rng)
        #self.RD = ranked_distance_matrix(self.P,self.P[:self.num_pioneers])
        if len(self.locality_mean) == 1:
            self.profile = locality_profile_unimodal(self.num_pioneers,
                        self.locality_mean[0],
                        self.locality_std[0])
        else:
            self.profile = locality_profile_bimodal(self.num_pioneers,
                        self.locality_mean,
                        self.locality_std)
        
        rng = np.random.default_rng(seed=self.locality_seed)
        draws = locality_profile_draw(self.num_agents,self.num_codes,self.profile,rng=rng)
        self.E = init_agents(self.Ep,self.RD,draws)
        self.E[:self.num_pioneers,:] = self.Ep
        self.Mf[:,3] = average_draw_distance(self.P,self.P[:self.num_pioneers,:],draws)
        catch_pioneers_not_being_targetted(self) 
        self.Mf[:,0] = molecular_specificity(self.Ep,self.E)
        #print(self.E)
    
    def pioneer_correlate(self):
        logging.debug('AgentEnoder: Pioneer correlation')
        if self.num_pioneers == 0 or self.pioneer_flips == 0: return 0 
        rng = np.random.default_rng(seed=self.pioneer_flips_seed)
        #flips = rng.integers(0,high = self.num_pioneers-1,size=self.pioneer_flips,dtype=int) 
        avg_flips = self.pioneer_flips / float(self.num_pioneers) 
        flips = specificity_draw(self.num_pioneers,avg_flips,self.num_pioneers,rng=rng,no_empties=False)
        if len(self.pioneer_locality_mean) == 1:
            self.pioneer_profile = locality_profile_unimodal(self.num_pioneers,
                        self.pioneer_locality_mean[0],
                        self.pioneer_locality_std[0])
        else:
            self.pioneer_profile = locality_profile_bimodal(self.num_pioneers,
                        self.pioneer_locality_mean,
                        self.pioneer_locality_std)
        rng = np.random.default_rng(seed=self.pioneer_locality_seed)
        draws = locality_profile_draw(self.num_pioneers,flips,self.pioneer_profile,rng=rng)
        A = init_agents(self.Ep,self.RD[:self.num_pioneers,:],draws)
        A += self.Ep
        A[A>0] = 1
        self.E[:self.num_pioneers,:] = A 
        self.Mf[:self.num_pioneers,3] = average_draw_distance(self.P[:self.num_pioneers,:],
                                                                self.P[:self.num_pioneers,:],draws)
        #self.E[:self.num_pioneers,:] = init_agents(self.Ep,self.RD[:self.num_pioneers,:],draws)
        self.Mf[:self.num_pioneers,0] = molecular_specificity(self.Ep,self.E[:self.num_pioneers,:])
    
    def _pioneer_correlate(self):
        logging.debug('AgentEnoder: Pioneer correlation')
         
        if self.num_pioneers == 0 or self.pioneer_flips == 0: return 0 
        rng = np.random.default_rng(seed=self.pioneer_flips_seed)
        flips = rng.integers(0,high = self.num_pioneers-1,size=self.pioneer_flips,dtype=int) 
         
        if len(self.pioneer_locality_mean) == 1:
            self.pioneer_profile = locality_profile_unimodal(self.num_pioneers,
                        self.pioneer_locality_mean[0],
                        self.pioneer_locality_std[0])
        else:
            self.pioneer_profile = locality_profile_bimodal(self.num_pioneers,
                        self.pioneer_locality_mean,
                        self.pioneer_locality_std)
        rng = np.random.default_rng(seed=self.pioneer_locality_seed)
        draws = locality_profile_draw(self.num_pioneers,flips,self.pioneer_fofile,rng=rng)
        self.E[:self.num_pioneer,:] = init_agents(self.Ep,self.RD[:self.num_pioneers,:],draws)
        self.Mf[:self.num_pioneers,0] = molecular_specificity(self.Ep,self.E[:self.num_pioneers,:])


    def init_specificity(self):
        logging.debug('AgentEnoder: Initialize:::Agent Encoding')
        if self.num_pioneers == 0: return 0 
        self.R = specificity(self.E[:self.num_pioneers,:],self.E)

def catch_pioneers_not_being_targetted(ae: AgentEnoder) -> None:
    not_targetted = ae.E[ae.num_pioneers:,:].sum(0)
    jnt = np.where(not_targetted == 0)[0]
    if len(jnt) > 0:
        D = cdist(ae.P[:ae.num_pioneers,:],ae.P[ae.num_pioneers:,:],metric='euclidean')
        for pdx in jnt:
            fdx = ae.num_pioneers + np.argmin(D[pdx,:])
            ae.E[fdx,:] = 0
            ae.E[fdx,pdx] = 1


def get_follower_pioneers(ae: AgentEnoder, aid: int) -> None:
    """
    A debugging tool
    """
    return np.where(ae.Ef[aid-ae.num_pioneers,:] == 1)[0] 
    

def init_agent_zone_positions(ae: AgentEnoder) -> None:
    """
    Initializes zone positions

    Pioneers placed in an inner zone

    Followers are then placed around the pioneers

    """

    i0,i1,j0,j1 = ae.pioneer_zone
    pioneer_pts = grid.sample_from_region(i0,i1,j0,j1,ae.num_pioneers,ae.pioneer_seed)
    for i in range(ae.num_pioneers):
        ae.P[i,:] = pioneer_pts[i]
    
    tmp = grid.shuffle_zone_positions(ae.grid,target_points=pioneer_pts,seed=ae.agent_pos_seed)
    
    for i in range(ae.num_followers):
        ae.P[i+ae.num_pioneers,:] = tmp[i]
    
    ae.P = ae.P.astype(int)

def init_agent_positions(ae: AgentEnoder) -> None:
    """
    Randomly initializes agent positions on the grid
    """
    positions = grid.get_positions(ae.grid)
    np.random.default_rng(ae.agent_pos_seed).shuffle(positions)
    for i in range(ae.num_agents):
        ae.P[i,:] = grid.index_to_coordinate(ae.grid,positions[i])
    ae.P = ae.P.astype(int)


def init_pioneers(num_pioneers: int) -> np.ndarray:
    """
    Initializes the pioneer encoder.

    Parameters:
    ----------- 
    num_pioneer: int, Number of pioneers

    Returns:
    --------
    Identity numpy array [num_pioneer x num_pioneers], ones on diagonal

    """
    return np.eye(num_pioneers,dtype=int)

def init_agents(code: np.ndarray, ranked_distance: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """
    Initializes agent encoding by assigning agents encoding based on draws and ranked_distance

    Parameters:
    -----------
    code: ndarray, Coding array
    ranked_distance: ndarray, Ranked distance matrix
    draws: ndarray, A random draws array

    Returns:
    --------
    Numpy array of agent encoding
    """
    
    A = np.zeros((draws.shape[0],code.shape[1]),dtype=int)
    for i in range(draws.shape[0]):
        for j in np.where(draws[i,:] == 1)[0]:
            A[i,:] += code[ranked_distance[i,j],:]
    A[A>0] = 1
    return A

def average_draw_distance(source: np.ndarray, target: np.ndarray, draws:np.ndarray) -> np.ndarray:
    """
    Average distance from source agent to target agents determined by draws

    Parameters:
    -----------
    source: numpy array, Source 2D grid positions
    target: = numpy array, Target 2D grid positions
    draws: ndararry, A random draws array

    Assumes draws array has at least one 1 in each row

    Returns:
    --------
    Numpy array size N of averge draw distance
    """
    
    D = cdist(source,target,metric='euclidean')
    D = np.sort(D,axis=1) 
    dmin = D[:,0].min()
    dmax = D[:,-1].max()
    A = np.zeros(source.shape[0])
    for i in range(source.shape[0]):  
        jdx = np.where(draws[i,:]==1)[0]
        if len(jdx)>0: A[i] = D[i,jdx].mean()
    #A = (A - dmin) / (dmax - dmin) 
    return A

def perturbed_pioneers(code: np.ndarray, pioneer_flips: np.ndarray, 
        ranked_distance: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """
    Initializes agent encoding by assigning agents encoding based on draws and ranked_distance

    Parameters:
    -----------
    code: ndarray, Coding array
    pioneers_flips: ndarray, Indicies of which pioneers to flip
    ranked_distance: ndarray, Ranked distance matrix
    draws: ndarray, A random draws array

    Returns:
    --------
    Numpy array of agent encoding
    """
    A = np.copy(code) 
    for i in range(draws.shape[0]):
        for j in np.where(draws[i,:] == 1)[0]:
            A[pioneer_flips[i],:] += code[ranked_distance[pioneer_flips[i],j],:]
    A[A>0] = 1
    return A


def locality_profile_unimodal(n: int, mu: float, std: float, rescale: bool=True) -> np.ndarray:
    """
    Returns a unimodal locality profile

    Parameters:
    -----------
    n: int, Length of the locality profile
    mu: float, mean of locality profile, typically in np.arange(n)
    std: float, standard deviation of locality profile, typically in np.arange(n)

    """
    if std == 0:
        pdf = np.zeros(n)
        pdf[int(mu)] = 1
    else: 
        x = np.arange(n)
        pdf = norm.pdf(x,loc=mu,scale=std)
    
    if rescale: pdf = pdf / pdf.max()
    return pdf 

def locality_profile_bimodal(n: int, mu: list or np.ndarray, std: list or np.ndarray) -> np.ndarray:
    """
    Returns a unimodal locality profile

    Parameters:
    -----------
    n: int, Length of the locality profile
    mu: float, mean of locality profile, typically in np.arange(n)
    std: float, standard deviation of locality profile, typically in np.arange(n)

    """
    x = np.arange(n)
    
    pdf0 = norm.pdf(x,loc=mu[0],scale=std[0])
    pdf1 = norm.pdf(x,loc=mu[1],scale=std[1])
    pdf = pdf0 + pdf1 

    return pdf / pdf.max() 


def locality_profile_draw(n: int, k: int or list or np.ndarray, 
        profile: np.ndarray, rng: np.random.Generator=None) -> np.ndarray:
    """ 
    Returns a 1/0 array of draws from a locality profile. 1 meaning that index is drawn 
    and 0 meaning that index is not drawn 
    
    Parameters:
    -----------
    n: int, Number of times to repeat draws
    k: int or array like, Number of draws per repeat. If array, should be of length n.
    profile: np.ndarray, locality profile
    rng: np.random.Generator, optional, default None. Numpy random generator
    """
    if rng is None: rng = np.random.default_rng()
    if isinstance(k,int): k = np.ones(n,dtype=int)*k

    m = len(profile)
    rand = rng.uniform(low=0,high=1.0,size=(n,m))
    test = np.argsort(-1*np.tile(profile,(n,1)) * rand,axis=1)
    draws = np.zeros((n,m),dtype=int) 
    for i in range(n): draws[i,test[i,:k[i]]] = 1 
    return draws
     
def specificity_draw(n: int, p: float, size: int=None, 
        no_empties: bool=True, rng: np.random.Generator=None) -> np.ndarray:
    """
    Returns random draws for number of molecules to respond to, drawn from a binomal distribution

    Parameters:
    -----------
    n: int, Number of pioneers
    p: float, probability of responding to pioneer
    size: int, number of draws
    no_empties: bool, optional, default: True, If True, any 0 values will be changed to 1. 
    rng: np.random.Generator, optional, default None. Numpy random generator
    """
    if rng is None: rng = np.random.default_rng()
    draws =  rng.binomial(n,p,size=size)
    if no_empties: draws[draws==0] = 1
    return draws

def specificity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Returns specificity matrix.

    (i,j) entry is the specificity of target[i] to source[j]
    
    Let s and t be a rows of source and target, respectively.

    Responsiveness of t to s -- R(s,t) -- is computed as:
    
    R(s,t) = Pr(s_i = 1 | t_i = 1)  
    
    Parameters:
    -----------
    source: np.ndarray, 1/0 matrix of source 
    target: np.ndarray, 1/0 matrix of the target

    """
    R = np.dot(target,source.transpose())
    tsum = target.sum(1)
    tsum[tsum == 0] = 1
    return R / tsum[:,None]

def ranked_distance_matrix(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """ 
    Creates a ranked distance matrix from source to target points

    Parameters:
    -----------
    source: numpy array, Source 2D grid positions
    target: = numpy array, Target 2D grid positions

    Returns:
    --------
    Numpy array of pioneer indices

    """
    D = cdist(source,target,metric='euclidean')
    return np.argsort(D,axis=1)

def molecular_specificity(pioneer: np.ndarray, follower: np.ndarray) -> np.ndarray:
    """
    Computes the molecular specificity of followers for a set of pioneers
    
    Computed that the probability that follower and pioneer express the same molecule


    Parameters:
    -----------
    pioneer: np.ndarray, (m,k) express array for pioneers
    follower: np.ndarray, (n,k) express array for followers
    
    Returns:
    --------
    size n array, molecular specificity for each follower 

    """
    S = np.dot(follower,pioneer.T)
    S[S>0] = 1
    return 1 - (S.sum(axis=1) / S.shape[1])

def average_molecular_specificity(pioneer: np.ndarray, follower: np.ndarray) -> float:
    """
    Computes the average molecular specificity
    """
    
    return molecular_specificity(pioneer,follower).mean() 

def average_pioneer_molecular_specificty(pioneer: np.ndarray) -> float: 
    return molecular_specificity(pioneer,pioneer - np.eye(pioneer.shape[1])).mean()

def total_correlation_binary(X: np.array) -> float:
    """
    Deprecated
    """

    X[X>0] = 1
    X = X.astype(int)

    (m,k) = X.shape
    p = np.zeros((X.shape[1],2))
    p[:,1] =  X.sum(axis=1) / float(m)
    p[:,0] = 1 - p[:,1]
    
    clabels = [binary_array_to_int(X[i,:]) for i in range(m)]
    ccount = defaultdict(float)
    for c in clabels: ccount[c] += 1
    for c in ccount.keys(): ccount[c] /= float(m) 
    
    print(ccount)
    psum = 0
    for c in ccount.keys():
        b = int_to_binary_array(c,k)
        den = np.prod([p[i,x] for (i,x) in enumerate(b)])
        psum += ccount[c] * np.log2(ccount[c] / den)
    
    print(psum)

def binary_array_to_int(a: np.ndarray) -> int:
    res = 0
    for ele in a:
        res = (res << 1) | ele
    return res

def int_to_binary_array(num,zfill):
    return np.array(list(np.binary_repr(num).zfill(zfill))).astype(np.int8)


