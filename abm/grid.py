"""
@name: space.py
@description:
    Classes for setting up spatial grids

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import numpy as np
import random
from scipy.stats import qmc
import time
from collections import namedtuple

def grid_tuple(nrows,ncols):
    """
    Convenience function to build a grid tuple
    
    Parameters:
    ----- 
    nrows: int
        Number of rows
    ncols: int
        Number of columns

    Returns
    --------
    Grid tuple ("Grid", "nrows rcols")

    """
    Grid = namedtuple("Grid", "nrows ncols")
    return Grid(nrows,ncols)


def index_to_coordinate(dim,z):
    """
    Converts grid index to (row,col) coordinate

    Parameters:
    -----
    dim: tuple (row,col)
    z : int
        Grid index positon

    Returns:
    --------
    (row,col) : (int,int)
        Row,Column index position as tuple

    """
    idx = z // dim[1]
    jdx = z%dim[1]
    return (idx,jdx)


def coordinate_to_index(dim,coord):
    """
    Converts (row,col) coordinate to grid index position

    Parameters:
    -----
    dim: tuple (row,col)
    (row,col): (int,int)
        Tuple specifying row and column index

    Returns:
    --------
    Grid index: int
    """
    return dim[1]*coord[0] + coord[1]

def neighborhood(dim,i,j,dw=1):
    """
    Iterates through the neighborhood of a given coordinate

    Parameters:
    -----------
    dim: tuple (row,col)
    i : int,
        Row index
    j : int,
        Column index
    dw: int
        Neighborhood radius

    """
    for _i in [i-dw,i,i+dw]:
        if _i < 0: continue
        if _i >= dim[0]: continue
        for _j in [j-dw,j,j+dw]:
            if _j < 0: continue
            if _j >= dim[1]: continue
            yield (_i,_j)

def empty_neighborhood(G,i,j):
    """
    Iterates through the neighborhood of a given coordinate

    Yields the unoccupied positions in the neighborhood

    Parameters:
    -----------
    G : numpy.array 
        The grid. O values are treated as unoccupied
    i : int,
        Row index
    j : int,
        Column index
    dw: int
        Neighborhood radius
    """
    dim = G.shape
    for (_i,_j) in neighborhood(dim,i,j):
        if G[_i,_j] != 0: continue
        yield (_i,_j)


def occupied_neighborhood(G,i,j):
    """
    Iterates through the neighborhood of a given coordinate

    Yields the unoccupied positions in the neighborhood

    Parameters:
    -----------
    G : numpy.array 
        The grid. O values are treated as unoccupied
    i : int,
        Row index
    j : int,
        Column index
    dw: int
        Neighborhood radius
    """
    dim = G.shape
    for (_i,_j) in neighborhood(dim,i,j):
        if G[_i,_j] == 0: continue
        yield (_i,_j)


def neighborhood_position(G,i,j,ndx,radius=3):
    """
    Returns a the global coordinate of the local neighborhood index position

    Parameters:
    -----------
    G : Grid object
    i : int
        Row index
    j : int
        Column index
    ndx: int
        local neighborhood index postion ndx in [0,radius**2]
    radius: int, optional (default = 3)

    """

    idx = ndx // radius
    jdx = ndx%radius
    shift = radius // 2
    idx -= shift
    jdx -= shift
    i = i + idx
    j = j + jdx 

    if (i < 0) or (i >= G.num_rows):
        return -1
    elif (j < 0) or (j >= G.num_cols):
        return -1
    else:
        return (i,j)

def get_positions(grid,skip=[]):
    """
    Retrieves list of grid index positions. Indices in skip are not returned
    
    Parameters:
    -----------
    grid: grid tuple
    skip: list, optional (default None)

    Returns list of positions

    """
    num_positions = grid.nrows * grid.ncols
    positions = list(range(num_positions))
    if skip: 
        positions = [p for p in self.positions if p not in skip]
    return positions

def sample_from_region(i0,i1,j0,j1,num_pts,seed=None):
    """
    Randomly picks grid points from a rectangular subregion

    Parameters:
    -----------
    i0 : int
        Lowest row index
    i1 : int
        Highest row index
    j0 : int
        Left column index
    j1 : int
        Right column index
    num_pts : int
        Number of points to select
    seed : int, optional (default None)
        Seed for random sampler
    """

    nrows = i1 - i0
    ncols = j1 - j0
    shift = np.array([i0,j0]) 
    sampler = qmc.Halton(d=2, scramble=True,seed=seed)
    sample = sampler.random(n=num_pts)
    sample = qmc.scale(sample,[0,0],[nrows,ncols])
    return [(s+shift).tolist() for s in np.around(sample).astype(int)]

def sample_field(field,sample_size=100,eps=1e-1):
    """
    Returns random row from self.get_local_field
    """

    fsum = field[:,1].sum()
    elements = []
    
    #If field sums to small value (<eps) then randomly choose position
    if fsum < eps:
        for i in range(field.shape[0]):
            elements.append(int(field[i,0]))
    else:
        field[:,1] /= fsum
        num_elements = (sample_size*field[:,1]).astype(int)
        for i in range(field.shape[0]):
            f = int(field[i,0])
            for _ in range(num_elements[i]):
                elements.append(f)
    e = random.choice(elements)
    return e


def shuffle_zone_positions(G,target_points=[],seed=-1):
    """ 
    Splits grid into 2 zones. 
    
    Zone 1 immediately surrounds the target_points. 
    Zone 2 immediately surrounds Zone 1.

    Parameters:
    -----------
    G : Grid object
    target_points: list
        list of tuple points that define zone 1
    seed: int

    """
    trow,tcol = zip(*target_points) 
    tdx = list(zip(trow,tcol))
    irange = [max(0,min(trow)-1),min(G.nrows-1,max(trow)+1)]
    jrange = [max(0,min(tcol)-1),min(G.ncols-1,max(tcol)+1)]
    
    ## Zone 1
    zone1 = []
    for i in range(irange[0],irange[1]+1):
        for j in range(jrange[0],jrange[1]+1):
            if (i,j) in tdx: continue
            zone1.append((i,j))
    
    ## Zone 2
    zone2 = []
    for i in range(G.nrows):
        for j in range(G.ncols):
            if (i,j) in tdx: continue
            if (i,j) in zone1: continue
            zone2.append((i,j))
    
    if seed != -1:
        random.Random(seed).shuffle(zone1)
        random.Random(seed).shuffle(zone2)
    else: 
        random.shuffle(zone1)
        random.shuffle(zone2)
    
    return zone1 + zone2

