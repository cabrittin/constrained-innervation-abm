"""
@name: signals.py                      
@description:                  
    Functions for generating signal patterns

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import numpy as np
from scipy.signal.windows import gaussian,tukey
from scipy.signal import convolve2d


def gaussian_kernel(kernel_size,sigma):
    """
    Returns a 2D guassian kernel

    Parameters:
    -----------
    kernel_size: int
        Size of gaussian kernel
    sigma: float
        Standard deviation of Gaussian kernel
    
    Returns:
    --------
        2D array gaussian kernel

    """
    kernel = gaussian(kernel_size,sigma)
    kernel = np.outer(kernel,kernel)
    return kernel 

def tukey_kernel(kernel_size,alpha):
    """
    Returns a 2D guassian kernel

    Parameters:
    -----------
    kernel_size: int
        Size of gaussian kernel
    sigma: float
        Standard deviation of Gaussian kernel
    
    Returns:
    --------
        2D array gaussian kernel

    """
    kernel = tukey(kernel_size,alpha)
    kernel = np.outer(kernel,kernel)
    return kernel 


def ring_kernel(ksize,radius,width):
    """
    Returns a 2D guassian kernel

    Parameters:
    -----------
    kernel_size: int
        Size of gaussian kernel
    sigma: float
        Standard deviation of Gaussian kernel
    
    Returns:
    --------
        2D array gaussian kernel

    """
    kernel1 = tophat_kernel(ksize,radius)
    kernel2 = tophat_kernel(ksize,radius+width)
    return kernel2 - kernel1

def tophat_kernel(ksize,width):
    """
    Returns a 2D guassian kernel

    Parameters:
    -----------
    kernel_size: int
        Size of gaussian kernel
    sigma: float
        Standard deviation of Gaussian kernel
    
    Returns:
    --------
        2D array gaussian kernel

    """
    w2 = ksize // 2 
    kernel = np.zeros(ksize)
    kernel[w2-width:w2+width] = 1
    kernel = np.outer(kernel,kernel)
    return kernel


def tukey_ring_kernel(kernel_size,sigma):
    """
    Returns a 2D guassian kernel

    Parameters:
    -----------
    kernel_size: int
        Size of gaussian kernel
    sigma: float
        Standard deviation of Gaussian kernel
    
    Returns:
    --------
        2D array gaussian kernel

    """
    kernel = 1 - tukey_kernel(kernel_size,sigma)
    return kernel


def point_signal(dim,pts):
    """
    Generates 2D array with points signals specified by pts

    Parameters:
    -----------
    dim: tuple,list
        Dimensions of 2D array (num rows, num columns)
    pts: list or list of lists
        If list, should be for [row index, col index]. Can specify multiple pts with list of lists
    """
    S = np.zeros(dim)
    if isinstance(pts[0],int):
        S[pts[0],pts[1]] = 1
    elif isinstance(pts[0],list):
        for p in pts: S[p[0],p[1]] = 1
    return S

def kernel_point_signal(dim,pts,kernel):
    """
    Generates 2D array with points signals specified by pts

    Parameters:
    -----------
    dim: tuple,list
        Dimensions of 2D array (num rows, num columns)
    pts: list or list of lists
        If list, should be for [row index, col index]. Can specify multiple pts with list of lists
    kernel: array
        2D kernel used to diffuse point signal    
    
    Returns:
    --------
    2D array

    """
    s = point_signal(dim,pts)
    s = convolve2d(s,kernel,boundary='symm',mode='same') 
    return s

def kernel_ring_signal(dim,pts,kernel):
    """
    Generates 2D array with points signals specified by pts

    Parameters:
    -----------
    dim: tuple,list
        Dimensions of 2D array (num rows, num columns)
    pts: list or list of lists
        If list, should be for [row index, col index]. Can specify multiple pts with list of lists
    kernel: array
        2D kernel used to diffuse point signal    
    
    Returns:
    --------
    2D array

    """

    s = point_signal(dim,pts)
    s = convolve2d(s,kernel,boundary='fill',mode='same') 
    return s


def slice_signal(dims,irange,jrange):
    """
    Generates 2D array with point signal along a continuous slice of 2D array

    Parameters:
    -----------
    dim: tuple,list
        Dimensions of 2D array (num rows, num columns)
    irange: tuple,list
        Row range of slice: (imin,imax)
    irange: tuple,list
        Column range of slice: (jmin,jmax)

    Returns:
    --------
    2D array
    """
    S = np.zeros(dims)
    islice = slice(irange[0],irange[1],None)
    jslice = slice(jrange[0],jrange[1],None)
    S[islice,jslice] = 1
    return S

def kernel_slice_signal(dim,irange,jrange,kernel):
    """
    Applies kernel to 2D array with point signal along a continuous slice of 2D array

    Parameters:
    -----------
    dim: tuple,list
        Dimensions of 2D array (num rows, num columns)
    irange: tuple,list
        Row range of slice: (imin,imax)
    irange: tuple,list
        Column range of slice: (jmin,jmax)
    kernel: array
        2D kernel used to diffuse point signal    
 
    Returns:
    --------
    2D array
    """
    s = slice_signal(dim,irange,jrange)
    s = convolve2d(s,kernel,boundary='fill',mode='same') 
    return s


