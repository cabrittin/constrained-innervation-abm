"""                            
@name: abm.scanalysis.measures.py
@description:                  
Measures for single-cell analysis

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""
from itertools import combinations
import numpy as np

def compute_discordance(cells,imap,G,verbose=False):
    P = np.zeros((len(cells),len(cells)))
    for (u,v) in combinations(cells,2):
        i = imap[u]
        j = imap[v]
        P[i,j] = log2_diff_expression(G[i,:],G[j,:])
        P[j,i] = P[i,j]
        if verbose: print(u,v,P[i,j])
    return P


def log2_diff_expression(Gi,Gj):
    psum = Gi + Gj
    psum[psum>1] = 1
    psum = psum.sum()
    nsum = np.dot(Gi,Gj)
    dsum = psum - nsum
    diff = 0 
    if dsum > 0 and nsum > 0: 
        diff = np.log2(dsum) - np.log2(nsum)
    elif nsum == 0:
        diff = 20
    elif dsum == 0:
        diff = 0

    return diff

def convert_to_similar(P,thresh):
    S = np.zeros(P.shape)
    S[P<=thresh] = 1
    return S


