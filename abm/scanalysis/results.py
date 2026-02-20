"""                            
@name: abm.scanalysis.results.py
@description:                  
Single-cell analysis results

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

from configparser import ConfigParser,ExtendedInterpolation
import numpy as np
from scipy.sparse import coo_array,save_npz
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.colors import LogNorm, PowerNorm
from scipy import stats
from itertools import combinations

from pycsvparser import read,write
from sctool import SingleCell
from sctool import scmod

from .measures import compute_discordance,convert_to_similar


def discordance_heatmap(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    pio = read.into_list(cfg['mat']['pioneers'])
    gene_keep = np.load(cfg['mat']['gene_keep'])
    
    sc = SingleCell(cfg)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
    print(f"Keeping cells in: {cfg['mat']['pioneers']}")
    scmod.select_cells_isin(sc,'cell_id',pio)
    print(f"Keeping genes in {cfg['mat']['gene_keep']}") 
    scmod.select_genes(sc,gene_keep)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
 
    imap = dict([(c,i) for (i,c) in enumerate(sc.cells['cell_id'])])
    G = sc.X.toarray()
    G[G>0] = 1
    P = compute_discordance(pio,imap,G)
    
    rcells = sc.cells['cell_id'].tolist()
    print(rcells)
    mdx = np.array(list(map(int,cfg['params']['rosette_order'].split(','))))
    rcells = [rcells[i] for i in mdx]
    print(rcells)

    P = P[mdx,:]
    P = P[:,mdx]

    fig,ax = plt.subplots(1,1,figsize=(3,2))
    sns.heatmap(P,ax=ax, cmap="magma",
                norm=PowerNorm(gamma=0.6, vmin=0, vmax=4.0))
                #cbar_kws={'label': 'Expression (log2, gamma-corrected)','fontsize':8})
    #ax.set_title("Heatmap with PowerNorm Gamma Correction")
    ax.set_yticks(np.arange(len(rcells))+0.5)
    ax.set_yticklabels(rcells,fontsize=6,rotation=0)
    ax.set_xticks(np.arange(len(rcells))+0.5)
    ax.set_xticklabels(rcells,fontsize=6)
    tick_values = [1, 2, 4, 8,16]
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(np.log2(tick_values))
    cbar.set_ticklabels([str(v) for v in tick_values])
    cbar.ax.tick_params(labelsize=6)       # tick label font size
    cbar.ax.set_ylabel("CAM discordance", fontsize=8) 
    plt.tight_layout() 
    

def discordance_rosettes(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    pio = read.into_list(cfg['mat']['pioneers'])
    pmap = read.into_dict(cfg['mat']['rosette'],dtype=int) 
    gene_keep = np.load(cfg['mat']['gene_keep'])
    
    sc = SingleCell(cfg)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
    print(f"Keeping cells in: {cfg['mat']['pioneers']}")
    scmod.select_cells_isin(sc,'cell_id',pio)
    print(f"Keeping genes in {cfg['mat']['gene_keep']}") 
    scmod.select_genes(sc,gene_keep)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")

    imap = dict([(c,i) for (i,c) in enumerate(sc.cells['cell_id'])])
    G = sc.X.toarray()
    G[G>0] = 1
    P = compute_discordance(pio,imap,G,verbose=False)
    #print('split') 
    data = []
    for (u,v) in combinations(pio,2):
        i = imap[u]
        j = imap[v]
        intra_rosette = int(pmap[u] == pmap[v])
        data.append([intra_rosette,P[i,j]])
        print(f"{u},{v},{intra_rosette},{P[i,j]}")
    
    df = pd.DataFrame(data=data,columns=['rosette','discordance'])
    
    group_a = df[df['rosette'] == 0]['discordance']
    group_b = df[df['rosette'] == 1]['discordance']
    t_statistic, p_value = stats.ttest_ind(group_a, group_b,equal_var=False)
    print(f"T-statistic: {t_statistic:.2f}")
    print(f"P-value: {p_value:.3f}")
    print(f"inter-rosettes n = {len(group_a)}")
    print(f"intra-rosettes n = {len(group_b)}")

    fig,ax = plt.subplots(1,1,figsize=(1.25,1.5))
    sns.boxplot(ax=ax,data=df,
                x='rosette',y='discordance',
                color='lightgray',width=0.5,order=[1,0])
    sns.stripplot(ax=ax,data=df,
                x='rosette',y='discordance',order=[1,0],
                color='k',size=2) 
    tick_values = [1, 2, 4, 8,16]
    ax.set_yticks(np.log2(tick_values))
    ax.set_yticklabels([str(v) for v in tick_values],fontsize=6)
    ax.set_xticks([0,1])
    ax.set_xticklabels(['intra','inter'],fontsize=8)
    ax.set_ylabel('CAM discordance',fontsize=8)
    ax.set_xlabel('Rosette',fontsize=8)
    plt.tight_layout()


def discordance_axon_distance(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    #pio = read.into_list(cfg['mat']['pioneers'])
    cmap = read.into_dict(cfg['mat']['cell_name_map']) 
    gene_keep = np.load(cfg['mat']['gene_keep'])

    df = pd.read_csv(cfg['mat']['pioneer_groups_fuzzy'])
    numpio = int(df['is_pioneer'].sum())
    pio = df[df['is_pioneer']==1]['cell'].tolist()
    numpio = len(pio)

    sc = SingleCell(cfg)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
    print(f"Keeping genes not in {cfg['mat']['gene_keep']}") 
    scmod.select_genes(sc,gene_keep)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
    ckeep = [cmap[c] for c in df[df['is_pioneer'] == 1]['cell'].tolist()]
    scmod.select_cells_isin(sc,'cell_id',ckeep)
    print(f"Removing cells not in {cfg['mat']['ref_cells']}") 
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
 
    imap = dict([(c,i) for (i,c) in enumerate(sc.cells['cell_id'])])
    imap_r = dict([(i,c) for (i,c) in enumerate(sc.cells['cell_id'])])
    

    gnames = sc.genes['gene_name'].tolist()  
    G = sc.X.toarray()
    G[G>0] = 1
    gsum = G.sum(axis=0)
    gsum[gsum>0] = 1
    cells = sc.cells['cell_id'].tolist()

    D = np.load(cfg['mat']['pioneer_distance'])
    d = 100*np.ones((G.shape[0],G.shape[0])) 
    for (i,j) in combinations(list(range(numpio)),2):
        try: 
            ci = imap[cmap[pio[i]]]
            cj = imap[cmap[pio[j]]]
            if ci == cj: continue
            #d[ci,cj] = max(d[ci,cj],D[i,j])
            d[ci,cj] = min(d[ci,cj],D[i,j])
            d[cj,ci] = d[ci,cj]
            #print(pio[i],pio[j],D[i,j],)
        except:
            pass

    d = np.floor(d)
    
    sdata = []
    for (i,j) in combinations(list(range(G.shape[0])),2):
        psum = G[i,:] + G[j,:]
        psum[psum>1] = 1
        psum = psum.sum()
        nsum = np.dot(G[i,:],G[j,:])
        dsum = psum - nsum
        nsum = max(1,nsum)
        s = np.log2(dsum)- np.log2(nsum)
        sdata.append([s,d[i,j]])
        print(f"{imap_r[i]},{imap_r[j]},{s},{int(d[i,j])}")
    
    spec = 1 - (G.sum(axis=1) / G.shape[1])
    x = np.linspace(0,1,101)
    y = np.array([np.sum(spec <= _x) / len(spec) for _x in x ])
    
    df = pd.DataFrame(data=sdata,columns=['discordance','separated'])

    group_0 = df[df['separated'] == 1]['discordance']
    group_1 = df[df['separated'] == 2]['discordance']
    group_2 = df[df['separated'] == 3]['discordance']
    
    
    print(f'n_0 = {len(group_0)}')
    print(f'n_1 = {len(group_1)}')
    print(f'n_2 = {len(group_2)}')

    print('Group 0-1')
    t_statistic, p_value = stats.ttest_ind(group_0, group_1,equal_var=False)
    print(f"T-statistic: {t_statistic:.2f}")
    print(f"P-value: {p_value:.3f}")
    
    print('Group 0-2')
    t_statistic, p_value = stats.ttest_ind(group_0, group_2,equal_var=False)
    print(f"T-statistic: {t_statistic:.2f}")
    print(f"P-value: {p_value:.3f}")
    
    fig,ax = plt.subplots(1,1,figsize=(1.25,1.5))
    sns.boxplot(ax=ax,data=df,
                x='separated',y='discordance',
                color='lightgray',width=0.5)
    sns.stripplot(ax=ax,data=df,
                x='separated',y='discordance',
                color='k',size=2) 
    tick_values = [1, 2, 4, 8,16]
    ax.set_yticks(np.log2(tick_values))
    ax.set_yticklabels([str(v) for v in tick_values],fontsize=6)
    #ax.set_xticks([0,1])
    #ax.set_xticklabels(['inter','intra'],fontsize=8)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels([0,1,2],fontsize=8)
    ax.set_ylabel('CAM discordance',fontsize=8)
    ax.set_xlabel('L4 PPD',fontsize=8)
    plt.tight_layout()


def pioneer_uniqueness_table(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    pio = read.into_list(cfg['mat']['pioneers'])
    gene_keep = np.load(cfg['mat']['gene_keep'])
    
    sc = SingleCell(cfg)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
    print(f"Keeping cells in: {cfg['mat']['pioneers']}")
    scmod.select_cells_isin(sc,'cell_id',pio)
    print(f"Keeping genes in {cfg['mat']['gene_keep']}") 
    scmod.select_genes(sc,gene_keep)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
     

    imap = dict([(c,i) for (i,c) in enumerate(sc.cells['cell_id'])])
    G = sc.X.toarray()
    G[G>0] = 1
    P = compute_discordance(pio,imap,G)
    
    discordance_thresh = float(args.discordance_thresh)
    S1 = convert_to_similar(P,discordance_thresh)
    fig,ax = plt.subplots(1,1,figsize=(1.5,1.5))
    ax.imshow(S1,cmap='binary')
    ax.set_xticks(np.arange(len(pio))) 
    ax.set_xticklabels(pio,fontsize=6,rotation=90)
    ax.set_yticks(np.arange(len(pio))) 
    ax.set_yticklabels(pio,fontsize=6)
    plt.tight_layout()
 
def broad_gene_expression(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    pio = read.into_list(cfg['mat']['pioneers'])
    broad_thresh = float(args.gene_thresh)

    sc = SingleCell(cfg)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
    print(f"Keeping cells in: {cfg['mat']['pioneers']}")
    scmod.select_cells_isin(sc,'cell_id',pio)

    G = sc.X.toarray()
    G[G>0] = 1
    gsum = G.sum(axis=0) 

    vals = gsum[:]
    x = np.arange(G.shape[0]+1)
    y = np.array([np.sum(vals <= _x) / len(vals) for _x in x ])
    fig,ax = plt.subplots(1,1,figsize=(1.5,1.5))
    ax.plot(x,y,'k')
    ax.axvline(broad_thresh,linestyle='--',color='r') 
    ax.set_xticks([0,2,4,6,8,10]) 
    ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0]) 
    ax.set_xlim([0,G.shape[0]+1])
    ax.set_ylim([0,1])
    ax.set_ylabel('ECDF',fontsize=8)
    ax.set_xlabel('# pioneers with expr.',fontsize=8)
    ax.tick_params(axis='both',labelsize=6)
    ax.set_title('CAM expr. breadth',fontsize=8) 
    plt.tight_layout() 

def binary_gene_expression(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    pio = read.into_list(cfg['mat']['pioneers'])
    gene_keep = np.load(cfg['mat']['gene_keep'])
    
    sc = SingleCell(cfg)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
    print(f"Keeping cells in: {cfg['mat']['pioneers']}")
    scmod.select_cells_isin(sc,'cell_id',pio)
    print(f"Keeping genes in {cfg['mat']['gene_keep']}") 
    scmod.select_genes(sc,gene_keep)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
    
    rcells = sc.cells['cell_id'].tolist()
    print(rcells)
    mdx = np.array(list(map(int,cfg['params']['rosette_order'].split(','))))
    rcells = [rcells[i] for i in mdx]
    print(rcells)

    imap = dict([(c,i) for (i,c) in enumerate(sc.cells['cell_id'])])
    G = sc.X.toarray()
    G[G>0] = 1
    G = G[mdx,:]
    
    linkage_cols = linkage(G.T, method='average', metric='jaccard')  
    col_order = leaves_list(linkage_cols)
    genes = sc.genes['gene_name'].tolist()
    genes = [genes[i] for i in col_order]

    G = G[:,col_order] 
    fig,ax = plt.subplots(1,1,figsize=(6,2))
    ax.imshow(G,cmap='binary')
    ax.set_yticks(np.arange(len(rcells)))
    ax.set_yticklabels(rcells,fontsize=6)
    ax.set_xticks(np.arange(G.shape[1]))
    ax.set_xticklabels(genes,fontsize=6,rotation=90)
    plt.tight_layout()


