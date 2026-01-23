"""                            
@name: abm.empirical.py
@description:                  
Function for dealing with empirical data

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""
from configparser import ConfigParser,ExtendedInterpolation
from itertools import combinations
import numpy as np
import networkx as nx
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind_from_stats
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from abm.proc_data import PipeObject
import toolbox.graphs.pop_communities as gpc

from pycsvparser import read
from .measures import initialize_graph_pioneers,tally_pioneer_contact_weights

CMAP = ['#9301E7','#E7A401','#5E7FF1','#FC0000','#1FFF00','#9b9b9b']

def intra_age_variance(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.cfg)
    
    chron = cfg['params']['data_chronology'].split(',')
    G = [nx.read_graphml(cfg['files'][c]) for c in chron]
    N = len(G)
    
    data = []
    for (i,j) in combinations(range(N),2):
        age_diff = abs(i-j)
        cons = fraction_conserved_edges(G[i],G[j])
        print(f"{chron[i]},{chron[j]},{age_diff},{cons}")
        data.append([age_diff,cons])

    df = pd.DataFrame(data=data,columns=['age_diff','similarity'])
    
    fig,ax = plt.subplots(1,1,figsize=(2,2))
    sns.boxplot(ax=ax,data=df,x='age_diff',y='similarity',
                color='lightgray',width=0.5)
    
    sns.stripplot(ax=ax,data=df,x='age_diff',y='similarity',
                  color='k',jitter=True,size=2)
    
    ax.set_ylim([0.45,0.75])
    ax.set_ylabel('Fraction conserved contacts',fontsize=8)
    ax.set_xlabel('Age difference', fontsize=8)
    ax.tick_params(axis='both',labelsize=6)
    plt.tight_layout()



def fraction_conserved_edges(G1, G2):
    """
    Compute the fraction of conserved (undirected) edges between two graphs,
    ignoring edge order.
    """
    # Normalize each edge so that ('A', 'B') and ('B', 'A') are treated the same
    edges1 = {tuple(sorted(e)) for e in G1.edges()}
    edges2 = {tuple(sorted(e)) for e in G2.edges()}

    conserved = edges1 & edges2
    if not edges1: return 0.0
    
    return len(conserved) / len(edges1 | edges2)
    

def reproducibility_shape(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.cfg)
    
    fin = cfg['files'][args.graph]
    G = PipeObject(fin=fin)

    R = [data['id'] for (u,v,data) in G.edges(data=True)]
    weights = np.ones_like(R) / len(R)
    
    bins = np.arange(0.5,7)
    fig,ax = plt.subplots(1,1,figsize=(1.5,1.2))
    ax.hist(R,bins=bins,weights=weights,rwidth=0.7,
            facecolor='lightgray',edgecolor='k')
    ax.set_ylim([0,0.5])
    ax.set_yticks([0,0.2,0.4])
    ax.set_xticks(bins)
    ax.set_xticks(np.arange(1,7))
    ax.tick_params(axis='both',labelsize=6)
    ax.set_ylabel('Frac. contacts', fontsize=7)
    #ax.set_xlabel('Reproducibility', fontsize=7)
    ax.set_xlabel('# datasets with contact', fontsize=7)
    ax.text(0.05,0.85,f'{args.label}',transform=ax.transAxes,fontsize=7)
    ax.text(0.05,0.65,f'n={len(R)}',transform=ax.transAxes,fontsize=7)
    plt.tight_layout()

def reproducibility_shape_unimodal(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.cfg)
    
    fin = cfg['files'][args.graph]
    G = PipeObject(fin=fin)

    R = [data['id'] for (u,v,data) in G.edges(data=True)]
    N = len(R)
    U = assign_gaussian_values(N)
    weights = np.ones_like(U) / N

    bins = np.arange(0.5,7)
    fig,ax = plt.subplots(1,1,figsize=(1.5,1.2))
    ax.hist(U,bins=bins,weights=weights,rwidth=0.7,
            facecolor='lightgray',edgecolor='k')
    ax.set_ylim([0,0.5])
    ax.set_yticks([0,0.2,0.4])
    ax.set_xticks(bins)
    ax.set_xticks(np.arange(1,7))
    ax.tick_params(axis='both',labelsize=6)
    ax.set_ylabel('Frac. contacts', fontsize=7)
    #ax.set_xlabel('Reproducibility', fontsize=7)
    ax.set_xlabel('# datasets with contact', fontsize=7)
    ax.text(0.05,0.85,f'{args.label}',transform=ax.transAxes,fontsize=7)
    ax.text(0.05,0.65,f'n={N}',transform=ax.transAxes,fontsize=7)
    plt.tight_layout()



def assign_gaussian_values(N, peak_value=6, peak_fraction=0.4):
    """
    Assigns N objects values from 1-6 such that ~peak_fraction of objects get peak_value,
    and the rest are distributed below the peak to give a mean near peak_value.
    """
    values = np.array([1,2,3,4,5,6])
    
    # Assign 40% of values to 6
    num_peak = int(N * peak_fraction)
    assigned = [6]*num_peak
    
    # Remaining objects
    num_rest = N - num_peak
    # Define probabilities for 1-5 using a truncated discrete Gaussian (centered near 5.5)
    x = np.array([1,2,3,4,5])
    mu = 5.5  # mean near 6
    sigma = 2  # controls spread
    probs = np.exp(-0.5*((x-mu)/sigma)**2)
    probs /= probs.sum()  # normalize

    # Randomly assign the rest
    assigned_rest = np.random.choice(x, size=num_rest, p=probs)
    
    # Combine and shuffle
    assigned.extend(assigned_rest)
    np.random.shuffle(assigned)
    return np.array(assigned)


def spatial_domains(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.cfg)
    
    fin = cfg['files'][args.graph]
    G = PipeObject(fin=fin)
    
    max_d = float(args.max_dendrogram)
    nodes = sorted(G.nodes())
    G.communities(tqdm_disable=False,nodes=nodes,
                  keep_pop=True,iters=100,max_d=max_d)
    G.index_communities()
    ncolor = [CMAP[G.comm_index[n]] for n in nodes]
    z = gpc.pop_comm_correlation(G.pop_comms)
    y = gpc.pop_comm_linkage(z)
    gpc.pop_comm_dendrogram(y,max_d=max_d,
                            truncate_mode='lastp',
                            p=12,
                            leaf_rotation=90.,
                            leaf_font_size=8.,
                            show_contracted=True,annotate_above=10)
    im = gpc.pop_comm_heatmap(z,y,yticklabels=nodes,
                              colors=ncolor,no_cbar=True)

def age_scores(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.cfg)
    
    df = pd.read_csv(cfg['files']['dataframe']) 
    
    y = df[args.dfcol].tolist()
    x = [0,1,2]
    fig,ax = plt.subplots(1,1,figsize=(1.1,1))
    ax.bar(x,y,width=0.5,facecolor='lightgray',edgecolor='k')
    ax.set_yticks(list(map(float,args.yticks.split(',')))) 
    ax.tick_params(axis='both',labelsize=6)
    ax.set_ylabel(args.ylabel,fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(['Y','M','A'],fontsize=8)
    plt.tight_layout()

def pioneer_follower_distance(args):
    df = pd.read_csv(args.df_emp)
    df = df[df['is_pioneer'] == 0]
    
    for _, row in df.iterrows():
        first_col = row.iloc[0]  
        nonzero = row[row != 0].drop(df.columns[0], errors='ignore')  
        pairs = [f"{col},{val}" for col, val in nonzero.items()][0]
        print(f"{first_col},{pairs}")


    df = df.iloc[:,2:-1].to_numpy()
    dist = np.max(df,axis=1)
    
    x = np.arange(70)
    y = _ecdf(dist,x)
    
    fig,ax = plt.subplots(1,1,figsize=(1.2,1.5))
    ax.plot(x,y,color='k')
    ax.set_xlim([1,5]) 
    ax.set_xticks([1,2,3,4,5]) 
    ax.set_xticklabels([0,1,2,3,4]) 
    ax.set_ylim([0,1])
    ax.set_ylabel('ECDF',fontsize=8)
    ax.set_xlabel('Pioneer-follower distance',fontsize=8)
    ax.tick_params(axis='y',labelsize=6)
    ax.tick_params(axis='x',labelsize=6)
    ax.set_title(f'n = {len(df)}',fontsize=6) 
    plt.tight_layout()

def _ecdf(data,dsort):
    return np.array([np.sum(data <= x) / len(data) for x in dsort])


def pioneer_contact_distribution(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)

    pct_thresh = cfg.getfloat('params','pg_pct_thresh')
    
    fig,ax = plt.subplots(1,3,figsize=(3.5,1.5),sharey=True)
    pio = sorted(read.into_list(cfg['mat']['pioneers']))
    for (idx,(gname,grp)) in enumerate(cfg['groups'].items()):
        print(f"Loading {cfg['files'][gname]}")
        G = PipeObject(fin=cfg['files'][gname])
        initialize_graph_pioneers(G,pio) 
        weight = np.log(np.array(tally_pioneer_contact_weights(G,pio)))
        z = (weight - weight.mean()) / weight.std()

        x = np.linspace(-4,4,801)
        y = np.array([np.sum(z <= _x) / len(weight) for _x in x])
        
        ydx = min(np.where(y>=pct_thresh)[0])
        
        ax[idx].plot(x,y)
        ax[idx].set_xlim([-3,3])
        ax[idx].set_ylim([0,1])
        ax[idx].set_xticks([-3,-2,-1,0,1,2,3])
        ax[idx].tick_params(axis='x',labelsize=6)
        ax[idx].set_xlabel('Pioneer contacts',fontsize=8)
        ax[idx].hlines(y[ydx],-3,x[ydx],'r',linestyle='--')
        ax[idx].vlines(x[ydx],0,y[ydx],'r',linestyle='--')

     
    ax[0].tick_params(axis='y',labelsize=6) 
    ax[0].set_ylabel('ECDF',fontsize=8) 
    plt.tight_layout() 
    #plt.savefig('results/pioneer_contact_thresh.svg') 
    #plt.show()

def assigned_pioneer_groups(args):
    from kmodes.kmodes import KModes
    df = pd.read_csv(args.df_emp)
    numpio = int(df['is_pioneer'].sum())
    pio = df[df['is_pioneer']==1]['cell'].tolist()
    fol = df[df['is_pioneer']==0]['cell'].tolist()
    
    X = df[df['is_pioneer']==0].iloc[:,2:].to_numpy()
    
    def cluster_reorder_binary_index(arr, k_clusters):
        # Cluster rows using K-modes
        km = KModes(n_clusters=k_clusters, init='Huang', n_init=20, verbose=0)
        clusters = km.fit_predict(arr)
        
        # Reorder the array based on cluster assignments
        return np.argsort(clusters)
    
    print(pio)
    idx = [5,6,4,7,13,3,8,9,10,11,12,1,2,0] 
    #k_clusters = 4
    #idx = cluster_reorder_binary_index(X.T, k_clusters)
    X = X[:,idx]
    pio = [pio[i] for i in idx]
    print(pio)

    k_clusters = 5
    idx = cluster_reorder_binary_index(X, k_clusters)
    X = X[idx,:]
    fol = [fol[i] for i in idx]


    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.imshow(X,cmap='binary', interpolation='nearest')
    ax.set_xticks(np.arange(len(pio))) 
    ax.set_xticklabels(pio,fontsize=8,rotation=90)
    ax.set_yticks(np.arange(len(fol))) 
    ax.set_yticklabels(fol,fontsize=8)
    plt.tight_layout()

def pg_synapse_fraction(args):
    Z = np.load(args.data_emp)
    N = Z['N']
    D = Z['D']
    
    print(N)
    print(D)

    def make_subplot(ax,d,n):
        x = np.arange(len(d))
        y0 = d[:,-1]
        y0sum = y0.sum()
        y0 /= y0sum 
        #y0 = d[:,1:].sum(axis=1)
        y1 = n[:,1:].sum(axis=1)
        y1sum = y1.sum() 
        y1 /= y1sum 
        y2 = n[:,-1]
        y2sum = y2.sum()
        y2 /= y2sum

        width = 0.25
        xticks = ["p-f","f-f","p-p'","p-f'","f-f'"] 
        ax.bar(x - width,y0,width,label=f'Cons. contacts: {int(y0sum)} ')
        ax.bar(x,y1,width,label=f'All synapses: {int(y1sum)}')
        ax.bar(x+width,y2,width,label=f'Cons. synapses: {int(y2sum)}')
        ax.set_xticks(x)
        ax.set_xticklabels(xticks,fontsize=8)
        ax.tick_params(axis='y',labelsize=8)
        ax.set_ylabel('Frac. contacts',fontsize=8)
        #ax.set_ylim([0,80])
        ax.set_ylim([0,0.5])
        ax.legend(loc='upper right',fontsize=6)
    
    fig,ax = plt.subplots(3,1,figsize=(2,4))
    for i in range(3): make_subplot(ax[i],D[i,:,:],N[i,:,:])
    plt.tight_layout()

def pg_synapse_fraction_distilled(args):
    Z = np.load(args.data_emp)
    N = Z['N']
    D = Z['D']
    R = Z['R'] 

    def scaled_all(D):
        dsum = D.sum(axis=2)
        dscale = dsum.sum(axis=1) 
        dsum = dsum / dscale[:,np.newaxis]
        return dsum.mean(axis=0)
    
    def scaled_conserved(D):
        dsum = D[:,:,-1]
        dscale = dsum.sum(axis=1)
        dsum = dsum / dscale[:,np.newaxis]
        return dsum.mean(axis=0)
    
    def scaled_rand(D):
        #dsum = D[:,:,-1]
        dsum = D.sum(axis=2)
        dscale = dsum.sum(axis=1)
        idx = np.where(dscale>0)[0] 
        dsum = dsum[idx,:]
        dscale = dscale[idx]
        dsum = dsum / dscale[:,np.newaxis]
        return dsum.mean(axis=0)
    
    #print(D) 
    #print(N)
    print('h',N.sum(axis=1))

    D = D[:,:,1:]
    z0 = scaled_conserved(D)

    N = N[:,:,1:]
    z1 = scaled_all(N)
    z2 = scaled_conserved(N)
    
    R = R[:,:,1:]
    z3 = scaled_rand(R)
    
    print(z0)
    print(z1)
    print(z2)
    print(z3) 
    #Z = np.stack((z0,z1,z2,z3),axis=0)
    Z = np.stack((z0,z1,z3),axis=0)
    Z = Z[:,[0,1,4]]

    fig,ax = plt.subplots(1,1,figsize=(2,2))
    img = ax.imshow(Z,cmap='plasma',vmin=0,vmax=0.3)
    ax.set_xticks([0,1,2])
    ax.set_xticklabels(['p-f','f-f',"f-f'"],fontsize=8)
    ax.set_yticks([0,1,2])
    #ax.set_yticklabels(['Cons. contacts','Cons. synapse','Rand. Null synapse'],fontsize=8)
    ax.set_yticklabels(['Cons. contacts','All synapses','Rand. Null synapse'],fontsize=8)
    cbar = fig.colorbar(img, ax=ax, 
                        orientation='horizontal', fraction=0.1, pad=0.1)
    cbar.ax.set_xticks([0,0.15,0.30]) 
    cbar.ax.tick_params(labelsize=6) # Adjust tick label font size
    cbar.set_label('Frac. contacts', fontsize=6)



    plt.tight_layout()

def pg_synapse_chi2(args):
    from scipy.stats import chisquare
    
    Z = np.load(args.data_emp)
    N = Z['N']
    D = Z['D']
    
    print(N.shape)

    def chi2_test(obs,sizes):
        n = obs.sum()
        probs = sizes / sizes.sum()
        exp = n * probs
        
        chi2_stat, p_value = chisquare(f_obs=obs, f_exp=exp)
        
        k = len(obs)
        cramers_v = np.sqrt(chi2_stat / (n * (k - 1)))
        
        print(f"Chi2 statistic: {chi2_stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print(f"Cramér's V (effect size): {cramers_v:.4f}")
        
        return obs,exp
   
    def make_chi2_plot(ax,obs,exp):
        x = np.arange(len(obs))
        width = 0.25
        xlabel = ["p-f","f-f","p-p'","p-f'","f-f'"] 
        
        ax.bar(x - width/2, obs, width, label='Observed')
        ax.bar(x + width/2, exp, width, label='Expected')
        ax.set_ylim([0,50])
        ax.set_yticks([0,10,20,30,40,50])
        ax.set_xticks(x)
        ax.set_xticklabels(xlabel,fontsize=8)
        #ax.set_ylabel("Counts",fontsize=8)
        ax.set_ylabel("# syn. contacts",fontsize=8)
        ax.tick_params(axis='y',labelsize=6)
        ax.legend(loc='upper center',fontsize=6)

    def make_residual_plot(ax,obs,exp):
        x = np.arange(len(obs))
        width = 0.25
        xlabel = ["p-f","f-f","p-p'","p-f'","f-f'"] 
        # standardized residuals: (obs - exp) / sqrt(exp)
        std_resid = (obs - exp) / np.sqrt(exp)
        ax.bar(x, std_resid)
        ax.axhline(0, color='k', linewidth=0.6)
        ax.set_xticks(x)
        ax.set_ylim([-3,3])
        ax.set_xticklabels(xlabel,fontsize=8)
        #ax.set_ylabel("Standardized residual",fontsize=8)
        ax.set_ylabel("Residual",fontsize=8)
        ax.axhline(2,linestyle='--',color='r')
        ax.axhline(-2,linestyle='--',color='r')
        ax.tick_params(axis='y',labelsize=6)

    fig,_ax = plt.subplots(3,2,figsize=(4,4))
    ax = _ax.ravel()
    #obs,exp = chi2_test(N[0,:,-1],D[0,:,-1])
    obs,exp = chi2_test(N[0,:,1:].sum(axis=1),D[0,:,-1])
    make_chi2_plot(ax[0],obs,exp)
    make_residual_plot(ax[1],obs,exp)

    #obs,exp = chi2_test(N[1,:,-1],D[1,:,-1])
    obs,exp = chi2_test(N[1,:,1:].sum(axis=1),D[1,:,-1])
    make_chi2_plot(ax[2],obs,exp)
    make_residual_plot(ax[3],obs,exp)

    #obs,exp = chi2_test(N[2,:,-1],D[2,:,-1])
    obs,exp = chi2_test(N[2,:,1:].sum(axis=1),D[2,:,-1])
    make_chi2_plot(ax[4],obs,exp)
    make_residual_plot(ax[5],obs,exp)

    ax[0].set_title("Observed vs Expected\n(proportional to size)",fontsize=8)
    ax[1].set_title("Standardized residuals\n(> ~2 notable deviation)",fontsize=8)
    
    plt.tight_layout()



