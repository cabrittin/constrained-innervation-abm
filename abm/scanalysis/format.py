"""                            
@name: abm.scanalysis.format.py
@description:                  
Formatting single-cell data

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

from configparser import ConfigParser,ExtendedInterpolation

import numpy as np
from scipy import sparse
from itertools import combinations,product
import csv
from scipy.sparse import coo_array,save_npz
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pycsvparser import read,write
from sctool import SingleCell
from sctool import scmod

from .measures import compute_discordance,convert_to_similar

def _format_binary(func):
    def inner(args): 
        cfg = ConfigParser(interpolation=ExtendedInterpolation())
        cfg.read(args.config)
        
        G,C,X = func(cfg)
        
        cell_file = cfg['files']['cells_meta']
        gene_file = cfg['files']['genes_meta']
        count_file = cfg['files']['count_matrix']

        print(f'Writing cells to {cell_file}')
        write.from_list(cell_file,C)
        print(f'Writing genes to {gene_file}')
        write.from_list(gene_file,G)
        
        X = coo_array(np.array(X))
        print(f'Writing counts to {count_file}')
        save_npz(count_file,X)
    
    return inner

@_format_binary
def format_packer_binary(cfg):
    cam = read.into_list(cfg['mat']['cam_genes'],multi_dim=True)
    wbid = [c[0] for c in cam]
    gmap = dict([(c,i) for (i,c) in enumerate(wbid)])
    
    time_keep = cfg['params']['time_keep'].split(',')
    regex_time_pattern = '|'.join(time_keep)

    source = cfg['files']['source']
    df = pd.read_csv(source, sep='\t', quoting=csv.QUOTE_NONE)
    print('Original shape',df.shape)
    df = df[df['gene.id'].isin(wbid)]
    print('After filtering for CAM genes:',df.shape)
    df = df[df['cell.bin'].str.contains(regex_time_pattern,na=False)]
    print('After filtering time bins:',df.shape)
    df = df[df['ci.95p.lb'] > 0]
    print('After removing low CI95',df.shape)
    
    unique_cells = df['cell.bin'].unique()
    cmap = dict([(c,i) for (i,c) in enumerate(unique_cells)])
    csplit = [c.split(':') for c in unique_cells] 

    X = np.zeros((len(unique_cells),len(cam)))

    for (idx,row) in df.iterrows():
        gene = row['gene.id']
        cell = row['cell.bin']
        tpm = row['adjusted.tpm.estimate']
        X[cmap[cell],gmap[gene]] = tpm
    
    G = [['Wormbase_ID','gene_name']] + cam
    C = [['cell_id','time_bin']] + csplit
    
    return G,C,X 

@_format_binary
def format_cengen_binary(cfg):
    cam = read.into_list(cfg['mat']['cam_genes'],multi_dim=True)
    wbid = [c[0] for c in cam]
    gmap = dict([(c,i) for (i,c) in enumerate(wbid)])
    
    source = cfg['files']['source']
    with open(source, mode='r', newline='') as file:
        reader = csv.reader(file)
        C = ['cell_id'] + next(reader)[3:]
        cmap = dict([(c,i) for (i,c) in enumerate(C[1:])])
        X = np.zeros((len(cmap),len(cam)))

        for row in reader:
            if row[2] not in wbid: continue 
            #genes.append(row[1:3])
            jdx = gmap[row[2]]
            X[:,jdx] = np.array(list(map(float,row[3:]))) 
    
    G = [['Wormbase_ID','gene_name']] + cam
    return G,C,X 

def filter_broad_genes(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    pio = read.into_list(cfg['mat']['pioneers'])
    gene_keep = cfg['mat']['gene_keep']
    broad_thresh = float(args.gene_thresh)

    sc = SingleCell(cfg)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
    print(f"Keeping cells in: {cfg['mat']['pioneers']}")
    scmod.select_cells_isin(sc,'cell_id',pio)

    G = sc.X.toarray()
    G[G>0] = 1
    gsum = G.sum(axis=0)
    gdx = np.where((gsum <= broad_thresh) & (gsum > 0))
    np.save(gene_keep,gdx[0])
    print(f'Writing filtered gene indices to: {gene_keep}')
    
    #gdx = np.where(gsum > 0)
    
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
    plt.show() 

def dataset_filter_compare(args):
    def load_sc(config): 
        cfg = ConfigParser(interpolation=ExtendedInterpolation())
        cfg.read(config)
        
        pio = read.into_list(cfg['mat']['pioneers'])
        gene_keep = np.load(cfg['mat']['gene_keep'])
        
        sc = SingleCell(cfg)
        print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
        print(f"Keeping cells in: {cfg['mat']['pioneers']}")
        scmod.select_cells_isin(sc,'cell_id',pio)
        print(f"Keeping genes in {cfg['mat']['gene_keep']}") 
        scmod.select_genes(sc,gene_keep)
        print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
        
        return sc
    
    def get_gene_list(config):
        cfg = ConfigParser(interpolation=ExtendedInterpolation())
        cfg.read(config)
        cam = read.into_list(cfg['mat']['cam_genes'],multi_dim=True)
        return cam

    sc1 = load_sc(args.config_1)
    sc2 = load_sc(args.config_2)

    g1 = set(sc1.genes['gene_name'].tolist())
    g2 = set(sc2.genes['gene_name'].tolist())
    

    shared = len(g1 & g2)
    total = len(g1 | g2)
    jaccard = float(shared) / total 
    g1_only = len(g1 - g2)
    g2_only = len(g2 - g1)

    print(f'Similarity: {jaccard}') 
    print(f'Shared: {shared}') 
    print(f'G1 only: {g1_only}') 
    print(f'G2 only: {g2_only}') 

    cam = get_gene_list(args.config_1)
    
    for (u,v) in cam:
        in_g1 = int(v in g1)
        in_g2 = int(v in g2)
        print(f"{u},{v},{in_g1},{in_g2}")

def supp_cell_by_gene_table_raw(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    pio = read.into_list(cfg['mat']['pioneers'])
    
    sc = SingleCell(cfg)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
    print(f"Keeping cells in: {cfg['mat']['pioneers']}")
    scmod.select_cells_isin(sc,'cell_id',pio)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
    
    G = sc.X.toarray()
    cells = sc.cells['cell_id'].tolist()
    genes = sc.genes['gene_name'].tolist()
    
    df = pd.DataFrame(data=G,columns=genes)
    df.insert(0,'neuron_class',cells)

    df.to_csv(args.fout,index=False)
    print(f'Wrote to: {args.fout}')

def supp_cell_by_gene_table_binary(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    pio = read.into_list(cfg['mat']['pioneers'])
    gene_keep = np.load(cfg['mat']['gene_keep'])
    
    sc = SingleCell(cfg)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
    print(f"Keeping cells in: {cfg['mat']['pioneers']}")
    scmod.select_cells_isin(sc,'cell_id',pio)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
    print(f"Keeping genes in {cfg['mat']['gene_keep']}") 
    scmod.select_genes(sc,gene_keep)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
 
    G = sc.X.toarray()
    G[G>0] = 1 
    cells = sc.cells['cell_id'].tolist()
    genes = sc.genes['gene_name'].tolist()
    
    df = pd.DataFrame(data=G,columns=genes)
    df.insert(0,'neuron_class',cells)

    df.to_csv(args.fout,index=False)
    print(f'Wrote to: {args.fout}')


def genes_filter_breakdown(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    
    pio = read.into_list(cfg['mat']['pioneers'])
    broad_thresh = float(args.gene_thresh)
    
    sc = SingleCell(cfg)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
    print(f"Keeping cells in: {cfg['mat']['pioneers']}")
    scmod.select_cells_isin(sc,'cell_id',pio)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
    
    G = sc.X.toarray()
    G[G>0] = 1 
    gsum = G.sum(axis=0)
    IDX = np.where(gsum == 0)[0]
    
    genes = sc.genes['gene_name'].tolist()
    print(f'Genes without cell expression')
    for i in IDX: print(genes[i])
    print(f'Num genes without expression: {len(IDX)}')
    
    idx = np.where(gsum > 0)[0]
    print(f"Removing genes without expression") 
    scmod.select_genes(sc,idx)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")
    
    G = sc.X.toarray()
    G[G>0] = 1
    gsum = G.sum(axis=0)
    IDX = np.where((gsum > broad_thresh))[0]

    genes = sc.genes['gene_name'].tolist()
    print(f'Genes without broad (>75%) expression')
    print([genes[i] for i in IDX])
    for i in IDX: print(genes[i])
    print(f'Num genes with broad expression: {len(IDX)}')
    
    idx = np.where(gsum <= broad_thresh)[0]
    print(f"Removing genes with broad expression") 
    scmod.select_genes(sc,idx)
    print(f"Cells: {sc.X.shape[0]}, Genes: {sc.X.shape[1]}")

    print('Remaining genes..')
    print(sc.genes['gene_name'].tolist()) 
    for g in sc.genes['gene_name'].tolist(): print(g)
   

def pioneer_uniqueness(args):
    """Computes pioneer distinctiveness, uniqueness is an old naming convention"""

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
    
    print(P)
    discordance_thresh = float(args.discordance_thresh)
    S1 = convert_to_similar(P,discordance_thresh)
    print(S1)
    u1 = 1-(S1.sum(axis=1) - 1) / (S1.shape[0] - 1)
    
    for (i,p) in enumerate(pio):
        print(f'{p},{u1[imap[p]]}')

    x = np.linspace(0,1,101)
    y = np.array([np.sum(u1 <= _x) / len(u1) for _x in x ])
    
    np.save(args.fout,y)
    print(f'Writing to: {args.fout}')

    fig,ax = plt.subplots(1,1,figsize=(2,2))
    ax.plot(x,1-y)
    ax.set_ylabel('1-ECDF',fontsize=8)
    ax.set_xlabel('Discordance',fontsize=8)
    ax.tick_params(axis='both',labelsize=6)

    plt.show()


def discordance_table(args):
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
    
    df = pd.DataFrame(data=P,columns=rcells)
    df.insert(0,'neuron_class',rcells)
    
    print(df)
    df.to_csv(args.fout,index=False)
    print(f'Wrote to: {args.fout}')

    udx = np.triu_indices(P.shape[0], k=1) 
    
    u = P[udx]
    x = np.linspace(0,4,401)
    y =np.array([np.sum(u <= _x) / len(u) for _x in x ])
    
    discordance_thresh = float(args.discordance_thresh)
    fig,ax = plt.subplots(1,1,figsize=(2,2))
    ax.plot(x,y)
    ax.set_ylabel('ECDF',fontsize=8)
    ax.set_xlabel('Discordance',fontsize=8)
    ax.tick_params(axis='both',labelsize=6)
    ax.axvline(discordance_thresh,color='r',linestyle='--')

    plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mode',
                        action = 'store',
                        choices = [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')
    
    parser.add_argument('--dir',
                        action = 'store',
                        dest = 'dir',
                        required = False,
                        default = None,
                        help = 'Directory path')
    
    parser.add_argument('-i',
                        action = 'store',
                        dest = 'fin',
                        required = False,
                        default = None,
                        help = 'Input file')

    parser.add_argument('-o',
                        action = 'store',
                        dest = 'fout',
                        required = False,
                        default = None,
                        help = 'Input file')


    args = parser.parse_args()
    eval(args.mode + '(args)')

