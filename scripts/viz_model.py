"""
@name: viz_model.py                        
@description:                  
    Function for visually verifying model setup

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import sys
import argparse
from configparser import ConfigParser,ExtendedInterpolation
from inspect import getmembers,isfunction
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.cm import get_cmap
from matplotlib.colors import ListedColormap, BoundaryNorm
import time
import numpy as np
import importlib

import abm.viz as av
from abm.viz import animate_volume_slice
from abm.model.model_base import get_position_encoding
from abm.batchrunner import batch_mp_vol,batch_mp_pipe
from abm.hooks import GraphHook as HK
import abm.proc_data
from abm.proc_data import process_graphs
from abm.proc_data import PipeObject

from pycsvparser import read

def load_model(args):
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    M = importlib.import_module(f"abm.model.{cfg['model']['model']}.model")
    return M.Model(cfg)

def viz_attractor_fields(args):
    """
    Plots heatmaps of the attractor fields
    """
    model = load_model(args)
    model.init()
    q,r = divmod(model.F.shape[2],5)
    print(model.num_pioneers)
    nrows = q + int(r>0) 
    fig,_ax = plt.subplots(nrows,5,figsize=(15,2*nrows))
    ax = _ax.flatten() 
    m = model.F.shape[0] 
    n = model.F.shape[1] 
    Z = np.zeros((m,n))
    for k in range(len(ax)): 
        ax[k].set_xticks([])
        ax[k].set_yticks([])
        ax[k].set_ylabel(f'ID: {k}',fontsize=10)
        if k < model.F.shape[2]: 
            Z += model.F[:,:,k]
            ax[k].imshow(model.F[:,:,k],cmap='plasma')
    
    Z[Z<1e-2] = 0
    plasma = get_cmap("plasma")
    colors = ["#d3d3d3"] + [plasma(i) for i in range(plasma.N)]
    cmap = ListedColormap(colors)
    
    bounds = np.concatenate(([0, 1e-9], np.linspace(1e-9, 1, plasma.N)))
    norm = BoundaryNorm(bounds, cmap.N)
    
    print(np.unique(Z))
    fig,ax = plt.subplots(1,1)
    ax.imshow(Z,cmap=cmap,vmin=0,vmax=1) 
    ax.set_xticks([])
    ax.set_yticks([])

    if args.fout is not None:
        plt.savefig(args.fout)
        print(f"Wrote to {args.fout}")

    plt.show()

def viz_grid(args):
    """
    View initial grid
    """
    model = load_model(args)
    model.init()
    av.show_slice(model,0,3) 

    if args.fout is not None:
        plt.savefig(args.fout)
        print(f"Wrote to {args.fout}")

    plt.show() 

def viz_position_encoding(args):
    """
    View initial grid
    """
    model = load_model(args)
    model.init()
    av.show_slice(model,0,3) 
    
    Z = np.zeros((2,10*20))
    get_position_encoding(model,Z[0,:])
    
    P = Z[0,:].reshape(10,20)

    fig,ax = plt.subplots(1,1,figsize=(8,4))
    p = np.zeros(P.shape)
    p[P==1] = 1
    ax.imshow(p,cmap='binary') 
    ax.set_xticks([])
    ax.set_yticks([])
    
    p = np.zeros(P.shape)
    p[P==2] = 1
    fig,ax = plt.subplots(1,1,figsize=(8,4))
    ax.imshow(p,cmap='binary') 
    ax.set_xticks([])
    ax.set_yticks([])


    if args.fout is not None:
        plt.savefig(args.fout)
        print(f"Wrote to {args.fout}")

    plt.show() 


def viz_simulation(args):
    """
    View staged innervation
    """
    if args.fout is not None:
        if '.gif' in args.fout:
            args.fout = args.fout.replace('.gif','_%d.gif')
        else:
            print(f"Output file path must contain '.gif' extension")


    model = load_model(args)
    model.init()

    for i in range(model.get_number_of_innervation_level()):
        fout = args.fout 
        if fout is not None:
            fout = fout%i
            print(f"Writing to {fout}")

        animate_volume_slice(model,display_val=args.display_val,interval=20,
                frames=model.num_steps,repeat=False,fout=fout)
        
        model.next_tlevel()


def viz_to_render(args):
    """
    View staged innervation
    """
    model = load_model(args)
    model.init()
    model.run_stages()

    print(model.V.shape,model.M.shape)
    print(model.M[:,1])  
    
    m = model.M.shape[0]
    n = model.V.shape[2]
    
    ihalf = model.V.shape[0] // 2
    jhalf = model.V.shape[1] // 2
    X = np.zeros((m,n,2))
    for (i, j, k), val in np.ndenumerate(model.V):
        if val == 0: continue
        aid = val - 1
        X[aid,k,:] = [j-jhalf,i-ihalf]

    
    if args.fout is not None:
        np.savez(args.fout,X=X,I=model.M[:,1])
        print(f'Wrote to {args.fout}')


def viz_run(args):
    IDX = 2
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    module = importlib.import_module(f"abm.model.{cfg['model']['model']}.model")
    M = getattr(module,'Model') 
    num_models = cfg.getint('model','num_models')
    num_pioneers = cfg.getint('model','num_pioneers') 

    models = [M(cfg,model_id=i) for i in range(num_models)]
    hooks = [[HK()] for i in range(num_models)]
    hooks = batch_mp_vol(models,hooks)
    G = [h[0]() for h in hooks]
    H = process_graphs(G)
    

    pipe_cols,pipe_str = zip(*read.into_list(cfg['analyze']['pipe_file'],multi_dim=True))
    pipe = [getattr(abm.proc_data,p) for p in list(pipe_str)]
    max_d = cfg.getint('analyze','max_dendrogram_distance')  
    
    P = PipeObject(graph=H)
    tmp = [f(P,num_pioneers=num_pioneers,deg=num_models,max_d=max_d,weight='wnorm') for f in pipe]
    P.index_communities()

    models[IDX].init()
    models[IDX].run_stages()
    
    cmap = mpl.colors.ListedColormap(['k', '#d8d8d8', 'blue', 'cyan','yellow','magenta','green'])
    doms = np.zeros((models[IDX].M.shape[0],1))
    for (k,v) in P.comm_index.items(): doms[k-1] = v 
    doms += 1
    doms[:num_pioneers] = -1
    doms[0] = -1 
    models[IDX].add_meta(doms)
    display_val = 4
    slice_num = 100
    
    av.show_custom_slice(models[IDX],slice_num,display_val,cmap)
    if args.fout is not None:
        plt.savefig(args.fout,dpi=300)
        print(f"Wrote to {args.fout}")
    
    plt.show()

def viz_color_test(args):
    import matplotlib as mpl
    def map_grid_to_display(aid,display_val=0):
        return display_map[aid, display_val]
   
    cmap = mpl.colors.ListedColormap(['red', '#d8d8d8', 'blue', 'cyan','yellow','magenta','green'])
    

    model = load_model(args)
    model.init()
    model.run_stages() 
    
    doms = np.random.randint(low=1,high=6,size=(model.M.shape[0],1))
    doms[:model.num_pioneers+1] = -1
    
    model.add_meta(doms)
    display_val = 4
    slice_num = 100
    
    av.show_custom_slice(model,slice_num,display_val,cmap)

    
    """
    if args.fout is not None:
        plt.savefig(args.fout)
        print(f"Wrote to {args.fout}")
    """
    plt.show() 


def viz_molecular_specificity(args):
    model = load_model(args)
    model.init()
    fs = model.average_molecular_specificity()
    ps = model.average_pioneer_specificity()
    
    print(model.locality())
    print(model.average_locality())

    fig,_ax = plt.subplots(1,2,figsize=(10,5))
    ax = _ax.flatten() 
    ax[0].hist(model.molecular_specificity(),bins=20,range=(0,1))    
    ax[0].set_ylabel('# follower agents',fontsize=12) 
    ax[0].set_xlabel('molecular specificity',fontsize=12) 
    ax[0].tick_params(axis='x',labelsize=10) 
    ax[0].tick_params(axis='y',labelsize=10) 
    ax[0].text(0.05,0.95,'fs=%1.2f'%fs,transform=ax[0].transAxes,fontsize=10)
    
    ax[1].hist(model.pioneer_specificity(),bins=20,range=(0,1))    
    ax[1].set_ylabel('# pioneer agents',fontsize=12) 
    ax[1].set_xlabel('molecular specificity',fontsize=12) 
    ax[1].tick_params(axis='x',labelsize=10) 
    ax[1].tick_params(axis='y',labelsize=10) 
    ax[1].text(0.05,0.95,'ps=%1.2f'%ps,transform=ax[1].transAxes,fontsize=10)
    plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mode',
                        action = 'store',
                        choices = [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')

    parser.add_argument('config',
                action = 'store',
                help = 'Config file')
    
    parser.add_argument('-o',
                action = 'store',
                dest = 'fout',
                required = False,
                default = None,
                help = 'Path to save output file')
    
    parser.add_argument('--display_value',
                action = 'store',
                dest = 'display_val',
                required = False,
                default = 1,
                type = int,
                help = 'Meta data column number for display')



    args = parser.parse_args()
    eval(args.mode + '(args)')

