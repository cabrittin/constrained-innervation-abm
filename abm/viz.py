"""
@name: viz.py
@description:
    Module for data vizualization

@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib import cm

#import abm.proc_data as pcd #CAN we remove this?

def get_current_display_slice(model,display_func,display_val=0):
    return display_func(model.V[:,:,model.cur_step-1],display_val=display_val)

def get_display_slice(model,slice_num,display_func,display_val=0):
    return display_func(model.V[:,:,slice_num],display_val=display_val)

def get_colormap(model,display_val=0):
    #Set colormap
    #viridis = cm.get_cmap('viridis', model.num_agents)
    viridis = cm.viridis
    _cmap = viridis(np.linspace(0, 1, (10*(10+model.num_agents))))
    cmap = ListedColormap(_cmap)
    
    ##Add pioneer to colormap
    if model.num_pioneers > 0 and display_val > 1: 
        red = np.array([1,0,0,1])
        _cmap[0] = red
        cmap = ListedColormap(_cmap)
    return cmap

def build_display_map(model):
    M = model.get_meta()
    #display_map = np.zeros((M.shape[0],4)) 
    display_map = np.copy(M) 
    display_map[:,0] = 10*(1+M[:,0])
    #display_map[:model.num_pioneers,1] = -10
    #display_map[model.num_pioneers:,1] = 10*(1+M[model.num_pioneers:,1])
    display_map[:,1] = 10*(10+M[:,2])
    
    display_map[:,2] = display_map[:,0]
    display_map[:model.num_pioneers,2] = -1
    
    display_map[:,3] = display_map[:,1]
    display_map[:model.num_pioneers,3] = -1
 
    display_map = np.insert(display_map,0,np.zeros(display_map.shape[1]),axis=0)
    return display_map

def show_slice(model,slice_num,display_val,ax=None):
    def map_grid_to_display(aid,display_val=0):
        return display_map[aid, display_val]
    
    cmap = get_colormap(model,display_val=display_val)
    display_map = build_display_map(model)
    display_func = np.vectorize(map_grid_to_display) 
    
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(8,4))
    
    """
    cmap = ListedColormap([
    "#d3d3d3",  # 0
    "#ffff00",  # 1
    "#eb5feb"   # 2
    ])
    """
    Z = get_display_slice(model,slice_num,map_grid_to_display,display_val=display_val)
    print(np.unique(model.V.shape))
    print(np.unique(model.V[:,:,:]))
    print(np.unique(Z))
    #ax.imshow(model.V[:,:,0],cmap=cmap)
    ax.imshow(Z,cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])

def show_custom_slice(model,slice_num,display_val,cmap,ax=None):
    def map_grid_to_display(aid,display_val=0):
        return display_map[aid, display_val]
 
    display_map = build_display_map(model)
    print(display_map.shape) 
    if ax is None: fig,ax = plt.subplots(1,1,figsize=(8,4))
    
    Z = get_display_slice(model,slice_num,map_grid_to_display,display_val=display_val)
    ax.imshow(Z,cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])


def build_response_map(model):
    M = model.get_meta()
    display_map = np.zeros((M.shape[0],4)) 
    display_map[:,0] = 10*(1+M[:,0])
    #display_map[:model.num_pioneers,1] = -10
    #display_map[model.num_pioneers:,1] = 10*(1+M[model.num_pioneers:,1])
    display_map[:,1] = 10*(10+M[:,1])
    
    display_map[:,2] = display_map[:,0]
    display_map[:model.num_pioneers,2] = -1
    
    display_map[:,3] = display_map[:,1]
    display_map[:model.num_pioneers,3] = -1
 
    display_map = np.insert(display_map,0,np.zeros(display_map.shape[1]),axis=0)
    return display_map


def show_responsiveness(model,slice_num,aid,ax=None):
    def map_grid_to_display(aid,display_val=0):
        return display_map[aid, display_val]
    
    viridis = cm.get_cmap('viridis', model.num_agents)
    _cmap = viridis(np.linspace(0, 1, model.num_pioneers+2))
    red = np.array([1,0,0,1])
    _cmap[0] = red
    cmap = ListedColormap(_cmap)
    
    display_map = np.zeros((model.num_agents+1,1))
    display_map[1:model.num_pioneers+1,0] = 1+(30*model.R[aid,:])
    display_map[aid+1,0] = -10
    display_func = np.vectorize(map_grid_to_display) 
    

    if ax is None:
        fig,ax = plt.subplots(1,1,figsize=(8,4))
    
    Z = get_display_slice(model,slice_num,display_func)
    ax.imshow(Z,cmap=cmap)
    ax.set_xticks([])
    ax.set_yticks([])


def animate_volume_slice(model,display_val=0,fout=None,**kwargs):
    def map_grid_to_display(aid,display_val=0):
        return display_map[aid, display_val]
    
    cmap = get_colormap(model,display_val=display_val)
    display_map = build_display_map(model)
    display_func = np.vectorize(map_grid_to_display) 
    
    fig,ax = plt.subplots(1,1,figsize=(8,4))
    im = ax.imshow(get_current_display_slice(model,display_func,display_val=display_val),
                    cmap=cmap,vmin=display_map.min(0)[display_val],
                    vmax=display_map.max(0)[display_val],animated=True)
    ax.set_title(f"Time step: 0")
    ax.set_xticks([])
    ax.set_yticks([])
    idx=0
    anim = animation.FuncAnimation(fig, update_vol_slice, fargs=(idx,model,ax,im,display_val,display_func),
            blit=False,**kwargs)
    
    if fout is not None:
        writervideo = animation.FFMpegWriter(fps=20)
        anim.save(fout, writer=writervideo)
    else: 
        plt.show()

def update_vol_slice(*args):
    args[2].step()
    args[4].set_array(get_current_display_slice(args[2],args[6],display_val=args[5]))
    args[3].set_title(f"Growth step: {args[0]}")
    return args[4],


def animate_volume(model,**kwargs):
    """
    3D animator
    """
    display_val = 1 
    num_steps = model.num_steps

    def update_lines(num, data, lines):
        for k,v in data.items():
            # NOTE: there is no .set_data() for 3 dim data...
            lines[k-1].set_data(v[1:num, :2].T)
            lines[k-1].set_3d_properties(v[1:num, 2])
        return lines

    # Attaching 3D axis to the figure
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(projection="3d")
    
    data = dict([(a.tid,np.zeros((model.V.shape[2],3))) for a,t in model.traverse_agents()])
    #data = {} 
    model.run_stage()
    model.run_stage()
    zscale = 5
    xscale = 2
    xmax,ymax,zmax = model.V.shape
    for k in range(model.V.shape[2]):
        for tid,d in data.items():
            idx = np.where(model.V[:,:,k]==tid)
            if len(idx[0]) > 0: 
                i = idx[0][0]
                j = idx[1][0]
                data[tid][k,:] = [float(j)/ymax,float(k)/zmax*zscale,float(i)/xmax]
            else:
                tmp = data[tid][k-1,:][:]
                tmp[1] = float(k)/zmax*zscale
                data[tid][k,:] = tmp
    #print(data) 
    # Create lines initially without data
    lines = [ax.plot([], [], [])[0] for _ in data]
    
    # Setting the axes properties
    ax.set(xlim3d=(0, 1), xlabel='X')
    ax.set(ylim3d=(0, zscale), ylabel='Z')
    ax.set(zlim3d=(0, 1), zlabel='Y') 
    ax.set_box_aspect((xscale,zscale,1)) 
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Creating the Animation object
    anim = animation.FuncAnimation(
            fig, update_lines, num_steps, fargs=(data, lines), interval=10,repeat=False)
    
    #fout = 'data/num_targets_k1.gif'
    fout = 'data/num_targets_k0.gif'
    png = 'data/num_targers_k0.png'

    print(animation)
    #plt.show()
    #writervideo = animation.FFMpegWriter(fps=10)
    #anim.save(fout, writer=writervideo)
    print('here') 
    #anim.save(fout, writer='imagemagick', fps=60)
    #plt.savefig(png) 
    #plt.close()

## Can we remove this in order to eliminate pcd dependency?
#def graph_variance(M):
#    data = pcd.variance_dist(M)
#    fig,ax = plt.subplots(1,1,figsize=(4,2))
#    plot_bar(data)

def plot_bar(ax,data):
    x = np.arange(len(data)) + 1
    ax.bar(x,data,color="0.5")
    ax.set_ylabel('Fraction',fontsize=10)
    ax.set_ylim([0,1]) 
    ax.set_xticks(x) 
    ax.tick_params(axis='x',labelsize=8)
    ax.tick_params(axis='y',labelsize=8)
    plt.tight_layout()

def run_variance(_ax,H):
    ax = _ax.flatten()
    for (i,h) in enumerate(H):
        h.variance_dist() 
        plot_bar(ax[i],h.vdist)

def degree_dist(G,key=None):
    #data = [w for (u,v,w) in G.edges(data='weight')]
    data = [G.degree(n) for n in G.nodes()]
    fig,ax = plt.subplots(1,1,figsize=(3,3))
    ax.hist(data,bins=200,histtype='step',cumulative=-1,density=True)
    ax.set_xlim([0,max(data)+5]) 

def plot_model_comparison(ax,x,Z,label,df,col):
    mu = Z.mean(0) 
    std = Z.std(0)
    sns.ecdfplot(ax=ax,data=df,x=col)
    ax.plot(x,mu,'k--',linewidth=1)
    ax.fill_between(x,mu+2*std,mu-2*std,color='r',alpha=0.1)
    ax.fill_between(x,mu+std,mu-std,color='r',alpha=0.2)
    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_xlabel(label,fontsize=6)
    ax.tick_params(axis='x',labelsize=6)
    ax.tick_params(axis='y',labelsize=6)
    ax.set_ylabel('Proportion',fontsize=8)

