"""
@name: batchrunner.py
@description:
    Modules for running models in batch
    
@author: Christopher Brittin
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05
"""

import multiprocessing_on_dill as mp
from tqdm import tqdm
import numpy as np

from abm.proc_data import PipeObject

def batch(num_steps,models,hooks):
    num_models = len(models)
    for (i,m) in tqdm(enumerate(models),total=num_models,desc='Models processed'):
        run(num_steps,m,hooks[i])
 
def run(model,hooks):
    model.init()
    for h in hooks: h.pre_action(model)
    model.run_stages() 
    for h in hooks: h.post_action(model)

def batch_mp_vol(models,hooks):
    """
    Note: I tested Pool vs Process. Process is faster
    """
    manager = mp.Manager()
    hook_dict = manager.dict()
    num_models = len(models)
    procs = [] 
    for idx in range(num_models):
        proc = mp.Process(target=run_mp_vol, args=(idx,models[idx],hooks[idx],hook_dict))
        procs.append(proc)
        proc.start()
    for proc in procs: proc.join()
    return [hook_dict[idx] for idx in range(num_models)]

def run_mp_vol(idx,model,hooks,hook_dict):
    model.init() 
    for h in hooks: h.pre_action(model) 
    model.run_stages()
    for h in hooks: h.post_action(model)
    hook_dict[idx] = hooks

def batch_mp_pipe(pg,pipe,din,**kwargs):
    manager = mp.Manager()
    rlist = manager.list(range(len(pg)))
    procs = []
    for (idx,p) in enumerate(pg):
        proc = mp.Process(target=run_mp_pipe, args=(idx,p,pipe,din,rlist),kwargs=kwargs)
        procs.append(proc)
        proc.start()
    for proc in procs: proc.join()
    return np.array(rlist)

def run_mp_pipe(idx,p,pipe,din,rlist,**kwargs):
    import warnings 
    warnings.filterwarnings('ignore')
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        fin = din + p[-1]
        P = PipeObject(fin,label=f'pipe_{p[0]}_{idx}')
        tmp = [f(P,label=idx,num_pioneers=p[3],**kwargs) for f in pipe]
        rlist[idx] = tmp


