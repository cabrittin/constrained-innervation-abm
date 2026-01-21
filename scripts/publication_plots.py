"""                            
@name: publication_plots.py
@description:                  
Script for generating all analysis derived plots for paper

Relative paths to pertinent files are defined in 

configs/publication_config.ini

You will need to adjust the all the paths under [main] to reflect where you installed the cebraindev_abm data directory.

If any other relative paths have been changed, then the file must be adjusted accordingly. 


To run
------

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""
import os
import sys
from configparser import ConfigParser,ExtendedInterpolation
import argparse
from inspect import getmembers,isfunction
import matplotlib.pyplot as plt
import importlib

CONFIG = 'configs/publication_config.ini'

def _save_plot(args):
    if args.fout is not None: 
        fout = os.path.join(args.dout,args.fout) 
        plt.savefig(fout,dpi=300)
        print(f"Wrote to {fout}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
                dest = 'mode',
                action = 'store',
                help = 'Name of figure to plot')
    
    parser.add_argument('-c','--config',
                dest = 'config',
                action = 'store',
                default = CONFIG,
                required = False,
                help = 'Config file')
    
    args = parser.parse_args()
    #eval(args.mode + '(args)')
    
    cfg = ConfigParser(interpolation=ExtendedInterpolation())
    cfg.read(args.config)
    cfg['main']['home'] = os.path.expanduser("~")

    if args.mode == "index":
        for key in cfg.keys():
            if 'fig' in key: print(key)
    else: 
        fig_args = argparse.Namespace(**cfg[args.mode]) 
        module_call = importlib.import_module(fig_args.module) 
        data = getattr(module_call,cfg[args.mode]['func'])(fig_args)
        _save_plot(fig_args)
        if fig_args.show_plot == "True": plt.show()


 
