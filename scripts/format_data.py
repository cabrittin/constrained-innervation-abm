"""                            
@name: scripts.format_data.py
@description:                  
Scripts calls for formatting data

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


CONFIG = 'configs/format_data.ini'

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

    fargs = argparse.Namespace(**cfg[args.mode]) 
     
    #data = getattr(results,cfg[args.mode]['func'])(fig_args)
    module_call = importlib.import_module(fargs.module) 
    data = getattr(module_call,cfg[args.mode]['func'])(fargs)
    


 
