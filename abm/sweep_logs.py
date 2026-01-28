"""
@name: sweep_logs.py                      
@description:                  
    Module for building log files for batch runs


@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import os
from configparser import ConfigParser,ExtendedInterpolation
import random
import csv
from collections import defaultdict,namedtuple


def format_sweep_range(sweep_str=None):
    """
    Formats the sweep range, the range to do the sweep of the parameter groups

    Input:
    ------
    sweep_str: str, optional (default None)
      String to specify the paramter group range 

      ':' token is used for intervals, e.g. 3:6 = 3,4,5

      ',' token used to seperate parameter groups, e.g. 3,10 = 3,10

      Tokens can be combined, e.g. 0,3:6,22 = 0,3,4,5,22
    
    Returns:
    --------
    list of ints for the parameter groups
    """
    pg = None 
    if isinstance(sweep_str,str):
        pg = []
        for s in sweep_str.split(','):
            if ':' in s:
                _s = s.split(':')
                pg += list(range(int(_s[0]),int(_s[1])))
            else:
                pg.append(int(s))
    return pg 

def list_to_sweep_range(lst):
    """
    Converts list of ints to a sweep range
    """
    ## Ensure sorted unique elements
    lst = sorted(list(set(lst)))
    
    ## Link concecutive ints
    search = {}
    for l in lst:
        linked = False
        for (k,v) in search.items():
            if l - v[-1]  == 1:
                search[k].append(l)
                linked = True
        if not linked:
            search[l] = [l]
    
    ## Format as string
    s = []
    for (k,v) in sorted(search.items()):
        if len(v) > 1:
            s.append(f'{v[0]}:{v[-1]+1}')
        else:
            s.append(str(v[0]))
    return ','.join(s)

def _iter_sweep(fname,sweep_range=None):
    if isinstance(sweep_range,list):
        sr = sweep_range[:]
    else: 
        sr = format_sweep_range(sweep_range)
    with open(fname,'r') as sfile:
        reader = csv.reader(sfile, delimiter=',')
        next(reader) #Skip header
        for row in reader:
            pg = int(row[0])
            if sr is not None and pg not in sr: continue 
            yield row

def get_header(fname):
    with open(fname,'r') as sfile:
        reader = csv.DictReader(sfile,delimiter=',')
        headers = reader.fieldnames
        return headers
        
def iter_sweep(fname,sweep_range=None):
    for row in _iter_sweep(fname,sweep_range=sweep_range):
        _row = [int(x) if x.is_integer() else x for x in map(float,row[:-1])] + [row[-1]]
        yield _row

def iter_sweep_by_pairs(fname,idx_pairs):
    with open(fname,'r') as sfile:
        reader = list(csv.reader(sfile, delimiter=','))[1:]
        for i,j in idx_pairs:
            yield i,j,reader[i],reader[j]

def sweep_length(fname,sweep_range=None):
    count = 0
    for row in _iter_sweep(fname,sweep_range=sweep_range): count += 1
    return count

def num_param_groups(fname):
    with open(fname, 'rb') as f:
        try:  # catch OSError in case of a one line file 
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b'\n':
                f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        last_line = f.readline().decode()
        last_line = last_line.split(',')
    return int(last_line[0])

def sweep_tree(fname,sweep_range=None):
    tree = defaultdict(list)
    for row in _iter_sweep(fname,sweep_range=sweep_range):
        _row = [int(x) if x.is_integer() else x for x in map(float,row[:-1])] + [row[-1]]
        tree[_row[0]].append(_row)
    return tree

def index_runs(fname,sweep_range=None):
    idx = 0 
    runs = []
    for row in _iter_sweep(fname,sweep_range=sweep_range):
        #_row = [idx] + list(map(int,row[:-1])) + [row[-1]]
        _row = [idx] + [int(x) if x.is_integer() else x for x in map(float,row[:-1])] + [row[-1]]
        runs.append(_row)
        idx += 1
    return runs

def pull_sweep(fname,cols,to_pull):
    with open(fname,'r') as sfile:
        reader = csv.DictReader(sfile,delimiter=',')
        pg_pull = [] 
        for line in reader:
            tmp = [int(line[c]) for c in cols]
            if tmp in to_pull: pg_pull.append(int(line['parameter_group']))
        pg_pull = list(map(str,sorted(list(set(pg_pull)))))
        return ','.join(pg_pull)


