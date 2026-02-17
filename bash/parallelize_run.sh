#!/bin/bash

###################################################################
#Description	: Parallizes the run function 
#Args           	:                                                                                           
#Author       	:Christopher Brittin
#Date           :2019-12-05                                                
#Email         	:"cabritin" <at> "gmail." "com"                                          
###################################################################

num_jobs=$1
din=$2

echo Parallelize $din across $num_jobs procs

splits=$(python scripts/run_model.py split_sweep_log --dir $din --num_splits=$num_jobs)

echo Log split: $splits

parallel -j$num_jobs --lb --plus "python scripts/run_model.py run --dir $din --sweep_range={1}" ::: $splits

echo Done!
