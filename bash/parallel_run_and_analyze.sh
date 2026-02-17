#!/bin/bash

###################################################################
#Script Name	:                                                                                           
#Description	: 
#Args           	:                                                                                           
#Author       	:Christopher Brittin
#Date           :2019-12-05                                                
#Email         	:"cabritin" <at> "gmail." "com"                                          
###################################################################

num_jobs=$1
model=$2

echo Run model....
bash/parallelize_run.sh $num_jobs $model

echo Run analysis...
bash/parallelize_analyze.sh $num_jobs $model

