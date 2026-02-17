#!/bin/bash

###################################################################
#Description	: Parallizes the analyze function 
#Args           	:                                                                                           
#Author       	:Christopher Brittin
#Date           :2019-12-05                                                
#Email         	:"cabritin" <at> "gmail." "com"                                          
###################################################################

num_jobs=$1
din=$2
fout=$din/dataframe.csv
split_tmp=tmp/split.tmp
SPACE=" "

echo Parallelize $sweep_log across $num_jobs
echo Saving to $fout

tfiles=()

for (( i=1; i<= $num_jobs; i++)); do
	#tfiles=${tfiles}${split_tmp}.${i}${SPACE}
	tfiles+=("${split_tmp}.${i}")
done

tfiles1=$(IFS=$SPACE ; echo "${tfiles[*]}")
splits=$(python scripts/run_model.py split_sweep_log --dir=$din --num_splits=$num_jobs)

echo Log split: $splits

parallel -j$num_jobs --lb --plus "python scripts/run_model.py analyze --dir=$din --sweep_range={1} --fout={2}" ::: $splits :::+ $tfiles1

tfiles2=$(IFS=, ; echo "${tfiles[*]}")
python scripts/run_model.py concat_dataframes --merge=$tfiles2 --fout=$fout

echo Wrote to $fout
echo Cleaning up....

for i in "${tfiles[@]}"; do
	rm $i
done

echo Done!
