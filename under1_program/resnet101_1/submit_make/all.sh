#!/bin/bash

### check argument ###
if [[ $# != 2 ]]; then
	echo "bad argument"
	op_mode=0
	seed=0
else
	echo "Your setting argument is below."
	op_mode=$1
	seed=$2
	echo "op_mode is ${op_mode}."
	echo "seed is ${seed}."
fi
##############################


### feature combine ###
./pre_combined.sh $op_mode
#######################


### submit_make ###
python submit_make.py --seed $seed --mode $op_mode --classifier catboost
#python submit_make.py --seed $seed --mode $op_mode --classifier svr
python submit_make.py --seed $seed --mode $op_mode --classifier linear
#######################
