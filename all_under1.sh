#!/bin/bash


echo "all_under1.sh start."

### check argument ###
if [[ $# != 1 ]]; then
	echo "Warning. bad argument!"
	echo "If you would like to assing using gpu number, please specify it as the first argument."
	gpu_num=0
	echo "GPU no ${gpu_num} will be used."
	echo ""
	read -p "ok? (y/n): " yn
	case "$yn" in 
		[yY]* ) ;; 
		*) echo "abort." ; exit ;; 
	esac
else
	echo "Your setting argument is below."
	gpu_num=$1
	echo "GPU no ${gpu_num} will be used."
fi
##############################



abs_path=$(pwd)

echo "absolute_path_is ${abs_path}."

cd ./under1_program

### build docker images ###
echo "-----------------------"
echo "build docker file start."
./all_build.sh
echo "build docker file end."
echo "-----------------------"
###########################


### Run docker images ###
echo "-----------------------"
echo "run docker file start."
./all_run.sh $gpu_num $abs_path
echo "run docker file end."
echo "-----------------------"
###########################

echo "all_resnet101.sh end."
