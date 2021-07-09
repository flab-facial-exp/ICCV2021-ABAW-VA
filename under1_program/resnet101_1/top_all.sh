#!/bin/bash


# target dataset #
#1:using
#2:not use

echo "shell script: top_all.sh start"
echo "shell script: -------------------------------------------"
echo "shell script: usage:top_all.sh dataset."


gpu_num=0
########################################

#for i in `seq 1 5`
#for i in `seq 1 1`
for i in `seq 1 1`
do
	echo "shell script: dataset or top_mode is ${i}."
	./pre_top_all.sh $i $gpu_num 2>&1 | tee result_top_dataset${i}.txt
done

echo "shell script: top_all.sh end"
