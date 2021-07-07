#!/bin/bash


# target dataset #
#1:abstract #train and test data are not assgned but decided at random, 5 fold cross validation
#2:artphoto #train and test data are not assigned but decided at random, 5 fold cross validation
#3:emotionroi #train and test data are assigned.
#4:FI_dataset #train and test data are not assigned but decided at random.
#5:twitter_I #train and test data are not assigned but decided at random.
#6:twitter_II #train and test data are asiggned. 5 fold cross validation

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
