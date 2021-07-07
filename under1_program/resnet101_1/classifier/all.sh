#!/bin/bash

### get available gpu number at random, automatically ###
### gpu will be used if you use catboost as classifier ###
### svm wii not use gpu.

dataset=$1
gpu_num=$2
#dataset=2
#gpu_num=0

TIME_A=`date +%s`   #
echo "classifier start."
./pre_classifier.sh 0 our_method $gpu_num $dataset $3 $4
echo "classifier end."
TIME_B=`date +%s`   #
PT=`expr ${TIME_B} - ${TIME_A}`
H=`expr ${PT} / 3600`
PT=`expr ${PT} % 3600`
M=`expr ${PT} / 60`
S=`expr ${PT} % 60`
echo "classifier processing time"
echo "${H}:${M}:${S}"
