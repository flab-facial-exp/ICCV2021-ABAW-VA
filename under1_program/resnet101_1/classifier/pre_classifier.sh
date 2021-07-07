#!/bin/bash

CUDA_VISIBLE_DEVICES=$3 python classifier.py --pca $1 --method $2 --dataset $4 --seed $5 --mode $6
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python classifier.py --pca $1 --method $2 --dataset $4 --seed $5 --mode $6
