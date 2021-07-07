#!/bin/bash


#if [[ $# != 4 ]]; then
#	echo "bad argument."
#	pca_dim=300
#	target_dir=nlp
#	random_seed=0
#	vec_flag=''
#	echo "Set pca dimension is ${pca_dim}."
#	echo "target_dir is ${target_dir}."
#	echo "random_seed is ${random_seed}."
#	echo "vec_flag is ${vec_flag}."
#
#else
#	echo "Your setting arguments are below."
#	pca_dim=$1
#	target_dir=$2
#	random_seed=$3
#	vec_flag=$4
#	echo "pca dimension is ${pca_dim}."
#	echo "target_dir is ${target_dir}."
#	echo "random_seed is ${random_seed}."
#	echo "vec_flag is ${vec_flag}."
#
#fi

pca_dim=$1
target_dir=$2
random_seed=$3
vec_flag=$4
echo "pca dimension is ${pca_dim}."
echo "target_dir is ${target_dir}."
echo "random_seed is ${random_seed}."
echo "vec_flag is ${vec_flag}."

echo "./pre_dim_reduce.sh start."
python dim_reduce.py --pca_dim $pca_dim --target_dir $target_dir --random_seed $random_seed $vec_flag
echo "./pre_dim_reduce.sh end."
