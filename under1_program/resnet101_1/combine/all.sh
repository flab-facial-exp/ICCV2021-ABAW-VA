#!/bin/bash

### check argument ###
if [[ $# != 4 ]]; then
	echo "bad argument"
	op_mode=0
	temp_mode=0
	gpu_num=0
else
	echo "Your setting argument is below."
	op_mode=$1
	temp_mode=$2
	gpu_num=$3
	dataset=$4
	echo "op_mode is ${op_mode}."
	echo "temp_mode is ${temp_mode}."
	echo "gpu_num is ${gpu_num}."
	echo "dataset is ${dataset}."
fi
##############################


if [[ $temp_mode == 0 ]]; then
	### feature extraction from image using ResNet50 + SENet ###
	# only one time is ok.
	if [ -d ./using_dataset/test/img_raw_feature/ ]; then
		echo "skip imagae feature extraction due to the existence of test/img_raw_feature."
	else
		./pre_img_feature.sh $gpu_num img_data #$1 is cuda_num $2 is target_part
	fi
	#############################################################
	if [[ $dataset == 1 ]];then
		pca_seed_img=100
		pca_seed_nlp=200
	elif [[ $dataset == 2 ]];then
		pca_seed_img=2061
		pca_seed_nlp=2061
	fi

	
	if [ -d ./using_dataset/test/nlp_rev_pca/ ]; then
		echo "skip pca process due to the existence of test/nlp_rev_pca."
	else
		./pre_dim_reduce.sh 300 img_raw_feature $pca_seed_img
		./pre_dim_reduce.sh 300 nlp_rev $pca_seed_nlp #nlp feature
	fi
	#./pre_dim_reduce.sh 300 nlp $pca_seed_nlp --vec_fea #vector feature


	### dimension reduction using afinn ###
	#./pre_afinn.sh train
	#./pre_afinn.sh test
	######################
fi

### feature combine ###
if [ -f ../classifier/test_combined.pickle ]; then
	echo "skip combine process due to the existence of test/nlp_rev_pca."
else
	./pre_combined.sh $op_mode
fi
#######################

#######################
