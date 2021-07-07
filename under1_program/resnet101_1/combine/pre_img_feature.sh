#!/bin/bash

#################################
#1:Feature extracting from image#
#################################
if [[ $# != 2 ]]; then
	echo "bad argument."
	target_part=img_data
	cuda_num=0
	echo "Set cuda_num is 0"
	echo "Set target_part is img_data"
else
	cuda_num=$1
	target_part=$2
	echo "Your setting arguments are below."
	echo "Using cuda_num is ${cuda_num}."
	echo "target_part is ${target_part}."
fi

#for train data image #
#CUDA_VISIBLE_DEVICES=$cuda_num python img_feature_ext.py --input_dir train_$target_part --transfer_model transfer_resnet_$target_part.model
CUDA_VISIBLE_DEVICES=$cuda_num python img_feature_ext.py --input_dir ./using_dataset/train/img --transfer_model transfer_resnet_$target_part.model
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python img_feature_ext.py --input_dir ./using_dataset/train/img --transfer_model transfer_resnet_$target_part.model
#mv class_label.pickle class_label_train.pickle
#######################
#for test data image #
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python img_feature_ext.py --input_dir ./using_dataset/test/img --transfer_model transfer_resnet_$target_part.model
CUDA_VISIBLE_DEVICES=$cuda_num python img_feature_ext.py --input_dir ./using_dataset/test/img --transfer_model transfer_resnet_$target_part.model
#mv class_label.pickle class_label_test.pickle
#######################

CUDA_VISIBLE_DEVICES=$cuda_num python img_feature_ext.py --input_dir ./using_dataset/submit_test/img --transfer_model transfer_resnet_$target_part.model

#mv class_label_train.pickle train_class_label.pickle
#mv class_label_test.pickle test_class_label.pickle
#################################
#################################
#################################
