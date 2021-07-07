#!/bin/bash

dataset=$1
gpu_num=$2
#
#dataset=1
#gpu_num=0

### setting parameter for transfer learning ###
if [[ $dataset == 1 ]]; then #1:shuffle
	epoch_num=2
	#batch_size=32
	batch_size=32
	epoch_train_step=60
	epoch_valid_step=20
	learning_rate=1e-8
elif [[ $dataset == 2 ]]; then #2:fix
	epoch_num=2
	batch_size=32
	epoch_train_step=6
	epoch_valid_step=6
	learning_rate=1e-8
fi
#########################


# making transfer_resnet_img_data.model #
#cp -r ./train train_ft
#cp -r ./test valid_ft
#cp -r ./train_img train_ft
#cp -r ./test_img valid_ft

./pre_transfer_learning.sh $gpu_num $epoch_num $batch_size $epoch_train_step $epoch_valid_step $learning_rate transfer_resnet_img_data.model #

#rm -r train_ft
#rm -r valid_ft

#mv train train_img_data
#mv test test_img_data
#mv train_img train_img_data
#mv test_img test_img_data
#############
