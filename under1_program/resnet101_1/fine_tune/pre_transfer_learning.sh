#!/bin/bash

echo "./pre_transfer_learning.sh start."
#CUDA_VISIBLE_DEVICES=$1 python transfer_learning.py --nb_epoch $2 --batch_size 32 --steps_per_epoch 580 --validation_steps 102 --retrain_epoch 0 --pre_epoch 0 --input_model resnet50 --plot --learning_rate $3 --output_model_file $4
#CUDA_VISIBLE_DEVICES=$1 python transfer_learning.py --nb_epoch $2 --batch_size $3 --steps_per_epoch $4 --validation_steps $5 --retrain_epoch 0 --pre_epoch 0 --input_model resnet101 --plot --learning_rate $6 --output_model_file $7
#CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python transfer_learning.py --nb_epoch $2 --batch_size $3 --steps_per_epoch $4 --validation_steps $5 --retrain_epoch 0 --pre_epoch 0 --input_model resnet101 --plot --learning_rate $6 --output_model_file $7
CUDA_VISIBLE_DEVICES=$1 python transfer_learning.py --nb_epoch $2 --batch_size $3 --steps_per_epoch $4 --validation_steps $5 --retrain_epoch 0 --pre_epoch 0 --input_model resnet101 --plot --learning_rate $6 --output_model_file $7
#CUDA_VISIBLE_DEVICES=$1 python transfer_learning.py --nb_epoch $2 --batch_size $3 --steps_per_epoch $4 --validation_steps $5 --retrain_epoch 0 --pre_epoch 0 --input_model densenet121 --plot --learning_rate $6 --output_model_file $7
echo "./pre_transfer_learning.sh end."
