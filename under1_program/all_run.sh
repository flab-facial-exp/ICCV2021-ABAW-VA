#!/bin/bash

gpu_num=$1
abs_path=$2

docker run --rm --runtime=nvidia --shm-size=32g -e NVIDIA_VISIBLE_DEVICES=$gpu_num \
-v /home/ubuntu/iccv2021_abaw/abaw_dataset/using_dataset:/home/ubuntu/iccv2021_abaw/abaw_dataset/using_dataset \
-v $abs_path/keras_models:/root/.keras \
-v $abs_path/under1_program/resnet101_1:/work \
-v $abs_path/docker_store_directory/under1:/root/docker_common_directory/under1 -it tyamamoto_under1_resnet101:1 /bin/bash
