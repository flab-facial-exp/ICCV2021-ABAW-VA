#!/bin/bash

echo "./pre_combined.sh start"
# using pca feature #
python combine_feature.py --input_dir ./using_dataset/submit_test --mode $1
#####################

echo "./pre_combined.sh end"
