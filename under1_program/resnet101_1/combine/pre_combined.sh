#!/bin/bash

echo "./pre_combined.sh start"
# using pca feature #
python combine_feature.py --input_dir ./using_dataset/train --img_pca_tf --nlp_pca_tf --mode $1
python combine_feature.py --input_dir ./using_dataset/test --img_pca_tf --nlp_pca_tf  --mode $1
#python combine_feature.py --input_dir ./using_dataset/submit_test --img_pca_tf --nlp_pca_tf  --mode $1
#####################

# using raw feature, no pca #
#python combine_feature.py --input_dir train --mode $op_mode
#python combine_feature.py --input_dir test  --mode $op_mode
#############################
echo "./pre_combined.sh end"
