#!/bin/bash

#$1:dataset
#$2:random_seed
#$3:op_mode
#$4:gpu_num

#mkdir temp_$1_$2_$3

#cp -r ./combine/test_img_data ./temp_$1_$2_$3/test_img_data

### classifier ###
echo "classifier start."
cd ./classifier_dataset$1_seed$2_comb$3
./all.sh $1 $4 $2 $3 2>&1 | tee ../result_dataset$1/result_seed_$2_mode_$3.txt
#mv test_result.txt test_result_$2_$3.txt
#mv test_result_*.txt test_result_*_$2_$3.txt
#echo test_result_*.txt | xargs -n 1 mv -v test_result_*_$2_$3.txt 
cp test_result_*.txt ../result_dataset$1/
cp submit_test_result_*.txt ../result_dataset$1/
cp classifier_model_valence_catboost.sav ../submit_make/
cp classifier_model_arousal_catboost.sav ../submit_make/
cp classifier_model_valence_svr.sav ../submit_make/
cp classifier_model_arousal_svr.sav ../submit_make/
#echo ../result_dataset$1/ | xargs -n 1 cp -v test_result_*_$2_$3.txt
cd ../
#python detect_wrong_img.py --input_file ./result_dataset$1/input_file_list_$2.txt --test_result ./result_dataset$1/test_result_$2_$3.txt --target_data $2 --mode_data $3 --dataset $1
echo "classifier end."

echo "classifier directory delete start."
rm -rf ./classifier_dataset$1_seed$2_comb$3
#rm -r ./temp_$1_$2_$3
echo "classifier directory delete end."

##################
