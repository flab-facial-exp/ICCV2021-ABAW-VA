# -*- coding: utf-8 -*-
import os
import sys
import shutil
from time import sleep
import random
import argparse
import numpy as np
import pickle
import glob
#####################

# print
print('-'*20)
print('combine_feature.py start\n')
print('This program load each feature txt files and combine them as one array. \n')
############

# argparse setting
a = argparse.ArgumentParser()
a.add_argument("--input_dir", default="train", help="target pickle file name")
a.add_argument("--mode", default=0,  help="operation_mode")
args = a.parse_args()

print('target input dir  is {}'.format(args.input_dir))

target_dir = args.input_dir

op_mode = args.mode
print('feature combine operation mode is {}'.format(op_mode))
############

dir_submit_pickle = './submit_pickle/'
if not os.path.exists(dir_submit_pickle):
    os.makedirs(dir_submit_pickle)

dir_img = target_dir + '/img_raw_feature_pca'  #pca img
dir_nlp = target_dir + '/nlp_rev_pca'  #pca confidence
dir_headpose = target_dir + '/headpose_rev'

### input_image ##############
for root, dirs, files in os.walk(dir_img):
    for dr in dirs:
        img_list = []
        nlp_list = []
        headpose_list = []

        img_list = glob.glob(os.path.join(root, dr + '/*.txt'))  #image              
        nlp_list = glob.glob(os.path.join(dir_nlp, dr + '/*.txt'))  #nlp
        headpose_list = glob.glob(os.path.join(dir_headpose, dr + '/*.txt'))  #headpose             

        img_list.sort()
        nlp_list.sort()
        headpose_list.sort()
        
        submit_combined_list = []

        for img_item, nlp_item, headpose_item in zip(img_list, nlp_list, headpose_list):
            temp_img_item = np.loadtxt(img_item)
            temp_nlp_item = np.loadtxt(nlp_item)
            temp_headpose_item = np.loadtxt(headpose_item)

            ### normalization ###
            la_norm_nlp = np.linalg.norm(temp_nlp_item, ord=2, keepdims=True) #L2_normalization
            if la_norm_nlp == 0:
                la_norm_nlp = 1
            norm_temp_nlp_item = temp_nlp_item/la_norm_nlp #normalization
            #####################

            ### changed ###
            if op_mode == 0 : #using all img and nlp  and vec feature
                combined_array = np.hstack((temp_img_item, norm_temp_nlp_item, temp_headpose_item))
            elif op_mode == 1 : #using only img feature
                combined_array = temp_img_item
            elif op_mode == 2 : #using only nlp feature
                combined_array = norm_temp_nlp_item
            elif op_mode == 3 : #using only headpose
                combined_array = temp_headpose_item
            elif op_mode == 4 : #using
                combined_array = np.hstack((temp_img_item, norm_temp_nlp_item))
            elif op_mode == 5 : #using
                combined_array = np.hstack((temp_img_item, temp_headpose_item))
            else : #using both img and nlp feature
                combined_array = np.hstack((temp_img_item, norm_temp_nlp_item, temp_headpose_item))
            ###############

            submit_combined_list.append(combined_array)
            ###############

        output_combined_feature   = str(dir_submit_pickle) + 'submit_test_' + str(dr) + '_combined.pickle'
        print('output_combined_feature file name is {0}'.format(output_combined_feature))

        feature_len = len(submit_combined_list)
        print('submit_combined_list len is {0}'.format(feature_len))

        feature_array = np.array(submit_combined_list)

        ###save feature_lists file ###
        print('Saving feature_array as {} file.'.format(output_combined_feature))
        with open(output_combined_feature, mode='wb') as f:
            pickle.dump(feature_array, f)
        ##############################

    break;
###############################################




print('combine_feature.py end\n')
print('-'*20)
