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
a.add_argument("--img_pca_tf", action="store_true", help="using img pca feature or not")
a.add_argument("--nlp_pca_tf", action="store_true", help="using nlp pca feature or not")
a.add_argument("--vec_pca_tf", action="store_true", help="using vec pca feature or not")
a.add_argument("--mode", default=0, help="feature combination mode", type=int)
args = a.parse_args()

print('target input dir  is {}'.format(args.input_dir))

target_dir = args.input_dir

op_mode = args.mode
print('feature combine operation mode is {}'.format(op_mode))
############

if args.img_pca_tf:  
    print('Using img pca features.')
    dir_img = target_dir + '/img_raw_feature_pca'  #pca img
else:
    print('Using img raw features.')
    dir_img = target_dir + '/img_raw_feature'  #img feature

if args.nlp_pca_tf:
    print('Using nlp pca features.')
    dir_nlp = target_dir + '/nlp_rev_pca'  #pca confidence
    #dir_vec = target_dir + '_nlp_raw_feature_pca'  #pca confidence
else:
    print('Using nlp raw features.')
    dir_nlp = target_dir + '/nlp_rev'  #confidence
    #dir_vec = target_dir + '_nlp_raw_feature'  #pca confidence

#dir_afinn = target_dir + '_nlp_raw_feature_afinn'  #afinn confidence

dir_headpose = target_dir + '/headpose_rev'

img_list = []
nlp_list = []
headpose_list = []
#vec_list = []

### input_image ##############
for root, dirs, files in os.walk(dir_img):
    for dr in dirs:
        dir_img_list = glob.glob(os.path.join(root, dr + '/*.txt'))  #input image              
        dir_img_list.sort()
        img_list += dir_img_list

        dir_nlp_list = glob.glob(os.path.join(dir_nl, dr + '/*.txt')) #nlp
        dir_nlp_list.sort()
        nlp_list += dir_nlp_list

        dir_headpose_list = glob.glob(os.path.join(dir_headpose, dr + '/*.txt'))  #input  headpose
        dir_headpose_list.sort()
        headpose_list += dir_headpose_list.sort()
    break;
###############################################

'''
### input_nlp ##############
for root, dirs, files in os.walk(dir_nlp):
    for dr in dirs:
        dir_nlp_list = glob.glob(os.path.join(root, dr + '/*.txt')) #nlp
        dir_nlp_list.sort()
        nlp_list += dir_nlp_list
    break;    
########################


### input_headpose ##############
for root, dirs, files in os.walk(dir_headpose):
    for dr in dirs:
        dir_headpose_list = glob.glob(os.path.join(root, dr + '/*.txt'))  #input  headpose
        dir_headpose_list.sort()
        headpose_list += dir_headpose_list.sort()
    break;
###############################################
'''

### input_vec ##############
#for root, dirs, files in os.walk(dir_vec):
#    for dr in dirs:
#        if 'pca' in dir_vec:
#            vec_list += glob.glob(os.path.join(root, dr + '/*_vector_sum_pca.txt')) #nlp
#        else:
#            vec_list += glob.glob(os.path.join(root, dr + '/*_vector_sum.txt')) #nlp
#    break;    
########################

### input_vec ##############
#for root, dirs, files in os.walk(dir_afinn):
#    for dr in dirs:
#        afinn_list += glob.glob(os.path.join(root, dr + '/*_afinn.txt')) #afinn
#    break;    
########################

#img_list.sort()
#nlp_list.sort()
#headpose_list.sort()
#vec_list.sort()
#afinn_list.sort()

train_combined_list = []

loop_cnt = 0

def min_max(x, axis=None):
    min = x.min(axis=axis, keepdims=True)
    max = x.max(axis=axis, keepdims=True)
    result = (x-min)/(max-min)
    return result

#for img_item, nlp_item, vec_item in zip(img_list, nlp_list, vec_list):
#for img_item, nlp_item in zip(img_list, nlp_list):
for img_item, nlp_item, headpose_item in zip(img_list, nlp_list, headpose_list):
#for img_item, nlp_item, vec_item, afinn_item in zip(img_list, nlp_list, vec_list, afinn_list):

    temp_img_item = np.loadtxt(img_item)
    temp_nlp_item = np.loadtxt(nlp_item)
    temp_headpose_item = np.loadtxt(headpose_item)
    #temp_vec_item = np.loadtxt(vec_item)
    #temp_afinn_item = np.loadtxt(afinn_item)
    
    ### normalization ###
    la_norm_nlp = np.linalg.norm(temp_nlp_item, ord=2, keepdims=True) #L2_normalization
    if la_norm_nlp == 0:
        la_norm_nlp = 1
        #print('Warning! nlp norm is 0!\n')
    norm_temp_nlp_item = temp_nlp_item/la_norm_nlp #normalization
    #####################

    ### normalization ###
    #la_norm_vec = np.linalg.norm(temp_vec_item, ord=2, keepdims=True) #L2_normalization
    #if la_norm_vec == 0:
    #    la_norm_vec = 1
    #    #print('Warning! vec norm is 0!\n')
    #norm_temp_vec_item = temp_vec_item/la_norm_vec #normalization
    ######################

    ### normalization ###
    #la_norm_afinn = np.linalg.norm(temp_afinn_item, ord=2, keepdims=True) #L2_normalization
    #if la_norm_afinn == 0:
    #    la_norm_afinn = 1
        #print('Warning! afinn norm is 0!\n')
    #norm_temp_afinn_item = temp_afinn_item/la_norm_afinn #normalization
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
    '''
    elif op_mode == 3 : #using only vec feature
        combined_array = norm_temp_vec_item
    elif op_mode == 4 : #using img and nlp feature
        combined_array = np.hstack((temp_img_item, norm_temp_nlp_item))
    elif op_mode == 5 : #using img and vec feature
        combined_array = np.hstack((temp_img_item, norm_temp_vec_item))
    elif op_mode == 6 : #using nlp and vec feature
        combined_array = np.hstack((temp_nlp_item, norm_temp_vec_item))
    #elif op_mode == 7 : #using all img and nlp  and vec feature, and pca
    #    combined_array = np.hstack((temp_img_item, norm_temp_nlp_item, norm_temp_vec_item))
    #elif op_mode == 8 : #afinn
    #    combined_array = np.hstack((temp_img_item, norm_temp_afinn_item))
    elif op_mode == 7 : #sum
        combined_array = temp_img_item + norm_temp_nlp_item
    elif op_mode == 8 : #sum
        combined_array = temp_img_item + norm_temp_vec_item
    elif op_mode == 9 : #sum
        combined_array = norm_temp_nlp_item + norm_temp_vec_item
    elif op_mode == 10 : #sum
        combined_array = temp_img_item + norm_temp_nlp_item + norm_temp_vec_item
    elif op_mode == 11 : #product
        combined_array = temp_img_item * norm_temp_nlp_item
    elif op_mode == 12 : #product
        combined_array = temp_img_item * norm_temp_vec_item
    elif op_mode == 13 : #product
        combined_array = norm_temp_nlp_item * norm_temp_vec_item
    elif op_mode == 14 : #product
        combined_array = temp_img_item * norm_temp_nlp_item * norm_temp_vec_item
    elif op_mode == 15 : #sum and product
        combined_array = np.hstack(((temp_img_item + norm_temp_nlp_item), (temp_img_item * norm_temp_nlp_item)))
    elif op_mode == 16 : #sum and product
        combined_array = np.hstack(((temp_img_item + norm_temp_vec_item), (temp_img_item * norm_temp_vec_item)))
    elif op_mode == 17 : #sum and product
        combined_array = np.hstack(((norm_temp_nlp_item + norm_temp_vec_item), (norm_temp_nlp_item * norm_temp_vec_item)))
    elif op_mode == 18 : #sum and product
        combined_array = np.hstack(((temp_img_item + norm_temp_nlp_item + norm_temp_vec_item), (temp_img_item * norm_temp_nlp_item * norm_temp_vec_item)))
    '''
    #combined_array = temp_img_item

    train_combined_list.append(combined_array)
    loop_cnt += 1
############

output_combined_feature   = target_dir + '_combined.pickle'

feature_len = len(train_combined_list)
print('target input dir  is {}'.format(args.input_dir))
print('combined_list len is {0}'.format(feature_len))


feature_h = len(train_combined_list[0])
feature_array = np.array(train_combined_list)

#######################################################################

###save feature_lists file ###
print('Saving feature_array as {} file.'.format(output_combined_feature))
with open(output_combined_feature, mode='wb') as f:
    pickle.dump(feature_array, f)
##############################
    
print('combine_feature.py end\n')
print('-'*20)
