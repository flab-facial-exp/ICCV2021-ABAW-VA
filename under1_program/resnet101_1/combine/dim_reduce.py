# -*- coding: utf-8 -*-
"""
Created on 2018/05/09
"""
######################
### import library ###
######################
import argparse
import numpy as np
from PIL import Image
import os
import sys
import glob

import pickle
from sklearn.decomposition import PCA

import pandas as pd
######################
######################
######################



########################
### argument setting ###
########################
print('\n\n')
print('-'*20)
print('dim_reduce.py program start.')
print('This program will reduce feature vector dimension by using PCA.\n')
#arg setting
a = argparse.ArgumentParser()
a.add_argument("--pca_dim", default=300, help="integer", type=int)
a.add_argument("--target_dir", default="nlp", help="feature txt file")
a.add_argument("--random_seed", default=0, help="random_seed_number", type=int)
a.add_argument("--vec_fea", action="store_true", help="using vector feature or not")
args = a.parse_args()

p_random_seed = args.random_seed
np.random.seed(p_random_seed)
########################
########################
########################


########################
target_dir = args.target_dir

feature_list = []

#target_train_dir = 'train_' + target_dir + '_raw_feature'
target_train_dir = './using_dataset/train/' + target_dir

print('Dealing feature target_dir is {}'.format(target_train_dir))


### input_nlp ##############
for root, dirs, files in os.walk(target_train_dir):
    for dr in dirs:
        feature_list += glob.glob(os.path.join(root, dr + '/*.txt')) #for nlp feature
    break;    
########################


feature_list.sort()

train_feature_list = []

for feature_item in feature_list:

    temp_feature_item = np.loadtxt(feature_item)

    ### added for converting nan to 0.00 ####
    if 'nlp' in target_train_dir and args.vec_fea :
        temp_feature_item_rev = pd.DataFrame(temp_feature_item)
        temp_feature_item_rev_2 = temp_feature_item_rev.fillna(0.0000)
        temp_feature_item = np.array(temp_feature_item_rev_2.values.flatten())
    ######################################################

    temp_norm = np.linalg.norm(temp_feature_item)
    #temp_feature_item = temp_feature_item/np.linalg.norm(temp_feature_item) #normalization
    if (temp_norm == 0):
        print('Warning! Norm is 0.')
        temp_norm = 1

    temp_feature_item = temp_feature_item/temp_norm #normalization

    train_feature_list.append(temp_feature_item)
############


train_feature_len = len(train_feature_list)
print('train_feature_list len is {0}'.format(train_feature_len))
train_feature_array = np.array(train_feature_list)

#######################################################################


#############################################################
### PCA ###
#############################################################
## for training list ##
train_feature_h, train_feature_w = train_feature_array.shape
pca_dim = min(train_feature_h, train_feature_w, args.pca_dim)

print('PCA start\n')
print('pca_dim is {}.'.format(pca_dim))
    
pca = PCA(n_components=pca_dim)
pca_train_feature_mat = pca.fit(train_feature_array)
    
print('\nPCA sum_ratio')
print(sum(pca.explained_variance_ratio_))

################
### function ###
################
def func_feature_pca(input_dir):
    cnt_dr = 0
    for root, dirs, files in os.walk(input_dir):
        print('Here, you have %d classes to feature\n' % len(dirs))
        dirs.sort()
        for dr in dirs:
            print('\n\n')
            print('-'*10)
            print('No %d class test start, %d/%d processing.' % (cnt_dr, cnt_dr+1, len(dirs)))

            print('No {0} class is {1}'.format(cnt_dr, dr))

            #if 'nlp' in input_dir and not args.vec_fea:
            #file_list = glob.glob(os.path.join(root, dr + '/*_confidence.txt'))
            file_list = glob.glob(os.path.join(root, dr + '/*.txt'))
            #elif 'img' in input_dir:
            #    file_list = glob.glob(os.path.join(root, dr + '/*.txt'))
            #elif 'nlp' in input_dir and args.vec_fea:
            #    file_list = glob.glob(os.path.join(root, dr + '/*_vector_sum.txt'))

            file_list.sort()

            #added, 2019/08/26#
            output_dir = input_dir + '_pca'
            dir_sub = output_dir + '/' + dr
            if not os.path.exists(dir_sub):
                os.makedirs(dir_sub)
            ###################
                    
            for item in file_list:
                ext_feature = np.loadtxt(item)

                ### added for converting nan to 0.00 ####
                if 'nlp' in target_train_dir and args.vec_fea :
                    temp_feature_item_rev = pd.DataFrame(ext_feature)
                    temp_feature_item_rev_2 = temp_feature_item_rev.fillna(0.0000)
                    ext_feature = np.array(temp_feature_item_rev_2.values.flatten())
                ######################################################

                re_ext_feature = ext_feature.reshape(1, -1)
                re_ext_feature = pca.transform(re_ext_feature)
                
                norm_feature_lists = re_ext_feature/np.linalg.norm(re_ext_feature)
                norm_feature_lists = norm_feature_lists.reshape(-1, 1)

                output_feature_txt = dir_sub + '/' + os.path.basename(item)[:-4] + '_pca' + '.txt'

                np.savetxt(output_feature_txt, norm_feature_lists)

            cnt_dr += 1
            
        break;
##############################
##############################
##############################

# for train #
input_dir = './using_dataset/train/' + target_dir
print('Target directory {0} pca start.\n'.format(input_dir))
func_feature_pca(input_dir)
print('Target directory {0} pca end.\n'.format(input_dir))
#############

# for test #
input_dir = './using_dataset/test/' + target_dir
print('Target directory {0} pca start.\n'.format(input_dir))
func_feature_pca(input_dir)
print('Target directory {0} pca end.\n'.format(input_dir))
#############

# for submit_test #
input_dir = './using_dataset/submit_test/' + target_dir
print('Target directory {0} pca start.\n'.format(input_dir))
func_feature_pca(input_dir)
print('Target directory {0} pca end.\n'.format(input_dir))
#############

    
print('dim_reduce.py end\n')
print('-'*20)
