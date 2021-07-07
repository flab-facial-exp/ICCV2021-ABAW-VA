# -*- coding: utf-8 -*-
"""
Created on 2018/05/08
"""


#######################################################
### import setting ###
#######################################################
import sys
import glob
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
import random as rn
import pickle
import shutil
import os
#######################################################
#######################################################
#######################################################



#######################################################
### parameter setting ###
#######################################################

#######################################################
#######################################################
#######################################################


#######################################################
### Fix random element in deep learning ###
#######################################################

#print('Fix random element in deep learning.\n')
#import os
#os.environ['PYTHONHASHSEED'] = '0'
#np.random.seed(42)
#rn.seed(123)

#######################################################
#######################################################
#######################################################


#dir_annotation = './annotation_rev/'
#shutil.rmtree(dir_annotation)
#os.makedirs(dir_annotation)

### for train ###
dir_train = './using_dataset/train/'
if not os.path.exists(dir_train):
    os.makedirs(dir_train)

dir_train_img = './using_dataset/train/img/'
if not os.path.exists(dir_train_img):
    os.makedirs(dir_train_img)

dir_train_nlp = './using_dataset/train/nlp/'
if not os.path.exists(dir_train_nlp):
    os.makedirs(dir_train_nlp)

dir_train_headpose = './using_dataset/train/headpose/'
if not os.path.exists(dir_train_headpose):
    os.makedirs(dir_train_headpose)
###################

### for test ###
dir_test = './using_dataset/test/'
if not os.path.exists(dir_test):
    os.makedirs(dir_test)

dir_test_img = './using_dataset/test/img/'
if not os.path.exists(dir_test_img):
    os.makedirs(dir_test_img)

dir_test_nlp = './using_dataset/test/nlp/'
if not os.path.exists(dir_test_nlp):
    os.makedirs(dir_test_nlp)

dir_test_headpose = './using_dataset/test/headpose/'
if not os.path.exists(dir_test_headpose):
    os.makedirs(dir_test_headpose)
######################

### for submit_test ###
dir_submit_test = './using_dataset/submit_test/'
if not os.path.exists(dir_submit_test):
    os.makedirs(dir_submit_test)

dir_submit_test_img = './using_dataset/submit_test/img/'
if not os.path.exists(dir_submit_test_img):
    os.makedirs(dir_submit_test_img)

dir_submit_test_nlp = './using_dataset/submit_test/nlp/'
if not os.path.exists(dir_submit_test_nlp):
    os.makedirs(dir_submit_test_nlp)

dir_submit_test_headpose = './using_dataset/submit_test/headpose/'
if not os.path.exists(dir_submit_test_headpose):
    os.makedirs(dir_submit_test_headpose)
######################

img_loop_num = 100 #get images per 100 frame
#img_loop_num = 1 #get images per 100 frame

print('Important information:frame skip is {0}.'.format(img_loop_num))

    
#######################################################
### function definition ### 
######################################################
def func_split_data(in_target_dir_img, in_target_dir_nlp, in_target_dir_headpose, in_txt_file):
    #print(in_txt_file)
    with open(in_txt_file, 'r', encoding='utf-8') as f:
        for i in f.read().splitlines():
            sub_dir_name = i[:-4]
            #print(sub_dir_name)
            
            source_img_path = './' + in_target_dir_img + '/' + str(sub_dir_name)
            source_nlp_path = './' + in_target_dir_nlp + '/' + str(sub_dir_name)
            source_headpose_path = './' + in_target_dir_headpose + '/' + str(sub_dir_name)
            
            if not os.path.exists(source_img_path):
                print('Python Warning: The img directory {0} is not exist and skip copy.'.format(source_img_path))
                continue;
                
            if not os.path.exists(source_nlp_path):
                print('Python Warning: The nlp directory {0} is not exist and skip copy.'.format(source_nlp_path))
                continue;
            
            if not os.path.exists(source_headpose_path):
                print('Python Warning: The headpose directory {0} is not exist and skip copy.'.format(source_headpose_path))
                continue;
            
            img_file_list = glob.glob(os.path.join(source_img_path + '/*.jpg'))
            img_file_list.sort()
            len_img_file_list = len(img_file_list)
            
            #img_loop_num = 100 #get images per 100 frame
            #img_loop_num = 1000 #get images per 100 frame
            ### for img ###
            flag_train_test = in_txt_file[:-9]
            if flag_train_test == 'train':
                target_img_path = dir_train_img + str(sub_dir_name)
            elif flag_train_test == 'test':
                target_img_path = dir_test_img + str(sub_dir_name)
            else: #for submit_test_list.txt
                target_img_path = dir_submit_test_img + str(sub_dir_name)
            
            if not os.path.exists(target_img_path):
                os.makedirs(target_img_path)

            if (flag_train_test == 'train' or flag_train_test == 'test'):
                for item in range(0, len_img_file_list, img_loop_num):
                    target_img_path_file = target_img_path + '/' + os.path.basename(img_file_list[item])
                    source_img_path_file = source_img_path + '/' + os.path.basename(img_file_list[item])
                    shutil.copy2(source_img_path_file, target_img_path_file)
            else:
                shutil.rmtree(target_img_path)
                shutil.copytree(source_img_path, target_img_path)
            #####################################

            ### for nlp ###
            if flag_train_test == 'train':
                #print('train')
                target_nlp_path = dir_train_nlp + str(sub_dir_name)
            elif flag_train_test == 'test':
                #print('test')
                target_nlp_path = dir_test_nlp + str(sub_dir_name)
            else:
                target_nlp_path = dir_submit_test_nlp + str(sub_dir_name)
            
            if not os.path.isdir(target_nlp_path):
                #print('directory does not exist and copy')
                shutil.copytree(source_nlp_path, target_nlp_path)
            else:
                #print('directory exist, delete and copy')
                shutil.rmtree(target_nlp_path)
                shutil.copytree(source_nlp_path, target_nlp_path)
            ##############


            ### for headpose ###
            if flag_train_test == 'train':
                #print('train')
                target_headpose_path = dir_train_headpose + str(sub_dir_name)
            elif flag_train_test == 'test':
                #print('test')
                target_headpose_path = dir_test_headpose + str(sub_dir_name)
            else:
                target_headpose_path = dir_submit_test_headpose + str(sub_dir_name)
            
            if not os.path.isdir(target_headpose_path):
                #print('directory does not exist and copy')
                shutil.copytree(source_headpose_path, target_headpose_path)
            else:
                #print('directory exist, delete and copy')
                shutil.rmtree(target_headpose_path)
                shutil.copytree(source_headpose_path, target_headpose_path)
            ##############
            

            
#######################################################
#######################################################
#######################################################




 
#######################################################
### function definition ###
#######################################################
def in_process(args):
    dataset_img = './using_dataset/dataset_img' 
    dataset_nlp = './using_dataset/dataset_nlp'
    dataset_headpose = './using_dataset/dataset_headpose'
    nb_epoch = int(args.nb_epoch)

    ### common ###
    target_dir_img = dataset_img
    target_dir_nlp = dataset_nlp
    target_dir_headpose = dataset_headpose
    ##############
    
    ### for train ###
    print('Traning data splitting start.\n')
    target_txt_file = 'train_list.txt'
    func_split_data(target_dir_img, target_dir_nlp, target_dir_headpose, target_txt_file)
    print('Traning data splitting end.\n')
    ########################
    
    ### for test ###
    print('Test data splitting start.\n')
    target_txt_file = 'test_list.txt'
    func_split_data(target_dir_img, target_dir_nlp, target_dir_headpose, target_txt_file)
    print('Test data splitting end.\n')
    ################
    
    ### for sumibt_test ###
    print('Submit test data splitting start.\n')
    target_txt_file = 'submit_test_list.txt'
    func_split_data(target_dir_img, target_dir_nlp, target_dir_headpose, target_txt_file)
    print('Submit test data splitting end.\n')
    ################
            
if __name__=="__main__":
    
    
    print('Program split_data.py start\n\n')
    a = argparse.ArgumentParser()
    a.add_argument("--nb_epoch", default=3)
    args = a.parse_args()

    #print('setting arguments are listed below.\n')
    #print('nb_epoch=%s' % args.nb_epoch)

    in_process(args)
    print('Program split_data.py End\n')
