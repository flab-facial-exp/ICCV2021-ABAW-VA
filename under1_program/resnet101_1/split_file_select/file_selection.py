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


dir_annotation = './using_dataset/annotation_rev/'
if os.path.exists(dir_annotation):
    shutil.rmtree(dir_annotation)
    os.makedirs(dir_annotation)
else:
    os.makedirs(dir_annotation)

dir_trainset = './using_dataset/train/nlp_rev/'
if not os.path.exists(dir_trainset):
    os.makedirs(dir_trainset)

dir_testset = './using_dataset/test/nlp_rev/'
if not os.path.exists(dir_testset):
    os.makedirs(dir_testset)

dir_headpose = './using_dataset/test/headpose_rev/'
if not os.path.exists(dir_headpose):
    os.makedirs(dir_headpose)

    
#######################################################
### function definition ### 
######################################################
#in_target_dir is target directory and should be image directory such as ./train/img/
#directory structure is ./train, ./test, and ./annotation.
#./train and ./test: Assuming that these two directory have img and nlp that have subdirectories with video frame name and frame images under those of subdirectories.
#./annotation: Assuming that this directory has only txt files under it.
#annotation_rev directory must be deleted before running this function due to the additional writing.
def func_arrange_data(in_target_dir):
    for root, dirs, files in os.walk(in_target_dir):
            for dr in dirs:
                img_file_list = glob.glob(os.path.join(root, dr + '/*.jpg'))
                img_file_list.sort()
                
                col_num_list = []
                rev_col_num_list =[]
                for item in img_file_list:
                    file_name = os.path.basename(item)
                    file_num = file_name[:-4]
                    col_num = int(file_num)
                    col_num_list.append(col_num)
                
                ### for valence and arousal txt data ###
                va_txt_file = './using_dataset/annotation/' + str(dr) + '.txt'
                if not os.path.exists(va_txt_file):
                    print('python error: {0} is not exist.'.format(va_txt_file))
                    print('The correspondence img file is deleted.')
                    not_exist_img_dir = str(root) + '/' + str(dr)
                    print(not_exist_img_dir)
                    shutil.rmtree(not_exist_img_dir)
                    continue;
                #print('va_txt_file is {}'.format(va_txt_file))
                line_cnt = 0
                col_cnt = 0
                output_anno_rev_valence_txt = dir_annotation + str(dr) + '_valence.txt'
                output_anno_rev_arousal_txt = dir_annotation + str(dr) + '_arousal.txt'
                valence_data = '0.44'
                arousal_data = '0.44'

                with open(va_txt_file, 'r', encoding='utf-8') as f:
                    for i in f.read().splitlines():
                        if not line_cnt == 0:
                            if (line_cnt - 1) == col_num_list[col_cnt]:
                                va_data = i.split(',')
                                valence_data = va_data[0] #valence
                                valence_data = valence_data + '\n'
                                arousal_data = va_data[1] #arousal
                                arousal_data = arousal_data + '\n'

                                if ( va_data[0] == '-5' or va_data[1] == '-5' ): 
                                    print('not annotated frame. delete correspondent img file.')
                                    original_img_num = '%06d.jpg' % int(col_num_list[col_cnt])
                                    not_anno_img_file_path = str(root) + '/' + str(dr) + '/' + original_img_num
                                    print('deleting files is {0}'.format(not_anno_img_file_path))
                                    os.remove(not_anno_img_file_path)
                                else:
                                    rev_col_num_list.append(col_num_list[col_cnt])
                                    with open(output_anno_rev_valence_txt, 'a') as f:
                                        f.write(valence_data)
                                    with open(output_anno_rev_arousal_txt, 'a') as f:
                                        f.write(arousal_data)
                                    
                                if col_cnt < len(col_num_list) -1:
                                    col_cnt += 1

                        line_cnt += 1

                ##### added for incorrespondence ###
                #if ((line_cnt - 1) < (col_cnt + 1)):
                if ((line_cnt - 1) < len(col_num_list)):
                    print('Warning:discrepancy between annotation.txt and img file.')
                    print('discrepancy is {0}'.format(dr))
                    dis_cnt = len(col_num_list) - line_cnt + 1
                    print('dis_cnt is {0}'.format(dis_cnt))
                    for i in range(dis_cnt):
                        rev_col_num_list.append(col_num_list[col_cnt+i])
                        with open(output_anno_rev_valence_txt, 'a') as f:
                            f.write(valence_data)
                        with open(output_anno_rev_arousal_txt, 'a') as f:
                            f.write(arousal_data)

                ####################################
                ##########################################
                        

                ### for nlp txt data ###
                learn_dir_name = in_target_dir[:-4]
                nlp_dir_path = './' + learn_dir_name + 'nlp/' + str(dr) 
                
                dir_nlp_subdir = './' + learn_dir_name + 'nlp_rev/' + str(dr) + '/' 
                if not os.path.exists(dir_nlp_subdir):
                    os.makedirs(dir_nlp_subdir)
                
                pre_corres_txt_name = '999999.txt' #for final version
                for i in rev_col_num_list:
                    corres_txt_name = '%06d.txt' % int(i)  #for final version
                    corres_txt_path = nlp_dir_path + '/' + corres_txt_name
                    if not os.path.exists(corres_txt_path):
                        print('python:nlp file_not exist')
                        corres_txt_path = nlp_dir_path + '/' + pre_corres_txt_name
                    else:
                        pre_corres_txt_name = corres_txt_name
                        
                    copy_corres_txt_path = dir_nlp_subdir + corres_txt_name
                    shutil.copy2(corres_txt_path, copy_corres_txt_path) 
                ##########################################


                ### for headpose txt data ###
                learn_dir_name = in_target_dir[:-4]
                headpose_dir_path = './' + learn_dir_name + 'headpose/' + str(dr) 
                
                dir_headpose_subdir = './' + learn_dir_name + 'headpose_rev/' + str(dr) + '/' 
                if not os.path.exists(dir_headpose_subdir):
                    os.makedirs(dir_headpose_subdir)
                
                pre_corres_txt_name = '999999.txt' #for final version
                for i in rev_col_num_list:
                    corres_txt_name = '%06d.txt' % int(i)  #for final version
                    corres_txt_path = headpose_dir_path + '/' + corres_txt_name
                    if not os.path.exists(corres_txt_path):
                        print('python:headpose file_not exist')
                        corres_txt_path = headpose_dir_path + '/' + pre_corres_txt_name
                    else:
                        pre_corres_txt_name = corres_txt_name
                        
                    copy_corres_txt_path = dir_headpose_subdir + corres_txt_name
                    shutil.copy2(corres_txt_path, copy_corres_txt_path) 
                ##########################################
                        
            break;
#######################################################
#######################################################
#######################################################



#######################################################
### function definition ### 
def func_arrange_submit(in_target_dir):
    for root, dirs, files in os.walk(in_target_dir):
            for dr in dirs:
                img_file_list = glob.glob(os.path.join(root, dr + '/*.jpg'))
                img_file_list.sort()
                
                col_num_list = []
                for item in img_file_list:
                    file_name = os.path.basename(item)
                    file_num = file_name[:-4]
                    col_num = int(file_num)
                    col_num_list.append(col_num)
                
                        
                ### for nlp txt data ###
                learn_dir_name = in_target_dir[:-4]
                nlp_dir_path = './' + learn_dir_name + 'nlp/' + str(dr) 
                
                dir_nlp_subdir = './' + learn_dir_name + 'nlp_rev/' + str(dr) + '/' 
                if not os.path.exists(dir_nlp_subdir):
                    os.makedirs(dir_nlp_subdir)
                
                pre_corres_txt_name = '999999.txt' #for final version
                for i in col_num_list:
                    corres_txt_name = '%06d.txt' % int(i)  #for final version
                    corres_txt_path = nlp_dir_path + '/' + corres_txt_name
                    if not os.path.exists(corres_txt_path):
                        print('python:submit_nlp file_not exist')
                        corres_txt_path = nlp_dir_path + '/' + pre_corres_txt_name
                    else:
                        pre_corres_txt_name = corres_txt_name
                        
                    copy_corres_txt_path = dir_nlp_subdir + corres_txt_name
                    shutil.copy2(corres_txt_path, copy_corres_txt_path) 
                ##########################################


                ### for headpose txt data ###
                learn_dir_name = in_target_dir[:-4]
                headpose_dir_path = './' + learn_dir_name + 'headpose/' + str(dr) 
                
                dir_headpose_subdir = './' + learn_dir_name + 'headpose_rev/' + str(dr) + '/' 
                if not os.path.exists(dir_headpose_subdir):
                    os.makedirs(dir_headpose_subdir)
                
                pre_corres_txt_name = '999999.txt' #for final version
                for i in col_num_list:
                    corres_txt_name = '%06d.txt' % int(i)  #for final version
                    corres_txt_path = headpose_dir_path + '/' + corres_txt_name
                    if not os.path.exists(corres_txt_path):
                        print('python:submit_headpose file_not exist')
                        corres_txt_path = headpose_dir_path + '/' + pre_corres_txt_name
                    else:
                        pre_corres_txt_name = corres_txt_name
                        
                    copy_corres_txt_path = dir_headpose_subdir + corres_txt_name
                    shutil.copy2(corres_txt_path, copy_corres_txt_path) 
                ##########################################
                        
            break;
#######################################################
#######################################################
#######################################################







 
#######################################################
### function definition ###
#######################################################
def in_process(args):
    train_img = 'using_dataset/train/img/' 
    test_img = 'using_dataset/test/img/'
    submit_test_img = 'using_dataset/submit_test/img/'
    #nb_epoch = int(args.nb_epoch)

    ### added 2021/06/10 ###
    print('train directoy start.')
    target_dir_train = train_img
    func_arrange_data(target_dir_train)
    print('train directoy end.')
    
    print('test directoy start.')
    target_dir_test = test_img
    func_arrange_data(target_dir_test)
    print('test directoy end.')
    ########################

    print('submit_test directoy start.')
    func_arrange_submit(submit_test_img)
    print('submit_test directoy end.')
            
if __name__=="__main__":
    
    
    print('Program file_selection.py start\n\n')
    a = argparse.ArgumentParser()
    a.add_argument("--nb_epoch", default=3)
    args = a.parse_args()

    #print('setting arguments are listed below.\n')
    #print('nb_epoch=%s' % args.nb_epoch)

    in_process(args)

    dir_done_flag = './using_dataset/done_flag_dir'
    if not os.path.exists(dir_done_flag):
        os.makedirs(dir_done_flag)
    print('Program file_selection.py End\n')
