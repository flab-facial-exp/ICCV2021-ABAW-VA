# -*- coding: utf-8 -*-
from sklearn import tree
import pydotplus
#from sklearn.externals.six import StringIO
import pandas as pd
import pickle
import argparse
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
#from sklearn.metrics import neg_mean_squared_error
from sklearn.metrics import mean_squared_error
import sys
#from sklearn.cross_validation import train_test_split
#from sklearn.grid_search import GridSearchCV
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC
from sklearn.svm import SVR
#from sklearn.externals import joblib
import joblib

from statistics import mean, variance, stdev

import itertools
from catboost import CatBoostRegressor
#from catboost import CatBoostClassifier
#import lightgbm as lgb
import glob
import os
#######################


# print
print('-'*40)
print('submit_make.py start\n')
##########################


#argparse setting
a = argparse.ArgumentParser()
#a.add_argument("--input_video_name", default='11', help="input_video_name")
a.add_argument("--seed", default=0, help="seed", type=int)
a.add_argument("--mode", default=0, help="mode", type=int)
a.add_argument("--classifier", default='catboost', help="classifier_name")
args = a.parse_args()

print('\nsetting arguments are listed below.')
#print('input_video_name is {0}'.format(args.input_video_name))
print('seed is {0}'.format(args.seed))
print('mode is {0}'.format(args.mode))
print('classifier is {0}'.format(args.classifier))
print('\n')

classifier_name = args.classifier
###########################################################

#input_submit_test_feature_file  = 'submit_test_combined.pickle'
#input_submit_test_feature_file  = str(args.input_video_name) + '.pickle'
#print('input_submit_test_feature_file is {0}'.format(input_submit_test_feature_file))

###load extracted features###
#with open(input_submit_test_feature_file, mode='rb') as f:
#    restored_submit_test_feature_list = pickle.load(f)
#############################

#dir_cal_submit = './submisssion_files/'
dir_cal_submit = './submisssion_files_' + str(classifier_name) + '/'
if not os.path.exists(dir_cal_submit):
    os.makedirs(dir_cal_submit)


##########################
####### function #########
##########################
#def func_submit_make(in_loaded_model, in_classifier_name, in_va_data, in_video_name='11', restored_submit_test_feature_list):
def func_submit_make(in_loaded_model, in_classifier_name, in_va_data, in_video_name, restored_submit_test_feature_list):
    
    ###for valence ####
    #model_valence_name = 'classifier_model_valence' + str(classifier) + '.sav'
    #loaded_valence_model = joblib.load(model_valence_name)

    predicted_submit_result = in_loaded_model.predict(restored_submit_test_feature_list)
    predicted_submit_result = [round(predicted_submit_result[n], 3) for n in range(len(predicted_submit_result))]

    output_submit_file_name = dir_cal_submit + 'submit_test_result_' + str(in_va_data) + '_' + str(in_classifier_name) + '_' + str(in_video_name) + '_' + str(args.seed) + '_' + str(args.mode) + '.txt'

    for i in range(len(predicted_submit_result)):
        write_txt = str(predicted_submit_result[i]) +  '\n'

        with open(output_submit_file_name, 'a') as f:
            f.write(write_txt)
    ############################################


    ###for arousal ####
    #model_arousal_name = 'classifier_model_arousal' + str(classifier) + '.sav'
    #loaded_arousal_model = joblib.load(model_arousal_name)

    #predicted_submit_result = input_model.predict(restored_submit_test_feature_list)
    #predicted_submit_result = [round(predicted_submit_result[n], 3) for n in range(len(predicted_submit_result))]

    #output_submit_file_name = 'submit_test_result_arousal_' + str(classifier) + '_' + str(input_video_name) + '_' + str(args.seed) + '_' + str(args.mode) + '.txt'

    #for i in range(len(predicted_submit_result)):
    #    write_txt = str(predicted_submit_result[i]) +  '\n'

    #    with open(output_submit_file_name, 'a') as f:
    #        f.write(write_txt)
    ############################################
    ############################################


    #print('\n')
    #print('-'*10, 'end', '-'*10)
##########################
##########################
##########################



#####################
###parameter setting#
#####################

#####################
#####################
#####################



##########################
##########################
##########################

target_dir = './submit_pickle/'

target_pickle_file_list = glob.glob(os.path.join(target_dir, '*.pickle'))
target_pickle_file_list.sort()
#print('target_pickle_file_list')
#print(target_pickle_file_list)

file_loop_cnt = 0

for item in target_pickle_file_list:
    file_name = os.path.basename(item)
    #print('file_name')
    #print(file_name)
    extract_va = file_name.split('_')
    if (len(extract_va) == 4): #normal case
        in_video_name = extract_va[2]
    elif (len(extract_va) == 5): #normal case
        in_video_name = extract_va[2] + '_' + extract_va[3]
    elif (len(extract_va) == 6): #normal case
        in_video_name = extract_va[2] + '_' + extract_va[3] + '_' + extract_va[4]
    else:
        print('python wanrning: pickle file {0} is not processed correctly.'.format(item))

    print('No{0}:video_name is {1}.\n'.format(file_loop_cnt, in_video_name))

    ### valence ###
    #model_valence_name = 'classifier_model_valence_catboost.sav'
    model_valence_name = 'classifier_model_valence_' + str(classifier_name) + '.sav'
    loaded_valence_model = joblib.load(model_valence_name)

    #model_arousal_name = 'classifier_model_arousal_catboost.sav'
    model_arousal_name = 'classifier_model_arousal_' + str(classifier_name) + '.sav'
    loaded_arousal_model = joblib.load(model_arousal_name)
    #va_data='valence'

    #classifier_name='catboost'

    with open(item, mode='rb') as f:
        restored_submit_test_feature_list = pickle.load(f)


    predicted_submit_valence_result = loaded_valence_model.predict(restored_submit_test_feature_list)
    predicted_submit_valence_result = [round(predicted_submit_valence_result[n], 3) for n in range(len(predicted_submit_valence_result))]
    
    predicted_submit_arousal_result = loaded_arousal_model.predict(restored_submit_test_feature_list)
    predicted_submit_arousal_result = [round(predicted_submit_arousal_result[n], 3) for n in range(len(predicted_submit_arousal_result))]

    #output_submit_file_name = dir_cal_submit + 'submit_test_result_' + str(in_va_data) + '_' + str(in_classifier_name) + '_' + str(in_video_name) + '_' + str(args.seed) + '_' + str(args.mode) + '.txt'
    output_submit_file_name = dir_cal_submit + str(in_video_name) + '.txt'
    print('Output submission file name is {0}.\n'.format(output_submit_file_name))

    write_txt = 'valence,arousal' + '\n'

    with open(output_submit_file_name, 'a') as f:
        f.write(write_txt)

    for i in range(len(predicted_submit_valence_result)):
        write_txt = str(predicted_submit_valence_result[i]) + ',' + str(predicted_submit_arousal_result[i]) + '\n'

        with open(output_submit_file_name, 'a') as f:
            f.write(write_txt)

    #func_submit_make(loaded_valence_model, classifier_name, va_data, in_video_name, restored_submit_test_feature_list)
    ###################

    file_loop_cnt += 1


    ### arousal ###
    #model_arousal_name = 'classifier_model_arousal_catboost.sav'
    #loaded_arousal_model = joblib.load(model_arousal_name)
    #va_data='arousal'
    #func_submit_make(loaded_arousal_model, classifier_name, va_data, in_video_name, restored_submit_test_feature_list)
    ###################



print('\n')
print('submit_make.py end\n')
print('-'*40)
