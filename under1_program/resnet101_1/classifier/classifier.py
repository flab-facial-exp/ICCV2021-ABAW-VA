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
from sklearn.linear_model import LinearRegression
#from catboost import CatBoostClassifier
#import lightgbm as lgb
#######################


# print
print('-'*40)
print('classifier.py start\n')
print('This program train and test for various classifier to calculate accuracy and so on.')
##########################


#argparse setting
a = argparse.ArgumentParser()
a.add_argument("--pca", default=1, help="integer:0 is no_pca, 1 is pca", type=int)
a.add_argument("--method", default='our_method', help="method:our_method or conv_method")
a.add_argument("--dataset", default=1, help="dataset", type=int)
a.add_argument("--seed", default=0, help="seed", type=int)
a.add_argument("--mode", default=0, help="mode", type=int)
args = a.parse_args()

print('\nsetting arguments are listed below.')
print('PCA setting is {}'.format(args.pca))
print('method is {0}'.format(args.method))
print('dataset is {0}'.format(args.dataset))
print('seed is {0}'.format(args.seed))
print('mode is {0}'.format(args.mode))
dataset_num = int(args.dataset)
print('\n')
###########################################################

if (args.pca == 1):
    if (args.method == 'our_method') :
        input_train_feature_file = 'norm_pca_train_combined_feature.pickle'
        input_test_feature_file  = 'norm_pca_test_combined_feature.pickle'
    elif (args.method == 'conv_method') :
        input_train_feature_file = 'norm_pca_train_total_feature.pickle'
        input_test_feature_file  = 'norm_pca_test_total_feature.pickle'
    else:
        print('method error. please check method argument')
else:
    if (args.method == 'our_method') :
        input_train_feature_file = 'train_combined.pickle'
        input_test_feature_file  = 'test_combined.pickle'
        #input_submit_test_feature_file  = 'submit_test_combined.pickle'
    elif (args.method == 'conv_method') :
        input_train_feature_file = 'train_total.pickle'
        input_test_feature_file  = 'test_total.pickle'
    else:
        print('method error. please check method argument')
    
print('input_train_feature_file is {0}'.format(input_train_feature_file))
print('input_test_feature_file is {0}'.format(input_test_feature_file))
#print('input_submit_test_feature_file is {0}'.format(input_submit_test_feature_file))

###load extracted features###
with open(input_train_feature_file, mode='rb') as f:
    restored_train_feature_list = pickle.load(f)

with open(input_test_feature_file, mode='rb') as f:
    restored_test_feature_list = pickle.load(f)

#with open(input_submit_test_feature_file, mode='rb') as f:
#    restored_submit_test_feature_list = pickle.load(f)

with open('train_class_valence_label.pickle', mode='rb') as f:
    restored_train_class_valence_list = pickle.load(f)
    
with open('test_class_valence_label.pickle', mode='rb') as f:
    restored_test_class_valence_list = pickle.load(f)

with open('train_class_arousal_label.pickle', mode='rb') as f:
    restored_train_class_arousal_list = pickle.load(f)
    
with open('test_class_arousal_label.pickle', mode='rb') as f:
    restored_test_class_arousal_list = pickle.load(f)
#############################



##########################
####### function #########
##########################
def func_classifier_analysis(input_model, method='our_method', classifier='svr'):
    
    ###for valence ####
    print('calculate valence start.')
    input_model.fit(restored_train_feature_list, restored_train_class_valence_list)

    model_valence_name = 'classifier_model_valence_' + str(classifier) + '.sav'
    joblib.dump(input_model, model_valence_name)

    predicted_result = input_model.predict(restored_test_feature_list)
    #predicted_submit_result = input_model.predict(restored_submit_test_feature_list)

    predicted_result = [round(predicted_result[n], 3) for n in range(len(predicted_result))]
    #predicted_submit_result = [round(predicted_submit_result[n], 3) for n in range(len(predicted_submit_result))]

    x_pred_list = predicted_result
    x_pred_mean = mean(x_pred_list)
    x_pred_var = variance(x_pred_list)
    x_pred_stdev = stdev(x_pred_list)
    y_true_list = [float(s) for s in restored_test_class_valence_list]
    y_true_mean = mean(y_true_list)
    y_true_var  = variance(y_true_list)
    y_true_stdev  = stdev(y_true_list)
    ##############################

    #sxy = mean((x_pred_list - x_pread_mean)*(y_true_list - y_pred_mean))

    ccc_calc_1 = (2*x_pred_stdev*y_true_stdev)/(x_pred_var + y_true_var + (x_pred_mean - y_true_mean)**2)

    t_sx_list = [(i - x_pred_mean) for i in x_pred_list]
    t_sy_list = [(i - y_true_mean) for i in y_true_list]
    sxy_mul_list = [t_sx_list[i]*t_sy_list[i] for i in range(len(x_pred_list))]
    sxy_mean = mean(sxy_mul_list)
    
    ccc_calc_2 = (2*sxy_mean)/(x_pred_var + y_true_var + (x_pred_mean - y_true_mean)**2)

    t_sx_s_list = [ i*i for i in t_sx_list]
    t_sy_s_list = [ i*i for i in t_sy_list]

    sum_t_sx_s_list = np.sqrt(sum(t_sx_s_list))*np.sqrt(sum(t_sy_s_list))
    sum_sxy_mul_list = sum(sxy_mul_list)
    rho = sum_sxy_mul_list / sum_t_sx_s_list

    ccc_calc_3 = (2*rho*x_pred_stdev*y_true_stdev)/(x_pred_var + y_true_var + (x_pred_mean - y_true_mean)**2)



    #ccc_calc = (2*sxy)/(x_pred_var + y_true_var + (x_pred_mean - y_true_mean)**2)
    mse_calc = mean_squared_error(restored_test_class_valence_list, predicted_result)

    output_file_name = 'test_result_valence_' + str(classifier) + '_' + str(args.seed) + '_' + str(args.mode) + '.txt'
    #output_submit_file_name = 'submit_test_result_valence_' + str(classifier) + '_' + str(args.seed) + '_' + str(args.mode) + '.txt'

    for i in range(len(predicted_result)):
        write_txt = str(predicted_result[i]) +  '\n'

        with open(output_file_name, 'a') as f:
            f.write(write_txt)

    #for i in range(len(predicted_submit_result)):
    #    write_txt = str(predicted_submit_result[i]) +  '\n'

    #    with open(output_submit_file_name, 'a') as f:
    #        f.write(write_txt)

    print('\n')
    print('-'*10, 'start', '-'*10)
    print('{0}_{1}_ccc_1 is {2:1.6f}'.format(method, classifier, ccc_calc_1))
    print('{0}_{1}_ccc_2 is {2:1.6f}'.format(method, classifier, ccc_calc_2))
    print('{0}_{1}_ccc_3 is {2:1.6f}'.format(method, classifier, ccc_calc_3))
    print('{0}_{1}_mse_is {2:1.6f}'.format(method, classifier, mse_calc))
    
    '''
    if (classifier =='resnet'):
        print('no best param')
    else:
        print('{0}_{1}_best_parameter_valence_is {2}. '.format(method, classifier, input_model.best_params_))
        #print('nothing')
    '''

    print('calculate valence end.')
    ############################################


    ###for arousal ####
    print('calculate arousal start.')
    input_model.fit(restored_train_feature_list, restored_train_class_arousal_list)

    model_arousal_name = 'classifier_model_arousal_' + str(classifier) + '.sav'
    joblib.dump(input_model, model_arousal_name)

    predicted_result = input_model.predict(restored_test_feature_list)
    #predicted_submit_result = input_model.predict(restored_submit_test_feature_list)

    predicted_result = [round(predicted_result[n], 3) for n in range(len(predicted_result))]
    #predicted_submit_result = [round(predicted_submit_result[n], 3) for n in range(len(predicted_submit_result))]


    x_pred_list = predicted_result
    x_pred_mean = mean(x_pred_list)
    x_pred_var = variance(x_pred_list)
    x_pred_stdev = stdev(x_pred_list)
    y_true_list = [float(s) for s in restored_test_class_arousal_list]
    y_true_mean = mean(y_true_list)
    y_true_var  = variance(y_true_list)
    y_true_stdev  = stdev(y_true_list)

    ccc_calc_1 = (2*x_pred_stdev*y_true_stdev)/(x_pred_var + y_true_var + (x_pred_mean - y_true_mean)**2)

    t_sx_list = [(i - x_pred_mean) for i in x_pred_list]
    t_sy_list = [(i - y_true_mean) for i in y_true_list]
    sxy_mul_list = [t_sx_list[i]*t_sy_list[i] for i in range(len(x_pred_list))]
    sxy_mean = mean(sxy_mul_list)
    
    ccc_calc_2 = (2*sxy_mean)/(x_pred_var + y_true_var + (x_pred_mean - y_true_mean)**2)

    t_sx_s_list = [ i*i for i in t_sx_list]
    t_sy_s_list = [ i*i for i in t_sy_list]

    sum_t_sx_s_list = np.sqrt(sum(t_sx_s_list))*np.sqrt(sum(t_sy_s_list))
    sum_sxy_mul_list = sum(sxy_mul_list)
    rho = sum_sxy_mul_list / sum_t_sx_s_list

    ccc_calc_3 = (2*rho*x_pred_stdev*y_true_stdev)/(x_pred_var + y_true_var + (x_pred_mean - y_true_mean)**2)

    mse_calc = mean_squared_error(restored_test_class_arousal_list, predicted_result)

    output_file_name = 'test_result_arousal_' + str(classifier) + '_' + str(args.seed) + '_' + str(args.mode) + '.txt'
    #output_submit_file_name = 'submit_test_result_arousal_' + str(classifier) + '_' + str(args.seed) + '_' + str(args.mode) + '.txt'

    for i in range(len(predicted_result)):
        write_txt = str(predicted_result[i]) +  '\n'

        with open(output_file_name, 'a') as f:
            f.write(write_txt)

    #for i in range(len(predicted_submit_result)):
    #    write_txt = str(predicted_submit_result[i]) +  '\n'

    #    with open(output_submit_file_name, 'a') as f:
    #        f.write(write_txt)

    print('\n')
    print('-'*10, 'start', '-'*10)
    print('{0}_{1}_ccc_1 is {2:1.6f}'.format(method, classifier, ccc_calc_1))
    print('{0}_{1}_ccc_2 is {2:1.6f}'.format(method, classifier, ccc_calc_2))
    print('{0}_{1}_ccc_3 is {2:1.6f}'.format(method, classifier, ccc_calc_3))
    print('{0}_{1}_mse_is {2:1.6f}'.format(method, classifier, mse_calc))
    
    '''
    if (classifier =='resnet'):
        print('no best param')
    else:
        print('{0}_{1}_best_parameter_arousal_is {2}. '.format(method, classifier, input_model.best_params_))
        #print('nothing')
    '''

    print('calculate arousal end.')
    ############################################








    #conf_mat = confusion_matrix(restored_test_class_list, predicted_result)

    #print('{0}_{1}_confusion_matrix_num_is \n{2}'.format(method, classifier, conf_mat))

    '''
    #confusion_matrix_accuracy
    conf_mat_acc = [0.0]*conf_mat.shape[0]*conf_mat.shape[1]
    conf_mat_acc = np.array(conf_mat_acc).reshape(conf_mat.shape[0], conf_mat.shape[1])
    

    for y in range(conf_mat.shape[0]):
        for x in range(conf_mat.shape[1]):
            row_sum = np.sum(conf_mat, axis=1)
            conf_mat_acc[y][x] = round((conf_mat[y][x]/row_sum[y])*100, 2)

    print('\n')
    print('{0}_{1}_confusion_matrix_acc_is \n{2}'.format(method, classifier, conf_mat_acc))

    if (dataset_num == 3):
        class_name = ['0:anger', '1:disgust', '2:fear', '3:joy', '4:sadness', '5:surprise']
    elif (dataset_num == 4):
        class_name = ['0:amusement', '1:anger', '2:awe', '3:contentment', '4:disgust', '5:excitement', '6:fear', '7:sadness']
    else:
        class_name = ['0:negative', '1:positive']


    class_report = classification_report(restored_test_class_list, predicted_result, target_names=class_name)
    '''
    print('\n')
    #print('{0}_{1}_class_report_is \n{2}'.format(method, classifier, class_report))
    print('-'*10, 'end', '-'*10)
##########################
##########################
##########################

###parameter setting#
#score = 'f1_micro'
#score = 'accuracy'
#score = 'mean_squared_error'
score = 'neg_mean_squared_error'
print('grid_search parameter is {0}'.format(score))
#####################

##########################
####### SVM linear###############
##########################
#model_svm = SVC(kernel='linear', random_state=0)
#func_classifier_analysis(model_svm, method='svm_linear')
#tuned_parameters = [  {'C': [1, 2, 5], 'kernel': ['linear']}, {'C': [1, 2, 5], 'kernel': ['rbf'], 'gamma': [0.001]}] #simple version

#tuned_parameters = [  {'C': [1, 1.1, 1.2], 'kernel': ['linear'], 'class_weight': [None ]  } ] #2019/11/12 version
#tuned_parameters = [  {'C': [1, 1.1, 1.2], 'kernel': ['linear'], 'class_weight': [None ]  }, {'C': [2], 'kernel': ['rbf'], 'gamma': [1]}] #2019/11/14 version, not use
#tuned_parameters = [  {'C': [1, 2, 5, 10], 'kernel': ['rbf'], 'gamma': [0.1, 1, 2, 5, 10]}] #2020/02/12 test version
#tuned_parameters = [  {'C': [1, 1.1, 1.2], 'kernel': ['linear'], 'class_weight': [None ]  }, {'C': [2, 5], 'kernel': ['rbf'], 'gamma': [0.1, 1, 2]}] #


#tuned_parameters = [  {'C': [1, 2, 5, 10], 'kernel': ['linear']}, {'C': [1, 2, 5, 10], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]}] # default
#tuned_parameters = [  {'C': [1], 'kernel': ['linear']} ] #test



#tuned_parameters = [  {'C': [1], 'kernel': ['linear']} ] #simple version, most
#model_svm = GridSearchCV(SVR(), tuned_parameters, cv=5, scoring=score ) #default
##########func_classifier_analysis(model_svm, method=args.method, classifier='svm')
#model_svm = SVR(kernel='linear', C=1)
#print('SVR start')
#func_classifier_analysis(model_svm, method=args.method, classifier='svr')
#print('SVR end')
##########################
##########################
##########################


##########################
####### catboost###############
##########################
######tuned_parameters = [  {'iterations': [10, 50, 100], 'learning_rate': [1, 1e-1, 1e-2], 'depth': [2, 3, 5, 10], 'task_type': ['GPU'] } ] #
######tuned_parameters = [  {'iterations': [50, 100, 500, 1000], 'learning_rate': [1e-1, 1e-2, 1e-3], 'depth': [6, 7, 8, 9, 10], 'task_type': ['GPU'] } ] #
######tuned_parameters = [  {'iterations': [100], 'learning_rate': [1e-1, 1e-2, 1e-3], 'depth': [5, 8, 10]  } ] #
#tuned_parameters = [  {'iterations': [50], 'learning_rate': [1, 1e-1], 'depth': [5, 8, 10]  } ] #



#tuned_parameters = [  {'iterations': [50], 'learning_rate': [1e-1], 'depth': [5]  } ] #
#tuned_parameters = [  {'iterations': [5000], 'use_best_model': [True], 'eval_metric' : [RMSE], 'learning_rate': [1e-2, 3e-2, 5e-3], 'depth': [5, 8, 10]  } ] #
#tuned_parameters = [  {'iterations': [5000], 'use_best_model': [True], 'learning_rate': [1e-2, 3e-2, 5e-3], 'depth': [5, 8, 10]  } ] #
#tuned_parameters = [  {'iterations': [5000], 'learning_rate': [1e-2, 3e-2, 5e-3], 'depth': [5, 8, 10]  } ] #
#tuned_parameters = [  {'iterations': [1000], 'learning_rate': [3e-2], 'depth': [8]  } ] #
#model_catboost = GridSearchCV(CatBoostRegressor(silent=True), tuned_parameters, cv=5, scoring=score ) #default
#model_catboost = CatBoostClassifier(iterations=100, learning_rate=1e-1, depth=5, loss_function='Logloss')
#model_catboost = CatBoostRegressor(iterations=1000, learning_rate=3e-1, depth=8)
#model_catboost = CatBoostRegressor(iterations=3000, learning_rate=3e-1, depth=8)
#model_catboost = CatBoostRegressor(iterations=5000, learning_rate=3e-1, depth=8)
#model_catboost = CatBoostRegressor(iterations=3000, learning_rate=3e-1, depth=8)
#model_catboost = CatBoostRegressor(iterations=100000, learning_rate=10e-1, depth=8)
model_catboost = CatBoostRegressor(iterations=1000, learning_rate=10e-1, depth=8)
#model_catboost = CatBoostRegressor(iterations=5000, learning_rate=3e-1, depth=8)
#model_catboost =  LinearRegression()
print('catboost start')
func_classifier_analysis(model_catboost, method=args.method, classifier='catboost')
print('catboost end')
##########################
##########################
##########################

model_linear =  LinearRegression()
print('linear start')
func_classifier_analysis(model_linear, method=args.method, classifier='linear')
print('linear end')
##########################


##########################
####### catboost###############
##########################
#tuned_parameters = [  {'max_depth': [5, 7, 9]  } ] #
#model_lgb = GridSearchCV(lgb(objective='regression'), tuned_parameters, cv=5, scoring=score ) #default
#model_lgb = GridSearchCV(lgb(objective='regression'), tuned_parameters, cv=5, scoring=score ) #default
#model_lgb = GridSearchCV(lgb, tuned_parameters, cv=5, scoring=score ) #default
#func_classifier_analysis(model_lgb, method=args.method, classifier='lgb')
##########################
##########################
##########################


##############
### DT ###
##############
#tuned_parameters = [  {'max_depth': [3, 4, 5], 'criterion': ['gini', 'entropy'], 'random_state': [0, 1]} ] #simple version
#tuned_parameters = [  {'max_depth': [3, 4, 5, 6, 7, 8], 'criterion': ['gini', 'entropy'], 'random_state': [0, 1, 2, 3]} ] #default
#tuned_parameters = [  {'max_depth': [3], 'criterion': ['gini', 'entropy'], 'random_state': [0]} ] #test


##model_dt = tree.DecisionTreeClassifier(max_depth=4, criterion='gini', random_state=0)
#model_dt = GridSearchCV(tree.DecisionTreeClassifier(), tuned_parameters, cv=5, scoring=score)
#func_classifier_analysis(model_dt, method=args.method, classifier='decision_tree')
##############
##############
##############
 

##############
### RF ###
##############
#tuned_parameters = [  {'n_estimators': [50, 300], 'criterion': ['gini', 'entropy'] ,'max_depth': [5, 10, 50], 'random_state': [0, 1]} ] #simple version
#tuned_parameters = [  {'n_estimators': [30, 50, 100, 300], 'criterion': ['gini', 'entropy'] ,'max_depth': [3, 4, 5, 10, 20], 'random_state': [0, 1, 2, 3]} ] #default
#tuned_parameters = [  {'n_estimators': [30], 'criterion': ['gini', 'entropy'] ,'max_depth': [3], 'random_state': [0]} ] #test

#model_rf = GridSearchCV(RandomForestClassifier(), tuned_parameters, cv=5, scoring=score)
#func_classifier_analysis(model_rf, method=args.method, classifier='random_forest')
#####################################################
print('\n')
print('classifier.py end\n')
print('-'*40)
