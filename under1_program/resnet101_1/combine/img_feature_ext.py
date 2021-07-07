# -*- coding: utf-8 -*-
"""
Created on 2018/05/09
"""

import argparse
import numpy as np
from PIL import Image
import sys
import glob

from keras.preprocessing import image
from keras.models import Model, load_model
#from keras.applications.resnet50 import preprocess_input
#from keras.applications.resnet50 import ResNet50
#from keras.applications.resnet import preprocess_input
#from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
from keras.applications.resnet import ResNet101
#from keras.applications.densenet import preprocess_input
#from keras.applications.densenet import DenseNet121
from keras.layers import Conv2D, Dense, AveragePooling2D, GlobalAveragePooling2D, Input, Flatten, Dropout, BatchNormalization, MaxPooling2D
from keras.utils import plot_model

import pickle
from sklearn.decomposition import PCA

import tensorflow as tf
import random as rn

import os


#target_size = (581, 484) #fixed size for InceptionV3 architecture
#IM_WIDTH, IM_HEIGHT = 581, 484 #fixed size for ResNet50
target_size = (300, 300) #fixed size for InceptionV3 architecture
IM_WIDTH, IM_HEIGHT = 300, 300 #fixed size for ResNet50


#######################################################
### Fix random element in deep learning ###
#######################################################

print('Fix random element in deep learning.\n')
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(123)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True, device_count={'GPU' : 1})
#session_conf.gpu_options.per_process_gpu_memory_fraction = 0.4
#session_conf.gpu_options.allow_growth=True
session_conf.gpu_options.allow_growth=False
from keras import backend as K
tf.set_random_seed(101)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

#test check
#######################################################
#######################################################
#######################################################
def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis = -1))


def func_feature_ext(model, img, target_size):
    """Run model prediction on image
    Args:
        model: keras model
        img: PIL format image
        target_size: (w,h) tuple
    Returns:
        list of predicted labels and their probabilities 
    """
    if img.size != target_size:
        img = img.resize(target_size)

    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature_output = model.predict(x)
    
    #return feature_output[0][0][0] #for notop output
    return feature_output[0] #for flattten output

"""feature_extraction layer"""
def func_feature_layer(base_model):
    #x = base_model.layers[178].output #notop output , (2, 3, 2048), for ResNet50
    x = base_model.layers[344].output #notop output , (2, 3, 2048), for ResNet101
    x = AveragePooling2D(6)(x) #added

    #x = base_model.layers[349].output #notop output , (2, 3, 2048), for ResNet101 + SeNet
    #x = base_model.layers[431].output #notop output , (3, 4, 1024), for DenseNet121

    #x = base_model.layers[426].output #notop output , (15, 18, 1024), for DenseNet121
    #x = AveragePooling2D(4)(x) #added

    x= Flatten()(x)
        
    feature_model = Model(inputs=base_model.input, outputs=x)
    return feature_model


if __name__=="__main__":
    print('\n\n')
    print('-'*20)
    print('feature_ext.py program start.')
    #print('This program will extract feature vector by using resnet50 model.\n')
    #print('This program will extract feature vector by using densenet121 model.\n')
    print('This program will extract feature vector by using ResNet101 model.\n')
    #arg setting
    a = argparse.ArgumentParser()
    a.add_argument("--input_dir", default="./train_resnet_total",help="path to image")
    a.add_argument("--correct_class_switch", action='store_false', help="correct_label.pickle or not")
    a.add_argument("--transfer_model", default='transfer_resnet.model', help="transfer_model")
    
    args = a.parse_args()
    

    #load model and save
    #if(K.image_dim_ordering() == 'th'):
    if(K.common.image_dim_ordering() == 'th'):
        input_tensor = Input(shape=(3, IM_HEIGHT, IM_WIDTH))
    else:
        input_tensor = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))

    #base_model = ResNet50(input_tensor = input_tensor,weights='imagenet', include_top=False) #to avoid warning, if you concern
    base_model = load_model(args.transfer_model)
    print('loaded model file is {}.'.format(args.transfer_model))


    #print('load base model from ResNet50')
    #print('ResNet50 layer number is %d\n' % (len(base_model.layers)))
    ###plot_model(base_model, to_file="base_model.jpg", show_shapes=True)
    
    model = func_feature_layer(base_model)
    ###plot_model(model, to_file="feature_ext_model.jpg", show_shapes=True)
    ###model.save(args.feature_model)
    
    target_dir = args.input_dir
    print('target directory is {}.'.format(args.input_dir))

    # added, 2019/08/26
    output_dir = target_dir + '_raw_feature'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    ###########
    
    #parameter setting
    cnt_dr = 0 #correct class number
        
    feature_lists = []
    correct_class_lists = []
    ####################
    
    ###feature extraction###
    for root, dirs, files in os.walk(target_dir):
        print('Here, you have %d classes to feature\n' % len(dirs))
        dirs.sort()
        for dr in dirs:
            print('\n\n')
            print('-'*10)
            print('No %d class test start, %d/%d processing.' % (cnt_dr, cnt_dr+1, len(dirs)))

            print('No {0} class is {1}'.format(cnt_dr, dr))
            
            #file_list = glob.glob(os.path.join(root, dr + '/*.png'))
            file_list = glob.glob(os.path.join(root, dr + '/*.jpg'))

            file_list.sort()

            #added, 2019/08/26#
            dir_sub = output_dir + '/' + dr
            if not os.path.exists(dir_sub):
                os.makedirs(dir_sub)
            ###################
                    
            for item in file_list:
                img = Image.open(item)
                ext_feature = func_feature_ext(model, img, target_size)
                
                norm_feature_lists = ext_feature/np.linalg.norm(ext_feature)

                feature_lists.append(norm_feature_lists)

                #saving features with same format of nlp, 2019/08/26#
                output_feature_txt = dir_sub + '/' + os.path.basename(item)[:-4] + '.txt'
                np.savetxt(output_feature_txt, norm_feature_lists)
                ###################

              
            cnt_dr += 1
            
        break;
    ##############################
    
    
    '''
    ###save feature_lists file ###
    print('Saving feature_lists')
    output_feature_file = str(args.input_dir) + '.pickle'
    print('feature_lists file name is {0}'.format(output_feature_file))
    #feature_mat = np.array(feature_lists)
    with open(output_feature_file, mode='wb') as f:
        #pickle.dump(feature_mat, f)
        pickle.dump(feature_lists, f)
    print('Done')
    '''
    
    '''
    if args.correct_class_switch:
        print('Saving class_class_lists')
        output_correct_file = 'class_label.pickle'
        #class_label_mat = np.array(correct_class_lists)
        with open(output_correct_file, mode='wb') as f:
            #pickle.dump(class_label_mat, f)
            pickle.dump(correct_class_lists, f)
        print('Done')
    '''

    print('Saving end.\n')
    ##############################
    
   
    
    
    print('feature_ext.py program end.\n\n')
###############################################
