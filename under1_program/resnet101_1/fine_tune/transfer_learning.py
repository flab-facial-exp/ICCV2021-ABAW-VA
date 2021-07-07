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
import tensorflow as tf
import random as rn
import pickle

from keras import __version__
#from keras.applications.inception_v3 import InceptionV3 
#from keras.applications.vgg19 import VGG19 
#from keras.applications.vgg16 import VGG16 
#from keras.applications.resnet import ResNet50
from keras.applications.resnet import ResNet101
#from keras.applications.densenet import DenseNet121
from keras.models import Model, load_model
from keras.layers import Conv2D, Dense, AveragePooling2D, GlobalAveragePooling2D, Input, Flatten, Dropout, BatchNormalization, MaxPooling2D, Multiply
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD, Adam
from keras.utils import plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler
#######################################################
#######################################################
#######################################################



#######################################################
### parameter setting ###
#######################################################
#IM_WIDTH, IM_HEIGHT = 214, 474 #fixed size for ResNet50
#IM_WIDTH, IM_HEIGHT = 581, 484 #fixed size for ResNet50, ResNet101, DenseNet121
IM_WIDTH, IM_HEIGHT = 300, 300 #fixed size for ResNet50, ResNet101, DenseNet121

#init_learn_rate = 5e-7 #initial learning rate


#######################################################
#######################################################
#######################################################


#######################################################
### Fix random element in deep learning ###
#######################################################

print('Fix random element in deep learning.\n')
import os
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(42)
rn.seed(123)

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True, device_count={'GPU' : 1})
session_conf.gpu_options.allow_growth=False
from keras import backend as K
tf.set_random_seed(101)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)
#######################################################
#######################################################
#######################################################


#######################################################
### function definition ###
#######################################################
'''
"""Get number of files by searching directory recursively"""
def get_nb_files(directory):
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt
'''

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis = -1))

def step_decay(epoch, init_learn_rate):
    #x=init_learn_rate
    x=5e-7

    if epoch >= 200 : x=1e-6
    if epoch >= 300 : x=5e-7
    if epoch >= 500 : x=1e-7

    print('learning_rate={}'.format(x))

    return x


"""Freeze all layers and compile the model"""
def setup_to_transfer_learn(transfer_model, base_model, init_learn_rate):
    for layer in transfer_model.layers:
        layer.trainable = False
        
    '''
    #for ResNet50
    transfer_model.layers[164].trainable = True #third from the last conv2D
    transfer_model.layers[165].trainable = True #
    transfer_model.layers[166].trainable = True #
    transfer_model.layers[167].trainable = True #second from the last conv2D
    transfer_model.layers[168].trainable = True #
    transfer_model.layers[169].trainable = True #
    transfer_model.layers[170].trainable = True #final conv2D
    transfer_model.layers[171].trainable = True
    transfer_model.layers[172].trainable = True
    transfer_model.layers[173].trainable = True
    transfer_model.layers[174].trainable = True
    transfer_model.layers[175].trainable = True
    transfer_model.layers[176].trainable = True
    transfer_model.layers[177].trainable = True
    transfer_model.layers[178].trainable = True
    transfer_model.layers[179].trainable = True
    transfer_model.layers[180].trainable = True
    transfer_model.layers[181].trainable = True
    transfer_model.layers[182].trainable = True
    '''

    #for ResNet101
    transfer_model.layers[335].trainable = True #
    transfer_model.layers[336].trainable = True #
    transfer_model.layers[337].trainable = True #
    transfer_model.layers[338].trainable = True #second from the last conv2D
    transfer_model.layers[339].trainable = True #
    transfer_model.layers[340].trainable = True #
    transfer_model.layers[341].trainable = True #final conv2D
    transfer_model.layers[342].trainable = True
    transfer_model.layers[343].trainable = True
    transfer_model.layers[344].trainable = True
    transfer_model.layers[345].trainable = True
    transfer_model.layers[346].trainable = True
    transfer_model.layers[347].trainable = True
    transfer_model.layers[348].trainable = True
    transfer_model.layers[349].trainable = True
    transfer_model.layers[350].trainable = True
    transfer_model.layers[351].trainable = True
    transfer_model.layers[352].trainable = True
    transfer_model.layers[353].trainable = True

    '''
    #for DenseNet121
    #transfer_model.layers[417].trainable = True #
    #transfer_model.layers[418].trainable = True #
    #transfer_model.layers[419].trainable = True #
    #transfer_model.layers[420].trainable = True #second from the last conv2D
    #transfer_model.layers[421].trainable = True #
    #transfer_model.layers[422].trainable = True #
    #transfer_model.layers[423].trainable = True #final conv2D
    #transfer_model.layers[424].trainable = True
    #transfer_model.layers[425].trainable = True
    #transfer_model.layers[426].trainable = True
    transfer_model.layers[427].trainable = True
    transfer_model.layers[428].trainable = True
    transfer_model.layers[429].trainable = True
    transfer_model.layers[430].trainable = True
    transfer_model.layers[431].trainable = True
    transfer_model.layers[432].trainable = True
    transfer_model.layers[433].trainable = True
    transfer_model.layers[434].trainable = True
    transfer_model.layers[435].trainable = True
    '''

    print('init_learn_rate is {}'.format(init_learn_rate))

    #transfer_model.compile(optimizer=Adam(lr=init_learn_rate), loss='categorical_crossentropy', metrics=['accuracy']) #good learning
    #transfer_model.compile(optimizer=Adam(lr=init_learn_rate), loss=root_mean_squared_error) #good learning
    transfer_model.compile(optimizer=Adam(lr=init_learn_rate), loss='mean_squared_error') #good learning
    #transfer_model.compile(optimizer=SGD(lr=1e-3, decay=5e-4, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    #transfer_model.compile(optimizer=SGD(lr=1e-3, decay=5e-4, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    #transfer_model.compile(optimizer=SGD(lr=init_learn_rate, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])



"""Add last layer to the convnet"""
#def add_new_last_layer(base_model, nb_classes):
def add_new_last_layer(base_model):
    x = base_model.output
    ####x = Flatten()(x)
    ####x = Dense(256, activation='relu')(x)
    ####x = Dropout(0.5)(x)
    ####predictions = Dense(nb_classes, activation='softmax')(x)

    ####input_x = base_model.layers[173].output #activation output, (none, 15, 7, 2048), for ResNet50
    #input_x = base_model.layers[344].output #activation output, (none, 16, 19, 2048), for ResNet101
    ####input_x = base_model.layers[426].output #activation output, (none, 15, 18, 1024), for DenseNet121

    # Squeeze
    #x = GlobalAveragePooling2D()(input_x)
    #### Excitation
    ####p_senet_comp = 4 #for densenet121
    #p_senet_comp = 8 #Resnet
    #x = Dense(2048//p_senet_comp, activation="relu")(x) #for ResNet
    #x = Dense(2048, activation="sigmoid")(x)            #for ResNet
    ####x = Dense(1024//p_senet_comp, activation="relu")(x)  #for DenseNet121
    ####x = Dense(1024, activation="sigmoid")(x)             #for DenseNet121
    #x = Multiply()([input_x,x])
    ########################

    x = AveragePooling2D(6)(x) #resnet
    ####x = AveragePooling2D(4)(x) #densenet
    x = Flatten()(x)
    #x = Dense(4096, activation='relu')(x)
    #x = Dropout(0.5)(x)
    #x = Dense(1024, activation='relu')(x)
    #x = Dropout(0.5)(x)
    ####predictions = Dense(nb_classes, activation='softmax')(x)
    #predictions = Dense(1)(x)

    #transfer_model = Model(inputs=base_model.input, outputs=predictions)
    transfer_model = Model(inputs=base_model.input, outputs=x)
    return transfer_model
    

"""Use transfer learning and fine-tuning to train a network on a new dataset"""
def train(args):
    print('train initial process start\n')
    #train_img = 'train/' 
    #train_img = 'train_ft/' 
    train_img = './using_dataset/train/img/' 
    #train_img = 'train_img/' 
    #validation_img = 'valid_ft/'
    validation_img = './using_dataset/test/img/'
    #validation_img = 'valid_img/'
    nb_epoch = int(args.nb_epoch)
    arg_batch_size = int(args.batch_size)
    arg_steps_per_epoch = int(args.steps_per_epoch)
    arg_validation_steps = int(args.validation_steps)
    arg_retrain_epoch = int(args.retrain_epoch)
    arg_pre_epoch = int(args.pre_epoch)
    #nb_train_samples = get_nb_files(train_img)
    #nb_classes = len(glob.glob(train_img + "/*"))
    learning_rate = float(args.learning_rate)

    ### added 2021/06/10 ###
    #train_image_list = []
    train_valence_list = []
    train_arousal_list = []

    target_dir = train_img
    for root, dirs, files in os.walk(target_dir):
            for dr in dirs:
                #file_list = glob.glob(os.path.join(root, dr + '/*.jpg'))
                #file_list.sort()
                                                
                '''
                for item in file_list:
                    #print('dealing file is {}'.format(item))
                    image = Image.open(item)
                    image = image.convert('RGB')
                    image = image.resize((IM_WIDTH, IM_HEIGHT))
                    image_data = np.asarray(image)
                    train_image_list.append(image_data)
                '''

                va_txt_file_valence = './using_dataset/annotation_rev/' + str(dr) + '_valence.txt'
                with open(va_txt_file_valence, 'r', encoding='utf-8') as f:
                    for i in f.read().splitlines():
                        train_valence_list.append(i)

                va_txt_file_arousal = './using_dataset/annotation_rev/' + str(dr) + '_arousal.txt'
                with open(va_txt_file_arousal, 'r', encoding='utf-8') as f:
                    for i in f.read().splitlines():
                        train_arousal_list.append(i)

            break;

    #train_image_list = np.array(train_image_list)
    train_valence_list = np.array(train_valence_list)
    train_arousal_list = np.array(train_arousal_list)

    #X_train = train_image_list.astype('float32') / 255
    Y_train_valence = train_valence_list
    Y_train_arousal = train_arousal_list

    output_class_valence_file = 'train_class_valence_label' + '.pickle'
    output_class_arousal_file = 'train_class_arousal_label' + '.pickle'
    print('train_class_valence_label name is {0}'.format(output_class_valence_file))
    print('train_class_arousal_label name is {0}'.format(output_class_arousal_file))

    with open(output_class_valence_file, mode='wb') as f:
        pickle.dump(Y_train_valence, f)

    with open(output_class_arousal_file, mode='wb') as f:
        pickle.dump(Y_train_arousal, f)
    #####################################################

    #####################################################
    #valid_image_list = []
    valid_valence_list = []
    valid_arousal_list = []

    target_dir = validation_img
    for root, dirs, files in os.walk(target_dir):
            for dr in dirs:
                #file_list = glob.glob(os.path.join(root, dr + '/*.jpg'))
                #file_list.sort()
                                                
                '''
                for item in file_list:
                    #print('dealing file is {}'.format(item))
                    image = Image.open(item)
                    image = image.convert('RGB')
                    image = image.resize((IM_WIDTH, IM_HEIGHT))
                    image_data = np.asarray(image)
                    valid_image_list.append(image_data)
                '''

                va_txt_file_valence = './using_dataset/annotation_rev/' + str(dr) + '_valence.txt'
                with open(va_txt_file_valence, 'r', encoding='utf-8') as f:
                    for i in f.read().splitlines():
                        valid_valence_list.append(i)

                va_txt_file_arousal = './using_dataset/annotation_rev/' + str(dr) + '_arousal.txt'
                with open(va_txt_file_arousal, 'r', encoding='utf-8') as f:
                    for i in f.read().splitlines():
                        valid_arousal_list.append(i)
            break;

    #valid_image_list = np.array(valid_image_list)
    valid_valence_list = np.array(valid_valence_list)
    valid_arousal_list = np.array(valid_arousal_list)

    #X_test = valid_image_list.astype('float32') / 255
    Y_test_valence = valid_valence_list
    Y_test_arousal = valid_arousal_list

    output_class_valence_file = 'test_class_valence_label' + '.pickle'
    output_class_arousal_file = 'test_class_arousal_label' + '.pickle'
    print('test_class_valence_label name is {0}'.format(output_class_valence_file))
    print('test_class_arousal_label name is {0}'.format(output_class_arousal_file))

    with open(output_class_valence_file, mode='wb') as f:
        pickle.dump(Y_test_valence, f)

    with open(output_class_arousal_file, mode='wb') as f:
        pickle.dump(Y_test_arousal, f)
    #####################################################



    ########################
            
    # data prep
    
    '''
    train_datagen = ImageDataGenerator(
        #rotation_range=40,
        #rotation_range=20,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #brightness_range=[0.8,1.0],
        #horizontal_flip=True,
        #featurewise_center=True,
        #featurewise_std_normalization=True,
        #zca_whitening=True,
        fill_mode='nearest')
    
    validation_datagen = ImageDataGenerator(
        #rotation_range=40,
        #width_shift_range=0.2,
        #height_shift_range=0.2,
        #rescale=1./255,
        #shear_range=0.2,
        #zoom_range=0.2,
        #horizontal_flip=True,
        fill_mode='nearest')
    '''

    '''
    print('train data generation')
    train_generator = train_datagen.flow_from_directory(
			train_img,
			target_size=(IM_HEIGHT, IM_WIDTH),
			batch_size=arg_batch_size,
			class_mode='categorical'
			)
    print('train_generator.class_indices')
    print(train_generator.class_indices)
    '''
    
    
    '''
    ###class weight ###
    class_weight_calc={}
    total_file_cnt = []
    
    for root, dirs, files in os.walk(train_img):
        for dr in dirs:
            #file_list = glob.glob(os.path.join(root, dr + '/*.png'))
            file_list = glob.glob(os.path.join(root, dr + '/*.jpg'))
            file_list.sort()
            total_file_cnt.append(len(file_list))
        break;
    print('total_file_cnt_list')
    print(total_file_cnt)
    
    l_cnt = 0
    for root, dirs, files in os.walk(train_img):
        for dr in dirs:
            #file_list = glob.glob(os.path.join(root, dr + '/*.png'))
            file_list = glob.glob(os.path.join(root, dr + '/*.jpg'))
            file_list.sort()
            class_file_cnt = len(file_list)
            
            class_ratio = (max(total_file_cnt)/class_file_cnt - 1.0 ) * 0.7 + 1.0

            if (class_ratio > 10.0):
                class_ratio = 10.0

            
            class_weight_calc[l_cnt] = class_ratio
            l_cnt +=1
        break;
    print('class_weight_calc')
    print(class_weight_calc)
    print('\n\n')
    ###########################
    '''
    
    '''
    print('validation data generation')
    validation_generator = validation_datagen.flow_from_directory(
			validation_img,
          target_size=(IM_HEIGHT, IM_WIDTH),
			batch_size=arg_batch_size,
			class_mode='categorical'
			)
    print('\n\n')
    '''
    
    
    #if(K.image_dim_ordering() == 'th'):
    if(K.common.image_dim_ordering() == 'th'):
        input_tensor = Input(shape=(3, IM_HEIGHT, IM_WIDTH))
    else:
        input_tensor = Input(shape=(IM_HEIGHT, IM_WIDTH, 3))
    
    print('train initial process end\n')
   
        
    
    print('------------------------------------------------------------')    
    print('load {0} as base_model start\n'.format(args.input_model))
    
    if (arg_retrain_epoch == 0): #train from start
        #base_model = ResNet50(input_tensor = input_tensor,weights='imagenet', include_top=False) #to avoid warning, if you concern
        base_model = ResNet101(input_tensor = input_tensor,weights='imagenet', include_top=False) #to avoid warning, if you concern
        #base_model = DenseNet121(input_tensor = input_tensor,weights='imagenet', include_top=False) #to avoid warning, if you concern
        #print('load base model from ResNet50')
        print('load base model from ResNet101')
        #print('load base model from DenseNet121')
        print('%s layer number is %d\n' % (str(args.input_model), len(base_model.layers)))
        
        print('make transfer_model start\n')
        #transfer_model = add_new_last_layer(base_model, nb_classes)
        transfer_model = add_new_last_layer(base_model)
        plot_model(transfer_model, to_file="transfer_model.jpg", show_shapes=True)
        print('Created transfer_model layer number is %d\n' % (len(transfer_model.layers)))
        print('make transfer_model end\n')

        # transfer learning setup
        print('transfer_model layer number is %d\n' % len(transfer_model.layers))
        #print('Setup transfer learning start\n')
        #setup_to_transfer_learn(transfer_model, base_model, learning_rate)
        #print('Setup transfer learning end\n')
        ##################
    else: #train from start
        filepath_model = 'transfer_model_epoch_' + '%05d' % arg_retrain_epoch + '_retrain' + str(arg_pre_epoch) + '.model'
        transfer_model = load_model(filepath_model)
        print('loaded trained model name is {}' .format(filepath_model))
        print('transfer_model layer number is %d\n' % len(transfer_model.layers))
    
    print('load {0} as base_model end\n'.format(args.input_model))
    print('------------------------------------------------------------')    
    #####################
    
    
    '''
    # transfer learning
    check_filepath = './transfer_model_epoch_{epoch:05d}_retrain%d.model' % (arg_retrain_epoch + arg_pre_epoch)
    checkpoint = ModelCheckpoint(filepath=check_filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=50)
    #checkpoint = ModelCheckpoint(filepath=check_filepath, monitor='val_loss', verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1)

    #even valid dataset
    #early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, mode='auto') #good result 52.17% ,61.34%, 70.588%

    early_stopping = EarlyStopping(monitor='val_loss', patience=20, verbose=1, mode='auto') #

    #start_rate = 1
    #stop_rate  = 1e-10
    #learning_rate = np.linspace(start_rate, stop_rate, nb_epoch)
    #print('learning_rate = {}'.format(learning_rate))
    #lr_cb = LearningRateScheduler(lambda epoch : float(learning_rate[epoch]))
    lr_cb = LearningRateScheduler(step_decay)

    print('Transfer learning start\n')
    #history_tl = transfer_model.fit_generator(train_generator,
    history_tl = transfer_model.fit_generator(train_datagen.flow(X_train, Y_train, batch_size = arg_batch_size),
                                   #samples_per_epoch=arg_samples_per_epoch,
                                   steps_per_epoch=arg_steps_per_epoch,
                                   #nb_epoch=nb_epoch,
                                   epochs=nb_epoch,
                                   #validation_data=validation_generator,
                                   validation_data = (X_test, Y_test),
                                   #class_weight=class_weight_calc,
                                   #callbacks=[checkpoint, early_stopping, lr_cb],
                                   #callbacks=[checkpoint, early_stopping],
                                   #callbacks=[checkpoint],
                                   #callbacks=[lr_cb],
                                   #nb_val_samples=arg_nb_val_samples) 
                                   #validation_steps=arg_validation_steps) 
                                   shuffle = True, 
                                   verbose = 1)
    print('history_tl.history\n')
    print(history_tl.history)
    print('Transfer learning end\n')
    '''
    
    print('Transfer model saving start\n')
    transfer_model.save(args.output_model_file)
    print('Transfer model saving end\n')
    
    '''
    print('Plot learning process if you use --plot.\n')
    if args.plot:
        plot_training(history_tl, args)
    '''
        
        
def plot_training(history, args):
    #acc = history.history['acc']
    #acc = history.history['accuracy']
    #val_acc = history.history['val_acc']
    #val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    #epochs = range(1, len(acc)+1)

    arg_epoch = int(args.nb_epoch)
    
    '''
    plt.plot(epochs, acc, 'g', label='training' )
    plt.plot(epochs, val_acc, 'r', label='validation')
    plt.title('Training and validation accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.xlim(1, arg_epoch)
    plt.xticks(range(1, arg_epoch+1, 5))
    plt.legend()
    plt.savefig('accuracy.png')
    
    plt.figure()
    '''

    #plt.plot(epochs, loss, 'g', label='training')
    #plt.plot(epochs, val_loss, 'r', label='validation')
    plt.plot(loss, 'g', label='training')
    plt.plot(val_loss, 'r', label='validation')
    plt.title('Training and validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.xlim(1, arg_epoch)
    plt.xticks(range(1, arg_epoch+1, 5))
    plt.legend()
    plt.savefig('loss.png')

if __name__=="__main__":
    
    
    print('Program transfer.py start\n\n')
    a = argparse.ArgumentParser()
    a.add_argument("--nb_epoch", default=3)
    a.add_argument("--batch_size", default=32)
    a.add_argument("--steps_per_epoch", default=1280)
    a.add_argument("--validation_steps", default=128)
    a.add_argument("--retrain_epoch", default=0)
    a.add_argument("--pre_epoch", default=0)
    a.add_argument("--plot", action="store_true")
    a.add_argument("--input_model", default="inceptionv3")
    a.add_argument("--output_model_file", default="transfer.model")
    a.add_argument("--learning_rate", default=1e-7, type=float)
    args = a.parse_args()

    print('setting arguments are listed below.\n')
    print('nb_epoch=%s' % args.nb_epoch)
    print('batch_size=%s' % args.batch_size)
    print('steps_per_epoch=%s' % args.steps_per_epoch)
    print('validation_steps=%s' % args.validation_steps)
    print('retrain_epoch=%s' % args.retrain_epoch)
    print('pre_epoch=%s' % args.pre_epoch)
    print('plot=%s' % args.plot)
    print('input_model=%s' % args.input_model)
    print('output_model_file=%s\n\n' % args.output_model_file)
    print('learning_rate={}\n\n' .format(args.learning_rate))
    
    train(args)
    print('Program transfer.py End\n')
