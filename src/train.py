#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
*Cyril Pecoraro*

https://github.com/cyril-p
"""
import sys
import pandas as pd
import cv2
import os 
import glob
import gc
from joblib import Parallel, delayed

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpli
import warnings
warnings.filterwarnings("ignore", module="matplotlib")
pd.options.mode.chained_assignment = None

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss

from keras.applications import vgg16
from keras.preprocessing import image
from keras import optimizers, callbacks
from keras.models import Sequential, load_model, Model
from keras.layers import Flatten, Dense 
from keras.utils import to_categorical
from tensorflow.python.client import device_lib
import tensorflow as tf

from mlflow import log_metric, log_param, log_artifact

def preprocess_image(img):
    """Preprocess an image according to VGG16 imagenet requirement (mean substraction)
    Args:
    - img: image in BGR format with shape: [w,h,channel]
    Return:
    - img: preprocessed image
    """   
    img = img.astype(np.float16)
    img[:, :, 0] -= 103.939
    img[:, :, 1] -= 116.779
    img[:, :, 2] -= 123.68
    
    return img

def load_image(img_file, img_reshape_size):
    """Load an image
    Args:
    - img_file: image file
    - img_reshape_size: shape(w,h) to resize the image
    Return:
    - img: image resized and preprocessed according to VGG16 standards
    """   
    img = cv2.imread(img_file)
    img = cv2.resize(img, img_reshape_size)
    img = preprocess_image(img)
    
    return img

def load_train_dataset(dataset_dir, img_reshape_size, nprocs=10):
    """Load the images located in the main folder dataset_dir Each class is in a separate subfolder
    Args:
    - dataset_dir: path to the directory containing subdirectories of images
    - img_reshape_size: shape(w,h) to resize the image
    - nprocs:Number of processors to use
    Return:
    - X: numpy array with each image data as a row
    - y: numpy array with each class as an integer for each image
    """
    X = []
    y = []
    # Train dataset
    for i in range(10):
        path = os.path.join(dataset_dir, 'c'+str(i),'*.jpg')
        files = glob.glob(path)

        X.extend(Parallel(n_jobs=nprocs)(delayed(load_image)(im_file, img_reshape_size) for im_file in files))
        y.extend([i]*len(files))
        print('folder train/c'+str(i), 'loaded')

    X = np.asarray(X, dtype=np.float16)
    y = np.asarray(y)
    return X, y

def create_VGG16_model(n_classes=10, n_layers_train=2, learning_rate=0.0001):
    """Load the images located in the main folder dataset_dir Each class is in a separate subfolder
    Args:
    - n_classes: number of classes to predict for the classifier
    - n_layers_train: number of last layers to train. The other layers will be frozen.
    Return:
    - model: Keras model
    """
    #Load the VGG model
    vgg16_base = vgg16.VGG16(weights='imagenet', include_top=True, input_shape=(224,224,3))    

    # Remove top layer and connect custom dense layer
    fc2 = vgg16_base.get_layer('fc2').output
    mypredictions = Dense(n_classes, activation='softmax', name='mypredictions')(fc2)
    model = Model(inputs=vgg16_base.input, outputs=mypredictions)
    
    model.summary()
    # Freeze the layers except the last n_layers_train layers
    for layer in model.layers[:-n_layers_train]:
        layer.trainable = False
    for layer in model.layers[:]:
        print(layer, layer.trainable)    

    model.compile(loss='categorical_crossentropy',
          optimizer=optimizers.Adam(lr=learning_rate),
          metrics=['accuracy'])

    model.summary()
    return model

class LogMlFlowMetrics(callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        log_metric("train_loss", logs.get('loss'))
        log_metric("train_acc", logs.get('acc'))
        log_metric("val_loss", logs.get('val_loss'))
        log_metric("val_acc", logs.get('val_acc'))

if __name__ == '__main__':
    # Parameters for the run
    from config import *

    # Using the MLFlow tracking 
    log_param("batch_size", batch_size)
    log_param("n_epoch", n_epoch)
    log_param("learning_rate", learning_rate)
    log_param("n_layers_train", n_layers_train)
    log_param("data_augmentation", data_augmentation)

    # Image sizes - Requirements of the CNN model
    img_reshape_size = (224,224)

    # Working directories
    dataset_dir_train = os.path.join('Data','train')

    # Load train dataset
    print('Loading dataset train...')
    X_train_, y_train_ = load_train_dataset(dataset_dir_train, img_reshape_size) 
    
    # Creation of a validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train_, y_train_, 
                                                      test_size=0.3, 
                                                      random_state=35)

    # One hot encoding of the classes
    y_train = to_categorical(y_train)
    y_val = to_categorical(y_val)

    # Shapes 
    print('X_train shape:', X_train.shape)
    print('y_train shape:', y_train.shape)#
    print('X_val shape:', X_val.shape)
    print('y_val shape:', y_val.shape)

    # Image Data Generator for preprocessing and data augmentation
    if(data_augmentation==True):
        train_datagen = image.ImageDataGenerator(preprocessing_function=preprocess_input_image,
                                                width_shift_range=0.1,
                                                height_shift_range=0.1,
                                                shear_range=0.1,
                                                rotation_range=8,
                                                fill_mode='nearest'
                                                )
    else:
        train_datagen = image.ImageDataGenerator()

    train_generator = train_datagen.flow(x=X_train, y=y_train,
                                batch_size=batch_size,
                                shuffle=False,
                                seed=42)
    
    # Create and compile model
    model = create_VGG16_model(n_layers_train=n_layers_train, learning_rate=learning_rate)

    # Tensorboard logs
    tensorboard_log_dir='VGG16_lr'+str(learning_rate)+'_train'+str(n_layers_train)+'_epochs'+str(n_epoch)+'_data_aug'+str(data_augmentation)

    tbCallBack = callbacks.TensorBoard(log_dir=os.path.join('Graph', tensorboard_log_dir),
                                       histogram_freq=0, 
                                       write_graph=True, 
                                       write_images=False)

    # Fit the model on batches with real-time data augmentation:
    model_history = model.fit_generator(train_generator,
                                        validation_data=(X_val, y_val),
                                        shuffle=True,
                                        epochs=n_epoch,
                                        steps_per_epoch=np.ceil(X_train.shape[0]//batch_size),
                                        callbacks=[tbCallBack, LogMlFlowMetrics()],
                                        verbose=1,
                                        use_multiprocessing=True)
    
    # Save model and Exit
    if not os.path.isdir('Model'):
        os.mkdir('Model')
    filename = 'VGG16_lr'+str(learning_rate)+'_train'+str(n_layers_train)+'_epochs'+str(n_epoch)+'_data_aug'+str(data_augmentation)+'.h5'

    model_file = os.path.join('Model', filename)
    model.save(model_file)
    log_artifact(model_file)
    print('File', filename, 'saved')
