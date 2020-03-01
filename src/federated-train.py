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
import keras
from keras.applications import vgg16
from keras.preprocessing import image
from keras import optimizers, callbacks
from keras.models import Sequential, load_model, Model
from keras.layers import Flatten, Dense 
from keras.utils import to_categorical
from tensorflow.python.client import device_lib
import tensorflow as tf

from mlflow import log_metric, log_param, log_artifact

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

    #model.compile(loss='categorical_crossentropy',
    #      optimizer=optimizers.Adam(lr=learning_rate),
    #      metrics=['accuracy'])

    model.summary()
    return model
def get_acc(predictions, actuals):
    cnt = 0
    for pred, act in zip(predictions, actuals):
        if (pred == act):
            cnt += 1
    return float(cnt) / batch_size

if __name__ == '__main__':
    # Parameters for the run
    from config import *

    # Image sizes - Requirements of the CNN model
    img_reshape_size = (224,224)

    # Working directories
    dataset_dir_train = os.path.join('../Human-Action-Recognition-with-Keras/imgs/','train')

    # Load train dataset
    
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

    train_generator = train_datagen.flow_from_directory(
        dataset_dir_train,
        target_size = img_reshape_size,
        batch_size = batch_size,
        class_mode='categorical')
    
    # Create and compile model
    model = create_VGG16_model(n_layers_train=n_layers_train, learning_rate=learning_rate)
    nb_train_samples = 22424
    steps_per_epoch = np.ceil(nb_train_samples / batch_size)
    print("each epoch needs", steps_per_epoch)
    # model.fit_generator(train_generator, 
    #                     epochs = 5, 
    #                     steps_per_epoch = steps_per_epoch, 
    #                     verbose= 1)

    #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    #loss = losses.categorical_crossentropy(model.output, )
    #gvs = optimizer.compute_gradients(loss)
    # Save model and Exit
    
    #loss_fn = tf.keras.losses.categorical_crossentropy()
    with tf.Session() as sess:
        optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        step = 0
        grad_add = 0
        for epoch in range(n_epoch):
            for x, y in train_generator:
                if (step == steps_per_epoch):
                    break
                with tf.GradientTape() as tape:
                    logits = model(tf.convert_to_tensor(x))
                    loss_value = tf.keras.losses.categorical_crossentropy(y, logits)
                    calc_output = tf.argmax(logits, 1)
                    expect_output = tf.argmax(tf.convert_to_tensor(y), 1)
                    #acc, acc_op = tf.metrics.accuracy( expect_output ,calc_output)
                #print("loss", loss_value.eval())
                #print("batch output", calc_output.eval())
                #print("batch label", expect_output.eval())
                if step % 20 == 0:
                    print("Epoch ", epoch, " step", step,  " loss\t", tf.reduce_mean(loss_value).eval(), "acc\t", get_acc(calc_output.eval(), expect_output.eval()))
                
                grads = tf.gradients(loss_value, model.trainable_weights)
                if step % n_clients == 0:
                    grad_add = grads
                else:
                    grads += grads
                #print("grad", grads)
                if (step+1) % n_clients == 0:
                    print("update @", step)
                    optimizer.apply_gradients(zip(grad_add, model.trainable_weights))
                step += 1
            

    if not os.path.isdir('Model'):
        os.mkdir('Model')
    filename = 'VGG16_lr'+str(learning_rate)+'_train'+str(n_layers_train)+'_epochs'+str(n_epoch)+'_data_aug'+str(data_augmentation)+'.h5'

    model_file = os.path.join('Model', filename)
    model.save(model_file)
    log_artifact(model_file)
    print('File', filename, 'saved')
