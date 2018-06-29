#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
*Cyril Pecoraro*

https://github.com/cyril-p
"""
import pandas as pd
import cv2
import os 
import glob
import datetime
import sys
from joblib import Parallel, delayed
import numpy as np
from sklearn.metrics import log_loss
from keras.models import load_model
from keras.applications import vgg16

def load_image(img_file, img_reshape_size):
    """Load an image
    Args:
    - img_file: image file
    - img_reshape_size: shape(w,h) to resize the image
    Return:
    - img: openCV image
    """   
    img = cv2.imread(img_file)
    img = cv2.resize(img, img_reshape_size)
    # Preprocess input according to VGG16
    vgg16.preprocess_input(img)
    return img

def load_test_dataset(dataset_dir, img_reshape_size, nprocs=10):
    """Load the images located in the main folder dataset_dir Each class is in a separate subfolder
    Args:
    - dataset_dir: path to the directory containing subdirectories of images
    - img_reshape_size: shape(w,h) to resize the image
    - nprocs:Number of processors to use
    Return:
    - X: numpy array with each image data as a row
    - y: numpy array with each class as an integer for each image
    - X_id: numpy array containing the name of the file corresponding to each row of y and X
    """
    X = []
    y = []
    X_id = []
    
    # Test dataset
    path = os.path.join(dataset_dir, '*.jpg')
    file_paths = glob.glob(path)
    file_names = os.listdir(dataset_dir)

    X_id.extend([im_file] for im_file in file_names)
    X.extend(Parallel(n_jobs=nprocs)(delayed(load_image)(im_file, img_reshape_size) for im_file in file_paths))
    print('folder test loaded')

    X = np.array(X)
    y = np.array(y)
    X_id = np.array(X_id).flatten()
    return X, y, X_id

def create_submission(predictions, test_id, info):
    """Create a .csv file for a submission on Kaggle platform. File will be saved in /Output
    Args:
    - predictions: numpy array of the prediction
    - test_id: nupy vector containing file name associated with each row of predictions
    - info: String with prefix file name. Enter params value for instance
    Return:
    /
    """    
    result = pd.DataFrame(predictions, columns=['c0', 'c1', 'c2', 'c3','c4', 
                                                'c5', 'c6', 'c7','c8', 'c9'])
    result.loc[:, 'img'] = pd.Series(test_id, index=result.index)
    
    now = datetime.datetime.now()
    if not os.path.isdir('Output'):
        os.mkdir('Output')
        
    # extract only name of the model
    info = info.split('/')[1].split('.h5')[0]
    
    filename = info + '_' + str(now.strftime("%Y-%m-%d-%H-%M"))
    sub_file = os.path.join('Output', 'submission-' + filename + '.csv')
    result.to_csv(sub_file, index=False)
    print('File', filename, 'saved')
    

if __name__ == '__main__':
    model_info = sys.argv[1]
    print('Working with model', model_info)
    
    # Working directories#  
    dataset_dir = 'Data'
    dataset_dir_train = os.path.join(dataset_dir, 'train')

    # Image sizes - Requirements of the CNN model
    img_reshape_size = (224,224)
    
    # Load test dataset
    dataset_dir_test = os.path.join(dataset_dir, 'test')
    print('Load dataset test')
    X_test, y_test, X_test_id = load_test_dataset(dataset_dir_test, img_reshape_size)

    # Shapes 
    print('X_test shape:', X_test.shape)
    
    # Predict
    model = load_model(model_info)
    y_pred_test = model.predict(X_test, verbose=1)
    create_submission(y_pred_test, X_test_id, model_info)
