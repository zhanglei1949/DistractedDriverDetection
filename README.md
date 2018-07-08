# DistractedDriverDetection
Deep Learning and Computer Vision to detect distracted drivers

The goal is to predict what the driver is doing based on pictures

This project is from a [Kaggle competition : State Farm Distracted Driver Detection
](https://www.kaggle.com/c/state-farm-distracted-driver-detection)


![alt text](https://github.com/cyril-p/DistractedDriverDetection/blob/master/misc/output_DEb8oT.gif)

There are 10 classes to predict:
* c0: safe driving
* c1: texting - right
* c2: talking on the phone - right
* c3: texting - left
* c4: talking on the phone - left
* c5: operating the radio
* c6: drinking
* c7: reaching behind
* c8: hair and makeup
* c9: talking to passenger

## Explanation of my work

I used a pre-trained VGG-16 Convolutional Neural Network as a base layer. I then removed the last layer (=top layer) and added a Dense layer with a softmax to output the classification. The optimization algorithm is Adam with a small learning rate: 0.00005. 


The used the weights trained VGG-16 on the image-net dataset for the initializatzion. All the layers beside the last 8 layers were frozen (3 Conv layers and 3 Fully Connected layers + 1 max pool + 1 flatten layer). I didn't do data augmentation. I fine tuned the model using the training set and a validation set (30% split)

I used the Keras library as much as possible in this project as it allows to prototype quickly and surely. I used the dataflow and datagen functions to load the file and preprocess them while feeding them by batches directly to the model.



9 epochs only were used. The learning_rate, number of epochs and numbers of layers to freeze values were determined by several experiments, not shown here. An early stopping was used.

The framework was run on a Google Cloud Virtual Machine with a Tesla K80 GPU.

### Result:
* Public Leaderboard: 0.71 (top 25%)
* Private Leaderboard 0.62 (top 20%)

# Setup

This project is optimized to run on a GPU. I use the Google Cloud Platform with a Tesla K80 for this project

* [Download the data](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data) and store under the folder `Data/`

* Create the environment:
    `conda env create -f myenv.yml `

* Activate the environment:
    - On Windows, in your Anaconda Prompt, run 
    `activate myenv`
    - On macOS and Linux, in your Terminal Window, run 
    `source activate myenv`

    - To update:
    `conda env update -f myenv.yml`
    
### Train and predict

#### Using MLFlow
* Specify parameters in the file `src/config.py`
* Run the project to train and predict
`mlflow run [your project directory]`

#### Using Pythons scripts
* Specify parameters in the file `src/config.py`
* Run the script to train.  It will save a trained model under the folder `Model/`.
`python src/train.py `

* Run the script to predict
`python src/predict.py [Model/your model name]`

#### Using Jupyter notebook
* Specify parameters in the first cell
* Run the notebook `main_training`. It will save a trained model under the folder `Model/`. It will also use the script `predict.py` and submit the .csv file to Kaggle directly.
