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

### Explanation of my work

# Setup

This project is optimized to run on a GPU. I use the Google Cloud Platform with a Tesla K80 for this project

* [Download the data](https://www.kaggle.com/c/state-farm-distracted-driver-detection/data)

* Create the environment:
    `conda env create -f myenv.yml `

* Activate the environment:
    - On Windows, in your Anaconda Prompt, run 
    `activate myenv`
    - On macOS and Linux, in your Terminal Window, run 
    `source activate myenv`

    - To update:
    `conda env update -f myenv.yml`

* Run the notebook `main.ipynb`
