# TensorFlowPlayGround
Tensorflow codes written as part of Advanced Machine Learning Course Work

### SimpleTensors.py

This code illustrates the use the symbolic constants and variables in tensorflow, 
interactive sessions which feed the data to the variables at runtime using two different approaches

### LinearRegressorEstimator
1. LinearRegressorEstimator.py
2. dataset.csv

A Linear Regressor is trained with a linear dataset using Tensorflow.
This code is mainly aimed to illustrates the use of:
- Dataset API
  - Provides an efficient input pipeline
  - Involves creating dataset instance from the data, creating an iterator and consuming data
- Pre-made Estimator
  - Provide a much higher conceptual level than the base TensorFlow APIs
  - Abstracts the creation of computational graph or sessions since Estimators handle all 

### EMNIST_DNNClassifier
1. DNN.py
2. Report.txt

A DNN Classifier which is a Premade Estimator is used to predict the class of EMNIST images.

Dataset - https://www.nist.gov/itl/iad/image-group/emnist-dataset

This code illustrates the following :
- Reading data directly from the ubyte file
- Data Preprocessing
- Creating an instance of Dataset API
- Using Shuffle and Pre-fetch functinalities of TensorFlow
- Using Pre-Made DNN Estimator to train and test
- Reporting metrics per class
- Reporting overall metrics

**Referred to http://cjalmeida.net/post/tensorflow-mnist/**

### Selfi-Dataset
DataSet - http://crcv.ucf.edu/data/Selfie/
- Selfie dataset contains 46,836 selfie images annotated with 36 different attributes divided into several categories as follows. 
- Gender: is female. Age: baby, child, teenager, youth, middle age, senior. Race: white, black, asian. Face shape: oval, round, heart. Facial gestures: smiling, frowning, mouth open, tongue out, duck face. Hair color: black, blond, brown, red. Hair shape: curly, straight, braid. Accessories: glasses, sunglasses, lipstick, hat, earphone. Misc.: showing cellphone, using mirror, having braces, partial face. Lighting condition: harsh, dim. 
1. multiclass.ipynb
  - This file contains the code to:
    - analysis of datapoints and augment images with lower distribution
    - Resnet50 architecture used as a base and last few layers added to it and trained
    - made to predict 36 attributes
2. multitask.ipynb
  - This file has two parts:
    - 2 output heads
      - input images and predict popularity score along with 7 attributes 
    - 2 input heads
      - input images and popularity scores, predict 36 attributes
   
    
3. popularity_score.ipynb
  - This file constains predicting the class of an image (great,avg,poor - based on popularity score- using simple fraction ethod to group classes.)  
4. popularity_score_qcut.ipynb
  - This file constains predicting the class of an image (great,avg,poor - based on popularity score- using q-cut approach to group classes.) 
5. analysis.ipynb



