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

**Referred to http://cjalmeida.net/post/tensorflow-mnist/**
- Data Preprocessing
- Creating an instance of Dataset API
- Using Shuffle and Pre-fetch functinalities of TensorFlow
- Using Pre-Made DNN Estimator to train and test
- Reporting metrics per class
- Reporting overall metrics



