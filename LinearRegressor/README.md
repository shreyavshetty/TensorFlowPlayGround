This folder contains two files:

1. LinearRegressorEstimator.py
2. dataset.csv

A Linear Regressor is trained with a linear dataset using Tensorflow. This code is mainly aimed to illustrates the use of:

- Dataset API
   - Provides an efficient input pipeline
   - Involves creating dataset instance from the data, creating an iterator and consuming data
-Pre-made Estimator
   - Provide a much higher conceptual level than the base TensorFlow APIs
   - Abstracts the creation of computational graph or sessions since Estimators handle all
