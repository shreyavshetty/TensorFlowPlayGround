'''
Importing Libraries 
'''
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
'''
Loading Dataset
'''
dataset = pd.read_csv('ds1.csv')
'''
Data Preprocessing
'''
meanx = np.mean(dataset['x'])
stdx = np.std(dataset['x'])
meany = np.mean(dataset['y'])
stdy = np.std(dataset['y'])
x = dataset['x']
y = dataset['y']
for i in range(x.size):
    x[i] = (x[i]-meanx)/stdx
for i in range(y.size):
    y[i] = (y[i]-meany)/stdy
'''
Creating Feature Columns and Label column
'''
feature = tf.feature_column.numeric_column('x', dtype=tf.float64, shape=())
label = tf.feature_column.numeric_column('y', dtype=tf.float64, shape=())
feature_cols = [feature]
feature_name = ['x']
label_name = 'y'
feature_ndarray = dataset[feature_name]
label_ndarray = dataset[label_name]
'''
Creating Train and Test Dataset
'''
x_train, x_test, y_train, y_test = train_test_split(feature_ndarray, label_ndarray, random_state=0, test_size=0.3)
'''
Functions to be fed to the Estimator
------------------------------------
train_input() and val_input() has the following functionality:
- We make use of the Dataset API here because it provides an effiecient way to pipeline the tensor inputs,
  enables parallel computations
- Create an object of the Tf Dataset class
- Use from_tensor_slices method to feed the dataset
- Create an iterator - make_one_shot_iterator - simplest iterator - iterates over the dataset in batches of 32 - _dataset.batch(32)
- To get the values as we iterate, make use of get_next()
'''
def train_input():
    _dataset = tf.data.Dataset.from_tensor_slices(({'x': x_train}, y_train))
    dataset = _dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels
def val_input():
    _dataset = tf.data.Dataset.from_tensor_slices(({'x': x_test}, y_test))
    dataset = _dataset.batch(32)
    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels
'''
Pre-made Linear Regressor Estimator provided by TensorFlow is instantiated
Estimator is trained,evaluated and with the dataset
'''
estimator = tf.estimator.LinearRegressor(feature_columns=feature_cols)
estimator.train(input_fn=train_input, steps=None)
train_eval = estimator.evaluate(input_fn=train_input)
test_eval = estimator.evaluate(input_fn=val_input)
preds = estimator.predict(input_fn=val_input)
predictions = np.array([item['predictions'][0] for item in preds])
'''
Printing the metrics
'''
print("train_metrics: ",train_e1)
print("test_metrics: ",test_e1)

