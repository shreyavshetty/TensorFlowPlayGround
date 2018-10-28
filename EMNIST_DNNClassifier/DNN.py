'''
Importing all the rquired libraries
'''
import tensorflow as tf
import numpy as np
import requests
import gzip
import shutil
import struct
import time
from subprocess import check_call
from urllib.parse import urlparse
from hashlib import md5
from pathlib import Path
from typing import Tuple
from sklearn.metrics import classification_report
'''
Defining all the required constants
'''
INPUT = Path('Dataset')
IMAGES = {'train': INPUT / 'emnist-byclass-train-images-idx3-ubyte',
          'val': INPUT / 'emnist-byclass-test-images-idx3-ubyte'}
LABELS = {'train': INPUT / 'emnist-byclass-train-labels-idx1-ubyte',
          'val': INPUT / 'emnist-byclass-test-labels-idx1-ubyte'}
BATCH_SIZE = 128
'''
Function to read EMNIST Dataset - x_train values
Create tf.data.Dataset out of emnist images data
:param split: one of 'train' or 'val' for training or validation data
Dataset description - Referred to - http://yann.lecun.com/exdb/mnist/
'''
def read_emnist_images(split):
    assert split in ['train', 'val']
    fd = IMAGES[split].open('rb')
    magic, size, h, w = struct.unpack('>iiii', fd.read(4 * 4))
    data = np.frombuffer(fd.read(), 'u1').reshape(size, h, w, 1)[0:70000]
    fd.close()
    print("data shape",data.shape)
    return data
'''
Function to read EMNIST Dataset - y_train values - labels
Create tf.data.Dataset out of emnist labels data
:param split: one of 'train' or 'val' for training or validation data
''' 
def read_emnist_labels(split):
    assert split in ['train', 'val']
    fd = LABELS[split].open('rb')
    magic, size, = struct.unpack('>ii', fd.read(2 * 4))
    data = np.frombuffer(fd.read(), 'u1').reshape(size,1)[0:70000]
    fd.close()
    print("data shape",data.shape)
    return data

'''
Fuction to normalize images
'''
def normalize(images):
    images = tf.cast(images, tf.float32)
    images /= 255.
    return images
'''
Function to transform emnist data for use in training.
To images: random zoom and crop to 28x28, then normalize 
To labels: one-hot encode.
'''
def transform_train(images, labels):
    zoom = 0.9 + np.random.random() * 0.2  # random between 0.9-1.1
    size = int(round(zoom * 28))
    images = tf.image.resize_bilinear(images, (size, size))
    images = tf.image.resize_image_with_crop_or_pad(images, 28, 28)
    images = normalize(images)
    labels = tf.cast(labels, tf.int32)
    print("imges shape ",images.shape)
    print("labels shape",labels.shape)
    return images, labels
'''
Function to normalize emnist images and one-hot encode labels
'''
def transform_val(images, labels):
    images = normalize(images)
    labels = tf.cast(labels, tf.int32)
    return images, labels
'''
Function to create Dataset for EMNIST Data - the correct tf.data.Dataset for a given split, transforms and
batch inputs.
'''
def create_emnist_dataset(BATCH_SIZE, split):
    images = read_emnist_images(split)
    labels = read_emnist_labels(split)
    def gen():
        for image, label in zip(images, labels):
            yield image, label
    ds = tf.data.Dataset.from_generator(gen, (tf.uint8, tf.uint8), ((28, 28, 1), (1,)))
    # iter = ds.make_one_shot_iterator()
    # imgs, labs = iter.get_next()
    if split == 'train':
        return ds.batch(BATCH_SIZE).map(transform_train), len(labels)
    elif split == 'val':
        return ds.batch(BATCH_SIZE).map(transform_val), len(labels)
'''
Function to create an input function which creates an instance of the training dataset 
Makes use of prefetch and shuffle
'''
def input_fn_train():
  train_ds,train_ds_size= create_emnist_dataset(BATCH_SIZE, 'train')
  dataset = train_ds
  dataset = dataset.shuffle(buffer_size=3) #FLAGS.shuffle_buffer_size
  dataset = dataset.prefetch(1)
  iter = dataset.make_one_shot_iterator()
  imgs, labs = iter.get_next()
  x={"x": imgs}
  print("x shape",x['x'].shape)
  print("y shape",labs.shape)
  return x,labs
'''
Function to create an input function which creates an instance of the test dataset 
Makes use of prefetch and shuffle
'''
def input_fn_test():
  test_ds,test_ds_size= create_emnist_dataset(70000, 'val')
  dataset = test_ds
  dataset = dataset.shuffle(buffer_size=3) #FLAGS.shuffle_buffer_size
  dataset = dataset.prefetch(1)
  iter = dataset.make_one_shot_iterator()
  imgs, labs = iter.get_next()
  x={"x": imgs}
  print("x shape",x['x'].shape)
  print("y shape",labs.shape)
  return x,labs
'''
Function to create an input function which creates an instance of the test dataset 
Makes use of prefetch and shuffle
'''
def input_fn_xtest():
  test_ds,test_ds_size= create_emnist_dataset(70000, 'val')
  # tf.estimator.inputs.numpy_input_fn(x={"x": images},y=labels, shuffle=True)
  dataset = test_ds
  dataset = dataset.shuffle(buffer_size=3) #FLAGS.shuffle_buffer_size
  #dataset = dataset.batch(batch_size=BATCH_SIZE)
  dataset = dataset.prefetch(1)
  iter = dataset.make_one_shot_iterator()
  imgs, labs = iter.get_next()
  x={"x": imgs}
  return x
'''
Function to create an input function which creates an instance of the test dataset 
Makes use of prefetch and shuffle
'''
def input_fn_ytest():
  test_ds,test_ds_size= create_emnist_dataset(70000, 'val')
  # tf.estimator.inputs.numpy_input_fn(x={"x": images},y=labels, shuffle=True)
  dataset = test_ds
  dataset = dataset.shuffle(buffer_size=3) #FLAGS.shuffle_buffer_size
  #dataset = dataset.batch(batch_size=BATCH_SIZE)
  dataset = dataset.prefetch(1)
  iter = dataset.make_one_shot_iterator()
  imgs, labs = iter.get_next()
  x={"x": imgs}
  return labs
'''
Function to create an instance of the pre-made DNN Estimator
Calculates the time taken for the computation
Calculates the train metrics 
'''
def dnn():
    feature_columns = [tf.feature_column.numeric_column("x",shape=[28, 28])]
    dnnEstimator = tf.estimator.DNNClassifier(feature_columns=feature_columns,hidden_units=[1024, 512, 256],activation_fn = tf.nn.relu,optimizer=tf.train.ProximalAdagradOptimizer(learning_rate=0.1),n_classes=62)
    start = time.time()
    dnnEstimator.train(input_fn=input_fn_train, steps=25000)
    end = time.time()
    train_metrics = dnnEstimator.evaluate(input_fn=input_fn_train)
    total_time = int(end-start)
    y_test = input_fn_ytest()
    y_pred = dnnEstimator.predict(input_fn = input_fn_xtest)
    test_metrics = dnnEstimator.evaluate(input_fn = input_fn_test)
    y_pred = np.array([p['class_ids'][0] for p in y_pred])
    sess = tf.Session()
    with sess.as_default():
        numpy_array_1 = y_test.eval()
        print(numpy_array_1.shape, y_pred.shape)
        print(classification_report(numpy_array_1, y_pred))
    return train_metrics,test_metrics,total_time
print(dnn())
