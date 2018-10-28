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

References :
 - http://yann.lecun.com/exdb/mnist/
 - http://cjalmeida.net/post/tensorflow-mnist/

Report.txt
 - Conatins class wise metrics reported - precision, recall , fscore, support
 - Contains Train and Test Accuracy and Loss
