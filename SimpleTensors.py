'''
This code illustrates the use the symbolic constants and variables in tensorflow, 
interactive sessions which feed the data to the variables at runtime using two different approaches
'''
import tensorflow as tf
x = tf.constant([1,2,3,4])
y = tf.constant([5,6,7,8])
z = tf.multiply(x,y)
print(z)
#need to run an interactive session in order to obtain the result
#approach 1
sess = tf.Session()
value = sess.run(z)
print(value)
sess.close()
#approach 2
with tf.Session() as sess:
    value = sess.run(z)
    print(value)
    
