import tensorflow as tf
import numpy
import matplotlib
import mkl
sess = tf.Session()
a = tf.constant(10)
b = tf.constant(22)
print(sess.run(a + b))