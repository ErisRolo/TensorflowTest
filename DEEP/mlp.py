import tensorflow as tf
import numpy as np

Xp=tf.placeholder(tf.float32,shape=[4,2])
Yp=tf.placeholder(tf.float32,shape=[4,1])

def weight_variable(shape):
    tw=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(tw)

def bais_varible(shape):
    tb=tf.constant(0.1,shape=shape)
    return tf.Variable(tb)

W1=weight_variable([2,3])
B1=bais_varible([3])
L1=tf.matmul(Xp,W1)+B1
L1=tf.nn.sigmoid(L1)

W2=weight_variable([3,1])
B2=bais_varible([1])
OUT=tf.matmul(L1,W2)+B2

print(OUT)
loss=tf.reduce_mean(tf.square(Yp-OUT))
TrainStep=tf.train.AdamOptimizer(0.1).minimize(loss)


sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init)

data=np.array([[0,0],[1,0],[0,1],[1,1]])
label=np.array([[0],[1],[1],[0]])



for i in range(2000):
    error,out,result=sess.run([loss,OUT,TrainStep],feed_dict={Xp:data,Yp:label})
    print(error,out)