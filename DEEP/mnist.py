import numpy as np
import tensorflow as tf
import struct
import matplotlib.pyplot as plt
from readmnist import DataUtils

trainfile_X = 'MNIST/train-images.idx3-ubyte'
trainfile_y = 'MNIST/train-labels.idx1-ubyte'
testfile_X = 'MNIST/t10k-images.idx3-ubyte'
testfile_y = 'MNIST/t10k-labels.idx1-ubyte'
train_X = DataUtils(filename=trainfile_X).getImage()
train_y = DataUtils(filename=trainfile_y).getLabel()
test_X = DataUtils(testfile_X).getImage()
test_y = DataUtils(testfile_y).getLabel()

#plt.imshow(np.reshape(train_X[2000,:],[28,28]))
#plt.show()


Xp=tf.placeholder(tf.float32,shape=[None,784])
Yp=tf.placeholder(tf.float32,shape=[None,10])
Xp1=tf.reshape(Xp,shape=[-1,28,28,1])

#Network Strat
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.01)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)



Wconv1=weight_variable([5,5,1,64])
Bconv1=bias_variable([64])
conv1=tf.nn.relu(tf.nn.conv2d(Xp1,Wconv1,strides=[1,1,1,1],padding='VALID')+Bconv1)
pool2=tf.nn.max_pool(conv1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

Wconv3=weight_variable([5,5,64,128])
Bconv3=bias_variable([128])
conv3=tf.nn.relu(tf.nn.conv2d(pool2,Wconv3,strides=[1,1,1,1],padding='VALID')+Bconv3)
pool4=tf.nn.max_pool(conv3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
pool4_line=tf.reshape(pool4,shape=[-1,2048])

Wfc5=weight_variable([2048,512])
Bfc5=bias_variable([512])
fc5=tf.nn.relu(tf.matmul(pool4_line,Wfc5)+Bfc5)

Wfc6=weight_variable([512,128])
Bfc6=bias_variable([128])
fc6=tf.nn.relu(tf.matmul(fc5,Wfc6)+Bfc6)

Wfc7=weight_variable([128,10])
Bfc7=bias_variable([10])
fc7=tf.matmul(fc6,Wfc7)+Bfc7
OUT=tf.nn.softmax(fc7)

loss=tf.reduce_mean(-tf.reduce_sum(Yp*tf.log(tf.clip_by_value(OUT,1e-10,1.0)),reduction_indices=[1]))
TrainStep=tf.train.AdamOptimizer(0.01).minimize(loss)

correct_prediction = tf.equal(tf.argmax(OUT,1), tf.argmax(Yp,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
res=tf.argmax(OUT,1)


def getBatch(Batch_num):
    Data=[]
    Label=[]
    for i in range(Batch_num):
        index=np.random.randint(60000)
        Data.append(train_X[index,:])

        labeltemp=np.zeros([10])
        for j in range(10):
            if j==train_y[index]:
                labeltemp[j]=1
        Label.append(labeltemp)
    return np.array(Data),np.array(Label)


#print(data.shape)
#print(label.shape)
#print(label)
sess=tf.Session()
init=tf.initialize_all_variables()
sess.run(init)

for i in range(2000):
    print(i)
    data,label=getBatch(64)
    error,out,result=sess.run([loss,OUT,TrainStep],feed_dict={Xp:data,Yp:label})
    print(error)

resall=[]
for i in range(60):
    RES=sess.run(res, feed_dict={Xp:train_X[i*1000:(i+1)*1000,:],Yp:np.zeros([1000,10])})
    resall.append(RES)

resall=np.array(resall)
resall=np.reshape(resall,[60000])

np.savetxt('label.csv',resall,delimiter=',')