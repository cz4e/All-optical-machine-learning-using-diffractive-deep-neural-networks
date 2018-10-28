#!/usr/bin/env python

from tensorflow.examples.tutorials.mnist  import  input_data
import tensorflow as tf
import numpy as np
import random as ra
import math

def w_imag(x,y,z,x0,y0,z0=0.0,lb=632.8e-7):
    r=np.sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0))
    answer = (z-z0)/(r*r)*(1.0/(2*np.pi*r)+1.0/1.0j*lb)*np.exp(1j*2*np.pi*r/lb)
    return np.imag(answer)

def w(x,y,z,x0,y0,z0=0.0,lb=1.0):
    r = np.sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)+(z-z0)*(z-z0))
    answer = (z-z0)/(r*r)*(1.0/(2*np.pi*r)+1.0/1.0j*lb)*np.exp(1j*2*np.pi*r/lb)
    return np.real(answer)

def set_w(z0,z,layer1,layer2):
    global lbda
    x=0
    y=0
    x0=0
    y0=0
    num1 = int(np.sqrt(layer1))
    num2 = int(np.sqrt(layer2))
    size1 = 8.0/num1
    size2 = 8.0/num2
    array=[]
    if layer2 != 10:
        array=np.array([w(x+i*size2/2.0,y+k*size2/2.0,z,x0+m*size1/2.0,y0+n*size1/2.0,z0,lb=lbda) for m in  range(num1) for n in range(num1) for i in range(num2) for k in range(num2)])
    else:
        for m in range(num1):
            for n in range(num1):
                l = 2
                while l<=6:
                   if l == 4:
                      array=array +[w(x+l,y+i,z,x0+m*size1/2.0,y0+n*size1/2.0,z0,lb=lbda) for i in range(1,8,2)]
                   else:
                      array = array + [w(x+l,y+i,z,x0+m*size1/2.0,y0+n*size1/2.0,z0,lb=lbda) for i in range(1,6,2)]
                   l+=2
       	array=np.array(array)
    return np.reshape(array,[layer1,layer2])

def set_w_imag(z0,z,layer1,layer2):
    global lbda
    x=0
    y=0
    x0=0
    y0=0
    num1 = int(np.sqrt(layer1))
    num2 = int(np.sqrt(layer2))
    size1 = 8.0/num1
    size2 = 8.0/num2
    array=[]
    if layer2 != 10:
        array=np.array([w_imag(x+i*size2/2.0,y+k*size2/2.0,z,x0+m*size1/2.0,y0+n*size1/2.0,z0,lb=lbda) for m in  range(num1) for n in range(num1) for i in range(num2) for k in range(num2)])
    else:
        for m in range(num1):
            for n in range(num1):
                l = 2
                while l<=6:
                   if l == 4:
                      array=array +[w_imag(x+l,y+i,z,x0+m*size1/2.0,y0+n*size1/2.0,z0,lb=lbda) for i in range(1,8,2)]
                   else:
                      array = array + [w_imag(x+l,y+i,z,x0+m*size1/2.0,y0+n*size1/2.0,z0,lb=lbda) for i in range(1,6,2)]
                   l+=2
       	array=np.array(array)
    return np.reshape(array,[layer1,layer2]) 
#def sum_w(x,y,N):
#    return np.sum(np.fromfunction(lambda i:w(x,y,0.2,i%N,i//N),(N*N,)))

def make_random(row):
    return np.pi*np.random.random(size = row)


#configure Holder
data_x = tf.placeholder(tf.float32,shape=[None,784])
data_y = tf.placeholder(tf.float32,shape=[None,10])


##define constant
n_input = 784
n_hidden_1 = 1024
n_hidden_2 = 1024
n_hidden_3 = 1024
n_hidden_4 = 1024
n_hidden_5 = 1024
n_output = 10
delta = 3
lbda = 632.8e-7
#configure network 
w1=set_w_imag(0,delta,n_input,n_hidden_1)
w2=set_w_imag(delta,2*delta,n_hidden_1,n_hidden_2)
w3=set_w_imag(2*delta,3*delta,n_hidden_2,n_hidden_3)
w4=set_w_imag(3*delta,4*delta,n_hidden_3,n_hidden_4)
w5=set_w_imag(4*delta,5*delta,n_hidden_4,n_hidden_5)
w6=set_w_imag(5*delta,6*delta,n_hidden_5,n_output)

weights = {
        1:tf.constant(set_w(0,delta,n_input,n_hidden_1),dtype=tf.float32,name = 'w1'), 
        2:tf.constant(set_w(delta,2*delta,n_hidden_1,n_hidden_2),dtype=tf.float32,name='w2'),
        3:tf.constant(set_w(2*delta,3*delta,n_hidden_2,n_hidden_3),dtype=tf.float32,name='w3'),
        4:tf.constant(set_w(3*delta,4*delta,n_hidden_3,n_hidden_4),dtype=tf.float32,name='w4'),
        5:tf.constant(set_w(4*delta,5*delta,n_hidden_4,n_hidden_5),dtype=tf.float32,name='w5'),
        6:tf.constant(set_w(5*delta,6*delta,n_hidden_5,n_output),dtype=tf.float32,name= 'output')
        }

biases = {
    #h2,h4 phase,h1,h3,h5 ,output A
        1:tf.constant(make_random(n_hidden_1),dtype=tf.float32,name = 'b1'),
        2:tf.Variable(make_random(n_hidden_2),dtype=tf.float32,name = 'b2'),
        3:tf.Variable(make_random(n_hidden_3),dtype=tf.float32,name = 'b3'),
        4:tf.Variable(make_random(n_hidden_4),dtype=tf.float32,name = 'b4'),
        5:tf.Variable(make_random(n_hidden_5),dtype=tf.float32,name = 'b5'),
        6:tf.constant(make_random(n_output),dtype=tf.float32,name = 'output')
        }

amplitudes = {
        1:tf.Variable(make_random(n_hidden_1),dtype=tf.float32,name='a1'),
        2:tf.constant(make_random(n_hidden_2),dtype=tf.float32,name='a2'),
        3:tf.constant(make_random(n_hidden_3),dtype=tf.float32,name='a3'),
        4:tf.constant(make_random(n_hidden_4),dtype=tf.float32,name='a4'),
        5:tf.constant(make_random(n_hidden_5),dtype=tf.float32,name='a5'),
        6:tf.Variable(make_random(n_output),dtype=tf.float32,name='output')
}
#define forword
#print(weights[6].shape())
def inference(input_h):  
    layer_1 = tf.matmul(tf.cast(input_h,dtype=tf.complex64),tf.cast(weights[1],dtype=tf.complex64)+1j*tf.cast(w1,dtype=tf.complex64))*tf.multiply(tf.cast(amplitudes[1],dtype=tf.complex64),tf.cast(tf.math.exp(1j*tf.cast(biases[1],dtype=tf.complex64)),dtype=tf.complex64))
    layer_2 = tf.matmul(layer_1,tf.cast(weights[2],dtype=tf.complex64)+1j*tf.cast(w2,dtype=tf.complex64))*tf.multiply(tf.cast(amplitudes[2],dtype=tf.complex64),tf.math.exp(1j*tf.cast(biases[2],dtype=tf.complex64)))
    layer_3 = tf.matmul(layer_2,tf.cast(weights[3],dtype=tf.complex64)+1j*tf.cast(w3,dtype=tf.complex64))*tf.multiply(tf.cast(amplitudes[3],dtype=tf.complex64),tf.math.exp(1j*tf.cast(biases[3],dtype=tf.complex64)))
    layer_4 = tf.matmul(layer_3,tf.cast(weights[4],dtype=tf.complex64)+1j*tf.cast(w4,dtype=tf.complex64))*tf.multiply(tf.cast(amplitudes[4],dtype=tf.complex64),tf.math.exp(1j*tf.cast(biases[4],dtype=tf.complex64)))
    layer_5 = tf.matmul(layer_4,tf.cast(weights[5],dtype=tf.complex64)+1j*tf.cast(w5,dtype=tf.complex64))*tf.multiply(tf.cast(amplitudes[5],dtype=tf.complex64),tf.math.exp(1j*tf.cast(biases[5],dtype=tf.complex64)))
    output_layer = tf.matmul(layer_5,tf.cast(weights[6],dtype=tf.complex64)+1j*tf.cast(w6,dtype=tf.complex64))*tf.multiply(tf.cast(amplitudes[6],dtype=tf.complex64),tf.math.exp(1j*tf.cast(biases[6],dtype=tf.complex64)))

    return output_layer

mnist = input_data.read_data_sets("/home/manjaro/Downloads", one_hot=True)
learning_rate = 20
logits = inference(data_x)
prediction = logits         

logits_abs = np.abs(logits)**2
data_y_abs = np.abs(data_y)**2
loss = tf.reduce_mean(np.sum(tf.square(logits_abs-data_y_abs))/n_output)
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss)

pre_correct = tf.equal(tf.argmax(data_y,1),tf.argmax(np.abs(prediction),1))
accuracy = tf.reduce_mean(tf.cast(pre_correct,tf.float32))   #tf.cast改變元素類型,tf.reduce_mean求數軸的平均值

init =  tf.global_variables_initializer()
train_epochs = 200
test_epochs = 10
batch_size = 64

with tf.Session() as session:
    session.run(init)
    total_batch = int(mnist.train.num_examples / batch_size)
    for epoch in range(train_epochs):
        for batch in range(total_batch):
            batch_x,batch_y = mnist.train.next_batch(batch_size)
            session.run(train_op,feed_dict={data_x:batch_x,data_y:batch_y})
        loss_,acc = session.run([loss,accuracy],feed_dict={data_x:batch_x,data_y:batch_y})
        print("epoch :{} lose:{:.4f} acc:{:.4f}".format(epoch,loss_,acc))

    print("Optimizer finished")
    for epoch in range(test_epochs):
        batch_x,batch_y = mnist.test.next_batch(batch_size)
        test_acc = session.run(accuracy,feed_dict={data_x:batch_x,data_y:batch_y})
        #test_acc = session.run(accuracy,feed_dict={data_x:mnist.test.images,data_y:batch_y})
        print("test accuracy:",test_acc)

