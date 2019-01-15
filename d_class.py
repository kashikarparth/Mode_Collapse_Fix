# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 19:27:18 2018

@author: Parth
"""
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

tf.reset_default_graph()

batch_size = 100
lr = 0.0002
train_epoch = 500


def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

def discriminator_class(x, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator_class', reuse=reuse):
        
        rs = tf.reshape(x,shape = [batch_size,4096])
        
        h1 = tf.layers.dense(rs,1024)
        lrel1 = lrelu(h1) 
        d1 = tf.layers.dropout(lrel1,0.3)
        
        
        h2 = tf.layers.dense(d1,1024)
        lrel2 = lrelu(h2) 
        d2 = tf.layers.dropout(lrel2,0.3)
        
        h3 = tf.layers.dense(d2,1024)
        lrel3 = lrelu(h3) 
        d3 = tf.layers.dropout(lrel3,0.3)
        
        
        h4 = tf.layers.dense(d3,1)
        
        return h4
                
X = tf.placeholder(tf.float32, shape=(None, None, None, None))  
Y = tf.placeholder(tf.float32, shape = (None,None) )
isTrain = tf.placeholder(dtype=tf.bool)

predicted = discriminator_class(X)
D_loss_class = tf.square(tf.subtract(Y,predicted))

T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator_class')]


with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss_class, var_list=D_vars)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
train_set = (train_set - 0.5) / 0.5  

train_hist = {}
train_hist['D_losses'] = []
    
print('training start!')

min_loss = 10000

for epoch in range(150):
    D_losses = []
    a = True
    for iter in range(mnist.train.num_examples // batch_size):
        x_ = train_set[iter*batch_size:(iter+1)*batch_size]
        y_ = np.reshape(np.nonzero(mnist.train.labels)[1][iter*batch_size:(iter+1)*batch_size],newshape = (100,1))
        if(a):
            print("Current Epoch "+ str(epoch),  end = " ")
            a = False
        if iter%50 == 0:
            print("=",end = "")
        
        loss_d_, _ = sess.run([D_loss_class, D_optim], {X: x_, Y: y_, isTrain: True})
        D_losses.append(loss_d_)
    print(" ")
    train_hist['D_losses'].append(np.mean(D_losses))
    
    if(np.mean(D_losses)<min_loss):
        layer = 0
        for i in range(len(D_vars)):
            a = np.asanyarray(D_vars[i].eval())
            if i%2 ==0:
                np.save(arr = a, file = "kernel" + str(layer))
            else:
                np.save(arr = a, file = "bias" + str(layer))
                layer = layer + 1
        print("saved the parameters")
        min_loss = np.mean(D_losses)
        
    print("Discriminator Loss after epoch: " + str(epoch) + " " + str(np.mean(D_losses)))

sess.close()
