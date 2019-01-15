# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 22:58:30 2018

@author: Parth
"""


import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import os,time
import matplotlib.pyplot as plt

tf.reset_default_graph()
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True, reshape=[])

batch_size = 100
lr = 0.0002
train_epoch = 20
random_space_dim = 20
guide_dim = 10
latent_space_dim = random_space_dim + guide_dim
z_input = np.random.normal(0,1,(55000,1,1,latent_space_dim))
target_classes_real = np.empty(shape = [mnist.train.num_examples])

for i in range(guide_dim):    
    z_input[:,0,0,random_space_dim+i] = np.nonzero(mnist.train.labels)[1]
target_classes_real = z_input[:,0,0,latent_space_dim-1]


weights_class = []
biases_class = []
layer = 0
for i in range(8):
    if i%2==0:
        weights_class.append(tf.convert_to_tensor(np.load(file = "kernel" + str(layer)+".npy"),dtype = tf.float32))
    else:
        biases_class.append(tf.convert_to_tensor(np.load(file = "bias" + str(layer)+".npy"),dtype = tf.float32))
        layer = layer + 1
    
    
def lrelu(x, th=0.2):
    return tf.maximum(th * x, x)

def class_predicted(x):
    for i in range(3):
        x = tf.add(tf.matmul(x,weights_class[i]),biases_class[i])
        x = lrelu(x)
    x = tf.add(tf.matmul(x,weights_class[3]),biases_class[3])
    return x


def class_cost(x,target_classes):
    x = tf.reshape(x,shape = [batch_size,4096])
    predicted_classes = class_predicted(x)
    cost = tf.reduce_mean(tf.square(tf.subtract(predicted_classes,target_classes)))
    cost = tf.log(cost)
    mean, var = tf.nn.moments(predicted_classes, axes=[1])
    diff = tf.square(tf.subtract(var,0.01*tf.ones(shape=[1]))) + tf.square(tf.subtract(mean,target_classes[0]))
    diff = tf.log(diff)
    return cost + diff


def generator(x, isTrain=True, reuse=False):
    with tf.variable_scope('generator', reuse=reuse):    
        rs = tf.reshape(x,shape=[batch_size,latent_space_dim])    
        h1 = tf.layers.dense(rs,256)
        lrel1 = lrelu(h1)    
        h2 = tf.layers.dense(lrel1,512)
        lrel2 = lrelu(h2)    
        h3 = tf.layers.dense(lrel2,1024)
        lrel3 = lrelu(h3)    
        h4 = tf.layers.dense(lrel3,4096)    
        o = tf.nn.tanh(h4)    
        o = tf.reshape(o, shape = [100,64,64,1])
        return o
    
def discriminator(x, isTrain=True, reuse=False):
    with tf.variable_scope('discriminator', reuse=reuse):
        
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
        o = tf.nn.sigmoid(h4)
        
        h4 = tf.reshape(h4,shape = [100,1,1,1])
        o = tf.reshape(o,shape = [100,1,1,1])
        
        return o,h4

x = tf.placeholder(tf.float32, shape=(None, 64, 64, 1))
z = tf.placeholder(tf.float32, shape=(None, 1, 1, latent_space_dim))
target_classes = tf.placeholder(tf.float32, shape = (None,None))

isTrain = tf.placeholder(dtype=tf.bool)


G_z = generator(z, isTrain)
G_z_reshaped = tf.reshape(tf.image.resize_images(G_z, [28, 28]),shape = (batch_size,784))
D_real, D_real_logits = discriminator(x, isTrain)
D_fake, D_fake_logits = discriminator(G_z, isTrain, reuse=True)

D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real_logits, labels=tf.ones([batch_size, 1, 1, 1])))
D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.zeros([batch_size, 1, 1, 1])))
guided = class_cost(G_z,target_classes)

D_loss = D_loss_real + D_loss_fake
G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake_logits, labels=tf.ones([batch_size, 1, 1, 1]))) + guided
    

T_vars = tf.trainable_variables()
D_vars = [var for var in T_vars if var.name.startswith('discriminator')]
G_vars = [var for var in T_vars if var.name.startswith('generator')]



with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
    D_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(D_loss, var_list=D_vars)
    G_optim = tf.train.AdamOptimizer(lr, beta1=0.5).minimize(G_loss, var_list=G_vars)

session = tf.InteractiveSession()
tf.global_variables_initializer().run()


train_set = tf.image.resize_images(mnist.train.images, [64, 64]).eval()
train_set = (train_set - 0.5) / 0.5  

train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []

saver = tf.train.Saver()


save_dir = 'checkpoints/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    
save_path = os.path.join(save_dir, 'best_validation_log_loss/')

if(0):
    saver.save(sess = session,save_path = save_path)
    print("Session Saved ***************************************")


if(0):
    saver.restore(sess = session,save_path = save_path)
    print("Session Restored********************************")
    
    
#plt.gray()
#z_ = z_input[2*batch_size:(3)*batch_size]
#for i in range(guide_dim):      
#    z_[:,0,0,random_space_dim+i] = 5
#print(np.shape(z_))
#z_ = tf.convert_to_tensor(z_,dtype = tf.float32)
#labels= np.nonzero(mnist.train.labels)[1][2*batch_size:(3)*batch_size] 
#images = session.run(generator(z_,reuse = True))
#predicted_classes = session.run(class_predicted(tf.reshape(images,shape=[batch_size,4096])))
#
#for i in range(10):
#    plt.imshow(np.reshape(images[i],newshape = [64,64]))
#    print(labels[i])
#    print("predicted_class " , str(predicted_classes[i]))
#    plt.show()

print('training start!')
start_time = time.time()
for epoch in range(100):
    z_input = np.random.normal(0,1,(55000,1,1,latent_space_dim))
    for i in range(guide_dim):    
        z_input[:,0,0,random_space_dim+i] = np.nonzero(mnist.train.labels)[1]
    G_losses = []
    D_losses = []
    epoch_start_time = time.time()
    a = True
    for iter in range(mnist.train.num_examples // batch_size):
        # update discriminator
        x_ = train_set[iter*batch_size:(iter+1)*batch_size]
        z_ = z_input[iter*batch_size:(iter+1)*batch_size]
        target_classes_ = np.reshape(target_classes_real[iter*batch_size:(iter+1)*batch_size],newshape = (batch_size,1))
        
        if(a):
            print("Current Epoch " + str(epoch) + " ")
            a = False
        if iter%50 == 0:
            print("Current Batch " + str(int(iter/50)))

        loss_d_, _ = session.run([D_loss, D_optim], {x: x_, z: z_, target_classes: target_classes_, isTrain: True})

        D_losses.append(loss_d_)

        # update generator
        z_ = z_input[iter*batch_size:(iter+1)*batch_size]
        loss_g_, _ = session.run([G_loss, G_optim], {z: z_, x: x_,  target_classes: target_classes_, isTrain: True})
        G_losses.append(loss_g_)
        
    saver.save(sess = session,save_path = save_path)
    print("Session Saved ***************************************")
    
    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time
    print('[%d/%d] - ptime: %.2f loss_d: %.3f, loss_g: %.3f' % ((epoch + 1), train_epoch, per_epoch_ptime, np.mean(D_losses), np.mean(G_losses)))


    train_hist['G_losses'].append(np.mean(G_losses))
    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)
    
    print("Discriminator Loss after epoch: " + str(epoch) + " " + str(np.mean(D_losses)))
    print("Generator Loss after epoch: " + str(epoch) + " " + str(np.mean(G_losses)))


end_time = time.time()
total_ptime = end_time - start_time
train_hist['total_ptime'].append(total_ptime)

print('Avg per epoch ptime: %.2f, total %d epochs ptime: %.2f' % (np.mean(train_hist['per_epoch_ptimes']), train_epoch, total_ptime))
print("Training finish!... save training results")

session.close()
    




