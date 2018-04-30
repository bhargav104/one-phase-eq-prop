#!/usr/bin/python
# coding: utf-8


import tensorflow as tf
import numpy as np
import gzip, pickle
import sys, os
import argparse
import matplotlib.pyplot as plt
  
# for visualization
#import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')


parser = argparse.ArgumentParser(description='One phase veqprop simulation')
parser.add_argument('--n_batch', type=int, default=100)
parser.add_argument('--n_train', type=int, default=50000)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--num_iter', type=int, default=50)
parser.add_argument('--warmup', type=float, default=0.25)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--data_dir', type=str, default='tmp')
args = parser.parse_args()

print('running with\n' + '\n'.join(
        ['%s: %s' % (k,v) for k,v in args._get_kwargs()]))

N = 1000 # neuron count
Nx = 392 # input neurons
Ny = 10 # output neurons
n_batch, n_train, n_epochs = args.n_batch, args.n_train, args.n_epochs
data_dir = args.data_dir
num_iter = args.num_iter
symmetric = True #run with symmetric interactions


# load mnist
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('%s/MNIST_data' % data_dir, one_hot=True)

# only use most active half of pixels
img_mean = mnist.train.images.mean(axis=0)
use_pixels = np.where(img_mean > np.median(img_mean))[0]
train_x, train_y = mnist.train.images[:, use_pixels], mnist.train.labels
test_x, test_y = mnist.test.images[:, use_pixels], mnist.test.labels

sess = tf.InteractiveSession()

# network state and parameter init
s_init = np.zeros((n_batch, N), dtype=np.float32)
#s_init = np.random.randn(n_batch,N)
w_init = np.zeros((N, N), dtype=np.float32)
b_init = np.zeros((n_batch, N), dtype=np.float32)

# random weight init
w_init[:] = np.random.uniform(-.001, .001, size=w_init.shape)
#w_init[:] = np.random.randn(w_init.shape)
w_init[np.eye(N).astype(bool)] = 0
w_init[:Nx, :Nx] = 0
w_init[-Ny:, -Ny:] = 0
#w_init[:Nx, -Ny:] = 0
w_init[-Ny:, :Nx] = 0
# w_init[-Ny:, :-100] = 0
# w_init[:-100, -Ny:] = 0
# w_init[Nx:100, Nx:100] = 0
# w_init[-100:-Ny, -100:-Ny] = 0

# w_init[784:900, 784:900] = 0

if symmetric:
    w_init = 0.5 * (w_init + w_init.T)

# set the learning rate for different parts of the matrix
w_mask = np.ones((N, N),dtype=np.float32)
w_mask[w_init == 0] = 0

# plt.figure()
# plt.imshow(w_mask)

# parameters
dt = tf.placeholder(tf.float32, shape=(), name='dt')
lr = tf.placeholder(tf.float32, shape=(), name='lr')
beta = tf.placeholder(tf.float32, shape=(), name='beta')
x = tf.placeholder(tf.float32, shape=((None, Nx)), name='x') # clamped input values
y = tf.placeholder(tf.float32, shape=((None, Ny)), name='y') # clamped target values
    
# create variables
s  = tf.Variable(s_init, name='s', dtype=tf.float32)
ds  = tf.Variable(np.zeros_like(s_init[:,Nx:]), name='ds', dtype=tf.float32)
w = tf.Variable(w_init, name='w', dtype=tf.float32)
b = tf.Variable(b_init, name='b', dtype=tf.float32)
dw = tf.Variable(np.zeros_like(w_init), name='dw', dtype=tf.float32)
mon_c = tf.Variable(0, name='mon_c', dtype=tf.float32)


# rho = tf.tanh
rho = tf.clip_by_value
phi = tf.square # does not include sum...

# define energy
E_ = 0.5 * tf.reduce_sum(tf.square(s)) - 0.5 * tf.reduce_sum(tf.multiply(rho(s, -1, 1), tf.matmul(rho(s, -1, 1), w, transpose_b=True) + b))

# define cost
s_y_ = tf.slice(s, (0, N - Ny), (-1, -1)) # network y values
C_ = tf.reduce_sum(tf.square(rho(s_y_, -1, 1) - y))
#C_=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=s_y_,labels=y))

# define global objective
#F_=beta*C_+E_
F_ = beta * C_ + tf.reduce_sum(phi(tf.gradients(E_, s)))
#F_=beta*C_
# compute dynamics
dFds_ = tf.reshape(tf.gradients(F_, s), (-1, N))
dFdw_ = tf.reshape(tf.gradients(F_, w), (N, N))
dFdb_ = tf.reshape(tf.gradients(F_, b), (-1, N))

# compute state update
dFds_notx_ = tf.slice(dFds_, (0, Nx), (-1, -1))
jj=tf.reduce_mean(dFds_notx_)
jj1=tf.reduce_max(dFds_notx_)
s_notx_ = tf.slice(s, (0, Nx), (-1, -1))
s_ = tf.concat((x, s_notx_ - dt * dFds_notx_), axis=1)

# compute weight update
dw_ = -tf.multiply(dFdw_, w_mask)
if symmetric:
    dw_ = 0.5 * (dw_ + tf.transpose(dw_))
w_ = w + lr / beta * dw_ # / tf.sqrt(dw_avg)
val1 = tf.reduce_mean(dw_)
b_ = b - lr / beta * dFdb_

dfds_ = tf.eye(N) - tf.multiply(w_mask, w)
dfds_inv_ = tf.matrix_inverse(dfds_)
dCds_ = tf.squeeze(tf.gradients(C_, s))

dJdw_ = tf.multiply(
        tf.expand_dims(dCds_, 1),
        tf.expand_dims(tf.matmul(s, dfds_inv_, transpose_b=True), 2))
dJdw_ = tf.reduce_mean(dJdw_, axis=0)
dJdw_ = -tf.multiply(w_mask, dJdw_)

ang_ = tf.reduce_sum(tf.multiply(dw_, dJdw_)) / (tf.norm(dw) * tf.norm(dJdw_))

# update state
step = tf.group(
    s.assign(s_),
    ds.assign(dFds_notx_),
    mon_c.assign(C_)
)

# update weight
learn = tf.group(
    w.assign(w_),
    b.assign(b_),
    dw.assign(dw_),
)

# re-initialize
reset_s = tf.group(
    s.assign(s_init),
)

# initialize state
tf.global_variables_initializer().run()
f=open('data.txt','w')  



def run_fit(x_clamp, y_clamp, timestep=0.01, learning_rate=None):
    learning_rate = learning_rate or args.lr
    reset_s.run()
    tab=[]
    for i in range(num_iter):
        out=sess.run(s_,feed_dict={dt: timestep, lr: learning_rate / n_batch, beta: 1, x: x_clamp, y: y_clamp})
        f.write(str(out[0][400])+'\n')
        step.run({dt: timestep, lr: learning_rate / n_batch, beta: 1, x: x_clamp, y: y_clamp})
        # update parameters
        if i >= args.warmup * num_iter:
            learn.run({dt: timestep, lr: learning_rate / n_batch, beta: 1, x: x_clamp, y: y_clamp})
            '''
            x1=sess.run(ret1,feed_dict={y:y_clamp})
            x2=sess.run(ret2,feed_dict={y:y_clamp})
            x3=sess.run(ret3,feed_dict={y:y_clamp})
            print x2[0].shape
            sys.exit(0)
            act_grads=np.zeros(N,N)


            cmp_grads=sess.run(w_,feed_dict={dt: timestep, lr: learning_rate / n_batch, beta: 1, x: x_clamp, y: y_clamp})*w_mask
            act_grads=act_grads.flatten()
            cmp_grads=cmp_grads.flatten()
            ans=np.dot(act_grads,cmp_grads)
            ans=ans/(np.linalg.norm(act_grads)*np.linalg.norm(cmp_grads))
            print ans
            '''


def run_predict(x_clamp, y_clamp, timestep=0.01):
    reset_s.run()
    for i in range(num_iter):
        step.run({dt: timestep, lr: 0, beta: 0, x: x_clamp, y: y_clamp})
    return s.eval()

def evaluate(x_clamp, y_clamp):
    if x_clamp.shape[0] <= n_batch:
        return run_predict(x_clamp, y_clamp)
    out = []
    for i in range(x_clamp.shape[0] // n_batch):
        out.append(run_predict(x_clamp[i*n_batch:(i+1)*n_batch], y_clamp[i*n_batch:(i+1)*n_batch]))
    return np.concatenate(out)


def get_acc(y1, y2):
    return np.square(y1[:1000, -10:].argmax(axis=1) == y2.argmax(axis=1)).mean()

def get_mse(y1, y2):
    return np.square(y1[:1000, -10:] - y2).mean()


if __name__ == '__main__':

    # run it
    history = {
            'train': [],
            'test': [],
            'w': [],
            }

    # test untrained performance
    print 'testing...'
    history['train'].append(evaluate(train_x[:1000], train_y[:1000]))
    history['test'].append(evaluate(test_x[:1000], test_y[:1000]))

    # train
    history['w'].append(w.eval())
    # while mnist.train.epochs_completed < n_epochs:
    for ep in range(n_epochs):
        print '\n== running epoch %s ==' % (ep + 1)
        for i in range(n_train // n_batch):
            smax = s.eval().max()
            dwmax = dw.eval().max()
            dsmax = ds.eval().max()
            ceval = mon_c.eval()
            eeval = E_.eval()
            print '  running batch', i + 1,'| dsmax ',dsmax, '| smax', smax, '| cost', ceval,'| energy',eeval
            x_clamp, y_clamp = train_x[i*n_batch:(i+1)*n_batch], train_y[i*n_batch:(i+1)*n_batch]
            run_fit(x_clamp, y_clamp, learning_rate=args.lr)
        history['w'].append(w.eval())

        print 'testing...'
        preds=evaluate(test_x, test_y)
        preds=preds[:,N-Ny:]
        preds=preds.argmax(axis=1)
        test=test_y.argmax(axis=1)
        ctr=0
        for i in xrange(test.shape[0]):
            #print preds[i],test[i]
            if preds[i]==test[i]:
                ctr+=1
        print 'accuracy',ctr,test.shape[0]
        history['train'].append(evaluate(train_x[:1000], 0*train_y[:1000]))
        history['test'].append(evaluate(test_x[:1000], 0*test_y[:1000]))

        print 'test acc', get_acc(history['test'][-1], test_y[:1000]) 
        print 'test mse', get_mse(history['test'][-1], test_y[:1000])

        print 'saving output...'
        for k in history:
            np.save('%s/hist_%s' % (data_dir, k), history[k])

    print 'done training'

