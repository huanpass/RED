import argparse
import pickle
import time
import numpy as np
import theano as th
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams
import lasagne
import lasagne.layers as ll
from lasagne.init import Normal
from lasagne.layers import dnn
import nn
import sys
import plotting
import scipy.io
import scipy.misc
import random
from noise import AddNoise

number_output = 100
batch_size = 100


for i in range(int(number_output/batch_size)):
    print(i+1)
    # fixed random seeds
    rng = np.random.RandomState(1)
    theano_rng = MRG_RandomStreams(rng.randint(2 ** 15))
    lasagne.random.set_rng(np.random.RandomState(rng.randint(2 ** 15)))
    #print("fixed random seads finished")

    # specify generative model
    noise_dim = (batch_size, 100)
    noise = theano_rng.uniform(size=noise_dim)
    gen_layers = [ll.InputLayer(shape=noise_dim, input_var=noise)]
    gen_layers.append(nn.batch_norm(ll.DenseLayer(gen_layers[-1], num_units=4*4*512, W=Normal(0.05), nonlinearity=nn.relu), g=None))
    gen_layers.append(ll.ReshapeLayer(gen_layers[-1], (batch_size,512,4,4)))
    gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (batch_size,256,8,8), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 4 -> 8
    gen_layers.append(nn.batch_norm(nn.Deconv2DLayer(gen_layers[-1], (batch_size,128,16,16), (5,5), W=Normal(0.05), nonlinearity=nn.relu), g=None)) # 8 -> 16
    gen_layers.append(nn.weight_norm(nn.Deconv2DLayer(gen_layers[-1], (batch_size,3,32,32), (5,5), W=Normal(0.05), nonlinearity=T.tanh), train_g=True, init_stdv=0.1)) # 16 -> 32
    gen_dat = ll.get_output(gen_layers[-1])

    samplefun = th.function(inputs=[],outputs=gen_dat)
    isnoise = 0
    #load models
    with np.load('./model_old/gen_params300.npz') as f:
        param_values = [f['arr_%d' % j] for j in range(len(f.files))]
    #for k in range(len(param_values)):
    #    print(param_values[k].shape)
    if(isnoise == 1):
        param_values[2] = AddNoise(param_values[2])
        param_values[4] = AddNoise(param_values[4])
        param_values[6] = AddNoise(param_values[6])
    #print(weight.type)    
    lasagne.layers.set_all_param_values(gen_layers[-1], param_values, trainable=True)
    sample_x = samplefun()
    img_bhwc = np.transpose(sample_x[:100,], (0, 2, 3, 1))
    img_tile = plotting.img_tile(img_bhwc, aspect_ratio=1.0, border_color=1.0, stretch=True)
    img = plotting.plot_img(img_tile, title='CIFAR10 samples')
    plotting.plt.savefig('test'+str(i+1)+'.png')
    plotting.plt.close('all')

    #savepath = "/home/hc218/Zichen_Fan/inception-score/data_fake_noise/"
    #for j in range(batch_size):
        #print(i*batch_size+j+1)
    #    scipy.misc.imsave(savepath+str(i*batch_size+j+1).zfill(5)+".png",img_bhwc[j,:,:,:])
#print flatten_img.reshape(2, 2, 64, 64, 3).shape

