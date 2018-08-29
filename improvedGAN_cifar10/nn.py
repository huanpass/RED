"""
Code modified from PyTorch DCGAN examples: https://github.com/pytorch/examples/tree/master/dcgan
"""
from __future__ import print_function
import argparse
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from utils import weights_init, compute_acc
from network import _netG, _netG_CIFAR10
from folder import ImageFolder
import plotting
import scipy.io
import scipy.misc


ngpu = 1
gpu_id = 2
cuda = 1
batchSize = 100
imageSize=32
# specify the gpu id if using only 1 gpu
if ngpu == 1:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

# some hyper parameters
nz = 110
num_classes = 10
nc = 3
numberOutput = 100
# Define the generator and initialize the weights

netG = _netG_CIFAR10(ngpu, nz)
netG.apply(weights_init)

for i in range(int(numberOutput/batchSize)):
    print(i+1)
    manualSeed = random.randint(1, 10000)
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    if cuda:
        torch.cuda.manual_seed_all(manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")



    # tensor placeholders
    input = torch.FloatTensor(batchSize, 3, imageSize, imageSize)
    noise = torch.FloatTensor(batchSize, nz, 1, 1)
    eval_noise = torch.FloatTensor(batchSize, nz, 1, 1).normal_(0, 1)
    dis_label = torch.FloatTensor(batchSize)
    aux_label = torch.LongTensor(batchSize)
    real_label = 1
    fake_label = 0

    # if using cuda
    if cuda:
        netG.cuda()
        dis_criterion.cuda()
        aux_criterion.cuda()
        input, dis_label, aux_label = input.cuda(), dis_label.cuda(), aux_label.cuda()
        noise, eval_noise = noise.cuda(), eval_noise.cuda()

    # define variables
    input = Variable(input)
    noise = Variable(noise)
    eval_noise = Variable(eval_noise)
    dis_label = Variable(dis_label)
    aux_label = Variable(aux_label)
    # noise for evaluation
    eval_noise_ = np.random.normal(0, 1, (batchSize, nz))
    eval_label = np.random.randint(0, num_classes, batchSize)
    eval_onehot = np.zeros((batchSize, num_classes))
    eval_onehot[np.arange(batchSize), eval_label] = 1
    eval_noise_[np.arange(batchSize), :num_classes] = eval_onehot[np.arange(batchSize)]
    eval_noise_ = (torch.from_numpy(eval_noise_))
    eval_noise.data.copy_(eval_noise_.view(batchSize, nz, 1, 1))


                
    netG.load_state_dict(torch.load('./output/netG_epoch_499.pth')) 
    fake = netG(eval_noise)
    fake_data_array = fake.data.cpu().numpy()
    img_bhwc = np.transpose(fake_data_array[:100,], (0, 2, 3, 1))
    savepath = "/home/hc218/Zichen_Fan/ACGAN-PyTorch/fakeimage/"
    
    for j in range(batchSize):
        #print(i*batch_size+j+1)
        scipy.misc.imsave(savepath+str(i*batchSize+j+1).zfill(5)+".png",img_bhwc[j,:,:,:])
        #plotting.plt.close('all')
