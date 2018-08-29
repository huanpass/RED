#!/usr/bin/env python
import numpy as np
import theano as th
import theano.tensor as T
import lasagne
from lasagne.layers import dnn
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

def input2padding(x,input_shape,filter_shape,stride):
    """
    """
    N = input_shape[0]
    C = input_shape[1]
    H = input_shape[2]
    W = input_shape[3]
    temp = np.zeros([N,C,H,W])
    temp = x.eval()
    print(temp.shape)
    K = filter_shape
    S = stride
    P = int(K/2)
    output_H = S*H+K-1
    output_W = S*W+K-1
    output = np.zeros([N,C,output_H,output_W])
    for n in range(N):
        for k in range(C):
            for i in range(output_H):
                for j in range(output_W):
                    if(((i>=P+1 and i<output_H-P) and ((i-P-1)%2==0)) and ((j>=P and j<output_W-P-1) and ((j-P)%2==0))):
                        #print(i,j)
                        output[n,k,i,j] = temp[n,k,int((i-P-1)/2),int((j-P)/2)]
    return output


def weight2crossbar(weight,weight_shape):
    N = weight_shape[0]
    C = weight_shape[1]
    H = weight_shape[2]
    W = weight_shape[3]
    weightcrossbar_H = N
    weightcrossbar_W = C*H*W
    weightcrossbar = np.zeros([weightcrossbar_H,weightcrossbar_W])
    for i in range(weightcrossbar_H):
        weightcrossbar[i,:] = weight[i,:].reshape((1,-1)).eval()
    return weightcrossbar




def input2column(x,HH,WW,stride):
    """
    Input:
    - x: input matrix to be translate into columes,(C,H,W)
    - HH, filter height
    - WW, filter width
    - H_R: output height
    - W_R: output width
#    - stride: stride
    Returns:
    -cols: (new_h*new_w, hh*ww*c) matrix, each column is a cube that will convlve with a filter
    """
    C,H,W = x.shape
    H_R=int((H-HH)/stride+1)
    W_R=int((W-WW)/stride+1)
    cols = np.zeros([H_R*W_R,HH*WW*C])
    for i in range(H_R):
        for j in range(W_R):
            patch = x[:,i*stride:i*stride+HH,j*stride:j*stride+WW]
            cols[i*W_R+j,:]=patch.reshape(-1)
    return cols


def conv_forward_fast(x,w,conv_param,ReRAM=None):
    """
    Input:
    - x: Input data of shape(N,C,H,W)
    - w: filters of shape(F,C,HH,WW)
    - b: biase of shape(F,)
    - conv_param: A dictionary with the following key:
    - 'stride': how much pixels the sliding window will travel
    - 'pad': The number of pixels that will be used to zero-pad the input
    N: Mini-batch size
    C: Input depth
    H/W: input image height/width
    F: number of filters 
    HH/WW: kernel height / weight
    Returns:
    - out: Output data, of shape(N,F,H',W')
    H' = (H+2*conv_param['pad']-HH)/conv_param['stride'] +1
    W' = (W+2*conv_param['pad']-WW)/conv_param['stride'] +1
    - cache: (x, w, out,conv_param)
    """
    N,C,H,W=x.shape
    F,C,HH,WW=w.shape
    S=conv_param['stride']
    P=conv_param['pad']
    assert (H+2*P-HH)%S==0
    assert (W+2*P-WW)%S==0
    H_prime=int((H+2*P-HH)/S+1)
    W_prime=int((W+2*P-WW)/S+1)
    out = np.zeros([N,F,H_prime,W_prime])
    print("H_prime ", H_prime, "W_prime ", W_prime)
    for i in range(N):
        im =x[i,:,:,:]
        im_pad=np.pad(im,((0,0),(P,P),(P,P)), 'constant', constant_values = (0,0))
        #im_x has a shape of (H_prime*W_prime,C*HH*WW)
        im_x = im2col(im_pad,HH,WW,S)
        #im_w has a shape of(F,C*WW*HH)
        im_w=np.reshape(w,(F,-1))
        #im_out has a shape of (H_prime*W_prime,F)
        im_out=np.dot(im_x,im_w.T)
        out[i,:,:,:,]=col2im(im_out,H_prime,W_prime,C)
    if ReRAM is not None:
        ReRAM.CONV_Layer('conv',im_x.T.shape,im_w.T.shape)
    cache=(x,w,b,conv_param)