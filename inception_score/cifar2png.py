
"""
Created on Thu Jun 15 21:56:21 2017
@author: liao
"""
import pickle as p
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as plimg
from PIL import Image
def load_CIFAR_batch(filename):
    """ load single batch of cifar """
    with open(filename, 'rb')as f:
#        datadict = p.load(f)
        datadict = p.load(f,encoding='latin1')
        X = datadict['data']
        Y = datadict['labels']
        X = X.reshape(10000, 3, 32, 32)
        Y = np.array(Y)
        return X, Y
 
def load_CIFAR_Labels(filename):
    with open(filename, 'rb') as f:
        lines = [x for x in f.readlines()]
        print(lines)
 
 
if __name__ == "__main__":
    number = 5
    for i in range(number):
        load_CIFAR_Labels("./cifar-10-batches-py/batches.meta")
        imgX, imgY = load_CIFAR_batch("./cifar-10-batches-py/data_batch_"+str(i+1))
        print(imgX.shape)
        print("saving pictures:")
#       for i in range(imgX.shape[0]):
        for j in range(imgX.shape[0]):
#           imgs = imgX[i - 1]#?
            imgs = imgX[j]
            img0 = imgs[0]
            img1 = imgs[1]
            img2 = imgs[2]
            i0 = Image.fromarray(img0)
            i1 = Image.fromarray(img1)
            i2 = Image.fromarray(img2)
            img = Image.merge("RGB",(i0,i1,i2))
            name = "img" + str(imgX.shape[0]*i+j).zfill(5)+'.png'
            img.save("./cifar10_png/"+name,"png")
            img.save("/home/hc218/Zichen_Fan/inception-score/data_real/"+name,"png")
    print("save finished")
