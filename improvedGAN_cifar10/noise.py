#!/usr/bin/env python


import numpy as np

np.set_printoptions(threshold='nan')


def NonlinearWeight(xPulse, maxNumLevel, A, B, minConductance):
    return B * (1 - np.exp(-xPulse/A)) + minConductance

def InvNonlinearWeight(conductance, maxNumLevel, A, B,minConductance):
    return -A * np.log(1-(conductance-minConductance)/B)

def truncate(x,numLevel):
    if (numLevel <=0):
        return x
    else:
        val = x * numLevel
        r_val = np.floor(val)
        return r_val  / numLevel

def AddNoise(weight,bitwidth):
        
    maxConductance = 3.8462e-8
    minConductance = 3.0769e-9
    maxNumLevelLTP = np.power(2,bitwidth)
    maxNumLevelLTD = np.power(2,bitwidth)
    paramALTP = 7.012968
    paramALTD = 7.012968
    paramBLTP = (maxConductance - minConductance) / (1 - np.exp(-maxNumLevelLTP/paramALTP))
    paramBLTD = (maxConductance - minConductance) / (1 - np.exp(-maxNumLevelLTD/paramALTD))
    sigmaCtoC = 0 * (maxConductance - minConductance)
    stuckat0_rate = 0.1
    stuckat1_rate = 0.1
    isstuckat = 0
    istruncate = 1
    weight_positive = (np.abs(weight) + weight)/2
    if istruncate ==1:
        weight_positive = truncate(weight_positive, maxNumLevelLTP);
    num_pulse_positive = weight_positive * maxNumLevelLTP
    conductance_positive = NonlinearWeight(num_pulse_positive, maxNumLevelLTP, paramALTP, paramBLTP, minConductance)
    conductance_positive += np.random.normal(0,sigmaCtoC,conductance_positive.shape) 
    conductance_positive = np.minimum(conductance_positive,maxConductance)
    conductance_positive = np.maximum(conductance_positive,minConductance)

    if isstuckat ==1:
        a = np.random.random_sample(np.shape(conductance_positive))
        b = (a<stuckat1_rate)
        conductance_positive = conductance_positive + b
        conductance_positive = np.minimum(conductance_positive,maxConductance)
        c = np.random.random_sample(np.shape(conductance_positive))
        d = (a<stuckat0_rate)
        conductance_positive = conductance_positive - d
        conductance_positive = np.maximum(conductance_positive,minConductance)

    num_pulse_positive = InvNonlinearWeight(conductance_positive, maxNumLevelLTP, paramALTP, paramBLTP, minConductance)
    weight_positive = num_pulse_positive / maxNumLevelLTP


    #negative weights
    weight_negative = (np.abs(weight) - weight)/2
    if istruncate ==1:
        weight_negative = truncate(weight_negative, maxNumLevelLTD);
    num_pulse_negative = weight_negative * maxNumLevelLTD
    conductance_negative = NonlinearWeight(num_pulse_negative, maxNumLevelLTD, paramALTD, paramBLTD, minConductance)
    conductance_negative += np.random.normal(0,sigmaCtoC,conductance_negative.shape) 
    conductance_negative = np.minimum(conductance_negative,maxConductance)
    conductance_negative = np.maximum(conductance_negative,minConductance)

    if isstuckat ==1:
        a = np.random.random_sample(np.shape(conductance_negative))
        b = (a<stuckat1_rate)
        conductance_negative = conductance_negative + b
        conductance_negative = np.minimum(conductance_negative,maxConductance)
        c = np.random.random_sample(np.shape(conductance_negative))
        d = (a<stuckat0_rate)
        conductance_negative = conductance_negative - d
        conductance_negative = np.maximum(conductance_negative,minConductance)        

    num_pulse_negative = InvNonlinearWeight(conductance_negative, maxNumLevelLTD, paramALTD, paramBLTD, minConductance)
    weight_negative = num_pulse_negative / maxNumLevelLTD

    #total weights
    weight_noise = weight_positive - weight_negative

    return weight_noise




