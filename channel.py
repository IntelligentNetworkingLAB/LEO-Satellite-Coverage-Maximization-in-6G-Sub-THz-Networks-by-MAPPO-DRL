import numpy as np
import math
from scipy.spatial import distance
import matplotlib.pyplot as plt
import random as rand

def TotalPathLosses(_d, _f):
    total = sum(_d)
    c = 1 # NEED
    return (4*math.pi*total*_f)**2/(c)

def RicianFading(_LoS, _NLoS, _Factor):
    # return complex number
    return math.sqrt(_Factor/(_Factor+1))*_LoS + math.sqrt(1/(_Factor+1))*_NLoS

def GetDataRate(_channel):
    bandwidth = 1e+10 # NEED
    data_rate = bandwidth * np.log2(1 + _channel)
    return data_rate

def GetReceivedSignal(_p, _g, _w, _PI, _H, _d, _f):
    # _p: 1 x 1
    # _g: 1 x N_r 
    # _w: 1 x 1
    # _PI: N_r x 1 x R_s
    # _H:  R_s x 1
    # d is R_s x 1
    # f is frequency

    N = 1e-10 # NEED
    Pdt = _PI[0] * _H[0]
    for i in range(1, len(_PI)):
        Pdt *=  _PI[i] * _H[i]
    return _p * (abs(_g*Pdt*_w)**2) / (N * TotalPathLosses(_d, _f))


# Main function -> Test for Channel Model
if __name__ == '__main__':
    