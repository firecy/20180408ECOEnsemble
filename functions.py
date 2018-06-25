#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import sys
import timeit
from datetime import datetime
import time

import numpy as np

def minmax_standardization(x, x_min, x_max):
    '''
    this function realizes data minmax standardization.
    x_nor = (x-x_min)/(x_max - x_min)
    '''
    for i in xrange(x.shape[0]):
        for j in xrange(x.shape[1]):
            if x[i, j] < x_min[j]:
                x[i, j] = x_min[j]
            if x[i, j] > x_max[j]:
                x[i, j] = x_max[j]
    x_nor = (x - x_min) / (x_max - x_min)
    return x_nor

def minmax_standardization4(x, x_min, x_max):
    for j in range(x.shape[1]):
        if x_min[j] < 0:
            x = np.hstack((x, np.zeros((len(x), 1))))
            for i in range(x.shape[0]):
                if x[i, j] <= x_min[j]:
                    x[i, j] = 1.
                    x[i, -1] = 1.
                elif x[i, j] > x_min[j] and x[i, j] < 0:
                    x[i, j] = 1 * x[i, j] / x_min[j]
                    x[i, -1] = 1.
                elif x[i, j] >= 0 and x[i, j] < x_max[j]:
                    x[i, j] = x[i, j] / x_max[j]
                else: x[i, j] = 1.
        else:
            for i in range(x.shape[0]):
                if x[i, j] < x_min[j]:
                    x[i, j] = 0.
                elif x[i, j] > x_max[j]:
                    x[i, j] = 1.
                else:
                    x[i, j] = (x[i, j] - x_min[j]) / (x_max[j] - x_min[j])
    return x

def minmax_standardization5(x, x_min, x_max):
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            if x[i, j] <= x_min[j]:
                x[i, j] = 0.
            elif x[i, j] >= x_max[j]:
                x[i, j] = 1.
            else: x[i, j] = (x[i, j] - x_min[j]) / (x_max[j] - x_min[j])
    return x

def fea_standardization(x, x_mean, x_std):
    '''
    this function realizes data feature standardization.
    The data is converted to a mean of 0 and variance data 1.
    '''
    x[:, 0: 61] -= x_mean[0: 61]
    x[:, 0: 61] /= x_std[0: 61]
    return x

def fea_stand_inverse(x, x_mean, x_std):
    x[:, 0: 61] *= x_std[0: 61]
    x[:, 0:61] += x_mean[0: 61]
    return x

def get_usv(x):
    x -= np.mean(x, axis=0)
    cov = np.dot(x.T, x) / x.shape[0]
    u, s, v = np.linalg.svd(cov)
    return u, s

def zca_whitening(x, u, s, x_mean, epsilon):
    '''
    this function is aimed to reduce the relevance of data and noises.
    '''
    x[:, 0: 61] -= x_mean
    #cov = np.dot(x.T, x)
    #U, S, V = np.linalg.svd(cov)
    xrot = np.dot(x[:, 0: 61], u)
    #xrot = np.dot(x, u)
    xpcawhite = xrot / np.sqrt(s + epsilon)
    xzcawhite = np.dot(xpcawhite, u.T)
    xzcawhite = np.hstack((xzcawhite, x[:, 61: 86]))
    return xzcawhite

def ts_ms(ts):
    fault_timestamp = str(ts)
    fault_timestamp_1 = datetime.strptime(fault_timestamp,'%Y_%m_%d_%H:%M:%S:%f')
    fault_timestamp_2 = fault_timestamp_1.strftime('%Y-%m-%d %H:%M:%S:%f')
    millisecond =  int(time.mktime(fault_timestamp_1.timetuple()))
    return fault_timestamp_2, millisecond
