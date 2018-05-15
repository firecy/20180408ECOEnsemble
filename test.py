#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

import timeit
from datetime import datetime
import time
import gc

import numpy as np



from functions import *
from preprocessing import *

from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split

from keras.layers import Input, Dense, LSTM, Concatenate, Lambda
from keras.models import Model, Sequential
from keras import regularizers
from keras.models import model_from_json
from keras.callbacks import EarlyStopping
from keras import backend as K
from keras.optimizers import SGD, RMSprop, Adam
from keras.utils import np_utils
import pickle
from sklearn.grid_search import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from scipy.spatial.distance import cosine
from CalCor import cal_DCor, RVmod, RV


def softmax(x):
    #Compute the softmax in a numerically stable way.
    for i in range(len(x)):
        #x[i] = x[i] - np.max(x[i])
        x[i] = np.exp(x[i])
        x[i] /= np.sum(x[i])
    return x

x = np.load('data/codedata/80001_x_train.npy')
t0 = x[0].shape[0]
t1 = 20
Diag = np.zeros((t1, t0))
I = np.ones((1, t0/t1))
print I.shape
for i in range(t1):
    Diag[i, t0/t1*i: t0/t1*(i+1)] = I
print Diag, Diag.shape
dcormat = np.zeros((t0, t0))
xnew = []
for j in range(t0):
    x3 = np.zeros((len(x), x[j+1].shape[1]))
    for j2 in range(x3.shape[0]):
        if x[j2].shape[0] > j:
            x3[j2] = x[j2][j]
    xnew.append(x3)
for i2 in range(t0):
    for i3 in range(t0-i2):
        x1 = xnew[i2]
        x2 = xnew[i2+i3]
        dcormat[i2, i2+i3] = cal_DCor(x1, x2)
print dcormat
A = softmax(np.dot(Diag, dcormat+np.transpose(dcormat)))
print A, A.shape
