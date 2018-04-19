#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import timeit
import time
from datetime import datetime
import os
import sys
import gc
import pygmo as pg
import GA
from IMDS import CSOSDG
from sklearn.metrics import roc_auc_score

# aim at solving binary imbalance classifier problem
def BECO_Ensemble_train(Minset, Majset, model_old, lr, epoch, batch_size, pop_size, gen_max, cx, mx):
    # define ranges of variables
    # Minset is minority dataset (x, y), x is 3D array, and y is 1D vector.
    # Majset is majority dataset (x, y), x is 3D array, and y is 1D vector.
    xMin, yMin = Minset
    xMaj, yMaj = Majset
    # get the best chromosomes population based on binary ga
    best_chromosomes_pop = GA.BGA_training(xMin=xMin, yMin=yMin,
                                           xMaj=xMaj, yMaj=yMaj,
                                           model_old=model_old, lr=lr,
                                           epoch=epoch, batch_size=batch_size,
                                           pop_size=pop_size, gen_max=gen_max,
                                           cx=cx, mx=mx)
    model_lists = []
    error_lists = []
    # train Ne(=pop_size) classifier
    for i in range(len(best_chromosomes_pop)):
        N_new = best_chromosomes_pop[0]
        k = best_chromosomes_pop[1]
        nn = best_chromosomes_pop[2]
        Cls = best_chromosomes_pop[3]
        CSO_type = best_chromosomes_pop[4]
        Min_new = CSOSDG(xMin, Cls, N_new, k, nn, CSO_type, batch_size)
        # construct new dataset
        x_new = np.vstack((self.xMaj, self.xMin, Min_new))
        y_new = np.hstack((self.yMaj, self.yMin, np.ones(N_new, )*self.yMin[0]))
        # finetune model based on new dataset
        model, error = model_train(x=x_new, y=y_new, model=self.model_old,
                         lr=self.lr, epoch=self.epoch, batch_size=self.batch_size)[1:3]
        model_lists.append(model)
        error_lists.append(error)
    return model_lists, error_lists

# construct BECO-ensemble-PFL model
def BECO_Ensemble_PFL(x, model_lists, error_lists):
    Ye = np.zeros((x.shape[0], 2))
    # ensemble with combined output
    for i in range(len(model_lists)):
        y_prob = model_lists[i].predict(x)
        w = error_lists[i] / np.sum(error_lists)
        Ye_prob += w * y
    return Ye_prob

# construct BECO-ensemble-test model
def BECO_Ensemble_test(x, y, model_lists, error_lists):
    y_prob = BECO_Ensemble_PFL(X, model_lists, error_lists)
    y_pre = np.argmax(y_prob, axis=1)
    auc = roc_auc_score(y, y_prob[0, :])
    return auc
