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
from sklearn.metrics import f1_score
from IMDS import CSOSDG
from PFL import *

# define a binary problem
class BPFL_fc_imb:
    def __init__(self, xMin, xMaj, yMin, yMaj, model_old, lr, epoch, batch_size):
        #print ('__init__')
        self.xMin = xMin
        self.yMin = yMin
        self.xMaj = xMaj
        self.yMaj = yMaj
        self.model_old = model_old
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size

    # define fitness
    def fitness(self, x):
        #N_new, k, nn, Cls, CSO_type
        #print ('fitness')
        # generate new data
        #Min_new = CSOSDG(self.xMin, int(N_new), int(k), int(nn), Cls, CSO_type)
        Min_new = CSOSDG(self.xMin, int(x[0]), int(x[1]), int(x[2]), x[3], x[4])
        # construct new dataset
        x_new = np.vstack((self.xMaj, self.xMin, Min_new))
        y_new = np.hstack((self.yMaj, self.yMin, np.ones(Min_new.shape[0], )*self.yMin[0]))
        print(x_new.shape, y_new.shape)
        # finetune model based on new dataset
        F1 = model_train(x=x_new, y=y_new, model_old=self.model_old,
                         lr=self.lr, epoch=self.epoch, batch_size=self.batch_size)[1]
        return F1

    # define bounds
    def get_bounds(self):
        #print ('define bouns')
        N_max = len(self.yMaj)
        N_min = len(self.yMin)
        return ([0, 0, 0, 0, 0], [N_max, N_min/2, N_min-1, 3, 1])

# coding GA training and get best Ne(=pop_size) chromosomes
def BGA_train(xMin, yMin, xMaj, yMaj, model_old, lr, epoch, batch_size, pop_size, gen_max, cx, mx):
    # define the problem
    print ('define the problem')
    prob = pg.problem(BPFL_fc_imb(xMin, xMaj, yMin, yMaj,
                                 model_old, lr, epoch, batch_size))
    # population initialization
    print ('population initialization')
    pop_init = pg.population(prob=prob, size=pop_size)
    # genetic algorithm construction
    print ('genetic algorithm construction')
    gaalgo = pg.algorithm(pg.sga(gen=gen_max, cr=cx, m=mx, param_s=2,
                             crossover="single", mutation="uniform",
                             selection="truncated"))
    # run ga and get the best population
    print ('run ga and get the best population')
    pop_new = gaalgo.evolve(pop_init).get_x()
    return pop_new
