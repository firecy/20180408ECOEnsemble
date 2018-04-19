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
from PFL import model_train

# define a binary problem
class BPFL_fc_imb:
    def __init__(self, xMin, xMaj, yMin, yMaj, model_old, lr, epoch, batch_size):
        self.xMin = xMin
        self.yMin = yMin
        self.xMaj = xMaj
        self.yMaj = yMaj
        self.model_old = model_old
        self.lr = lr
        self.epoch = epoch
        self.batch_size = batch_size

    # define fitness
    def fitness(self, N_new, k, nn, Cls, CSO_type):
        # generate new data
        Min_new = CSOSDG(self.xMin, Cls, N_new, k, nn, CSO_type, batch_size)
        # construct new dataset
        x_new = np.vstack((self.xMaj, self.xMin, Min_new))
        y_new = np.hstack((self.yMaj, self.yMin, np.ones(N_new, )*self.yMin[0]))
        # finetune model based on new dataset
        F1 = model_train(x=x_new, y=y_new, model=self.model_old,
                         lr=self.lr, epoch=self.epoch, batch_size=self.batch_size)[0]
        return F1

    # define bounds
    def get_bounds(self):
        N_max = len(self.yMaj)
        N_min = len(self.yMin)
        return ([0, N_max], [0, N_min/2], [0, N_min-1], [0, 2], [0, 1])

# coding GA training and get best Ne(=pop_size) chromosomes
def BGA_train(xMin, yMin, xMaj, yMaj, model_old, lr, epoch, batch_size, pop_size, gen_max, cx, mx):
    # define the problem
    prob = pg.problem(BPFL_fc_imb(xMin, xMaj, yMin, yMaj,
                                 model_old, lr, epoch, batch_size))
    # population initialization
    pop_init = pg.population(prob=prob, size=pop_size)
    # genetic algorithm construction
    gaalgo = pg.algorithm(sga((gen=gen_max, cr=cx, m=mx, param_s=2,
                             crossover="single", mutation="uniform",
                             selection="truncated", seed=random)))
    # run ga and get the best population
    pop_new = gaalgo.evolve(pop_init).get_x()
    return pop_new
