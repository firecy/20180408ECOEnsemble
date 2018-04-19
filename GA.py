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

class pfl_fc_imb:
    def __init__(self, Min, y, N_max, N_min):
        self.Min = Min
        self.y = y
        self.N_max = N_max
        self.N_min = N_min
    def fitness(self, N_new, k, nn, cls, CSO_type):
        Min_new = CSOSDG(self.Min, cls, N_new, k, nn, CSO_type, batch_size)
        F1 = f1_score(self.y, y_test_pred, average='macro')
        return F1
    def get_bounds(self):
        return ([0, N_max], [0, N_min/2], [0, N_min-1], [0, 2], [0, 1])

def GA_training(dataset, y, pop_size, gen_max, cx, mx):
    # define ranges of variables
    # X is (data0set, data1set, ... ), dataset is array.
    # y is (label0set, label1set, ...), labelset is vector.
    num_x = []
    for i in range(len(y)):
        num_x.append(len(y[i]))
    N_max = np.max(num_x)
    N_min = np.min(num_x)
    # define the problem
    prob = pg.problem(pfl_fc_imb(dataset, y, N_max, N_min))
    # population initialization
    pop_init = pg.population(prob=prob, size=pop_size)
    # genetic algorithm construction
    gaalgo = pg.algorithm(sga((gen=gen_max, cr=cx, m=mx, param_s=2,
                             crossover="single", mutation="uniform",
                             selection="truncated", seed=random)))
    # run ga and get the best population
    pop_new = gaalgo.evolve(pop_init).get_x()
    return pop_new
