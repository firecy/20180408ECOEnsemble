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

from gaft.engine import GAEngine
from gaft.components.decimal_individual import DecimalIndividual
from gaft.components.population import Population
from gaft.operators.selection.roulette_wheel_selection import RouletteWheelSelection
from gaft.operators.crossover.uniform_corssover import UniformCrossover
from gaft.operators.mutation.flip_bit_mutation import FlipBitMuation
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis
from gaft.analysis.fitness_store import FitnessStore

def ECO_Ensemble(X, y, pop_size, gen_max, cx, mx, Ef, TC):
    # define ranges of variables
    # X is (data0set, data1set, ... ), dataset is array.
    # y is (label0set, label1set, ...), labelset is vector.
    num_x = []
    for i in range(len(y)):
        num_x.append(len(y[i]))
    N_max = np.max(num_x)
    N_min = np.min(num_x)

    # population initialization
    indv_template = DecimalIndividual(ranges=[(0, N_max), (0, 2), (0, N_min/2),
                                              (0, N_min-1), (0, 1)], eps=1)
    population_size = N_max * 3 * (N_min/2) * (N_min-1) * 2
    population = Population(indv_template=indv_template, size=population_size)
    population.init()

    # genetic operators
    selection = RouletteWheelSelection()
    crossover = UniformCrossover(pc=cx)
    mutation = FlipBitMuation(pm=mx)

    # create genetic algorithm engine
    Pool = GAEngine(population=population, selection=selection,
                    crossover=crossover, mutation=mutation,
                    analysis=[OnTheFlyAnalysis, FitnessStore])

    #define fitness evaluation
    @engine.fitness_register
    def fitness(indv):
        x
