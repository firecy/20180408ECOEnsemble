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
from PFL import *

# aim at solving binary imbalance classifier problem
def BECO_Ensemble_train(Minset, Majset, model_old, lr, epoch, batch_size, pop_size, gen_max, cx, mx):
    # define ranges of variables
    # Minset is minority dataset (x, y), x is 3D array, and y is 1D vector.
    # Majset is majority dataset (x, y), x is 3D array, and y is 1D vector.
    xMin, yMin = Minset
    xMaj, yMaj = Majset
    # get the best chromosomes population based on binary ga
    print ('GA implementation')
    best_chromosomes_pop = GA.BGA_train(xMin=xMin, yMin=yMin,
                                           xMaj=xMaj, yMaj=yMaj,
                                           model_old=model_old, lr=lr,
                                           epoch=epoch, batch_size=batch_size,
                                           pop_size=pop_size, gen_max=gen_max,
                                           cx=cx, mx=mx)
    model_lists = []
    error_lists = []
    # train Ne(=pop_size) classifier
    print(best_chromosomes_pop)
    for i in range(len(best_chromosomes_pop)):
        N_new = best_chromosomes_pop[i][0]
        k = best_chromosomes_pop[i][1]
        nn = best_chromosomes_pop[i][2]
        Cls = best_chromosomes_pop[i][3]
        CSO_type = best_chromosomes_pop[i][4]
        Min_new = CSOSDG(xMin, N_new=int(N_new), k=int(k), nn=int(nn), Cls=Cls, CSO_type=CSO_type)
        # construct new dataset
        x_new = np.vstack((xMaj, xMin, Min_new))
        y_new = np.hstack((yMaj, yMin, np.ones(len(Min_new), )*yMin[0]))
        # finetune model based on new dataset
        model, error = model_train(x=x_new, y=y_new, model_old=model_old,
                         lr=lr, epoch=epoch, batch_size=batch_size)
        model_lists.append(model)
        error_lists.append(error)
    save_results(model_lists, error_lists)
    return model_lists, error_lists

# construct BECO-ensemble-PFL model
def BECO_Ensemble_PFL(x, model_lists, error_lists):
    Ye_prob = np.zeros((x.shape[0], 2))
    # ensemble with combined output
    for i in range(len(model_lists)):
        y_prob = model_lists[i].predict(x, batch_size=10)
        w = error_lists[i] / np.sum(error_lists)
        Ye_prob += w * y_prob
    return Ye_prob

# construct BECO-ensemble-test model
def BECO_Ensemble_test(x, y, model_lists, error_lists):
    y_prob = BECO_Ensemble_PFL(x, model_lists, error_lists)
    y_pre = np.argmax(y_prob, axis=1)
    auc = roc_auc_score(y, y_pre)
    return auc

def save_results(model_lists, error_lists):
    for i in range(len(model_lists)):
        model_path = 'ECOR/orginal_lstmsaeatt_clf80001nor_architechture'
        weights_path = 'ECOR/original_lstmsaeatt_clf80001nor_weights'
        model_path = model_path + str(i) + '.json'
        weights_path = weights_path + str(i) + '.h5'
        save_model(model_lists[i], model_path, weights_path)
    np.save('ECOR/original_error_lists_80001nor', error_lists)

def main():
    print ('load data')
    minority_x = np.load('dataset/codedata/80001/80001_xnor_pos_train2.npy')
    minority_y = np.load('dataset/codedata/80001/80001_y_pos_train2.npy')
    print(minority_y, minority_x.shape)
    majority_x = np.load('dataset/codedata/80001/80001_xnor_neg_train2.npy')
    majority_y = np.load('dataset/codedata/80001/80001_y_neg_train2.npy')
    print(majority_y, majority_x.shape)
    model_path = 'model/encoder_lstmsaeatt_nor_architechture2.json'
    weights_path = 'model/encoder_lstmsaeatt_nor_weights2.h5'
    encoder = load_model(model_path, weights_path)
    model_path2 = 'model/encoder_lstmsaeatt_clf80001nor_architechture2.json'
    weights_path2 = 'model/encoder_lstmsaeatt_clf80001nor_weights2.h5'
    clf = load_model(model_path2, weights_path2)
    print ('BECO_Ensemble_train')
    model_lists, error_lists = BECO_Ensemble_train(Minset=(minority_x, minority_y),
                                Majset=(majority_x, majority_y),
                                model_old=(model_path2, weights_path2),
                                lr=0.0003,
                                epoch=2000,
                                batch_size=10,
                                pop_size=2,
                                gen_max=2,
                                cx=0.5,
                                mx=0.2)
    print(model_lists, len(model_lists))
    print(error_lists, len(error_lists))
    test_x = np.load('dataset/codedata/80001/80001_xnor_test.npy')
    test_y = np.load('dataset/codedata/80001/80001_y_test.npy')
    test_x = missdata_implement(test_x, test_x[0].shape[1])
    test_x = np.vstack((test_x, test_x, test_x, test_x, test_x))
    test_y = np.hstack((test_y, test_y, test_y, test_y, test_y))
    encoder = Model(inputs=encoder.input, outputs=encoder.layers[4].output)
    test_x = encoder.predict(test_x, batch_size=25)

    auc = BECO_Ensemble_test(test_x, test_y, model_lists, error_lists)
    print(auc)

if __name__ == '__main__':
    main()
