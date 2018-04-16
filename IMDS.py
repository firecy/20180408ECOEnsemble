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

from sklearn.clusters import MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.clusters import DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance

# Calculate distances matrix between Minority samples and centers
def mc_distance(Min, C, distype):
    mcdist = np.zeros((len(Min), len(C)))
    for i in range(len(Min)):
        for j in range(len(C)):
            if distype == 'euclideane': mcdist[i, j] = distance.euclideane(Min[i], C[j])
            if distype == 'cosine': mcdist[i, j] = distance.cosine(Min[i], C[j])
    return mcdist

# Compute clustering with MiniBatchKMeans(2d), and return k clustering centers C
# and nn nearest samples of each clustering center clsnn
def train_Kmeans(Min, k, nn, batch_size=5):
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=k,
                          batch_size=batch_size, n_init=10,
                          max_no_improvement=10, verbose=0)
    mbk.fit(Min)
    # get cluster centers C: R(k x d)
    C = mbk.cluster_centers_
    # calculate distances between samples and clustering centers mcdist: R(len(Min) x k)
    mcdist = mc_distance(Min, C, distype='euclideane')
    # get nn nearest samples index of each clustering centers clsnn: Z(nn x k)
    clsnn = np.argsort(mcdist, axis=0)[0: nn]
    return C, clsnn

# Compute clustering with AgglomerativeClustering(2d), and return k clustering centers C
# and nn nearest samples of each clustering center clsnn
def train_AgglomerativeClustering(Min, k, nn, batch_size=5):
    average_linkage = AgglomerativeClustering(linkage="average", affinity="cosine",
        n_clusters=k)
    average_linkage.fit(Min)
    # get cluster centers C: R(k x d)
    C = ?
    # calculate distances between samples and clustering centers mcdist: R(len(Min) x k)
    mcdist = mc_distance(Min, C, distype='cosine')
    # get nn nearest samples index of each clustering centers clsnn: Z(nn x k)
    clsnn = np.argsort(mcdist, axis=0)[0: nn]
    return C, clsnn
