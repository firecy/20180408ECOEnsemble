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

from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering, Birch
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance

# Calculate distances matrix between Minority samples and centers
def mc_distance(Min, C, distype):
    mcdist = np.zeros((len(Min), len(C)))
    for i in range(len(Min)):
        for j in range(len(C)):
            if distype == 'euclidean': mcdist[i, j] = distance.euclidean(Min[i], C[j])
            if distype == 'cosine': mcdist[i, j] = distance.cosine(Min[i], C[j])
    return mcdist

# Compute clustering with MiniBatchKMeans(2d), and return k clustering centers C
# and nn nearest samples of each clustering center clsnn
def train_Kmeans(Min, k, nn, batch_size=4):
    mbk = MiniBatchKMeans(init='k-means++', n_clusters=k,
                          batch_size=batch_size, n_init=10,
                          max_no_improvement=10, verbose=0)
    mbk.fit(Min)
    # get cluster centers C: R(k x d)
    C = mbk.cluster_centers_
    # calculate distances between samples and clustering centers mcdist: R(len(Min) x k)
    mcdist = mc_distance(Min, C, distype='cosine')
    # get nn nearest samples index of each clustering centers clsnn: Z(nn x k)
    clsnn = np.argsort(mcdist, axis=0)[0: nn]
    return C, clsnn

# Compute clustering with AgglomerativeClustering(2d), and return k clustering centers C
# and nn nearest samples of each clustering center clsnn
def train_AgglomerativeClustering(Min, k, nn, batch_size=None):
    average_linkage = AgglomerativeClustering(linkage="average", affinity="cosine",
        n_clusters=k)
    average_linkage.fit(Min)
    # get cluster centers C: R(k x d)
    minlabels = average_linkage.labels_
    C = np.zeros((k, Min.shape[1]))
    for i in range(k):
        C[i, :] = np.mean(Min[np.where(minlabels==i)])
    # calculate distances between samples and clustering centers mcdist: R(len(Min) x k)
    mcdist = mc_distance(Min, C, distype='euclidean')
    # get nn nearest samples index of each clustering centers clsnn: Z(nn x k)
    clsnn = np.argsort(mcdist, axis=0)[0: nn]
    return C, clsnn

def train_BIRCH(Min, k, nn, batch_size=None):
    birch = Birch(branching_factor=50, n_clusters=k, threshold=0.5, compute_labels=True)
    birch.fit(Min)
    # get cluster centers C：R(k x d)
    minlabels = birch.labels_
    C = np.zeros((k, Min.shape[1]))
    for i in range(k):
        C[i, :] = np.mean(Min[np.where(minlabels==i)])
    # calculate distances between samples and clustering centers mcdist: R(len(Min) x k)
    mcdist = mc_distance(Min, C, distype='euclidean')
    # get nn nearest samples index of each clustering centers clsnn: Z(nn x k)
    clsnn = np.argsort(mcdist, axis=0)[0: nn]
    return C, clsnn

def DGWCC(C, clsnn, n_new, k, nMin, Min):
    for i in range(k):
        idx = np.random.choice(clsnn[:, i], n_new)
        for j in range(n_new):
            x_new = C[i, :] + np.random.rand() * (Min[idx[j], :] - C[i, :])
            nMin.append(x_new)
    return nMin

def DGWDWC(C, clsnn, n_new, k, nMin, Min):
    for i in range(k):
        idx1 = np.random.choice(clsnn[:, i], n_new)
        idx2 = np.random.choice(clsnn[:, i], n_new)
        for j in range(n_new):
            x_new = Min[idx1[j], :] + np.random.rand() * (Min[idx2[j], :] - Min[idx1[j], :])
            nMin.append(x_new)
    return nMin

def CSOSDG(Min, N_new, k, nn, Cls, CSO_type):
    n0 = Min.shape[0]
    n1 = Min.shape[1]
    n2 = Min.shape[2]
    Min = np.reshape(Min, (n0, n1*n2))
    if (Cls >= 0 and Cls < 1): #if Cls == 'Kmeans':
        C, clsnn = train_Kmeans(Min=Min, k=k, nn=nn)
    elif (Cls >= 1 and Cls < 2): #elif Cls == 'HAC':
        C, clsnn = train_AgglomerativeClustering(Min=Min, k=k, nn=nn)
    else: #Cls == 'BRICH'
        C, clsnn = train_BIRCH(Min=Min, k=k, nn=nn)
    n_new = N_new // k
    nMin = []
    if (CSO_type >= 0 and CSO_type < 0.5): #if CSO_type == 'CSO1':
        nMin = DGWCC(C, clsnn, n_new, k, nMin, Min)
    else: #
        nMin = DGWDWC(C, clsnn, n_new, k, nMin, Min)
    nMin = np.array(nMin)
    nMin = np.reshape(nMin, (nMin.shape[0], n1, n2))
    return nMin

def main():
    minority = np.load('dataset/codedata/80001/80001_xnor_pos_train2.npy')
    n0 = minority.shape[0]
    n1 = minority.shape[1]
    n2 = minority.shape[2]
    print (minority.shape)
    minority = np.reshape(minority, (n0, n1*n2))
    print (minority.shape)
    minority_new = CSOSDG(Min=minority, Cls='HAC', N_new=100, k=2, nn=5, CSO_type='CSO2')
    minority_new = np.array(minority_new)
    print (minority_new.shape)
    minority_new = np.reshape(minority_new, (minority_new.shape[0], n1, n2))
    np.save('dataset/codedata/80001/80001_xnor_posnew_train2', minority_new)
    print(len(minority_new))

if __name__ == '__main__':
    main()
