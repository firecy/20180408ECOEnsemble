#!/usr/bin/python
# -*- coding: utf-8 -*-

import os
import sys

import timeit
from datetime import datetime
import time
import gc

import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from pandas.tools.plotting import parallel_coordinates
from pandas.tools.plotting import radviz

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

def attention(x, c_num):
    c = []
    for i in range(len(x)):
        attention_weights = # size:cxt
        c.append(np.dot(attent_weights, x[i]))
    return np.array(c)

def attention_inverse(x, timesteps):
    h = []
    for i in range(len(x)):
        attention_inverse_weights = # size:txc
        h.append(np.dot(attention_inverse_weights, x[i]))
    return np.array(h)

def create_lstmsae_model(input_dim, hidden_unit, initializer, batch_size, timesteps,
                 c_num, lr):
    # build the encoder
    model = Sequential()
    model2 = Sequential()
    model.add(LSTM(hidden_unit, return_sequences=True,
                    stateful=True,
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10),
                    batch_input_shape=(batch_size, timesteps, input_dim)))
    model2.add(LSTM(hidden_unit, return_sequences=True,
                    stateful=True, go_backwards=True,
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10),
                    batch_input_shape=(batch_size, timesteps, input_dim)))
    model.add(Concatenate([encoder, encoder2]))
    model.add(Lambda(attention(c_num=c_num), output_shape=(batch_size, c_num, hidden_unit)))
    # build decoder
    model.add(Lambda(attention_inverse(timesteps=timesteps),
                    output_shape=(batch_size, timesteps, hidden_unit))
    model.add(LSTM(input_dim, return_sequences=True,
                   stateful=True, kernel_initializer=initializer,
                   activity_regularizer=regularizers.l2(10e-10)))
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mse')
    return model

def train_lstmsae_model(trainset, hidden_unit, initializer, batch_size, c_num, lr, epoch):
    x = trainset
    model = KerasRegressor(build_fn=create_lstmsae_model, input_dim=x.shape[2],
                           timesteps=x.shape[1], verbose=0)
    hidden_unit = hidden_unit
    initializer = initializer
    batch_size = batch_size
    c_num = c_num
    lr = lr
    epoch = epoch
    param_grid = dict(lr=lr, batch_size=batch_size, initializer=initializer,
                      nb_epoch=epoch, hidden_unit= hidden_unit, c_num=c_num,
                      batch_size=batch_size)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,
                        scoring='r2', cv=10)
    print "pr no fault"
    grid_result = grid.fit(x, x)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    del grid, grid_result
    gc.collect()
    return grid_result.best_estimator_.model, grid_result.best_params_

def create_lstmclf_model(input_dim, timesteps, batch_size, output_dim, lr):
    model = Sequential()
    model.add(LSTM(input_dim,
                    stateful=True,
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10),
                    batch_input_shape=(batch_size, timesteps, input_dim)))
    model.add(Dense(output_dim, activation='softmax'))
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_lstmclf_model(trainset, batch_size, output_dim, lr, fine_epoch, initializer):
    x, y = trainset
    nb_classes = len(set(y))
    model = KerasClassifier(build_fn=create_lstmclf_model, input_dim=x.shape[2],
                           timesteps=x.shape[1], verbose=0)
    lr = lr
    epoch = fine_epoch
    batch_size = batch_size
    param_grid = dict(lr=lr, batch_size=batch_size, initializer=initializer,
                      nb_epoch=epoch, batch_size=batch_size)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,
                        scoring='accuracy', cv=10)
    print "pr no fault"
    grid_result = grid.fit(x, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    del grid, grid_result
    gc.collect()
    return grid_result.best_estimator_.model, grid_result.best_params_

def train_pflt_model1(dataset):

def train_lstmsae_pflt_model2(dataset, hidden_unit, initializer, c_num, lr1,
                            batch_size, pre_epoch, lr2, fine_epoch):
    x, y = dataset
    nb_classes = len(set(y))
    y_htc = np_utils.to_categorical(y, nb_classes)
    timesteps = x.shape[1]
    input_dim = x.shape[2]
    batch_size = batch_size
    # build encoder
    main_input = Input(batch_shape=(batch_size, timesteps, input_dim))
    encoder1 = LSTM(hidden_unit, return_sequences=True,
                    stateful=True,
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10))(main_input)
    encoder2 = LSTM(hidden_unit, return_sequences=True,
                    stateful=True, go_backwards=True,
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10))(main_input)
    encoder = Concatenate([encoder1, encoder2])
    encoder = Lambda(attention(c_num=c_num),
                    output_shape=(c_num, hidden_unit))(encoder)
    # build encoder
    decoder = Lambda(attention_inverse(timesteps=timesteps),
                    output_shape=(batch_size, timesteps, hidden_unit))(encoder)
    decoder = LSTM(input_dim, return_sequences=True,
                   stateful=True, kernel_initializer=initializer,
                   activity_regularizer=regularizers.l2(10e-10))(decoder)
    Decoder = Model(inputs=main_input, outputs=decoder)
    Encoder = Model(inputs=main_input, outputs=encoder)
    # compile model
    adam = Adam(lr=lr1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    Decoder.compile(optimizer=adam, loss='mse')
    # train model
    epochs = pre_epoch
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    Decoder.fit(x=x, y=x, batch_size=batch_size, epochs=pre_epoch,
                verbose=1, callbacks=[early_stopping], validation_split=0.3,
                shuffle=True)
    # build classifier
    classifier = LSTM(hidden_unit,
                    stateful=True,
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10))(encoder)
    classifier = Dense(nb_classes, activation='softmax')
    PFL_model = Model(inputs=main_input, outputs=classifier)
    # compile PFL model
    adam = Adam(lr=lr2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    PFL_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # train PFL model
    epochs = fine_epoch
    PFL_model.fit(x=x, y=y_htc, batch_size=batch_size, epochs=fine_epoch,
                  verbose=1, callbacks=[early_stopping], validation_split=0.3,
                  shuffle=True)
    return Encoder, Decoder, PFL_model

def sampling(args, batch_size=1380, latent_dim=24, eplsion_std=0.01):
    z_mean, z_log_var = args
    eplsion = K.randon_normal(shape=(batch_size, latent_dim,
                              mean=0., std=eplsion_std))
    return z_mean + K.exp(z_log_var / 2.) * epsilon

def vale_loss(x, decoder_mean):
    xent_loss =
    kl_loss = -0.5 * K.sum(1 + encoder_z_log_var - K.square(encoder_z_mean)
              - K.exp(encoder_z_log_var), axis=-1)
    return xent_loss + kl_loss

def train_lstmvae_pflt_model2(dataset, hidden_list, initializer, c_num, lr1,
                            batch_size, pre_epoch, lr2, fine_epoch):
    x, y = dataset
    nb_classes = len(set(y))
    y_htc = np_utils.to_categorical(y, nb_classes)
    timesteps = x.shape[1]
    input_dim = x.shape[2]
    batch_size = batch_size
    # build encoder
    main_input = Input(batch_shape=(batch_size, timesteps, input_dim))
    encoder_x1 = LSTM(hidden_list[0], return_sequences=True,
                    stateful=True,
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10))(main_input)
    encoder_x2 = LSTM(hidden_list[0], return_sequences=True,
                    stateful=True, go_backwards=True,
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10))(main_input)
    encoder_h = Concatenate([encoder_x1, encoder_x2])
    encoder_h = Lambda(attention(c_num=c_num),
                    output_shape=(batch_size, c_num, hidden_unit))(encoder_h)
    encoder_z_mean = LSTM(hidden_list[1], return_sequences=True,
                    stateful=True,
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10))(encoder_h)
    encoder_z_log_var = LSTM(hidden_list[1], return_sequences=True,
                        stateful=True,
                        kernel_initializer=initializer,
                        activity_regularizer=regularizers.l2(10e-10))(encoder_h)
    encoder = Lambda(sampling(batch_size=batch_size, latent_dim=hidden_list[1]),
        output_shape=(c_num, hidden_list[1]))([encoder_z_mean, encoder_z_log_var])
    # build encoder
    decoder = Lambda(attention_inverse(timesteps=timesteps),
                    output_shape=(timesteps, hidden_unit[1]))(encoder)
    decoder = LSTM(hidden_list[0], return_sequences=True,
                    stateful=True, kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10))(decoder)
    decoder_mean = LSTM(input_dim, return_sequences=True,
                   stateful=True, kernel_initializer=initializer,
                   activity_regularizer=regularizers.l2(10e-10))(decoder)
    VAE = Model(inputs=main_input, outputs=decoder_mean)
    Encoder = Model(inputs=main_input, outputs=encoder)
    # compile model
    adam = Adam(lr=lr1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    VAE.compile(optimizer=adam, loss=vae_loss)
    # train model
    epochs = pre_epoch
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    VAE.fit(x=x, y=x, batch_size=batch_size, epochs=pre_epoch,
                verbose=1, callbacks=[early_stopping], validation_split=0.3,
                shuffle=True)
    # build classifier
    classifier = LSTM(hidden_list[1],
                    stateful=True,
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10))(encoder)
    classifier = Dense(nb_classes, activation='softmax')
    PFL_model = Model(inputs=main_input, outputs=classifier)
    # compile PFL model
    adam = Adam(lr=lr2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    PFL_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # train PFL model
    epochs = fine_epoch
    PFL_model.fit(x=x, y=y_htc, batch_size=batch_size, epochs=fine_epoch,
                  verbose=1, callbacks=[early_stopping], validation_split=0.3,
                  shuffle=True)
    return Encoder, Decoder, PFL_model
