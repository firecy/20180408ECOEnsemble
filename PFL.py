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
from sklearn.metrics import f1_score, recall_score, accuracy_score
from sklearn.cross_validation import train_test_split

from keras.layers import Input, Dense, LSTM, Concatenate, Lambda, merge, RepeatVector, Masking
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.losses import MSLE
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
from attention import *

def extract_feature(model_path, weights_path, dataset):
    model = model_from_json(open(model_path).read())
    model.load_weights(weights_path)
    k = len(model.layers)
    x_new = []
    for i in range(len(dataset)):
        x_new0 = get_output_of_layer(model, dataset[i], k-2)
        x_new.append(x_new0)
    return x_new

def repeatmat(C, timesteps, c_num):
    reapeat_num = timesteps // c_num
    H2 = K.repeat_elements(C, reapeat_num, axis=1)
    print (H2.shape)
    return H2

def create_lstmsae_model(input_dim, hidden_unit, initializer, batch_size, timesteps,
                 c_num, lr, x_ins):
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
    model.add(Concatenate([model, model2]))
    model.add(Lambda(attention(c_num=c_num, x_ins=x_ins), output_shape=(c_num, hidden_unit)))
    # build decoder
    model.add(Lambda(repeatmat(timesteps=timesteps), output_shape=(timesteps, hidden_unit)))
    model.add(LSTM(input_dim, return_sequences=True,
                   stateful=True, kernel_initializer=initializer,
                   activity_regularizer=regularizers.l2(10e-10)))
    adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='mse')
    return model

def train_lstmsae_model(trainset, hidden_unit, initializer, batch_size, c_num, lr, epoch):
    x = trainset
    model = KerasRegressor(build_fn=create_lstmsae_model, input_dim=x[0].shape[1],
                           timesteps=x[0].shape[0], verbose=0, x_ins=x)
    hidden_unit = hidden_unit
    initializer = initializer
    batch_size = batch_size
    c_num = c_num
    lr = lr
    epoch = epoch
    param_grid = dict(lr=lr, batch_size=batch_size, initializer=initializer,
                      nb_epoch=epoch, hidden_unit=hidden_unit, c_num=c_num)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,
                        scoring='r2', cv=10)
    print("pr no fault")
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
                      nb_epoch=epoch)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1,
                        scoring='accuracy', cv=10)
    print("pr no fault")
    grid_result = grid.fit(x, y)
    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    del grid, grid_result
    gc.collect()
    return grid_result.best_estimator_.model, grid_result.best_params_

def train_lstmsae_pflt_model(dataset, hidden_unit, initializer, c_num, lr1,
                            batch_size, pre_epoch, lr2, fine_epoch):
    x, y = dataset
    nb_classes = len(set(y))
    y_htc = np_utils.to_categorical(y, nb_classes)
    timesteps = x[0].shape[0]
    input_dim = x[0].shape[1]
    batch_size = batch_size #[5, 10, 20]
    hidden_unit = hidden_unit #[24, 48, 96, 192, 384]
    c_num = c_num #[10, 20, 23, 46, 60, 69, 138, 345]
    lr1 = lr1 #[0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    pre_epoch = pre_epoch #[200, 500, 800, 1000]
    initializer = initializer #['orthogonal']
    fine_epoch = fine_epoch #[2000]
    model, overparams = train_lstmsae_model(trainset=x, hidden_unit=hidden_unit,
                    initializer=initializer, batch_size=batch_size, c_num=c_num,
                    lr=lr1, epoch=pre_epoch)
    for i in range(2):
        model.pop()
    model.add(LSTM(overparams['hidden_unit'],
                    kernel_initializer='orthogonal',
                    activity_regularizer=regularizers.l2(10e-10)))
    model.add(Dense(nb_classes, activation='softmax'))
    adam = Adam(lr=lr2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(x=x, y=y_htc, batch_size=overparams['batch_size'], epochs=fine_epoch,
                  verbose=1, callbacks=[early_stopping], validation_split=0.3,
                  shuffle=True)
    return model, overparams

# lstm, attention, sae, pfl
def train_lstmsae_pflt_model1(dataset, hidden_unit, initializer, c_num, lr1,
                            batch_size, pre_epoch, lr2, fine_epoch):
    x, y = dataset
    nb_classes = len(set(y))
    y_htc = np_utils.to_categorical(y, nb_classes)
    timesteps = x.shape[1]
    input_dim = x.shape[2]
    batch_size = batch_size #[5, 10, 20]
    hidden_unit = hidden_unit #[24, 48, 96, 192, 384]
    c_num = c_num #[10, 20, 23, 46, 60, 69, 138, 345]
    lr1 = lr1 #[0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    pre_epoch = pre_epoch #[200, 500, 800, 1000]
    initializer = initializer #['orthogonal']
    fine_epoch = fine_epoch #[2000]
    # build the encoder
    main_input = Input(batch_shape=(batch_size, timesteps, input_dim))
    main_input = Masking(mask_value=-5)(main_input)
    encoder_x1 = LSTM(hidden_unit, return_sequences=True,
                    stateful=True, activation='softsign',
                    kernel_initializer=initializer)(main_input)
    encoder_x2 = LSTM(hidden_unit, return_sequences=True,
                    stateful=True, go_backwards=True, activation='softsign',
                    kernel_initializer=initializer)(main_input)
    encoder_h = merge([encoder_x1, encoder_x2], mode='concat')
    encoder_c = Attention(c_num=c_num, timesteps=timesteps,
                    x_dim=input_dim, h_dim=hidden_unit)([main_input, encoder_h])
    # build decoder
    decoder = Lambda(repeatmat, arguments={'timesteps':timesteps, 'c_num':c_num},
                    output_shape=(timesteps, hidden_unit*2))(encoder_c)
    decoder = LSTM(input_dim, return_sequences=True, activation='softsign',
                   stateful=True, kernel_initializer=initializer)(decoder)
    model = Model(inputs=main_input, outputs=decoder)
    rmsprop = RMSprop(lr=lr1)
    #optimizer = RMSprop(lr=lr1)
    model.compile(optimizer=rmsprop, loss='msle', metrics=['mae', 'mse'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(x=x, y=x, batch_size=batch_size, epochs=pre_epoch,
                  verbose=1, validation_split=0.2,
                  shuffle=True,
                  callbacks=[early_stopping ])
    # build clf
    cfl = Lambda(repeatmat, arguments={'timesteps':c_num, 'c_num':c_num},
                    output_shape=(timesteps, hidden_unit*2))(encoder_c)
    cfl = LSTM(hidden_unit, return_sequences=False, activation='softsign',
                    kernel_initializer='orthogonal')(cfl)
    #cfl = Attention(c_num=1, timesteps=c_num,
    #                x_dim=hidden_unit*2, h_dim=hidden_unit)([encoder_c, cfl])
    cfl = Dense(nb_classes, activation='softmax')(cfl)
    pfl = Model(inputs=main_input, outputs=cfl)
    for i in range(len(pfl.layers)-2):
        pfl.layers[i+1].trainable = False
    rmsprop = RMSprop(lr=lr2)
    pfl.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    pfl.fit(x=x, y=y_htc, batch_size=batch_size, epochs=fine_epoch,
                  verbose=1, callbacks=[early_stopping], validation_split=0.2,
                  shuffle=True)
    return pfl

def train_lstmsaet_model1(dataset, hidden_unit, initializer, c_num, lr1,
                            batch_size, pre_epoch):
    x = dataset
    timesteps = x.shape[1]
    input_dim = x.shape[2]
    batch_size = batch_size #[5, 10, 20]
    hidden_unit = hidden_unit #[24, 48, 96, 192, 384]
    c_num = c_num #[10, 20, 23, 46, 60, 69, 138, 345]
    lr1 = lr1 #[0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    pre_epoch = pre_epoch #[200, 500, 800, 1000]
    initializer = initializer #['orthogonal']
    # build the encoder
    main_input = Input(batch_shape=(batch_size, timesteps, input_dim))
    encoder_x1 = LSTM(hidden_unit, return_sequences=True,
                    stateful=True, activation='softsign',
                    kernel_initializer=initializer)(main_input)
    encoder_x2 = LSTM(hidden_unit, return_sequences=True,
                    stateful=True, go_backwards=True, activation='softsign',
                    kernel_initializer=initializer)(main_input)
    encoder_h = merge([encoder_x1, encoder_x2], mode='concat')
    encoder_c = Attention(c_num=c_num, timesteps=timesteps,
                    x_dim=input_dim, h_dim=hidden_unit)([main_input, encoder_h])
    # build decoder
    decoder = Lambda(repeatmat, arguments={'timesteps':timesteps, 'c_num':c_num},
                    output_shape=(timesteps, hidden_unit*2))(encoder_c)
    decoder = LSTM(input_dim, return_sequences=True, activation='softsign',
                   stateful=True, kernel_initializer=initializer)(decoder)
    model = Model(inputs=main_input, outputs=decoder)
    rmsprop = RMSprop(lr=lr1)
    #optimizer = RMSprop(lr=lr1)
    model.compile(optimizer=rmsprop, loss='msle', metrics=['mae', 'mse'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(x=x, y=x, batch_size=batch_size, epochs=pre_epoch,
                  verbose=1, validation_split=0.2,
                  shuffle=True,
                  callbacks=[early_stopping ])
    #encoder = Model(inputs=main_input, outputs=encoder_c)
    return model

def train_lstmsaet_clf_model1(dataset, model, hidden_unit, batch_size, lr2, fine_epoch):
    x, y = dataset
    hid0 = x.shape[2]
    model = Model(inputs=model.input, outputs=model.layers[4].output)
    x = model.predict(x, batch_size=25)
    nb_classes = len(set(y))
    y_htc = np_utils.to_categorical(y, nb_classes)
    x, y_htc = shuffle(x, y_htc)
    timesteps = x.shape[1]
    input_dim = x.shape[2]
    batch_size = batch_size #[5, 10, 20]
    hidden_unit = hidden_unit #[24, 48, 96, 192, 384]
    fine_epoch = fine_epoch #[2000]
    # build the encoder
    main_input = Input(batch_shape=(batch_size, timesteps, input_dim))
    #main_input = Masking(mask_value=0)(main_input)
    # build clf
    #clf = BatchNormalization()(main_input)
    clf = LSTM(hid0, return_sequences=True, activation='softsign',
                    kernel_initializer='orthogonal')(main_input)
    #clf = BatchNormalization()(clf)
    clf = LSTM(hidden_unit, return_sequences=False, activation='softsign',
                    kernel_initializer='orthogonal')(clf)
    clf = BatchNormalization()(clf)
    #cfl = Attention(c_num=1, timesteps=c_num,
    #                x_dim=hidden_unit*2, h_dim=hidden_unit)([encoder_c, cfl])
    clf = Dense(nb_classes, activation='softmax')(clf)
    pfl = Model(inputs=main_input, outputs=clf)
    rmsprop = RMSprop(lr=lr2)
    pfl.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    pfl.fit(x=x, y=y_htc, batch_size=batch_size, epochs=fine_epoch,
                  verbose=1, callbacks=[early_stopping], validation_split=0.2,
                  shuffle=True)
    return pfl
# lstm, vector, sae, pfl
def train_lstmsae_pflt_model2(dataset, hidden_unit, initializer, lr1,
                            batch_size, pre_epoch, lr2, fine_epoch):
    x, y = dataset
    nb_classes = len(set(y))
    y_htc = np_utils.to_categorical(y, nb_classes)
    timesteps = x[0].shape[0]
    input_dim = x[0].shape[1]
    batch_size = batch_size #[5, 10, 20]
    hidden_unit = hidden_unit #[24, 48, 96, 192, 384]
    lr1 = lr1 #[0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    pre_epoch = pre_epoch #[200, 500, 800, 1000]
    initializer = initializer #['orthogonal']
    fine_epoch = fine_epoch #[2000]
    # build the encoder
    main_input = Input(batch_shape=(batch_size, timesteps, input_dim))
    main_input = Masking(mask_value=-5)(main_input)
    encoder_x1 = LSTM(hidden_unit, return_sequences=True, activation='softsign',
                    stateful=True,
                    kernel_initializer=initializer)(main_input)
    encoder_x2 = LSTM(hidden_unit, return_sequences=True, activation='softsign',
                    stateful=True, go_backwards=True,
                    kernel_initializer=initializer)(main_input)
    encoder_h = merge([encoder_x1, encoder_x2], mode='concat')
    encoder_c = LSTM(hidden_unit, return_sequences=False, activation='softsign',
                    kernel_initializer=initializer)(encoder_h)
    # build decoder
    decoder = RepeatVector(timesteps)(encoder_c)
    decoder = LSTM(input_dim, return_sequences=True, activation='softsign',
                   stateful=True, kernel_initializer=initializer)(decoder)
    model = Model(inputs=main_input, outputs=decoder)
    rmsprop = RMSprop(lr=lr1)
    #optimizer = RMSprop(lr=lr1)
    model.compile(optimizer=rmsprop, loss='msle', metrics=['mae', 'mse'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(x=x, y=x, batch_size=batch_size, epochs=pre_epoch,
                  verbose=1, validation_split=0.2, callbacks=[early_stopping],
                  shuffle=True)
    # build clf
    cfl = Dense(nb_classes, activation='softmax')(encoder_c)
    pfl = Model(inputs=main_input, outputs=cfl)
    for i in range(len(pfl.layers)-2):
        pfl.layers[i+1].trainable = False
    rmsprop = RMSprop(lr=lr2)
    pfl.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    pfl.fit(x=x, y=y_htc, batch_size=batch_size, epochs=fine_epoch,
                  verbose=1, callbacks=[early_stopping], validation_split=0.3,
                  shuffle=True)
    return pfl

def train_lstmsaet_model2(dataset, hidden_unit, initializer, lr1,
                            batch_size, pre_epoch):
    x = dataset
    timesteps = x[0].shape[0]
    input_dim = x[0].shape[1]
    batch_size = batch_size #[5, 10, 20]
    hidden_unit = hidden_unit #[24, 48, 96, 192, 384]
    lr1 = lr1 #[0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    pre_epoch = pre_epoch #[200, 500, 800, 1000]
    initializer = initializer #['orthogonal']
    # build the encoder
    main_input = Input(batch_shape=(batch_size, timesteps, input_dim))
    encoder_x1 = LSTM(hidden_unit, return_sequences=True, activation='softsign',
                    stateful=True,
                    kernel_initializer=initializer)(main_input)
    encoder_x2 = LSTM(hidden_unit, return_sequences=True, activation='softsign',
                    stateful=True, go_backwards=True,
                    kernel_initializer=initializer)(main_input)
    encoder_h = merge([encoder_x1, encoder_x2], mode='concat')
    encoder_c = LSTM(hidden_unit, return_sequences=False, activation='softsign',
                    kernel_initializer=initializer)(encoder_h)
    # build decoder
    decoder = RepeatVector(timesteps)(encoder_c)
    decoder = LSTM(input_dim, return_sequences=True, activation='softsign',
                   stateful=True, kernel_initializer=initializer)(decoder)
    model = Model(inputs=main_input, outputs=decoder)
    rmsprop = RMSprop(lr=lr1)
    #optimizer = RMSprop(lr=lr1)
    model.compile(optimizer=rmsprop, loss='msle', metrics=['mae', 'mse'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(x=x, y=x, batch_size=batch_size, epochs=pre_epoch,
                  verbose=1, validation_split=0.2, callbacks=[early_stopping],
                  shuffle=True)
    encoder = Model(inputs=main_input, outputs=encoder_c)
    return encoder

def train_lstmsaet_clf_model2(dataset, model, hidden_unit, batch_size, lr2, fine_epoch):
    x, y = dataset
    hid0 = x.shape[2]
    model = Model(inputs=model.input, outputs=model.layers[4].output)
    x = model.predict(x, batch_size=25)
    print (x.shape)
    nb_classes = len(set(y))
    y_htc = np_utils.to_categorical(y, nb_classes)
    x, y_htc = shuffle(x, y_htc)
    input_dim = x.shape[1]
    batch_size = batch_size #[5, 10, 20]
    hidden_unit = hidden_unit #[24, 48, 96, 192, 384]
    fine_epoch = fine_epoch #[2000]
    # build the encoder
    main_input = Input(batch_shape=(batch_size, input_dim))
    clf = Dense(hid0, activation='relu')(main_input)
    clf = Dense(hidden_unit, activation='relu')(clf)
    clf = BatchNormalization()(clf)
    clf = Dense(nb_classes, activation='softmax')(clf)
    pfl = Model(inputs=main_input, outputs=clf)
    rmsprop = RMSprop(lr=lr2)
    pfl.compile(optimizer=rmsprop, loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    pfl.fit(x=x, y=y_htc, batch_size=batch_size, epochs=fine_epoch,
                  verbose=1, callbacks=[early_stopping], validation_split=0.2,
                  shuffle=True)
    return pfl

def sampling(args, batch_size=1380, latent_dim=24, timesteps=1380, epsilon_std=0.01):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, timesteps, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2.) * epsilon

# lstm, attention, vae, pfl
def train_lstmvae_pflt_model1(dataset, hidden_list, initializer, c_num, lr1,
                            batch_size, pre_epoch, lr2, fine_epoch):
    x, y = dataset
    nb_classes = len(set(y))
    y_htc = np_utils.to_categorical(y, nb_classes)
    timesteps = x.shape[1]
    input_dim = x.shape[2]
    batch_size = batch_size
    # build encoder
    main_input = Input(batch_shape=(batch_size, timesteps, input_dim))
    main_input = Masking(mask_value=-5)(main_input)
    encoder_x1 = LSTM(hidden_list[0], return_sequences=True,
                    stateful=True,
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10))(main_input)
    encoder_x2 = LSTM(hidden_list[0], return_sequences=True,
                    stateful=True, go_backwards=True,
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10))(main_input)
    encoder_h = merge([encoder_x1, encoder_x2], mode='concat')
    encoder_c = Attention(c_num=c_num, timesteps=timesteps,
                    x_dim=input_dim, h_dim=hidden_list[0])([main_input, encoder_h])
    encoder_c = Lambda(repeatmat, arguments={'timesteps':c_num, 'c_num':c_num},
                    output_shape=(timesteps, hidden_list[0]*2))(encoder_c)
    encoder_z_mean = LSTM(hidden_list[1], return_sequences=True,
                    stateful=True,
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10))(encoder_c)
    encoder_z_log_var = LSTM(hidden_list[1], return_sequences=True,
                        stateful=True,
                        kernel_initializer=initializer,
                        activity_regularizer=regularizers.l2(10e-10))(encoder_c)
    encoder = Lambda(sampling, arguments={'batch_size':batch_size,
                    'latent_dim': hidden_list[1], 'timesteps':c_num},
                    output_shape=(c_num, hidden_list[1]))([encoder_z_mean, encoder_z_log_var])
    # build encoder
    decoder = Lambda(repeatmat, arguments={'timesteps':timesteps, 'c_num':c_num},
                    output_shape=(timesteps, hidden_list[1]))(encoder)
    decoder = LSTM(hidden_list[0]*2, return_sequences=True,
                    stateful=True, kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10))(decoder)
    decoder_mean = LSTM(input_dim, return_sequences=True,
                   stateful=True, kernel_initializer=initializer,
                   activity_regularizer=regularizers.l2(10e-10))(decoder)
    VAE = Model(inputs=main_input, outputs=decoder_mean)
    Encoder = Model(inputs=main_input, outputs=encoder)
    # compile model
    adam = Adam(lr=lr1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    def vae_loss(x, decoder_mean):
        xent_loss = K.sum(MSLE(x, decoder_mean))
        kl_loss = -0.5 * K.sum(K.sum(1 + encoder_z_log_var - K.square(encoder_z_mean)
                  - K.exp(encoder_z_log_var), axis=-1))
        return xent_loss + kl_loss
    VAE.compile(optimizer=adam, loss=vae_loss, metrics=['msle', 'mse'])
    # train model
    epochs = pre_epoch
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    VAE.fit(x=x, y=x, batch_size=batch_size, epochs=pre_epoch,
                verbose=1, callbacks=[early_stopping], validation_split=0.2,
                shuffle=True)
    # build classifier
    #classifier = Lambda(repeatmat, arguments={'timesteps':c_num, 'c_num':c_num},
    #                output_shape=(c_num, hidden_list[1]))(encoder)
    classifier = LSTM(hidden_list[1], return_sequences=False,
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10))(encoder)
    classifier = Dense(nb_classes, activation='softmax')(classifier)
    PFL_model = Model(inputs=main_input, outputs=classifier)
    for i in range(len(PFL_model.layers)-2):
        PFL_model.layers[i+1].trainable = False
    # compile PFL model
    adam = Adam(lr=lr2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    PFL_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # train PFL model
    epochs = fine_epoch
    PFL_model.fit(x=x, y=y_htc, batch_size=batch_size, epochs=fine_epoch,
                  verbose=1, callbacks=[early_stopping], validation_split=0.2,
                  shuffle=True)
    return PFL_model

def train_lstmvaet_model1(dataset, hidden_list, initializer, c_num, lr1,
                            batch_size, pre_epoch):
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
    encoder_h = merge([encoder_x1, encoder_x2], mode='concat')
    encoder_c = Attention(c_num=c_num, timesteps=timesteps,
                    x_dim=input_dim, h_dim=hidden_list[0])([main_input, encoder_h])
    encoder_c = Lambda(repeatmat, arguments={'timesteps':c_num, 'c_num':c_num},
                    output_shape=(timesteps, hidden_list[0]*2))(encoder_c)
    encoder_z_mean = LSTM(hidden_list[1], return_sequences=True,
                    stateful=True,
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10))(encoder_c)
    encoder_z_log_var = LSTM(hidden_list[1], return_sequences=True,
                        stateful=True,
                        kernel_initializer=initializer,
                        activity_regularizer=regularizers.l2(10e-10))(encoder_c)
    encoder = Lambda(sampling, arguments={'batch_size':batch_size,
                    'latent_dim': hidden_list[1], 'timesteps':c_num},
                    output_shape=(c_num, hidden_list[1]))([encoder_z_mean, encoder_z_log_var])
    # build encoder
    decoder = Lambda(repeatmat, arguments={'timesteps':timesteps, 'c_num':c_num},
                    output_shape=(timesteps, hidden_list[1]))(encoder)
    decoder = LSTM(hidden_list[0]*2, return_sequences=True,
                    stateful=True, kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-10))(decoder)
    decoder_mean = LSTM(input_dim, return_sequences=True,
                   stateful=True, kernel_initializer=initializer,
                   activity_regularizer=regularizers.l2(10e-10))(decoder)
    VAE = Model(inputs=main_input, outputs=decoder_mean)
    Encoder = Model(inputs=main_input, outputs=encoder)
    # compile model
    adam = Adam(lr=lr1, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    def vae_loss(x, decoder_mean):
        xent_loss = K.sum(MSLE(x, decoder_mean))
        kl_loss = -0.5 * K.sum(K.sum(1 + encoder_z_log_var - K.square(encoder_z_mean)
                  - K.exp(encoder_z_log_var), axis=-1))
        return xent_loss + kl_loss
    VAE.compile(optimizer=adam, loss=vae_loss, metrics=['msle', 'mse'])
    # train model
    epochs = pre_epoch
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    VAE.fit(x=x, y=x, batch_size=batch_size, epochs=pre_epoch,
                verbose=1, callbacks=[early_stopping], validation_split=0.2,
                shuffle=True)
    return VAE

def missdata_implement(x, d):
    print(d)
    x2 = np.zeros((len(x), 1380, d))
    for i in range(len(x)):
        if x[i].shape[0] == 1380:
            x2[i] = x[i]
        else:
            x1 = x[i]
            x3 = -1 * np.ones((1380-x1.shape[0], x1.shape[1]))
            x2[i] = np.vstack((x1, x3))
    return x2

def save_model(model, model_path, weights_path):
    model = model
    json_string = model.to_json()
    open(model_path, 'w').write(json_string)
    model.save_weights(weights_path)

def load_model(model_path, weights_path):
    model = model_from_json(open(model_path).read(), custom_objects={'Attention': Attention,
                        'c_num':345, 'x_dim':96, 'h_dim':192, 'timesteps':1380})
    model.load_weights(weights_path)
    return model

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def model_train(x, y, model_old, lr, epoch, batch_size=25):
    #encoder, clf_old = model_old
    clf_old = load_model(model_old[0], model_old[1])
    #encoder = Model(inputs=encoder.input, outputs=encoder.layers[4].output)
    #x2 = encoder.predict(x, batch_size=batch_size)
    nb_classes = len(set(y))
    y_htc = np_utils.to_categorical(y, nb_classes)
    x, y_htc = shuffle(x, y_htc)
    k = len(clf_old.layers)
    #timesteps = x.shape[1]
    #input_dim = x.shape[2]
    #main_input = Input(shape=(x.shape[0], timesteps, input_dim))
    #clf = LSTM(hid0, return_sequences=True, activation='softsign',
    #                kernel_initializer='orthogonal')(main_input)
    #clf = LSTM(hidden_unit, return_sequences=False, activation='softsign',
    #                kernel_initializer='orthogonal')(clf)
    #clf = BatchNormalization()(clf)
    #clf = Dense(nb_classes, activation='softmax')(clf)
    #pfl = Model(inputs=main_input, outputs=clf)
    clf_new = Model(inputs=clf_old.input, outputs=clf_old.layers[k-1].output)
    rmsprop = RMSprop(lr=lr)
    clf_new.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=[f1])
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    history = clf_new.fit(x=x, y=y_htc, batch_size=batch_size, epochs=epoch,
                  verbose=1, callbacks=[early_stopping], validation_split=0.2,
                  shuffle=True)
    F1 = clf_new.evaluate(x, y_htc, batch_size=batch_size)[1]
    return clf_new, [1. / F1]

def main():
    #x = np.load('dataset/codedata/xnor_train2.npy', encoding='latin1')[0:2500]
    x1 = list(np.load('dataset/codedata/80001/80001_xnor_train2.npy', encoding='latin1'))
    y1 = np.load('dataset/codedata/80001/80001_y_train2.npy', encoding='latin1')
    print (y1, len(y1), len(x1))
    for i in range(6):
        x1.append(x1[243-i])
        y1 = np.hstack((y1, y1[243-i]))
    x3 = list(np.load('dataset/codedata/80001/80001_xnor_test.npy', encoding='latin1'))
    y3 = np.load('dataset/codedata/80001/80001_y_test.npy', encoding='latin1')
    #print (len(x1), y1.shape, len(x3), y3.shape)

    x2 = missdata_implement(x1, x1[0].shape[1])
    x4 = missdata_implement(x3, x3[0].shape[1])
    x4 = np.vstack((x4, x4, x4, x4, x4))
    y3 = np.hstack((y3, y3, y3, y3, y3))
    print('finish data')
    #y4 = np_utils.to_categorical(y3, 2)
    #print(len(x3))
    #dataset = [x2, y]
    batch_size = 10#[5, 10, 20]
    hidden_list = [192, 96]
    hidden_unit = 40 #[24, 48, 96, 192, 384]
    c_num =345  #[10, 20, 23, 46, 60, 69, 138, 345]
    lr1 = 0.0003#[0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    pre_epoch = 1000 #[200, 500, 800, 1000]
    lr2 = 0.0003
    fine_epoch = 1000
    initializer = 'orthogonal'
    start_time = timeit.default_timer()

    '''
    model = train_lstmsae_pflt_model1(dataset=x2,
                                    hidden_unit=hidden_unit,
                                    initializer=initializer,
                                    c_num=c_num,
                                    lr1=lr1,
                                    batch_size=batch_size,
                                    pre_epoch=pre_epoch,
                                    lr2=lr2,
                                    fine_epoch=fine_epoch)

    model = train_lstmsae_pflt_model2(dataset=dataset,
                            hidden_unit=hidden_unit,
                            initializer=initializer,
                            lr1=lr1,
                            batch_size=batch_size,
                            pre_epoch=pre_epoch,
                            lr2=lr2,
                            fine_epoch=fine_epoch)


    model = train_lstmvae_pflt_model1(dataset=dataset,
                                hidden_list=hidden_list,
                                initializer=initializer,
                                c_num=c_num,
                                lr1=lr1,
                                batch_size=batch_size,
                                pre_epoch=pre_epoch,
                                lr2=lr2,
                                fine_epoch=fine_epoch)
    '''
    '''
    encoder = train_lstmsaet_model1(dataset=x,
                          hidden_unit=hidden_unit,
                          initializer=initializer,
                          c_num=c_num,
                          lr1=lr1,
                          batch_size=batch_size,
                          pre_epoch=pre_epoch)
    '''

    #end_time = timeit.default_timer()
    #print('train model ran for %.2fmin' %((end_time - start_time)/60.))
    model_path = 'model/encoder_lstmsaeatt_nor_architechture2.json'
    weights_path = 'model/encoder_lstmsaeatt_nor_weights2.h5'
    #save_model(encoder, model_path, weights_path)

    model = load_model(model_path, weights_path)
    clf = train_lstmsaet_clf_model1(dataset=(x2,y1),
                          model=model,
                          hidden_unit=hidden_unit,
                          lr2 = lr2,
                          batch_size=batch_size,
                          fine_epoch=fine_epoch)
    end_time = timeit.default_timer()
    print('train model ran for %.2fmin' %((end_time - start_time)/60.))

    model_path2 = 'model/encoder_lstmsaeatt_clf80001nor_architechture2.json'
    weights_path2 = 'model/encoder_lstmsaeatt_clf80001nor_weights2.h5'
    save_model(clf, model_path2, weights_path2)
    #clf = load_model(model_path2, weights_path2)
    start_time = timeit.default_timer()
    model = Model(inputs=model.input, outputs=model.layers[4].output)
    x4 = model.predict(x4, batch_size=25)
    y4_pre = np.argmax(clf.predict(x4, batch_size=10), axis=1)
    end_time = timeit.default_timer()
    print('train model ran for %.2fmin' %((end_time - start_time)/60.))
    print(y4_pre[0: 10])
    print(y3[0: 10])
    print(accuracy_score(y3[0:10], y4_pre[0:10]))
    print(f1_score(y3[0: 10], y4_pre[0: 10]))

    #model = load_model(model_path, weights_path)
    #model = Model(inputs=model.input, outputs=model.layers[4].output)
    #x2 = model.predict(x2, batch_size=25)
    #print (x2.shape)
    #np.save('dataset/codedata/80001/80001_xnor_neg_train2', x2)
    #np.save('dataset/codedata/80001/80001_y_neg_train2', y)
    #print (np.reshape(x2, (x2.shape[0], x2.shape[1]*x2.shape[2])).shape)



if __name__ == '__main__':
    main()
