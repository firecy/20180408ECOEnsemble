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
from sklearn.metrics import f1_score
from sklearn.cross_validation import train_test_split

from keras.layers import Input, Dense, LSTM, Concatenate, Lambda, merge, RepeatVector
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

def train_lstmsae_pflt_model1(dataset, hidden_unit, initializer, c_num, lr1,
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

def train_lstmsae_pflt_model3(dataset, hidden_unit, initializer, c_num, lr1,
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
    # build the encoder
    main_input = Input(batch_shape=(batch_size, timesteps, input_dim))
    encoder_x1 = LSTM(hidden_unit, return_sequences=True,
                    stateful=True,
                    kernel_initializer=initializer)(main_input)
    encoder_x2 = LSTM(hidden_unit, return_sequences=True,
                    stateful=True, go_backwards=True,
                    kernel_initializer=initializer)(main_input)
    encoder_h = merge([encoder_x1, encoder_x2], mode='concat')
    encoder_c = Attention(c_num=c_num, timesteps=timesteps,
                    x_dim=input_dim, h_dim=hidden_unit)([main_input, encoder_h])
    # build decoder
    decoder = Lambda(repeatmat, arguments={'timesteps':timesteps, 'c_num':c_num},
                    output_shape=(timesteps, hidden_unit*2))(encoder_c)
    decoder = LSTM(input_dim, return_sequences=True,
                   stateful=True, kernel_initializer=initializer)(decoder)
    model = Model(inputs=main_input, outputs=decoder)
    adam = Adam(lr=lr1, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
    #optimizer = RMSprop(lr=lr1)
    model.compile(optimizer=adam, loss='msle', metrics=['mae'])
    #early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    model.fit(x=x, y=x, batch_size=batch_size, epochs=pre_epoch,
                  verbose=1, validation_split=0.2,
                  shuffle=True)
    # build clf
    cfl = Lambda(repeatmat, arguments={'timesteps':c_num, 'c_num':c_num},
                    output_shape=(timesteps, hidden_unit*2))(encoder_c)
    cfl = LSTM(hidden_unit, return_sequences=False,
                    kernel_initializer='orthogonal')(cfl)
    cfl = Dense(nb_classes, activation='softmax')(cfl)
    pfl = Model(inputs=main_input, outputs=cfl)
    adam = Adam(lr=lr2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    pfl.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    pfl.fit(x=x, y=y_htc, batch_size=batch_size, epochs=fine_epoch,
                  verbose=1, callbacks=[early_stopping], validation_split=0.2,
                  shuffle=True)
    return pfl

def train_lstmsae_pflt_model4(dataset, hidden_unit, initializer, c_num, lr1,
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
    # build the encoder
    main_input = Input(batch_shape=(batch_size, timesteps, input_dim))
    encoder_x1 = LSTM(hidden_unit, return_sequences=True, activation='softsign',
                    stateful=True,
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-7))(main_input)
    encoder_x2 = LSTM(hidden_unit, return_sequences=True, activation='softsign',
                    stateful=True, go_backwards=True,
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-7))(main_input)
    encoder_h = merge([encoder_x1, encoder_x2], mode='concat')
    encoder_c = LSTM(hidden_unit, return_sequences=False, activation='softsign',
                    kernel_initializer=initializer,
                    activity_regularizer=regularizers.l2(10e-7))(encoder_h)
    # build decoder
    decoder = RepeatVector(timesteps)(encoder_c)
    decoder = LSTM(input_dim, return_sequences=True, activation='softsign',
                   stateful=True, kernel_initializer=initializer,
                   activity_regularizer=regularizers.l2(10e-7))(decoder)
    model = Model(inputs=main_input, outputs=decoder)
    adam = Adam(lr=lr1, beta_1=0.9, beta_2=0.999, epsilon=1e-8, decay=0.0)
    #optimizer = RMSprop(lr=lr1)
    model.compile(optimizer=adam, loss='msle', metrics=['mae'])
    model.fit(x=x, y=x, batch_size=batch_size, epochs=pre_epoch,
                  verbose=1, validation_split=0.2,
                  shuffle=True)
    # build clf
    cfl = Dense(nb_classes, activation='softmax')(encoder_c)
    pfl = Model(inputs=main_input, outputs=cfl)
    adam = Adam(lr=lr2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    pfl.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    pfl.fit(x=x, y=y_htc, batch_size=batch_size, epochs=fine_epoch,
                  verbose=1, callbacks=[early_stopping], validation_split=0.3,
                  shuffle=True)
    return pfl

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

def sampling(args, batch_size=1380, latent_dim=24, timesteps=1380, epsilon_std=0.01):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(batch_size, timesteps, latent_dim),
                              mean=0., stddev=epsilon_std)
    return z_mean + K.exp(z_log_var / 2.) * epsilon

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
    # compile PFL model
    adam = Adam(lr=lr2, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    PFL_model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    # train PFL model
    epochs = fine_epoch
    PFL_model.fit(x=x, y=y_htc, batch_size=batch_size, epochs=fine_epoch,
                  verbose=1, callbacks=[early_stopping], validation_split=0.2,
                  shuffle=True)
    return PFL_model

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

def main():
    x = np.load('dataset/codedata/80001/80001_xzca_train.npy', encoding='latin1')
    y = np.load('dataset/codedata/80001/80001_y_train.npy', encoding='latin1')
    x3 = np.load('dataset/codedata/80001/80001_xzca_test.npy', encoding='latin1')
    y3 = np.load('dataset/codedata/80001/80001_y_test.npy', encoding='latin1')
    x2 = missdata_implement(x, x[0].shape[1])
    x4 = missdata_implement(x3, x3[0].shape[1])

    print('finish data')
    #x2 = np.zeros((len(x), x[0].shape[0], x[0].shape[1]))
    #x4 = np.zeros((len(x3), x3[0].shape[0], x3[0].shape[1]))
    #print y.shape
    #for i in range(len(x)):
    #    x2[i] = x[i]
    #for i in range(len(x3)):
    #    x4[i] = x3[i]
    y4 = np_utils.to_categorical(y3, 2)
    print(len(x3))
    dataset = [x2, y]
    batch_size = 1 #[5, 10, 20]
    hidden_unit = [192, 10] #[24, 48, 96, 192, 384]
    c_cum = 690 #[10, 20, 23, 46, 60, 69, 138, 345]
    lr1 = 0.0003#[0.001, 0.003, 0.01, 0.03, 0.1, 0.3]
    pre_epoch = 1 #[200, 500, 800, 1000]
    lr2 = 0.003
    fine_epoch = 1
    initializer = 'orthogonal'

    model = train_lstmvae_pflt_model2(dataset=dataset,
                    hidden_list=hidden_unit, initializer=initializer,
                    c_num=c_cum, lr1=lr1, batch_size=batch_size,
                    pre_epoch=pre_epoch, lr2=lr2, fine_epoch=fine_epoch)
    score = model.evaluate(x4, y4, batch_size=1)
    y3_pre = np.argmax(model.predict(x4, batch_size=1), axis=1)
    print(y3_pre)
    print(y3)
    print(score)


if __name__ == '__main__':
    main()
