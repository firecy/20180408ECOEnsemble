#! -*- coding: utf-8 -*-

from keras import backend as K
from keras.engine.topology import Layer

class Attention(Layer):

    def __init__(self, c_num=345, timesteps=1380, x_dim=96, h_dim=192, **kwargs):
        self.c_num = c_num
        self.timesteps = timesteps
        self.x_dim = x_dim
        self.h_dim = h_dim
        self.output_dim = h_dim
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W1 = self.add_weight(name='W1',
                                  shape=(input_shape[0][-1], input_shape[1][-1]),
                                  initializer='orthogonal',
                                  trainable=True)
        print(self.W1.shape)
        self.W2 = self.add_weight(name='W2',
                                  shape=(self.c_num, self.timesteps),
                                  initializer='orthogonal',
                                  trainable=True)
        super(Attention, self).build(input_shape)
        print(self.W2.shape)

    def call(self, x):
        X, H = x
        X = K.dot(X, self.W1)
        #X = K.permute_dimensions(X, (0,2,1))
        print(H.shape, X.shape)
        A = K.batch_dot(X, H, axes=[2, 2]) / ((self.x_dim**0.5) * (self.h_dim**0.5))
        print(A.shape)
        A = K.dot(self.W2, A)
        print(A.shape)
        A = K.permute_dimensions(A, (1,0,2))
        print(A.shape)
        A = K.softmax(A)
        C = K.batch_dot(A, H, axes=[2, 1])
        print(C.shape)
        return C

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], self.c_num, self.output_dim)
