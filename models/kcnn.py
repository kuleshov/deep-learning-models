import os
import time

import numpy as np
import keras
import keras.backend as K
if K.backend() == 'tensorflow':
  import tensorflow as tf
elif K.backend() == 'theano':
  import theano.tensor as T

from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers import Input
from keras.models import Model

# ----------------------------------------------------------------------------

class KCNN(object):
  """CNN implemented in Keras"""

  def __init__(self, n_dim, n_out, n_chan=1, n_superbatch=12800, model='mnist', opt_alg='adam',
               opt_params={'lr' : 1e-3, 'b1': 0.9, 'b2': 0.99}):

    # create input vars
    X = Input(shape=(n_chan, n_dim, n_dim))

    # create network
    x = Dropout(0.2)(X)

    x = Convolution2D(nb_filter=64, nb_row=5, nb_col=5, 
          activation='relu', border_mode='same', init='glorot_uniform',
          dim_ordering='th')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), dim_ordering='th')(x)

    x = Convolution2D(nb_filter=64, nb_row=5, nb_col=5, 
          activation='relu', border_mode='same', init='glorot_uniform',
          dim_ordering='th')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), dim_ordering='th')(x)

    x = Convolution2D(nb_filter=128, nb_row=5, nb_col=5, 
          activation='relu', border_mode='same', init='glorot_uniform',
          dim_ordering='th')(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), dim_ordering='th')(x)

    x = Flatten()(x)
    x = Dense(1000, activation='relu')(x)
    x = Dropout(0.2)(x)

    y = Dense(n_out, activation='softmax')(x)

    # create model
    self.model = Model(input=X, output=y)

    # create optimizer
    lr, b1, b2 = opt_params['lr'], opt_params['b1'], opt_params['b2']
    optimizer = keras.optimizers.Adam(lr=lr, beta_1=b1, beta_2=b2)

    # compile model
    self.model.compile(optimizer=optimizer, loss='categorical_crossentropy', 
                       metrics=['accuracy'])

  def fit(self, X_train, Y_train, X_val, Y_val, n_epoch=10, n_batch=128, logname='./run'):
    if len(Y_train.shape) == 1 or Y_train.shape[1] == 1:
      from keras.utils.np_utils import to_categorical
      Y_train = to_categorical(Y_train, nb_classes=None)
      Y_val   = to_categorical(Y_val, nb_classes=None)
    self.model.fit(X_train, Y_train, batch_size=n_batch, nb_epoch=n_epoch,
                   validation_data=(X_val, Y_val))