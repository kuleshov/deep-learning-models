import time
import pickle
import numpy as np

import theano
import theano.tensor as T
import lasagne

from model import Model

# ----------------------------------------------------------------------------

class CNN(Model):
  def __init__(self, n_dim, n_out, n_chan=1, n_superbatch=12800, model='mnist', opt_alg='adam',
               opt_params={'lr' : 1e-3, 'b1': 0.9, 'b2': 0.99}):
    # save model that wil be created
    self.model = model

    # invoke parent constructor
    Model.__init__(self, n_dim, n_chan, n_out, n_superbatch, opt_alg, opt_params)
    
  def create_model(self, X, Y, n_dim, n_out, n_chan=1):
    if self.model == 'mnist':
      return self.create_mnist_model(X, Y, n_dim, n_out, n_chan)
    elif self.model == 'cifar10':
      return self.create_cifar10_model(X, Y, n_dim, n_out, n_chan)
    else:
      raise ValueError('Invalid CNN model type')

  def create_mnist_model(self, X, Y, n_dim, n_out, n_chan=1):
    l_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim), 
                                     input_var=X)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    # l_in_drop = l_in

    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in_drop, num_filters=64, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        pad='same', W=lasagne.init.GlorotUniform())
    l_conv1 = lasagne.layers.MaxPool2DLayer(
        l_conv1, pool_size=(3, 3), stride=(2,2))

    l_conv2 = lasagne.layers.Conv2DLayer(
        l_conv1, num_filters=64, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        pad='same', W=lasagne.init.GlorotUniform())
    l_conv2 = lasagne.layers.MaxPool2DLayer(
        l_conv2, pool_size=(3, 3), stride=(2,2))

    l_conv3 = lasagne.layers.Conv2DLayer(
        l_conv2, num_filters=128, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        pad='same', W=lasagne.init.GlorotUniform())
    l_conv3 = lasagne.layers.MaxPool2DLayer(
        l_conv3, pool_size=(3, 3), stride=(2,2))

    l_hid = lasagne.layers.DenseLayer(
        l_conv3, num_units=1000,
        nonlinearity=lasagne.nonlinearities.rectify)
    l_hid_drop = lasagne.layers.DropoutLayer(l_hid, p=0.5)
    # l_hid_drop = l_hid

    l_out = lasagne.layers.DenseLayer(
            l_hid_drop, num_units=n_out,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out

  def create_cifar10_model(self, X, Y, n_dim, n_out, n_chan=1):
    l_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim), 
                                     input_var=X)

    # input layer
    network = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim), 
                                     input_var=X)
    
    # CCP units
    ccp_num_filters = (64, 128)
    ccp_filter_size = 3
    for num_filters in ccp_num_filters:
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=num_filters,
            filter_size=(ccp_filter_size, ccp_filter_size),
            pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=num_filters,
            filter_size=(ccp_filter_size, ccp_filter_size),
            pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    
    # FC layers
    fc_num_units = (256, 256)
    for num_units in fc_num_units:
        network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=num_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
    
    # output layer
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    
    return network