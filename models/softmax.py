import time
import pickle
import numpy as np

import theano
import theano.tensor as T
import lasagne

from model import Model

# ----------------------------------------------------------------------------

class Softmax(Model):
  def __init__(self, n_dim, n_out, n_chan=1, n_superbatch=12800, model='mnist', opt_alg='adam',
               opt_params={'lr' : 1e-3, 'b1': 0.9, 'b2': 0.99}):
    Model.__init__(self, n_dim, n_chan, n_out, n_superbatch, opt_alg, opt_params)

  def create_model(self, X, Y, n_dim, n_out, n_chan=1):
    l_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim), 
                                     input_var=X)
    l_out = lasagne.layers.DenseLayer(
            l_in, num_units=n_out,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out