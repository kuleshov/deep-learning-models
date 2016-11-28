import time
import pickle
import numpy as np

import theano
import theano.tensor as T
import lasagne

from model import Model

# ----------------------------------------------------------------------------

class MLP(Model):
  def __init__(self, n_dim, n_out, n_chan=1, n_superbatch=12800, n_hidden=[1000,1000], opt_alg='adam',
               opt_params={'lr' : 1e-3, 'b1': 0.9, 'b2': 0.99}):
    # store model params
    self.n_hidden = n_hidden

    # call parent
    Model.__init__(self, n_dim, n_chan, n_out, n_superbatch, opt_alg, opt_params)

  def update_cv2(self):
    params = lasagne.layers.get_all_param_values(self.network)
    lasagne.layers.set_all_param_values(self.network_cv, params)

  def create_model(self, X, Y, n_dim, n_out, n_chan=1):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, n_dim, n_dim), 
                                     input_var=X)
    l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)
    # l_in_drop = l_in

    l_hid1 = lasagne.layers.DenseLayer(
        l_in_drop, num_units=self.n_hidden[0],
        nonlinearity=lasagne.nonlinearities.rectify,
        W=lasagne.init.GlorotUniform())

    l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)
    # l_hid1_drop = l_hid1

    l_hid2 = lasagne.layers.DenseLayer(
            l_hid1_drop, num_units=self.n_hidden[1],
            nonlinearity=lasagne.nonlinearities.rectify)

    l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)
    # l_hid2_drop = l_hid2

    l_out = lasagne.layers.DenseLayer(
            l_hid2_drop, num_units=n_out,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_out
