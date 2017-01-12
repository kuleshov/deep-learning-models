import time
import pickle
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
import lasagne

from helpers import *

# ----------------------------------------------------------------------------

class SemiSupModel(object):
  """Model superclass that includes training code"""
  def __init__(self, X_labeled, y_labeled, n_out, n_superbatch, opt_alg, opt_params):
    # add variable for labeled data that is set once and stays the same during all of training
    # (at initialization); create shared variable with that data already
    # experiment with batch training on that variable
    # need to create function that will return the dataset for you in the right split
    # need to update fit()
    # need to update training cost
    # need to implement predictions

    # create shared data variables for labeled data
    n_lab, n_chan, n_dim, _ = X_labeled.shape
    lab_train_set_x = theano.shared(X_labeled)
    lab_train_set_y = theano.shared(y_labeled)
    lab_train_set_y_int = T.cast(lab_train_set_y, 'int32')

    # create shared data variables for unlabeled data
    unl_train_set_x = theano.shared(np.empty((n_superbatch, n_chan, n_dim, n_dim), dtype=theano.config.floatX))
    unl_val_set_x = theano.shared(np.empty((n_superbatch, n_chan, n_dim, n_dim), dtype=theano.config.floatX))

    # create y-variables for unlabeled data
    unl_train_set_y = theano.shared(np.empty((n_superbatch,), dtype=theano.config.floatX))
    unl_val_set_y = theano.shared(np.empty((n_superbatch,), dtype=theano.config.floatX))
    unl_train_set_y_int, unl_val_set_y_int = T.cast(unl_train_set_y, 'int32'), T.cast(unl_val_set_y, 'int32')

    # create input vars
    X = T.tensor4(dtype=theano.config.floatX) # unlabeled X
    L = T.tensor4(dtype=theano.config.floatX) # labeled X
    yu = T.ivector() # for unlabeled data (will be taking expectation over all labels)
    yl = T.ivector() # for labeled data
    idx1, idx2 = T.lscalar(), T.lscalar()
    self.inputs = (X, yu, idx1, idx2)

    # create lasagne model
    self.network = self.create_model(L, yl, n_dim, n_out, n_chan)

    # create objectives
    loss, acc            = self.create_objectives(X, L, yu, yl, deterministic=False)
    loss_test, acc_test  = self.create_objectives(X, L, yu, yl, deterministic=True)
    self.objectives      = (loss, acc)
    self.objectives_test = (loss_test, acc_test)

    # load params
    params = self.get_params()

    # create gradients
    grads      = self.create_gradients(loss, deterministic=False)
    grads_test = self.create_gradients(loss_test, deterministic=True)
    
    # create updates
    alpha = T.scalar(dtype=theano.config.floatX) # adjustable learning rate
    updates = self.create_updates(grads, params, alpha, opt_alg, opt_params)

    # create methods for training / prediction
    var_assignments = {
      L  : lab_train_set_x,
      yl : lab_train_set_y_int,
      # X  : unl_train_set_x[idx1:idx2], 
      # yu : unl_train_set_y_int[idx1:idx2]
    }

    self.train = theano.function([idx1, idx2, alpha], [loss, acc], updates=updates,
                                 givens=var_assignments, on_unused_input='warn')
    self.loss = theano.function([L, yl], [loss, acc], on_unused_input='warn')

    # save config
    self.n_dim = n_dim
    self.n_out = n_out
    self.n_superbatch = n_superbatch
    self.alg = opt_alg

    # save data variables
    self.train_set_x = unl_train_set_x
    self.train_set_y = unl_train_set_y
    self.val_set_x = unl_val_set_x
    self.val_set_y = unl_val_set_y
    self.data_loaded = False

    # save neural network
    self.params = self.get_params()

  def create_objectives(self, deterministic=False):
    # load network
    l_out = self.network
    Y = self.inputs[1]

    # create predictions
    P = lasagne.layers.get_output(l_out)
    P_test = lasagne.layers.get_output(l_out, deterministic=True)

    # create loss
    loss = lasagne.objectives.categorical_crossentropy(P, Y).mean()
    loss_test = lasagne.objectives.categorical_crossentropy(P_test, Y).mean()

    # measure accuracy
    top = theano.tensor.argmax(P, axis=-1)
    top_test = theano.tensor.argmax(P_test, axis=-1)
    acc = theano.tensor.eq(top, Y).mean()
    acc_test = theano.tensor.eq(top_test, Y).mean()

    if deterministic:
      return loss_test, acc_test
    else:
      return loss, acc

  def create_gradients(self, loss, deterministic=False):
    params = self.get_params()
    return theano.grad(loss, params)

  def create_updates(self, grads, params, alpha, opt_alg, opt_params):
    scaled_grads = [grad * alpha for grad in grads]
    lr = opt_params.get('lr', 1e-3)
    if opt_alg == 'sgd':
      grad_updates = lasagne.updates.sgd(scaled_grads, params, learning_rate=lr)
    elif opt_alg == 'adam':
      b1, b2 = opt_params.get('b1', 0.9), opt_params.get('b2', 0.999)
      grad_updates = lasagne.updates.adam(scaled_grads, params, learning_rate=lr, 
                                          beta1=b1, beta2=b2)
    else:
      grad_updates = OrderedDict()

    return grad_updates

  def get_params(self):
    l_out = self.network
    return lasagne.layers.get_all_params(l_out, trainable=True)

  def fit(self, X_train, Y_train, X_val, Y_val, n_epoch=10, n_batch=100, logname='run'):
    """Train the model"""

    alpha = 1.0 # learning rate, which can be adjusted later
    n_data = len(X_train)
    n_superbatch = self.n_superbatch

    for epoch in range(n_epoch):
      # In each epoch, we do a full pass over the training data:
      train_batches, train_err, train_acc = 0, 0, 0
      start_time = time.time()

      # iterate over superbatches to save time on GPU memory transfer
      for X_sb, Y_sb in self.iterate_superbatches(X_train, Y_train, n_superbatch, datatype='train', shuffle=True):
        for idx1, idx2 in iterate_minibatch_idx(len(X_sb), n_batch):
          err, acc = self.train(idx1, idx2, alpha)

          # collect metrics
          train_batches += 1
          train_err += err 
          train_acc += acc
          if train_batches % 100 == 0:
            n_total = epoch * n_data + n_batch * train_batches
            metrics = [n_total, train_err / train_batches, train_acc / train_batches]
            log_metrics(logname, metrics)

      print "Epoch {} of {} took {:.3f}s ({} minibatches)".format(
          epoch + 1, n_epoch, time.time() - start_time, train_batches)

      # make a full pass over the training data and record metrics:
      train_err, train_acc = evaluate(self.loss, X_train, Y_train, batchsize=1000)
      val_err, val_acc = evaluate(self.loss, X_val, Y_val, batchsize=1000)

      print "  training loss/acc:\t\t{:.6f}\t{:.6f}".format(train_err, train_acc)
      print "  validation loss/acc:\t\t{:.6f}\t{:.6f}".format(val_err, val_acc)

      metrics = [ epoch, train_err, train_acc, val_err, val_acc ]
      log_metrics(logname + '.val', metrics)

  def dump(self, fname):
    """Pickle weights to a file"""
    params = lasagne.layers.get_all_param_values(self.network)
    with open(fname, 'w') as f:
      pickle.dump(params, f)

  def load(self, params):
    """Load pickled network"""
    with open(fname) as f:
      params = pickle.load(f)
    self.load_params(params)

  def load_params(self, params):
    """Load a given set of parameters"""
    lasagne.layers.set_all_param_values(self.network, params)

  def dump_params(self):
    """Dump a given set of parameters"""
    return lasagne.layers.get_all_param_values(self.network)

  def load_data(self, X, Y, dest='train'):
    assert dest in ('train', 'val')
    if dest == 'train':
        self.train_set_x.set_value(X)
        if Y is not None:
          self.train_set_y.set_value(Y)
    elif dest == 'val':
        self.val_set_x.set_value(X)
        if Y is not None:
          self.val_set_y.set_value(Y)

  def iterate_superbatches(self, X, Y, batchsize, datatype='train', shuffle=False):
    assert datatype in ('train', 'val')
    assert len(X) == len(Y)
    assert batchsize <= len(X)

    # if we are loading entire dataset, only load it once
    if batchsize == len(X):
      if not self.data_loaded:
        self.load_data(X, Y, dest=datatype)
        self.data_loaded = True
      yield X, Y
    else:
      # otherwise iterate over superbatches
      for superbatch in iterate_minibatches(X, Y, batchsize, shuffle=shuffle):
        inputs, targets = superbatch
        self.load_data(inputs, targets, dest=datatype)
        yield inputs, targets
