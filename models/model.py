import time
import pickle
from collections import OrderedDict

import numpy as np
import theano
import theano.tensor as T
import lasagne

from helpers import *

# ----------------------------------------------------------------------------

class Model(object):
  """Model superclass that includes training code"""
  def __init__(self, n_dim, n_chan, n_out, n_superbatch, opt_alg, opt_params):
    # create shared data variables
    train_set_x = theano.shared(np.empty((n_superbatch, n_chan, n_dim, n_dim), dtype=theano.config.floatX), borrow=False)
    val_set_x = theano.shared(np.empty((n_superbatch, n_chan, n_dim, n_dim), dtype=theano.config.floatX), borrow=False)

    # create y-variables
    train_set_y = theano.shared(np.empty((n_superbatch,), dtype=theano.config.floatX), borrow=False)
    val_set_y = theano.shared(np.empty((n_superbatch,), dtype=theano.config.floatX), borrow=False)
    train_set_y_int, val_set_y_int = T.cast(train_set_y, 'int32'), T.cast(val_set_y, 'int32')

    # create input vars
    X = T.tensor4(dtype=theano.config.floatX)
    Y = T.ivector()
    idx1, idx2 = T.lscalar(), T.lscalar()
    self.inputs = (X, Y, idx1, idx2)

    # create lasagne model
    self.network = self.create_model(X, Y, n_dim, n_out, n_chan)

    # create objectives
    loss, acc            = self.create_objectives(deterministic=False)
    loss_test, acc_test  = self.create_objectives(deterministic=False)
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
    # from theano.compile.nanguardmode import NanGuardMode
    self.train = theano.function([idx1, idx2, alpha], [loss, acc], updates=updates,
                                 givens={X : train_set_x[idx1:idx2], Y : train_set_y_int[idx1:idx2]},
                                 on_unused_input='warn')
                                 # mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=False))
    self.loss = theano.function([X, Y], [loss, acc], on_unused_input='warn')
    
    # # TODO: implement a create_predictions method
    # self.predict = theano.function([X], P)

    # save config
    self.n_dim = n_dim
    self.n_out = n_out
    self.n_superbatch = n_superbatch
    self.alg = opt_alg

    # save data variables
    self.train_set_x = train_set_x
    self.train_set_y = train_set_y
    self.val_set_x = val_set_x
    self.val_set_y = val_set_y
    self.data_loaded = False

    # save neural network
    self.params = self.get_params()
    self.grads = (grads, grads_test)

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

    if hasattr(self, 'gen_samples'): import scipy.misc

    # test_one_true_sample = np.array(X_train[0]).flatten()
    # scipy.misc.imsave('test_one_true_sample.png', test_one_true_sample.reshape((3,32,32))[0])

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
      train_err, train_acc = evaluate(self.loss, X_train, Y_train, batchsize=n_batch)
      val_err, val_acc = evaluate(self.loss, X_val, Y_val, batchsize=n_batch)

      print "  training loss/acc:\t\t{:.6f}\t{:.6f}".format(train_err, train_acc)
      print "  validation loss/acc:\t\t{:.6f}\t{:.6f}".format(val_err, val_acc)

      if hasattr(self, 'gen_samples'):
        X_sam = self.gen_samples(100)
        scipy.misc.imsave(logname + '.png', X_sam)

      metrics = [ epoch, train_err, train_acc, val_err, val_acc ]
      log_metrics(logname + '.val', metrics)
      self.dump(logname  + '.pkl')

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
        self.train_set_x.set_value(X, borrow=False)
        if Y is not None:
          self.train_set_y.set_value(Y, borrow=False)
    elif dest == 'val':
        self.val_set_x.set_value(X, borrow=False)
        if Y is not None:
          self.val_set_y.set_value(Y, borrow=False)

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
