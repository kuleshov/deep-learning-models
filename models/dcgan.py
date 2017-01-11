import time
import pickle
import numpy as np
import matplotlib.pyplot as plt

from collections import OrderedDict

import theano
import theano.tensor as T
import lasagne

from lasagne.layers import batch_norm

from model import Model
from helpers import *

# ----------------------------------------------------------------------------

class DCGAN(Model):
  def __init__(self, n_dim, n_out, n_chan=1, n_superbatch=12800, model='bernoulli',
                opt_alg='adam', opt_params={'lr' : 1e-3, 'b1': 0.9, 'b2': 0.99}):
    # save model that wil be created
    self.model = model

    # create shared data variables
    train_set_x = theano.shared(np.empty((n_superbatch, n_chan, n_dim, n_dim), dtype=theano.config.floatX), borrow=False)
    val_set_x = theano.shared(np.empty((n_superbatch, n_chan, n_dim, n_dim), dtype=theano.config.floatX), borrow=False)

    # create y-variables
    train_set_y = theano.shared(np.empty((n_superbatch,), dtype=theano.config.floatX), borrow=False)
    val_set_y = theano.shared(np.empty((n_superbatch,), dtype=theano.config.floatX), borrow=False)
    train_set_y_int, val_set_y_int = T.cast(train_set_y, 'int32'), T.cast(val_set_y, 'int32')

    # create input vars
    X = T.tensor4(dtype=theano.config.floatX)
    Z = T.matrix(dtype=theano.config.floatX)
    idx1, idx2 = T.lscalar(), T.lscalar()
    self.inputs = (X, Z, idx1, idx2)

    # create lasagne model
    self.network = self.create_model(X, Z, n_dim, n_out, n_chan)
    l_g, l_d = self.network

    # create objectives
    loss_g, loss_d, p_real, p_fake = self.create_objectives(deterministic=False)
    _, _, p_real_test, p_fake_test = self.create_objectives(deterministic=True)

    # load params
    params_g, params_d = self.get_params()

    # create gradients
    grads_g = theano.grad(loss_g, params_g)
    grads_d = theano.grad(loss_d, params_d)
    
    # create updates
    alpha = T.scalar(dtype=theano.config.floatX) # adjustable learning rate
    updates_g = self.create_updates(grads_g, params_g, alpha, opt_alg, opt_params)
    updates_d = self.create_updates(grads_d, params_d, alpha, opt_alg, opt_params)
    updates = OrderedDict(updates_g.items() + updates_d.items())

    # create methods for training / prediction
    self.train = theano.function([Z, idx1, idx2, alpha], [p_real, p_fake],
                                 updates=updates, givens={X : train_set_x[idx1:idx2]})
    self.loss = theano.function([X, Z], [p_real, p_fake])
    self.loss_test = theano.function([X, Z], [p_real_test, p_fake_test])
    self.gen = theano.function([Z], lasagne.layers.get_output(l_g, deterministic=True))

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

  def fit(self, X_train, Y_train, X_val, Y_val, n_epoch=10, n_batch=100, logname='run'):
    """Train the model"""

    alpha = 1.0 # learning rate, which can be adjusted later
    n_data = len(X_train)
    n_superbatch = self.n_superbatch

    for epoch in range(n_epoch):
      # In each epoch, we do a full pass over the training data:
      train_batches, train_err, train_acc = 0, 0, 0
      start_time = time.time()

      if epoch >= n_epoch // 2:
        progress = float(epoch) / n_epoch
        alpha = 2*(1 - progress)

      # iterate over superbatches to save time on GPU memory transfer
      for X_sb, Y_sb in self.iterate_superbatches(X_train, Y_train, n_superbatch, datatype='train', shuffle=True):
        for idx1, idx2 in iterate_minibatch_idx(len(X_sb), n_batch):
          noise = lasagne.utils.floatX(np.random.rand(n_batch, 100))
          p_real, p_fake = self.train(noise, idx1, idx2, alpha)

          # collect metrics
          train_batches += 1
          train_err += p_real 
          train_acc += p_fake
          if train_batches % 5 == 0:
            n_total = epoch * n_data + n_batch * train_batches
            metrics = [n_total, train_err / train_batches, train_acc / train_batches]
            log_metrics(logname, metrics)

            samples = self.gen(lasagne.utils.floatX(np.random.rand(42, 100)))
            plt.imsave('mnist_samples.png',
                       (samples.reshape(6, 7, 28, 28)
                               .transpose(0, 2, 1, 3)
                               .reshape(6*28, 7*28)),
                       cmap='gray')

      print "Epoch {} of {} took {:.3f}s ({} minibatches)".format(
          epoch + 1, n_epoch, time.time() - start_time, train_batches)

      # make a full pass over the training data and record metrics:
      Z_train = lasagne.utils.floatX(np.random.rand(len(X_train), 100))
      Z_val   = lasagne.utils.floatX(np.random.rand(len(X_val), 100))
      # train_err, train_acc = evaluate(self.loss, X_train, Z_train, batchsize=1000)
      train_err /= train_batches
      train_acc /= train_batches
      val_err, val_acc = evaluate(self.loss_test, X_val, Z_val, batchsize=1000)

      print "  training loss/acc:\t\t{:.6f}\t{:.6f}".format(train_err, train_acc)
      print "  validation loss/acc:\t\t{:.6f}\t{:.6f}".format(val_err, val_acc)

      metrics = [ epoch, train_err, train_acc, val_err, val_acc ]
      log_metrics(logname + '.val', metrics)
  
  def create_model(self, X, Z, n_dim, n_out, n_chan=1):
    # params
    n_lat = 100 # latent variables
    n_g_hid1 = 1024 # size of hidden layer in generator layer 1
    n_g_hid2 = 128 # size of hidden layer in generator layer 2
    n_out = n_dim * n_dim * n_chan # total dimensionality of output

    if self.model == 'gaussian': 
      raise Exception('Gaussian variables currently nor supported in GAN')

    # create the generator network
    l_g_in = lasagne.layers.InputLayer(shape=(None, n_lat), input_var=Z)
    l_g_hid1 = batch_norm(lasagne.layers.DenseLayer(l_g_in, n_g_hid1))
    l_g_hid2 = batch_norm(lasagne.layers.DenseLayer(l_g_hid1, n_g_hid2*7*7))
    l_g_hid2 = lasagne.layers.ReshapeLayer(l_g_hid2, ([0], n_g_hid2, 7, 7))
    l_g_dc1 = batch_norm(Deconv2DLayer(l_g_hid2, 64, 5, stride=2, pad=2))
    l_g = Deconv2DLayer(l_g_dc1, n_chan, 5, stride=2, pad=2, 
            nonlinearity=lasagne.nonlinearities.sigmoid)
    print ("Generator output:", l_g.output_shape)

    # create the discriminator network
    lrelu = lasagne.nonlinearities.LeakyRectify(0.2)
    l_d_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim), 
                                       input_var=X)
    l_d_hid1 = batch_norm(lasagne.layers.Conv2DLayer(
        l_d_in, num_filters=64, filter_size=5, stride=2, pad=2,
        nonlinearity=lrelu, name='l_d_hid1'))
    l_d_hid2 = batch_norm(lasagne.layers.Conv2DLayer(
        l_d_hid1, num_filters=128, filter_size=5, stride=2, pad=2,
        nonlinearity=lrelu, name='l_d_hid2'))
    l_d_hid3 = batch_norm(lasagne.layers.DenseLayer(l_d_hid2, 1024, nonlinearity=lrelu))
    l_d = lasagne.layers.DenseLayer(l_d_hid3, 1, nonlinearity=lasagne.nonlinearities.sigmoid)
    print ("Discriminator output:", l_d.output_shape)

    return l_g, l_d

  def create_objectives(self, deterministic=False):
    # load network
    l_g, l_d = self.network

    # load ouput
    g      = lasagne.layers.get_output(l_g, deterministic=deterministic)
    d_real = lasagne.layers.get_output(l_d, deterministic=deterministic)
    d_fake = lasagne.layers.get_output(l_d, g, deterministic=deterministic)

    # define loss
    loss_g = lasagne.objectives.binary_crossentropy(d_fake, 1).mean()
    loss_d = ( lasagne.objectives.binary_crossentropy(d_real, 1)
             + lasagne.objectives.binary_crossentropy(d_fake, 0) ).mean()

    # compute and store discriminator probabilities
    p_real = (d_real > 0.5).mean()
    p_fake = (d_fake < 0.5).mean()

    return loss_g, loss_d, p_real, p_fake

  def get_params(self):
    l_g, l_d = self.network

    params_g = lasagne.layers.get_all_params(l_g, trainable=True)
    params_d = lasagne.layers.get_all_params(l_d, trainable=True)

    return params_g, params_d

class Deconv2DLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
            nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        self.W = self.add_param(lasagne.init.Orthogonal(),
                (self.input_shape[1], num_filters) + self.filter_size,
                name='W')
        self.b = self.add_param(lasagne.init.Constant(0),
                (num_filters,),
                name='b')
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        shape = tuple(i*s - 2*p + f - 1
                for i, s, p, f in zip(input_shape[2:],
                                      self.stride,
                                      self.pad,
                                      self.filter_size))
        return (input_shape[0], self.num_filters) + shape

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
            subsample=self.stride, border_mode=self.pad)
        conved = op(self.W, input, self.output_shape[2:])
        if self.b is not None:
            conved += self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(conved)    