import time
import pickle
import numpy as np

import theano
import theano.tensor as T
import lasagne

from model import Model

from layers import GaussianSampleLayer

# ----------------------------------------------------------------------------

class VAE(Model):
  """Variational Autoencoder with Gaussian visible and latent variables"""
  def __init__(self, n_dim, n_out, n_chan=1, n_superbatch=12800, model='bernoulli',
                opt_alg='adam', opt_params={'lr' : 1e-3, 'b1': 0.9, 'b2': 0.99}):
    # save model that wil be created
    self.model = model

    # invoke parent constructor
    Model.__init__(self, n_dim, n_chan, n_out, n_superbatch, opt_alg, opt_params)
  
  def create_model(self, X, Y, n_dim, n_out, n_chan=1):
    # params
    n_lat = 200 # latent stochastic variabels
    n_hid = 500 # size of hidden layer in encoder/decoder
    n_out = n_dim * n_dim * n_chan # total dimensionality of ouput
    hid_nl = lasagne.nonlinearities.tanh if self.model == 'bernoulli' \
             else T.nnet.softplus

    # create the encoder network
    l_q_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim), 
                                     input_var=X)
    l_q_hid = lasagne.layers.DenseLayer(
        l_q_in, num_units=n_hid,
        nonlinearity=hid_nl)
    l_q_mu = lasagne.layers.DenseLayer(
        l_q_hid, num_units=n_lat,
        nonlinearity=None)
    l_q_logsigma = lasagne.layers.DenseLayer(
        l_q_hid, num_units=n_lat,
        nonlinearity=None)

    # create the decoder network
    l_p_z = GaussianSampleLayer(l_q_mu, l_q_logsigma)
    
    l_p_hid = lasagne.layers.DenseLayer(
        l_p_z, num_units=n_hid,
        nonlinearity=hid_nl,
        W=lasagne.init.GlorotUniform())
    l_p_mu, l_p_logsigma = None, None

    if self.model == 'bernoulli':
      l_sample = lasagne.layers.DenseLayer(l_p_hid, num_units=n_out,
          nonlinearity = lasagne.nonlinearities.sigmoid,
          W=lasagne.init.GlorotUniform(),
          b=lasagne.init.Constant(0.))

    elif self.model == 'gaussian':
      l_p_mu = lasagne.layers.DenseLayer(
          l_p_hid, num_units=n_out,
          nonlinearity=None)
      # relu_shift is for numerical stability - if training data has any
      # dimensions where stdev=0, allowing logsigma to approach -inf
      # will cause the loss function to become NAN. So we set the limit
      # stdev >= exp(-1 * relu_shift)
      relu_shift = 10
      l_p_logsigma = lasagne.layers.DenseLayer(
          l_p_hid, num_units=n_out,
          nonlinearity = lambda a: T.nnet.relu(a+relu_shift)-relu_shift)

      l_sample = GaussianSampleLayer(l_p_mu, l_p_logsigma)

    return l_p_mu, l_p_logsigma, l_q_mu, l_q_logsigma, l_sample

  def create_objectives(self, deterministic=False):
    # load network input
    X = self.inputs[0]

    # load network output
    if self.model == 'bernoulli':
      q_mu, q_logsigma, sample \
          = lasagne.layers.get_output(self.network[2:], deterministic=deterministic)
    elif self.model == 'gaussian':
      p_mu, p_logsigma, q_mu, q_logsigma, _ \
          = lasagne.layers.get_output(self.network, deterministic=deterministic)

    # first term of the ELBO: kl-divergence (using the closed form expression)
    kl_div = 0.5 * T.sum(1 + 2*q_logsigma - T.sqr(q_mu) 
                         - T.exp(2 * q_logsigma), axis=1).mean()

    # second term: log-likelihood of the data under the model
    if self.model == 'bernoulli':
      logpxz = -lasagne.objectives.binary_crossentropy(sample, X.flatten(2)).sum(axis=1).mean()
    elif self.model == 'gaussian':
      def log_lik(x, mu, log_sig):
          return T.sum(-(np.float32(0.5 * np.log(2 * np.pi)) + log_sig)
                        - 0.5 * T.sqr(x - mu) / T.exp(2 * log_sig), axis=1)
      logpxz = log_lik(X.flatten(2), p_mu, p_logsigma).mean()

    loss = -1 * (logpxz + kl_div)

    # we don't use the spearate accuracy metric right now
    return loss, -kl_div

  def get_params(self):
    _, _, _, _, l_sample = self.network
    return lasagne.layers.get_all_params(l_sample, trainable=True)