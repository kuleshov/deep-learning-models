import time
import pickle
import numpy as np

import theano
import theano.tensor as T
import lasagne
from lasagne.layers import batch_norm

from model import Model

from layers import GaussianSampleLayer
from layers import extras as nn
from distributions import log_bernoulli, log_normal, log_normal2

# ----------------------------------------------------------------------------

class VAE(Model):
  """Variational Autoencoder with Gaussian visible and latent variables"""
  def __init__(self, n_dim, n_out, n_chan=1, n_batch=128, n_superbatch=12800, model='bernoulli',
                opt_alg='adam', opt_params={'lr' : 1e-3, 'b1': 0.9, 'b2': 0.99}):
    # save model that wil be created
    self.model = model
    self.n_batch = n_batch
    self.n_lat = 100
    self.n_dim = n_dim
    self.n_chan = n_chan

    # invoke parent constructor
    Model.__init__(self, n_dim, n_chan, n_out, n_superbatch, opt_alg, opt_params)

    # sample generation
    Z = T.matrix(dtype=theano.config.floatX) # noise matrix
    _, _, _, _, l_sample, l_p_z = self.network
    sample = lasagne.layers.get_output(l_sample,  {l_p_z : Z}, deterministic=True)
    self.sample = theano.function([Z], sample, on_unused_input='warn')

  def create_model(self, X, Y, n_dim, n_out, n_chan=1):
      return self.create_std_model(X, Y, n_dim, n_out, n_chan)

  def create_std_model(self, X, Y, n_dim, n_out, n_chan=1):
    # params
    n_lat = self.n_lat # latent stochastic variabels
    n_hid = 500 # size of hidden layer in encoder/decoder
    n_out = n_dim * n_dim * n_chan # total dimensionality of output
    hid_nl = lasagne.nonlinearities.tanh if self.model == 'bernoulli' \
             else T.nnet.softplus
    # hid_nl = lasagne.nonlinearities.rectified

    # create the encoder network
    l_q_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim), 
                                     input_var=X)
    l_q_hid = lasagne.layers.DenseLayer(
        l_q_in, num_units=n_hid,
        nonlinearity=hid_nl, name='q_hid')
    l_q_mu = lasagne.layers.DenseLayer(
        l_q_hid, num_units=n_lat,
        nonlinearity=None, name='q_mu')
    l_q_logsigma = lasagne.layers.DenseLayer(
        l_q_hid, num_units=n_lat,
        nonlinearity=None, name='q_logsigma')

    # create the decoder network
    l_p_z = GaussianSampleLayer(l_q_mu, l_q_logsigma)
    
    l_p_hid = lasagne.layers.DenseLayer(
        l_p_z, num_units=n_hid,
        nonlinearity=hid_nl,
        W=lasagne.init.GlorotUniform(), name='p_hid')
    l_p_mu, l_p_logsigma = None, None

    if self.model == 'bernoulli':
      l_sample = lasagne.layers.DenseLayer(l_p_hid, num_units=n_out,
          nonlinearity = lasagne.nonlinearities.sigmoid,
          W=lasagne.init.GlorotUniform(),
          b=lasagne.init.Constant(0.), name='p_sigma')

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

    return l_p_mu, l_p_logsigma, l_q_mu, l_q_logsigma, l_sample, l_p_z


  def create_objectives(self, deterministic=False):
    """ELBO objective with the analytic expectation trick"""
    # load network input
    X = self.inputs[0]

    # load network output
    if self.model == 'bernoulli':
      q_mu, q_logsigma, sample, _ \
          = lasagne.layers.get_output(self.network[2:], deterministic=deterministic)
    elif self.model == 'gaussian':
      p_mu, p_logsigma, q_mu, q_logsigma, _, _ \
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

  def _get_net_params(self):
    # load network output
    if self.model == 'bernoulli':
      _, _, l_p_mu, _ = self.network[2:]
    elif self.model == 'gaussian':
      raise NotImplementedError()

    # load params
    params  = lasagne.layers.get_all_params(l_p_mu, trainable=True)

    p_params = [p for p in params if p.name.startswith('p_')]
    q_params = [p for p in params if p.name.startswith('q_')]

    return p_params, q_params

  def get_params(self):
    p_params, q_params = self._get_net_params()
    return p_params + q_params

  def gen_samples(self, n_sam):
    n_lat, n_dim, n_chan = self.n_lat, self.n_dim, self.n_chan
    noise = np.random.randn(n_sam, n_lat).astype(theano.config.floatX)
    # noise = np.zeros((n_sam, n_lat))
    # noise[range(n_sam), np.random.choice(n_lat, n_sam)] = 1

    assert np.sqrt(n_sam) == int(np.sqrt(n_sam))
    n_side = int(np.sqrt(n_sam))

    p_mu = self.sample(noise)
    p_mu = p_mu.reshape((n_side, n_side, n_chan, n_dim, n_dim))
    p_mu = p_mu[:,:,0,:,:] # keep the first channel
    # split into n_side (1,n_side,n_dim,n_dim,) images,
    # concat along columns -> 1,n_side,n_dim,n_dim*n_side
    p_mu = np.concatenate(np.split(p_mu, n_side, axis=0), axis=3)
    # split into n_side (1,1,n_dim,n_dim*n_side) images,
    # concat along rows -> 1,1,n_dim*n_side,n_dim*n_side
    p_mu = np.concatenate(np.split(p_mu, n_side, axis=1), axis=2)
    return np.squeeze(p_mu)