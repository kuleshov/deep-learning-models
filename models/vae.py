import time
import pickle
import numpy as np

import theano
import theano.tensor as T
import lasagne

from model import Model

from layers import GaussianSampleLayer
from distributions import log_bernoulli, log_normal, log_normal2

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
    return self.create_objectives_analytic(deterministic)

  def create_objectives_analytic(self, deterministic=False):
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

  def create_objectives_elbo(self, deterministic=False):
    """ELBO objective without the analytic expectation trick"""
    # load network input
    X = self.inputs[0]
    x = X.flatten(2)

    # load network output
    if self.model == 'bernoulli':
      q_mu, q_logsigma, p_mu, z \
           = lasagne.layers.get_output(self.network[2:], deterministic=deterministic)
    elif self.model == 'gaussian':
      raise NotImplementedError()

    # entropy term
    log_qz_given_x = log_normal2(z, q_mu, q_logsigma).sum(axis=1)

    # expected p(x,z) term
    z_prior_sigma = T.cast(T.ones_like(q_logsigma), dtype=theano.config.floatX)
    z_prior_mu = T.cast(T.zeros_like(q_mu), dtype=theano.config.floatX)
    log_pz = log_normal(z, z_prior_mu,  z_prior_sigma).sum(axis=1)     
    log_px_given_z = log_bernoulli(x, p_mu).sum(axis=1)
    log_pxz = log_pz + log_px_given_z

    elbo = (log_pxz - log_qz_given_x).mean()

    # we don't use the spearate accuracy metric right now
    return -elbo, -log_qz_given_x.mean()  

  def create_gradients(self, loss, deterministic=False):
    from theano.gradient import disconnected_grad as dg

    # load network input
    X = self.inputs[0]
    x = X.flatten(2)

    # load network output
    if self.model == 'bernoulli':
      q_mu, q_logsigma, p_mu, z \
           = lasagne.layers.get_output(self.network[2:], deterministic=deterministic)
    elif self.model == 'gaussian':
      raise NotImplementedError()

    # load params
    p_params, q_params = self._get_net_params()

    # entropy term
    log_qz_given_x = log_normal2(z, q_mu, q_logsigma).sum(axis=1)

    # expected p(x,z) term
    z_prior_sigma = T.cast(T.ones_like(q_logsigma), dtype=theano.config.floatX)
    z_prior_mu = T.cast(T.zeros_like(q_mu), dtype=theano.config.floatX)
    log_pz = log_normal(z, z_prior_mu, z_prior_sigma).sum(axis=1)     
    log_px_given_z = log_bernoulli(x, p_mu).sum(axis=1)
    log_pxz = log_pz + log_px_given_z

    # compute learning signals
    l = log_pxz - log_qz_given_x 
    # l_avg, l_std = l.mean(), T.maximum(1, l.std())
    # c_new = 0.8*c + 0.2*l_avg
    # v_new = 0.8*v + 0.2*l_std
    # l = (l - c_new) / v_new
  
    # compute grad wrt p
    p_grads = T.grad(-log_pxz.mean(), p_params)

    # compute grad wrt q
    # q_target = T.mean(dg(l) * log_qz_given_x)
    # q_grads = T.grad(-0.2*q_target, q_params) # 5x slower rate for q
    log_qz_given_x = log_normal2(dg(z), q_mu, q_logsigma).sum(axis=1)
    q_target = T.mean(dg(l) * log_qz_given_x)
    q_grads = T.grad(-0.2*q_target, q_params) # 5x slower rate for q
    # q_grads = T.grad(-l.mean(), q_params) # 5x slower rate for q

    # # compute grad of cv net
    # cv_target = T.mean(l**2)
    # cv_grads = T.grad(cv_target, cv_params)

    # combine and clip gradients
    clip_grad = 1
    max_norm = 5
    grads = p_grads + q_grads
    mgrads = lasagne.updates.total_norm_constraint(grads, max_norm=max_norm)
    cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]

    return cgrads  

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