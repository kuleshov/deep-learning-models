import time
import pickle
import numpy as np
from collections import OrderedDict

import theano
import theano.tensor as T
import lasagne

from model import Model

from layers import BernoulliSampleLayer
from distributions import log_bernoulli

# ----------------------------------------------------------------------------

class SBN(Model):
  """Sigmoid Belief Network trained using Neural Variational Inference"""
  def __init__(self, n_dim, n_out, n_chan=1, n_superbatch=12800, opt_alg='adam', 
              opt_params={'lr' : 1e-3, 'b1': 0.9, 'b2': 0.99}):

    # invoke parent constructor
    Model.__init__(self, n_dim, n_chan, n_out, n_superbatch, opt_alg, opt_params)
  
  def create_model(self, X, Y, n_dim, n_out, n_chan=1):
    # params
    n_lat    = 200 # latent stochastic variables
    n_hid    = 500 # size of hidden layer in encoder/decoder
    n_hid_cv = 500 # size of hidden layer in control variate net
    n_out    = n_dim * n_dim * n_chan # total dimensionality of ouput
    hid_nl   = lasagne.nonlinearities.tanh

    # create the encoder network
    l_q_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim), 
                                     input_var=X)
    l_q_hid = lasagne.layers.DenseLayer(
        l_q_in, num_units=n_hid,
        nonlinearity=hid_nl)
    l_q_out = lasagne.layers.DenseLayer(
        l_q_hid, num_units=n_lat,
        nonlinearity=None)
    l_q_mu = lasagne.layers.DenseLayer(
        l_q_hid, num_units=n_lat,
        nonlinearity=T.nnet.sigmoid)
    l_q_sample = BernoulliSampleLayer(l_q_mu)

    # create the decoder network
    # note that we currently only handle Bernoulli x variables
    l_p_in = lasagne.layers.InputLayer((None, n_lat))
    l_p_hid = lasagne.layers.DenseLayer(
        l_p_in, num_units=n_hid,
        nonlinearity=hid_nl,
        W=lasagne.init.GlorotUniform())
    l_p_mu = lasagne.layers.DenseLayer(l_p_hid, num_units=n_out,
        nonlinearity = lasagne.nonlinearities.sigmoid,
        W=lasagne.init.GlorotUniform(),
        b=lasagne.init.Constant(0.))

    # create control variate (baseline) network
    l_cv_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim), 
                                        input_var=X)
    l_cv_hid = lasagne.layers.DenseLayer(
        l_cv_in, num_units=n_hid_cv,
        nonlinearity=hid_nl)
    l_cv = lasagne.layers.DenseLayer(
        l_cv_hid, num_units=1,
        nonlinearity=None)

    # create variables for centering signal
    c = theano.shared(np.zeros((1,1), dtype=np.float32), broadcastable=(True,True))
    v = theano.shared(np.zeros((1,1), dtype=np.float32), broadcastable=(True,True))

    return l_p_mu, l_q_mu, l_q_sample, l_cv, c, v

  def _create_components(self, deterministic=False):
    # load network input
    X = self.inputs[0]
    x = X.flatten(2)

    # load networks
    l_p_mu, l_q_mu, l_q_sample, _, _, _ = self.network

    # load network output
    z, q_mu = lasagne.layers.get_output([l_q_sample, l_q_mu], deterministic=deterministic)
    p_mu = lasagne.layers.get_output(l_p_mu, z, deterministic=deterministic)

    # entropy term
    log_qz_given_x = log_bernoulli(z, q_mu).sum(axis=1)

    # expected p(x,z) term
    z_prior = T.ones_like(z)*np.float32(0.5)
    log_pz = log_bernoulli(z, z_prior).sum(axis=1)
    log_px_given_z = log_bernoulli(x, p_mu).sum(axis=1)
    log_pxz = log_pz + log_px_given_z

    # save them for later
    self.log_pxz = log_pxz
    self.log_qz_given_x = log_qz_given_x

    return log_pxz.flatten(), log_qz_given_x.flatten()

  def create_objectives(self, deterministic=False):
    # load probabilities
    log_pxz, log_qz_given_x = self._create_components(deterministic=deterministic)

    # compute the lower bound
    elbo = T.mean(log_pxz - log_qz_given_x)

    # we don't use the second accuracy metric right now
    return -elbo, -T.mean(log_qz_given_x)

  def create_gradients(self, loss, deterministic=False):
    from theano.gradient import disconnected_grad as dg

    # load networks
    l_p_mu, l_q_mu, _, l_cv, c, v = self.network

    # load params
    p_params  = lasagne.layers.get_all_params(l_p_mu, trainable=True)
    q_params  = lasagne.layers.get_all_params(l_q_mu, trainable=True)
    cv_params = lasagne.layers.get_all_params(l_cv, trainable=True)

    # load neural net outputs (probabilities have been precomputed)
    log_pxz, log_qz_given_x = self.log_pxz, self.log_qz_given_x
    cv = T.addbroadcast(lasagne.layers.get_output(l_cv),1)

    # compute learning signals
    l = log_pxz - log_qz_given_x - cv
    l_avg, l_std = l.mean(), T.maximum(1, l.std())
    c_new = 0.8*c + 0.2*l_avg
    v_new = 0.8*v + 0.2*l_std
    l = (l - c_new) / v_new
  
    # compute grad wrt p
    p_grads = T.grad(-log_pxz.mean(), p_params)

    # compute grad wrt q
    q_target = T.mean(dg(l) * log_qz_given_x)
    q_grads = T.grad(-0.2*q_target, q_params) # 5x slower rate for q

    # compute grad of cv net
    cv_target = T.mean(l**2)
    cv_grads = T.grad(cv_target, cv_params)

    # combine and clip gradients
    clip_grad = 1
    max_norm = 5
    grads = p_grads + q_grads + cv_grads
    mgrads = lasagne.updates.total_norm_constraint(grads, max_norm=max_norm)
    cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]

    return cgrads

  def get_params(self):
    l_p_mu, l_q_mu, _, l_cv, _, _ = self.network
    p_params  = lasagne.layers.get_all_params(l_p_mu, trainable=True)
    q_params  = lasagne.layers.get_all_params(l_q_mu, trainable=True)
    cv_params = lasagne.layers.get_all_params(l_cv, trainable=True)
    return p_params + q_params + cv_params #+ [c]

  def create_updates(self, grads, params, alpha, opt_alg, opt_params):
    # call super-class to generate SGD/ADAM updates
    grad_updates = Model.create_updates(self, grads, params, alpha, opt_alg, opt_params)

    # create updates for centering signal

    # load neural net outputs (probabilities have been precomputed)
    _, _, _, l_cv, c, v = self.network
    log_pxz, log_qz_given_x = self.log_pxz, self.log_qz_given_x
    cv = T.addbroadcast(lasagne.layers.get_output(l_cv),1)

    # compute learning signals
    l = log_pxz - log_qz_given_x - cv
    l_avg, l_std = l.mean(), T.maximum(1, l.std())
    c_new = 0.8*c + 0.2*l_avg
    v_new = 0.8*v + 0.2*l_std

    # compute update for centering signal
    cv_updates = {c : c_new, v : v_new}

    return OrderedDict( grad_updates.items() + cv_updates.items() )
    