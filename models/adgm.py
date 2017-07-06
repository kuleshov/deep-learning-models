import time
import pickle
import numpy as np

import theano
import theano.tensor as T
import lasagne

from model import Model

from lasagne.layers import batch_norm
from layers.sampling import GaussianSampleLayer
from layers.shape import RepeatLayer
from layers import extras as nn

from distributions import log_bernoulli, log_normal, log_normal2

# ----------------------------------------------------------------------------

class ADGM(Model):
  """Auxiliary Deep Generative Model (unsupervised version)"""
  def __init__(self, n_dim, n_out, n_chan=1, n_batch=128, n_superbatch=12800, model='bernoulli',
                opt_alg='adam', opt_params={'lr' : 1e-3, 'b1': 0.9, 'b2': 0.99}):
    # save model that wil be created
    self.model = model
    self.n_sample = 1 # adjustable parameter, though 1 works best in practice

    self.n_batch = n_batch
    self.n_lat = 200
    self.n_dim = n_dim
    self.n_chan = n_chan
    self.n_batch = n_batch

    Model.__init__(self, n_dim, n_chan, n_out, n_superbatch, opt_alg, opt_params)

    # sample generation
    Z = T.matrix(dtype=theano.config.floatX) # noise matrix
    l_px_mu, l_px_logsigma, l_pa_mu, l_pa_logsigma, \
        l_qz_mu, l_qz_logsigma, l_qa_mu, l_qa_logsigma, \
        l_qa, l_qz  = self.network
    sample = lasagne.layers.get_output(l_px_mu,  {l_qz : Z}, deterministic=True)
    self.sample = theano.function([Z], sample, on_unused_input='warn')
  
  def create_model(self, X, Y, n_dim, n_out, n_chan=1):
    # params
    n_lat = 200 # latent stochastic variables
    n_aux = 10  # auxiliary variables
    n_hid = 499 # size of hidden layer in encoder/decoder
    n_sam = self.n_sample # number of monte-carlo samples
    n_out = n_dim * n_dim * n_chan # total dimensionality of ouput
    hid_nl = lasagne.nonlinearities.rectify
    # relu_shift = lambda av: T.nnet.relu(av+10)-10 # for numerical stability
    # relu_shift = lambda av: 5*T.tanh(av) # for numerical stability
    relu_shift = lambda av: T.clip(av, -5, 5)  # for numerical stability    

    # create the encoder network

    # create q(a|x)
    l_qa_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim), 
                                     input_var=X)
    l_qa_hid1 = batch_norm(lasagne.layers.DenseLayer(
        l_qa_in, num_units=n_hid,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=hid_nl))
    # l_qa_hid2 = (lasagne.layers.DenseLayer(
    #     l_qa_hid1, num_units=n_hid,
    #     W=lasagne.init.Orthogonal(),
    #     b=lasagne.init.Constant(0.0),
    #     nonlinearity=hid_nl))
    l_qa_mu = lasagne.layers.DenseLayer(
        l_qa_hid1, num_units=n_aux,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=None)
    l_qa_logsigma = lasagne.layers.DenseLayer(
        l_qa_hid1, num_units=n_aux,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=relu_shift)
    l_qa_mu = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qa_mu, n_ax=1, n_rep=n_sam),
        shape=(-1, n_aux))
    l_qa_logsigma = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qa_logsigma, n_ax=1, n_rep=n_sam),
        shape=(-1, n_aux))
    l_qa = GaussianSampleLayer(l_qa_mu, l_qa_logsigma)

    # create q(z|a,x)
    l_qz_hid1a = batch_norm(lasagne.layers.DenseLayer(
        l_qa, num_units=n_hid,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=hid_nl))
    l_qz_hid1b = batch_norm(lasagne.layers.DenseLayer(
        l_qa_in, num_units=n_hid,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=hid_nl))
    l_qz_hid1b = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qz_hid1b, n_ax=1, n_rep=n_sam),
        shape=(-1, n_hid))
    l_qz_hid2 = lasagne.layers.ElemwiseSumLayer([l_qz_hid1a, l_qz_hid1b])
    l_qz_hid2 = lasagne.layers.NonlinearityLayer(l_qz_hid2, hid_nl)
    # l_qz_hid3 = (lasagne.layers.DenseLayer(
    #     l_qz_hid2, num_units=n_hid,
    #     W=lasagne.init.Orthogonal(),
    #     b=lasagne.init.Constant(0.0),
    #     nonlinearity=hid_nl))
    l_qz_mu = lasagne.layers.DenseLayer(
        l_qz_hid2, num_units=n_lat,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=None)
    l_qz_logsigma = lasagne.layers.DenseLayer(
        l_qz_hid2, num_units=n_lat,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=relu_shift)
    l_qz = GaussianSampleLayer(l_qz_mu, l_qz_logsigma, name='l_qz')

    # create the decoder network

    # create p(x|z)
    l_px_hid1 = batch_norm(lasagne.layers.DenseLayer(
        l_qz, num_units=n_hid,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=hid_nl))
    # l_px_hid2 = (lasagne.layers.DenseLayer(
    #     l_px_hid1, num_units=n_hid,
    #     W=lasagne.init.Orthogonal(),
    #     b=lasagne.init.Constant(0.0),
    #     nonlinearity=hid_nl))
    l_px_mu, l_px_logsigma = None, None

    if self.model == 'bernoulli':
      l_px_mu = lasagne.layers.DenseLayer(l_px_hid1, num_units=n_out,
          nonlinearity = lasagne.nonlinearities.sigmoid,
          W=lasagne.init.GlorotUniform(),
          b=lasagne.init.Constant(0.0))
    elif self.model == 'gaussian':
      l_px_mu = lasagne.layers.DenseLayer(
          l_px_hid1, num_units=n_out,
          W=lasagne.init.Orthogonal(),
          b=lasagne.init.Constant(0.0),
          nonlinearity=None)
      l_px_logsigma = lasagne.layers.DenseLayer(
          l_px_hid1, num_units=n_out,
          W=lasagne.init.Orthogonal(),
          b=lasagne.init.Constant(0.0),
          nonlinearity=relu_shift)

    # create p(a|x,z)
    # l_pa_hid1a = lasagne.layers.DenseLayer(
    #   l_qa_in, num_units=n_hid,
    #   nonlinearity=hid_nl,
    #   W=lasagne.init.Orthogonal(),
    #   b=lasagne.init.Constant(0.0))
    # l_pa_hid1b = lasagne.layers.DenseLayer(
    #   l_qz, num_units=n_hid,
    #   nonlinearity=hid_nl,
    #   W=lasagne.init.Orthogonal(),
    #   b=lasagne.init.Constant(0.0))
    # l_pa_hid2 = lasagne.layers.ElemwiseSumLayer([l_pa_hid1a, l_pa_hid1b])
    # l_pa_hid2 = lasagne.layers.NonlinearityLayer(l_pa_hid2, hid_nl)
    # l_pa_mu = lasagne.layers.DenseLayer(
    #     l_pa_hid2, num_units=n_aux,
    #     W=lasagne.init.Orthogonal(),
    #     b=lasagne.init.Constant(0.0),
    #     nonlinearity=None)
    # l_pa_logsigma = lasagne.layers.DenseLayer(
    #     l_pa_hid2, num_units=n_aux,
    #     W=lasagne.init.Orthogonal(),
    #     b=lasagne.init.Constant(0.0),
    #     nonlinearity=relu_shift)

    # create p(a|z)
    l_pa_hid1 = batch_norm(lasagne.layers.DenseLayer(
      l_qz, num_units=n_hid,
      nonlinearity=hid_nl,
      W=lasagne.init.Orthogonal(),
      b=lasagne.init.Constant(0.0)))
    # l_pa_hid2 = batch_norm(lasagne.layers.DenseLayer(
    #   l_pa_hid1, num_units=n_hid,
    #   nonlinearity=hid_nl,
    #   W=lasagne.init.Orthogonal(),
    #   b=lasagne.init.Constant(0.0)))
    l_pa_mu = lasagne.layers.DenseLayer(
        l_pa_hid1, num_units=n_aux,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=None)
    l_pa_logsigma = lasagne.layers.DenseLayer(
        l_pa_hid1, num_units=n_aux,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=relu_shift)

    return l_px_mu, l_px_logsigma, l_pa_mu, l_pa_logsigma, \
           l_qz_mu, l_qz_logsigma, l_qa_mu, l_qa_logsigma, \
           l_qa, l_qz

  def create_objectives(self, deterministic=False):
    # load network input
    X = self.inputs[0]
    x = X.flatten(2)

    # duplicate entries to take into account multiple mc samples
    n_sam = self.n_sample
    n_out = x.shape[1]
    x = x.dimshuffle(0,'x',1).repeat(n_sam, axis=1).reshape((-1, n_out))

    # load network
    l_px_mu, l_px_logsigma, l_pa_mu, l_pa_logsigma, \
      l_qz_mu, l_qz_logsigma, l_qa_mu, l_qa_logsigma, \
      l_qa, l_qz = self.network
    
    # load network output
    pa_mu, pa_logsigma, qz_mu, qz_logsigma, qa_mu, qa_logsigma, a, z \
      = lasagne.layers.get_output(
          [ l_pa_mu, l_pa_logsigma, l_qz_mu, l_qz_logsigma, 
            l_qa_mu, l_qa_logsigma, l_qa, l_qz ], 
          deterministic=deterministic)

    if self.model == 'bernoulli':
      px_mu = lasagne.layers.get_output(l_px_mu, deterministic=deterministic)
    elif self.model == 'gaussian':
      px_mu, px_logsigma = lasagne.layers.get_output([l_px_mu, l_px_logsigma], 
                                                     deterministic=deterministic)

    # entropy term
    log_qa_given_x  = log_normal2(a, qa_mu, qa_logsigma).sum(axis=1)
    log_qz_given_ax = log_normal2(z, qz_mu, qz_logsigma).sum(axis=1)
    log_qza_given_x = log_qz_given_ax + log_qa_given_x

    # log-probability term
    z_prior_sigma = T.cast(T.ones_like(qz_logsigma), dtype=theano.config.floatX)
    z_prior_mu = T.cast(T.zeros_like(qz_mu), dtype=theano.config.floatX)
    log_pz = log_normal(z, z_prior_mu,  z_prior_sigma).sum(axis=1)
    log_pa_given_z = log_normal2(a, pa_mu, pa_logsigma).sum(axis=1)

    if self.model == 'bernoulli':
      log_px_given_z = log_bernoulli(x, px_mu).sum(axis=1)
    elif self.model == 'gaussian':
      log_px_given_z = log_normal2(x, px_mu, px_logsigma).sum(axis=1)

    log_paxz = log_pa_given_z + log_px_given_z + log_pz

    # # experiment: uniform prior p(a)
    # a_prior_sigma = T.cast(T.ones_like(qa_logsigma), dtype=theano.config.floatX)
    # a_prior_mu = T.cast(T.zeros_like(qa_mu), dtype=theano.config.floatX)
    # log_pa = log_normal(a, a_prior_mu,  a_prior_sigma).sum(axis=1)
    # log_paxz = log_pa + log_px_given_z + log_pz

    # compute the evidence lower bound
    elbo = T.mean(log_paxz - log_qza_given_x)

    # we don't use a spearate accuracy metric right now
    return -elbo, T.max(qz_logsigma)

  def create_gradients(self, loss, deterministic=False):
    grads = Model.create_gradients(self, loss, deterministic)

    # combine and clip gradients
    clip_grad = 1
    max_norm = 5
    mgrads = lasagne.updates.total_norm_constraint(grads, max_norm=max_norm)
    cgrads = [T.clip(g, -clip_grad, clip_grad) for g in mgrads]

    return cgrads

  def get_params(self):
    l_px_mu = self.network[0]
    l_pa_mu = self.network[2]
    params  = lasagne.layers.get_all_params(l_px_mu, trainable=True)
    params0 = lasagne.layers.get_all_params(l_pa_mu, trainable=True)
    for param in params0:
      if param not in params: params.append(param)
    
    return params

  def gen_samples(self, n_sam):
    n_lat, n_dim, n_chan, n_batch = self.n_lat, self.n_dim, self.n_chan, self.n_batch
    noise = np.random.randn(n_batch, n_lat).astype(theano.config.floatX)
    # noise = np.zeros((n_sam, n_lat))
    # noise[range(n_sam), np.random.choice(n_lat, n_sam)] = 1

    assert np.sqrt(n_sam) == int(np.sqrt(n_sam))
    n_side = int(np.sqrt(n_sam))

    p_mu = self.sample(noise)
    p_mu = p_mu[:n_sam]
    p_mu = p_mu.reshape((n_side, n_side, n_chan, n_dim, n_dim))
    p_mu = p_mu[:,:,0,:,:] # keep the first channel

    # split into n_side (1,n_side,n_dim,n_dim,) images,
    # concat along columns -> 1,n_side,n_dim,n_dim*n_side
    p_mu = np.concatenate(np.split(p_mu, n_side, axis=0), axis=3)
    # split into n_side (1,1,n_dim,n_dim*n_side) images,
    # concat along rows -> 1,1,n_dim*n_side,n_dim*n_side
    p_mu = np.concatenate(np.split(p_mu, n_side, axis=1), axis=2)
    return np.squeeze(p_mu)