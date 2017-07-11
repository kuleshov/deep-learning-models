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
from layers.normalization import weight_norm
from distributions import log_bernoulli, log_normal, log_normal2

import scipy.misc

# ----------------------------------------------------------------------------

class ConvVAE(Model):
  """Variational Autoencoder with Gaussian visible and latent variables"""
  def __init__(self, n_dim, n_out, n_chan=1, n_batch=128, n_superbatch=12800, model='bernoulli',
                opt_alg='adam', opt_params={'lr' : 1e-3, 'b1': 0.9, 'b2': 0.99}):
    # save model that wil be created
    self.model = model
    self.n_batch = n_batch
    self.n_lat = 100
    self.n_dim = n_dim
    self.n_chan = n_chan
    self.n_batch = n_batch

    # invoke parent constructor
    Model.__init__(self, n_dim, n_chan, n_out, n_superbatch, opt_alg, opt_params)

    # sample generation
    Z = T.matrix(dtype=theano.config.floatX) # noise matrix
    _, _, _, _, l_sample, l_p_z = self.network
    sample = lasagne.layers.get_output(l_sample,  {l_p_z : Z}, deterministic=True)
    self.sample = theano.function([Z], sample, on_unused_input='warn')

  def create_model(self, X, Y, n_dim, n_out, n_chan=1):
    return self.create_deconv_model(X, Y, n_dim, n_out, n_chan)

  def create_deconv_model(self, X, Y, n_dim, n_out, n_chan=1):
    # params
    n_lat = 100 # latent stochastic variabels
    n_out = n_dim * n_dim * n_chan # total dimensionality of output
    hid_nl = lasagne.nonlinearities.rectify
    safe_nl = lambda av: T.clip(av, -7, 1)  # for numerical stability 

    # create the encoder network
    l_q_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim), 
                                     input_var=X)

    l_q_conv1 = weight_norm(lasagne.layers.Conv2DLayer(
        l_q_in, num_filters=128, filter_size=(5, 5), stride=2,
        nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2),
        pad='same', W=lasagne.init.Normal(5e-2)))

    l_q_conv2 = weight_norm(lasagne.layers.Conv2DLayer(
        l_q_conv1, num_filters=256, filter_size=(5, 5), stride=2,
        nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2),
        pad='same', W=lasagne.init.Normal(5e-2)))

    l_q_conv3 = weight_norm(lasagne.layers.Conv2DLayer(
        l_q_conv2, num_filters=512, filter_size=(5, 5), stride=2,
        nonlinearity=lasagne.nonlinearities.LeakyRectify(0.2),
        pad='same', W=lasagne.init.Normal(5e-2)))

    l_q_mu = weight_norm(lasagne.layers.DenseLayer(
        l_q_conv3, num_units=n_lat, nonlinearity=None,
        W=lasagne.init.Normal(5e-2)))

    l_q_logsigma = weight_norm(lasagne.layers.DenseLayer(
        l_q_conv3, num_units=n_lat, nonlinearity=safe_nl,
        W=lasagne.init.Normal(5e-2)))

    # create the decoder network
    l_p_z = GaussianSampleLayer(l_q_mu, l_q_logsigma)

    l_p_hid1 = weight_norm(lasagne.layers.DenseLayer(
        l_p_z, num_units=4*4*512, nonlinearity=hid_nl, 
        W=lasagne.init.Normal(5e-2)))
    l_p_hid1 = lasagne.layers.ReshapeLayer(l_p_hid1, (-1, 512, 4, 4))
    
    l_p_hid2 = lasagne.layers.Upscale2DLayer(l_p_hid1, 2)
    l_p_hid2 = weight_norm(lasagne.layers.Conv2DLayer(l_p_hid2, 
      num_filters=256, filter_size=(5,5), pad='same',
      nonlinearity=hid_nl))

    l_p_hid3 = lasagne.layers.Upscale2DLayer(l_p_hid2, 2)
    l_p_hid3 = weight_norm(lasagne.layers.Conv2DLayer(l_p_hid3, 
      num_filters=128, filter_size=(5,5), pad='same',
      nonlinearity=hid_nl))

    l_p_up = lasagne.layers.Upscale2DLayer(l_p_hid3, 2)
    l_p_mu = lasagne.layers.flatten(
      weight_norm(lasagne.layers.Conv2DLayer(l_p_up, 
      num_filters=3, filter_size=(5,5), pad='same',
      nonlinearity=lasagne.nonlinearities.sigmoid)))
    l_p_logsigma = lasagne.layers.flatten(
      weight_norm(lasagne.layers.Conv2DLayer(l_p_up, 
      num_filters=3, filter_size=(5,5), pad='same',
      nonlinearity=safe_nl)))

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
    elif self.model in ('gaussian', 'svhn'):
      p_mu, p_logsigma, q_mu, q_logsigma, _, _ \
          = lasagne.layers.get_output(self.network, deterministic=deterministic)

    # first term of the ELBO: kl-divergence (using the closed form expression)
    kl_div = 0.5 * T.sum(1 + 2*q_logsigma - T.sqr(q_mu) 
                         - T.exp(2 * T.minimum(q_logsigma,50)), axis=1).mean()

    # second term: log-likelihood of the data under the model
    if self.model == 'bernoulli':
      logpxz = -lasagne.objectives.binary_crossentropy(sample, X.flatten(2)).sum(axis=1).mean()
    elif self.model in ('gaussian', 'svhn'):
      # def log_lik(x, mu, log_sig):
      #     return T.sum(-(np.float32(0.5 * np.log(2 * np.pi)) + log_sig)
      #                   - 0.5 * T.sqr(x - mu) / T.exp(2 * log_sig), axis=1)
      # logpxz = log_lik(X.flatten(2), p_mu, p_logsigma).mean()
      logpxz = log_normal2(X.flatten(2), p_mu, p_logsigma).sum(axis=1).mean()

    loss = -1 * (logpxz + kl_div)

    # we don't use the spearate accuracy metric right now
    return loss, -kl_div

  def gen_samples(self, n_sam):
    n_lat, n_dim, n_chan, n_batch = self.n_lat, self.n_dim, self.n_chan, self.n_batch
    noise = np.random.randn(n_batch, n_lat).astype(theano.config.floatX)
    # noise = np.zeros((n_sam, n_lat))
    # noise[range(n_sam), np.random.choice(n_lat, n_sam)] = 1

    assert np.sqrt(n_sam) == int(np.sqrt(n_sam))
    n_side = int(np.sqrt(n_sam))

    p_mu = self.sample(noise)
    one_sample = p_mu[0]
    p_mu = p_mu[:n_sam]
    p_mu = p_mu.reshape((n_side, n_side, n_chan, n_dim, n_dim))
    p_mu = p_mu[:,:,0,:,:] # keep the first channel

    scipy.misc.imsave('test_one_sample.png', one_sample.reshape((n_chan,n_dim,n_dim))[0])

    # split into n_side (1,n_side,n_dim,n_dim,) images,
    # concat along columns -> 1,n_side,n_dim,n_dim*n_side
    p_mu = np.concatenate(np.split(p_mu, n_side, axis=0), axis=3)
    # split into n_side (1,1,n_dim,n_dim*n_side) images,
    # concat along rows -> 1,1,n_dim*n_side,n_dim*n_side
    p_mu = np.concatenate(np.split(p_mu, n_side, axis=1), axis=2)
    return np.squeeze(p_mu)

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