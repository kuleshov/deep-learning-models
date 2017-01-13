import time
import pickle
import numpy as np

import theano
import theano.tensor as T
import lasagne

from semisup_model import SemiSupModel

from lasagne.layers import batch_norm
from layers.sampling import GaussianSampleLayer
from layers.shape import RepeatLayer

from distributions import log_bernoulli, log_normal, log_normal2

# ----------------------------------------------------------------------------

class SSADGM(SemiSupModel):
  """Auxiliary Deep Generative Model (semi-supervised version)"""
  def __init__(self, X_labeled, y_labeled, n_out, n_superbatch=12800, model='bernoulli',
                opt_alg='adam', opt_params={'lr' : 1e-3, 'b1': 0.9, 'b2': 0.99}):
    # save model that wil be created
    self.model = model
    self.n_out = n_out
    self.n_sample = 3 # monte-carlo samples; need to make this a command-line param

    SemiSupModel.__init__(self, X_labeled, y_labeled, n_out, n_superbatch, opt_alg, opt_params)
  
  def create_model(self, L, yl, n_dim, n_out, n_chan=1):
    # params
    n_lat = 100 # latent stochastic variables
    n_aux = 10 # auxiliary variables
    n_hid = 100 # size of hidden layer in encoder/decoder
    n_sam = self.n_sample # number of monte-carlo samples
    n_vis = n_dim * n_dim * n_chan # total dimensionality of ouput
    hid_nl = lasagne.nonlinearities.rectify
    relu_shift = lambda av: T.nnet.relu(av+10)-10 # for numerical stability

    # save this for later (hack; should be saved elsewhere)
    self.n_out = n_out
    self.n_aux = n_aux

    # create input layers
    l_qx_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim))
    l_qy_in = lasagne.layers.InputLayer(shape=(None, n_out))

    # create q(a|x)
    l_qa_hid1 = (lasagne.layers.DenseLayer(
        l_qx_in, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl))
    l_qa_hid2 = (lasagne.layers.DenseLayer(
        l_qa_hid1, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl))
    l_qa_mu = lasagne.layers.DenseLayer(
        l_qa_hid2, num_units=n_aux,
        W=lasagne.init.GlorotNormal(),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=None)
    l_qa_logsigma = lasagne.layers.DenseLayer(
        l_qa_hid2, num_units=n_aux,
        W=lasagne.init.GlorotNormal(),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=relu_shift)
    l_qa_mu = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qa_mu, n_ax=1, n_rep=n_sam),
        shape=(-1, n_aux))
    l_qa_logsigma = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qa_logsigma, n_ax=1, n_rep=n_sam),
        shape=(-1, n_aux))
    l_qa = GaussianSampleLayer(l_qa_mu, l_qa_logsigma)

    # create q(y|a,x)
    l_qy_hid1a = lasagne.layers.DenseLayer(
        l_qa, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl)
    l_qy_hid1b = lasagne.layers.DenseLayer(
        l_qx_in, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl)
    l_qy_hid1b = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qy_hid1b, n_ax=1, n_rep=n_sam),
        shape=(-1, n_hid))
    l_qy_hid2 = (lasagne.layers.ElemwiseSumLayer(
        [l_qy_hid1a, l_qy_hid1b]))
    l_qy_hid2 = lasagne.layers.NonlinearityLayer(l_qy_hid2, hid_nl)
    l_qy_hid3 = (lasagne.layers.DenseLayer(
        l_qy_hid2, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl))
    l_qy_mu = lasagne.layers.DenseLayer(
        l_qy_hid3, num_units=n_out,
        W=lasagne.init.GlorotNormal(),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=lasagne.nonlinearities.softmax)

    # create q(z|a,x,y)
    l_qz_hid1a = lasagne.layers.DenseLayer(
        l_qa, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl)
    l_qz_hid1b = lasagne.layers.DenseLayer(
        l_qx_in, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl)
    l_qz_hid1b = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qz_hid1b, n_ax=1, n_rep=n_sam),
        shape=(-1, n_hid))
    l_qz_hid1c = lasagne.layers.DenseLayer(
        l_qy_in, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl)
    l_qz_hid1c = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qz_hid1c, n_ax=1, n_rep=n_sam),
        shape=(-1, n_hid))
    l_qz_hid2 = (lasagne.layers.ElemwiseSumLayer(
        [l_qz_hid1a, l_qz_hid1b]))
        # [l_qz_hid1a, l_qz_hid1b, l_qz_hid1c]))
    l_qz_hid2 = lasagne.layers.NonlinearityLayer(l_qz_hid2, hid_nl)
    # l_qz_hid3 = (lasagne.layers.DenseLayer(
    #     l_qz_hid2, num_units=n_hid,
    #     W=lasagne.init.GlorotNormal('relu'),
    #     b=lasagne.init.Normal(1e-3),
    #     nonlinearity=hid_nl))
    l_qz_mu = lasagne.layers.DenseLayer(
        l_qz_hid2, num_units=n_lat,
        W=lasagne.init.GlorotNormal(),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=None)
    l_qz_logsigma = lasagne.layers.DenseLayer(
        l_qz_hid2, num_units=n_lat,
        W=lasagne.init.GlorotNormal(),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=relu_shift)
    l_qz = GaussianSampleLayer(l_qz_mu, l_qz_logsigma)

    # create the decoder network

    # create p(x|z,y)
    l_px_hid1a = lasagne.layers.DenseLayer(
        l_qz, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl)
    l_px_hid1b = lasagne.layers.DenseLayer(
        l_qy_in, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl)
    l_px_hid1b = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_px_hid1b, n_ax=1, n_rep=n_sam),
        shape=(-1, n_hid))
    l_px_hid2 = (lasagne.layers.ElemwiseSumLayer(
        [l_px_hid1a]))
        # [l_px_hid1a, l_px_hid1b]))
    l_px_hid2 = lasagne.layers.NonlinearityLayer(l_px_hid2, hid_nl)
    # l_px_hid3 = (lasagne.layers.DenseLayer(
    #     l_px_hid2, num_units=n_hid,
    #     W=lasagne.init.GlorotNormal('relu'),
    #     b=lasagne.init.Normal(1e-3),
    #     nonlinearity=hid_nl))
    l_px_mu, l_px_logsigma = None, None

    if self.model == 'bernoulli':
      l_px_mu = lasagne.layers.DenseLayer(l_px_hid2, num_units=n_vis,
          nonlinearity = lasagne.nonlinearities.sigmoid,
          W=lasagne.init.GlorotUniform(),
          b=lasagne.init.Normal(1e-3))
    elif self.model == 'gaussian':
      l_px_mu = lasagne.layers.DenseLayer(
          l_px_hid2, num_units=n_vis,
          nonlinearity=None)
      l_px_logsigma = lasagne.layers.DenseLayer(
          l_px_hid2, num_units=n_vis,
          nonlinearity=relu_shift)

    # create p(a|z,x,y)
    l_pa_hid1a = lasagne.layers.DenseLayer(
        l_qz, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl)
    l_pa_hid1b = lasagne.layers.DenseLayer(
        l_qx_in, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl)
    l_pa_hid1b = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_pa_hid1b, n_ax=1, n_rep=n_sam),
        shape=(-1, n_hid))
    l_pa_hid1c = lasagne.layers.DenseLayer(
        l_qy_in, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl)
    l_pa_hid1c = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_pa_hid1c, n_ax=1, n_rep=n_sam),
        shape=(-1, n_hid))
    l_pa_hid2 = (lasagne.layers.ElemwiseSumLayer(
        [l_pa_hid1a, l_pa_hid1b]))
        # [l_pa_hid1a, l_pa_hid1b, l_pa_hid1c]))
    l_pa_hid2 = lasagne.layers.NonlinearityLayer(l_pa_hid2, hid_nl)
    # l_pa_hid3 = (lasagne.layers.DenseLayer(
    #   l_pa_hid2, num_units=n_hid,
    #   nonlinearity=hid_nl,
    #   W=lasagne.init.GlorotNormal('relu'),
    #   b=lasagne.init.Normal(1e-3)))
    l_pa_mu = lasagne.layers.DenseLayer(
        l_pa_hid2, num_units=n_aux,
        W=lasagne.init.GlorotNormal(),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=None)
    l_pa_logsigma = lasagne.layers.DenseLayer(
        l_pa_hid2, num_units=n_aux,
        W=lasagne.init.GlorotNormal(),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=relu_shift)

    return l_px_mu, l_px_logsigma, l_pa_mu, l_pa_logsigma, \
           l_qz_mu, l_qz_logsigma, l_qa_mu, l_qa_logsigma, \
           l_qy_mu, l_qa, l_qz, l_qx_in, l_qy_in

  def create_model2(self, X, Y, n_dim, n_out, n_chan=1):
    # params
    n_lat = 200 # latent stochastic variables
    n_aux = 10  # auxiliary variables
    n_hid = 100 # size of hidden layer in encoder/decoder
    n_sam = self.n_sample # number of monte-carlo samples
    n_out = n_dim * n_dim * n_chan # total dimensionality of ouput
    hid_nl = lasagne.nonlinearities.rectify
    relu_shift = lambda av: T.nnet.relu(av+10)-10 # for numerical stability

    # create the encoder network

    self.n_aux = n_aux
    self.n_out = n_out

    # create input layers
    l_qx_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim))
    l_qy_in = lasagne.layers.InputLayer(shape=(None, n_out))

    # create q(a|x)
    l_qa_hid1 = (lasagne.layers.DenseLayer(
        l_qx_in, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl))
    # l_qa_hid2 = (lasagne.layers.DenseLayer(
    #     l_qa_hid1, num_units=n_hid,
    #     W=lasagne.init.GlorotNormal('relu'),
    #     b=lasagne.init.Normal(1e-3),
    #     nonlinearity=hid_nl))
    l_qa_mu = lasagne.layers.DenseLayer(
        l_qa_hid1, num_units=n_aux,
        W=lasagne.init.GlorotNormal(),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=None)
    l_qa_logsigma = lasagne.layers.DenseLayer(
        l_qa_hid1, num_units=n_aux,
        W=lasagne.init.GlorotNormal(),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=relu_shift)
    l_qa_mu = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qa_mu, n_ax=1, n_rep=n_sam),
        shape=(-1, n_aux))
    l_qa_logsigma = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qa_logsigma, n_ax=1, n_rep=n_sam),
        shape=(-1, n_aux))
    l_qa = GaussianSampleLayer(l_qa_mu, l_qa_logsigma)

    # create q(z|a,x,y)
    l_qz_hid1a = lasagne.layers.DenseLayer(
        l_qa, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl)
    l_qz_hid1b = lasagne.layers.DenseLayer(
        l_qx_in, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl)
    l_qz_hid1b = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qz_hid1b, n_ax=1, n_rep=n_sam),
        shape=(-1, n_hid))
    l_qz_hid1c = lasagne.layers.DenseLayer(
        l_qy_in, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl)
    l_qz_hid1c = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qz_hid1c, n_ax=1, n_rep=n_sam),
        shape=(-1, n_hid))
    l_qz_hid2 = (lasagne.layers.ElemwiseSumLayer(
        [l_qz_hid1a, l_qz_hid1b]))
        # [l_qz_hid1a, l_qz_hid1b, l_qz_hid1c]))
    l_qz_hid2 = lasagne.layers.NonlinearityLayer(l_qz_hid2, hid_nl)
    # l_qz_hid3 = (lasagne.layers.DenseLayer(
    #     l_qz_hid2, num_units=n_hid,
    #     W=lasagne.init.GlorotNormal('relu'),
    #     b=lasagne.init.Normal(1e-3),
    #     nonlinearity=hid_nl))
    l_qz_mu = lasagne.layers.DenseLayer(
        l_qz_hid2, num_units=n_lat,
        W=lasagne.init.GlorotNormal(),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=None)
    l_qz_logsigma = lasagne.layers.DenseLayer(
        l_qz_hid2, num_units=n_lat,
        W=lasagne.init.GlorotNormal(),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=relu_shift)
    l_qz = GaussianSampleLayer(l_qz_mu, l_qz_logsigma)

    # create the decoder network

    # create p(x|z)
    l_px_hid = lasagne.layers.DenseLayer(
        l_qz, num_units=n_hid,
        W=lasagne.init.GlorotNormal('relu'),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=hid_nl)
    l_px_mu, l_px_logsigma = None, None

    if self.model == 'bernoulli':
      l_px_mu = lasagne.layers.DenseLayer(l_px_hid, num_units=n_out,
          nonlinearity = lasagne.nonlinearities.sigmoid,
          W=lasagne.init.GlorotUniform(),
          b=lasagne.init.Normal(1e-3))
    elif self.model == 'gaussian':
      l_px_mu = lasagne.layers.DenseLayer(
          l_px_hid, num_units=n_out,
          nonlinearity=None)
      l_px_logsigma = lasagne.layers.DenseLayer(
          l_px_hid, num_units=n_out,
          nonlinearity=relu_shift)

    # create p(a|z)
    l_pa_hid = lasagne.layers.DenseLayer(
      l_qz, num_units=n_hid,
      nonlinearity=hid_nl,
      W=lasagne.init.GlorotNormal('relu'),
      b=lasagne.init.Normal(1e-3))
    l_pa_mu = lasagne.layers.DenseLayer(
        l_pa_hid, num_units=n_aux,
        W=lasagne.init.GlorotNormal(),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=None)
    l_pa_logsigma = lasagne.layers.DenseLayer(
        l_pa_hid, num_units=n_aux,
        W=lasagne.init.GlorotNormal(),
        b=lasagne.init.Normal(1e-3),
        nonlinearity=relu_shift)

    return l_px_mu, l_px_logsigma, l_pa_mu, l_pa_logsigma, \
           l_qz_mu, l_qz_logsigma, l_qa_mu, l_qa_logsigma, \
           l_qa, l_qa, l_qz, l_qx_in, None

  def create_objectives(self, X, L, yu, yl, deterministic=False):
    # load data dimensions
    n_lbl, n_chan, n_dim, _ = L.shape
    n_vis = n_dim * n_dim * n_chan
    n_unl = X.shape[0]
    n_sam = self.n_sample
    n_out = self.n_out
    n_aux = self.n_aux

    # duplicate entries to take into account multiple mc samples
    x = L.flatten(2)
    x = x.dimshuffle(0,'x',1).repeat(n_sam, axis=1).reshape((-1, n_vis))
    yl = lasagne.utils.one_hot(yl, m=n_out)
    yl = yl.dimshuffle(0,'x',1).repeat(n_sam, axis=1).reshape((-1, n_out))

    # load network
    l_px_mu, l_px_logsigma, l_pa_mu, l_pa_logsigma, \
      l_qz_mu, l_qz_logsigma, l_qa_mu, l_qa_logsigma, \
      l_qy_mu, l_qa, l_qz, l_qx_in, l_qy_in = self.network

    # first, we construct the ELBO on the labeled examples
    
    # load network output
    # input_asgn = { l_qx_in : L, l_qy_in : yl }
    input_asgn = { l_qx_in : L }
    pa_mu, pa_logsigma, qz_mu, qz_logsigma, qa_mu, qa_logsigma, qy_mu, a, z \
      = lasagne.layers.get_output(
          [ l_pa_mu, l_pa_logsigma, l_qz_mu, l_qz_logsigma, 
            l_qa_mu, l_qa_logsigma, l_qy_mu, l_qa, l_qz ],
          input_asgn, deterministic=deterministic)

    if self.model == 'bernoulli':
      px_mu = lasagne.layers.get_output(l_px_mu, input_asgn, deterministic=deterministic)
    elif self.model == 'gaussian':
      px_mu, px_logsigma = lasagne.layers.get_output([l_px_mu, l_px_logsigma], input_asgn,
                                                     deterministic=deterministic)

    # entropy term
    log_qa_given_x   = log_normal2(a, qa_mu, qa_logsigma).sum(axis=1)
    log_qz_given_ayx = log_normal2(z, qz_mu, qz_logsigma).sum(axis=1)
    log_qza_given_xy = log_qz_given_ayx + log_qa_given_x

    # log-probability term
    z_prior_sigma  = T.cast(T.ones_like(qz_logsigma), dtype=theano.config.floatX)
    z_prior_mu     = T.cast(T.zeros_like(qz_mu), dtype=theano.config.floatX)
    y_prior        = T.cast(T.ones((n_lbl*n_sam, n_out)) / n_out, dtype=theano.config.floatX)
    log_pz         = log_normal(z, z_prior_mu,  z_prior_sigma).sum(axis=1)
    log_pa_given_z = log_normal2(a, pa_mu, pa_logsigma).sum(axis=1)
    log_py         = -lasagne.objectives.categorical_crossentropy(y_prior, yl)

    if self.model == 'bernoulli':
      log_px_given_z = log_bernoulli(x, px_mu).sum(axis=1)
    elif self.model == 'gaussian':
      log_px_given_z = log_normal2(x, px_mu, px_logsigma).sum(axis=1)

    log_paxzy = log_pa_given_z + log_px_given_z + log_pz + log_py

    # compute the evidence lower bound
    elbo_lbl = T.mean(log_paxzy - log_qza_given_xy, axis=0)

    # # next, we build the elbo on the unlabeled examples

    # # we are going to replicate the batch n_out times, once for each label
    # I = T.eye(n_out)
    # t = I.reshape((n_out, 1, n_out)).repeat(n_unl, axis=1).reshape((-1, n_out))
    # U = X.reshape((1, n_unl, n_chan, n_dim, n_dim)).repeat(n_out, axis=0) \
    #      .reshape((-1, n_chan, n_dim, n_dim))
    # u = U.flatten(2)

    # # load network output
    # # not going to try to be fancy right now (commenting this out):
    # # a_unl = get_output(l_qa, X)
    # # a_unl_rep = a_unl.reshape((1, n_unl*n_sam, n_aux)) \
    # #                  .repeat(n_out, axis=0).reshape((-1, n_aux))

    # input_asgn = { l_qx_in : U, l_qy_in : t }
    # pa_mu, pa_logsigma, qz_mu, qz_logsigma, qa_mu, qa_logsigma, qy_mu, a, z \
    #   = lasagne.layers.get_output(
    #       [ l_pa_mu, l_pa_logsigma, l_qz_mu, l_qz_logsigma, 
    #         l_qa_mu, l_qa_logsigma, l_qy_mu, l_qa, l_qz ],
    #       input_asgn, deterministic=deterministic)

    # if self.model == 'bernoulli':
    #   px_mu = lasagne.layers.get_output(l_px_mu, input_asgn, deterministic=deterministic)
    # elif self.model == 'gaussian':
    #   px_mu, px_logsigma = lasagne.layers.get_output([l_px_mu, l_px_logsigma], input_asgn,
    #                                                  deterministic=deterministic)

    # # entropy term
    # log_qa_given_x   = log_normal2(a, qa_mu, qa_logsigma).sum(axis=1)
    # log_qz_given_ayx = log_normal2(z, qz_mu, qz_logsigma).sum(axis=1)
    # log_qy_given_ax  = log_bernoulli(t, qy_mu).sum(axis=1)
    # log_qza_given_xy = log_qz_given_ayx + log_qa_given_x + log_qy_given_ax

    # # log-probability term
    # z_prior_sigma  = T.cast(T.ones_like(qz_logsigma), dtype=theano.config.floatX)
    # z_prior_mu     = T.cast(T.zeros_like(qz_mu), dtype=theano.config.floatX)
    # y_prior        = T.cast(T.ones((n_out*n_unl*n_sam, n_out)) / n_out, dtype=theano.config.floatX)
    # log_pz         = log_normal(z, z_prior_mu,  z_prior_sigma).sum(axis=1)
    # log_pa_given_z = log_normal2(a, pa_mu, pa_logsigma).sum(axis=1)
    # log_py         = -lasagne.objectives.categorical_crossentropy(y_prior, t)

    # if self.model == 'bernoulli':
    #   log_px_given_z = log_bernoulli(u, px_mu).sum(axis=1)
    # elif self.model == 'gaussian':
    #   log_px_given_z = log_normal2(u, px_mu, px_logsigma).sum(axis=1)

    # log_paxzy = log_pa_given_z + log_px_given_z + log_pz + log_py

    # # compute the evidence lower bound
    # elbo_unl = T.mean(log_paxzy - log_qza_given_xy, axis=0)

    # compute the total lower bound
    elbo = elbo_lbl #+ elbo_unl

    # in case we want regularization:
    # l2_reg = 0.0
    # for p in self.get_params():
    #   if 'W' not in str(p): continue
    #   l2_reg += log_normal(p, 0, 1).sum()
    # elbo_lbl += l2_reg 

    # we don't use a spearate accuracy metric right now
    return -elbo, -elbo_lbl

  def create_gradients(self, loss, deterministic=False):
    grads = SemiSupModel.create_gradients(self, loss, deterministic)

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