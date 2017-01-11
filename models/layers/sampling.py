import numpy as np
import theano.tensor as T
import lasagne

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

# ----------------------------------------------------------------------------

class GaussianSampleLayer(lasagne.layers.MergeLayer):
    def __init__(self, mu, logsigma, rng=None, **kwargs):
        self.rng = rng if rng else RandomStreams(lasagne.random.get_rng().randint(1,2147462579))
        super(GaussianSampleLayer, self).__init__([mu, logsigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs
        shape=(self.input_shapes[0][0] or inputs[0].shape[0],
                self.input_shapes[0][1] or inputs[0].shape[1])
        if deterministic:
            return mu
        return mu + T.exp(logsigma) * self.rng.normal(shape)

class GaussianMultiSampleLayer(lasagne.layers.MergeLayer):
    def __init__(self, mu, logsigma, n_samples, rng=None, **kwargs):
        self.rng = rng if rng else RandomStreams(lasagne.random.get_rng().randint(1,2147462579))
        self.n_samples = n_samples
        super(GaussianMultiSampleLayer, self).__init__([mu, logsigma], **kwargs)

    def get_output_shape_for(self, input_shapes):
        n_batch, n_dim = input_shapes[0]
        return (n_batch, self.n_samples, n_dim)

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        mu, logsigma = inputs

        # get dimensions
        n_bat = self.input_shapes[0][0] or inputs[0].shape[0]
        n_dim = self.input_shapes[0][1] or inputs[0].shape[1]
        n_sam = self.n_samples

        # reshape inputs into rank-3 tensors
        mu3 = mu.reshape((n_bat,1,n_dim)).repeat(n_sam, axis=1)
        ls3 = logsigma.reshape((n_bat,1,n_dim)).repeat(n_sam, axis=1)
        
        # return reshape means if layer is deterministic
        if deterministic: return mu3

        # otherwise, take samples
        shape = (n_bat, n_sam, n_dim)    
        return mu3 + T.exp(ls3) * self.rng.normal(shape)        

class BernoulliSampleLayer(lasagne.layers.Layer):
    def __init__(self, mean,
                 seed=lasagne.random.get_rng().randint(1, 2147462579),
                 **kwargs):
        super(BernoulliSampleLayer, self).__init__(mean, **kwargs)
        self._srng = RandomStreams(seed)

    def seed(self, seed=lasagne.random.get_rng().randint(1, 2147462579)):
        self._srng.seed(seed)

    def get_output_shape_for(self, input_shape):
        return input_shape

    def get_output_for(self, mu, **kwargs):
        return self._srng.binomial(size=mu.shape, p=mu, dtype=mu.dtype)