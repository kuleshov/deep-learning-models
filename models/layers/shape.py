import numpy as np
import theano.tensor as T
import lasagne

# ----------------------------------------------------------------------------

class RepeatLayer(lasagne.layers.Layer):
    def __init__(self, l_in, n_ax, n_rep, rng=None, **kwargs):
        self.n_ax  = n_ax
        self.n_rep = n_rep
        super(RepeatLayer, self).__init__(l_in, **kwargs)

    def get_output_shape_for(self, input_shapes):
        return tuple(np.insert(input_shapes, self.n_ax, self.n_rep))

    def get_output_for(self, inputs, deterministic=False, **kwargs):
        return T.shape_padaxis(inputs, axis=self.n_ax).repeat(self.n_rep, self.n_ax)