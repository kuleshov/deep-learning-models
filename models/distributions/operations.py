import numpy as np
import theano.tensor as T

# ----------------------------------------------------------------------------
# (stolen from the parmesan lib)

def log_sum_exp(A, axis=None, sum_op=T.sum):
    """Computes `log(exp(A).sum(axis=axis))` avoiding numerical issues using the log-sum-exp trick.
    Direct calculation of :math:`\log \sum_i \exp A_i` can result in underflow or overflow numerical 
    issues. Big positive values can cause overflow :math:`\exp A_i = \inf`, and big negative values 
    can cause underflow :math:`\exp A_i = 0`. The latter can eventually cause the sum to go to zero 
    and finally resulting in :math:`\log 0 = -\inf`.
    The log-sum-exp trick avoids these issues by using the identity,
    .. math::
        \log \sum_i \exp A_i = \log \sum_i \exp(A_i - c) + c, \text{using},  \\
        c = \max A.
    This avoids overflow, and while underflow can still happen for individual elements it avoids 
    the sum being zero.
     
    Parameters
    ----------
    A : Theano tensor
        Tensor of which we wish to compute the log-sum-exp.
    axis : int, tuple, list, None
        Axis or axes to sum over; None (default) sums over all axes.
    sum_op : function
        Summing function to apply; default is T.sum, but can also be T.mean for log-mean-exp.
    
    Returns
    -------
    Theano tensor
        The log-sum-exp of `A`, dimensions over which is summed will be dropped.
    """
    A_max = T.max(A, axis=axis, keepdims=True)
    B = T.log(sum_op(T.exp(A - A_max), axis=axis, keepdims=True)) + A_max

    if axis is None:
        return B.dimshuffle(())  # collapse to scalar
    else:
        if not hasattr(axis, '__iter__'): axis = [axis]
        return B.dimshuffle([d for d in range(B.ndim) if d not in axis])  # drop summed axes

def log_mean_exp(A, axis=None):
    """Computes `log(exp(A).mean(axis=axis))` avoiding numerical issues using the log-sum-exp trick.
    See also
    --------
    log_sum_exp
    """
    return log_sum_exp(A, axis, sum_op=T.mean)