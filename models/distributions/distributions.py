import math
import theano.tensor as T

# ----------------------------------------------------------------------------
# this is all taken from the parmesan lib

c = - 0.5 * math.log(2*math.pi)

def log_bernoulli(x, p, eps=1e-5):
    """
    Compute log pdf of a Bernoulli distribution with success probability p, at values x.
        .. math:: \log p(x; p) = \log \mathcal{B}(x; p)
    Parameters
    ----------
    x : Theano tensor
        Values at which to evaluate pdf.
    p : Theano tensor
        Success probability :math:`p(x=1)`, which is also the mean of the Bernoulli distribution.
    eps : float
        Small number used to avoid NaNs by clipping p in range [eps;1-eps].
    Returns
    -------
    Theano tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.
    """
    p = T.clip(p, eps, 1.0 - eps)
    return -T.nnet.binary_crossentropy(p, x)

def log_normal(x, mean, std, eps=1e-5):
    """
    Compute log pdf of a Gaussian distribution with diagonal covariance, at values x.
    Variance is parameterized as standard deviation.
        .. math:: \log p(x) = \log \mathcal{N}(x; \mu, \sigma^2I)
    
    Parameters
    ----------
    x : Theano tensor
        Values at which to evaluate pdf.
    mean : Theano tensor
        Mean of the Gaussian distribution.
    std : Theano tensor
        Standard deviation of the diagonal covariance Gaussian.
    eps : float
        Small number added to standard deviation to avoid NaNs.
    Returns
    -------
    Theano tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.
    See also
    --------
    log_normal1 : using variance parameterization
    log_normal2 : using log variance parameterization
    """
    std += eps
    return c - T.log(T.abs_(std)) - (x - mean)**2 / (2 * std**2)    

def log_normal2(x, mean, log_var, eps=1e-5):
    """
    Compute log pdf of a Gaussian distribution with diagonal covariance, at values x.
    Variance is parameterized as log variance rather than standard deviation, which ensures :math:`\sigma > 0`.
        .. math:: \log p(x) = \log \mathcal{N}(x; \mu, \sigma^2I)
    
    Parameters
    ----------
    x : Theano tensor
        Values at which to evaluate pdf.
    mean : Theano tensor
        Mean of the Gaussian distribution.
    log_var : Theano tensor
        Log variance of the diagonal covariance Gaussian.
    eps : float
        Small number added to denominator to avoid NaNs.
    Returns
    -------
    Theano tensor
        Element-wise log probability, this has to be summed for multi-variate distributions.
    See also
    --------
    log_normal : using standard deviation parameterization
    log_normal1 : using variance parameterization
    """
    lv_clip = T.clip(log_var, -10., 10.)
    return c - log_var/2 - (x - mean)**2 / (2 * T.exp(lv_clip) + eps)    