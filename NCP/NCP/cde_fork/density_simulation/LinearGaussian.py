import scipy.stats as stats
import numpy as np
from .BaseConditionalDensitySimulation import BaseConditionalDensitySimulation
from scipy.stats import norm

class LinearGaussian(BaseConditionalDensitySimulation):
  """
  A simple, Gaussian conditional distribution where

  x = U(-1,1)
  y = N(y | mean = mu_slope*x+mu, scale = std_slope*x+std)

  Args:
    ndim_x: number of dimensions of x
    mu: the intercept of the mean line
    mu_slope: the slope of the mean line
    std: the intercept of the std dev. line
    std_slope: the slope of the std. dev. line
    random_seed: seed for the random_number generator
  """

  def __init__(self, ndim_x=1, mu=0.0, mu_slope=0.005, std=0.01, std_slope=0.002, random_seed=None):
    assert std > 0
    self.random_state = np.random.RandomState(seed=random_seed)
    self.random_seed = random_seed

    self.mu = mu
    self.base_std = std
    self.mu_slope = mu_slope
    self.std_slope = std_slope
    self.ndim_x = ndim_x
    self.ndim_y = 1
    self.ndim = self.ndim_x + self.ndim_y

    # approximate data statistics
    self.y_mean, self.y_std = self._compute_data_statistics()

    self.has_cdf = True
    self.has_pdf = True
    self.can_sample = True

  def pdf(self, X, Y):
    """ Conditional probability density function p(y|x) of the underlying probability model

    Args:
      X: x to be conditioned on - numpy array of shape (n_points, ndim_x)
      Y: y target values for witch the pdf shall be evaluated - numpy array of shape (n_points, ndim_y)

    Returns:
      p(X|Y) conditional density values for the provided X and Y - numpy array of shape (n_points, )
    """
    X, Y = self._handle_input_dimensionality(X, Y)
    mean = self._mean(X)
    p = np.squeeze(stats.norm.pdf((Y-mean)/self._std(X)) / self._std(X))
    # assert p.shape == (X.shape[0],)
    return p

  def cdf(self, X, Y):
    """ Conditional cumulated probability density function P(Y < y | x) of the underlying probability model

       Args:
         X: x to be conditioned on - numpy array of shape (n_points, ndim_x)
         Y: y target values for witch the cdf shall be evaluated - numpy array of shape (n_points, ndim_y)

       Returns:
        P(Y < y | x) cumulated density values for the provided X and Y - numpy array of shape (n_points, )
    """
    X, Y = self._handle_input_dimensionality(X, Y)
    mean = self._mean(X)
    return np.squeeze(stats.norm.cdf((Y-mean)/self._std(X)))

  def simulate_conditional(self, X):
    """ Draws random samples from the conditional distribution

    Args:
      X: x to be conditioned on when drawing a sample from y ~ p(y|x) - numpy array of shape (n_samples, ndim_x)

    Returns:
      Conditional random samples y drawn from p(y|x) - numpy array of shape (n_samples, ndim_y)
    """
    X = self._handle_input_dimensionality(X)

    n_samples = X.shape[0]
    Y = self._mean(X) + self._std(X) * self.random_state.normal(size=n_samples)
    X, Y = X.reshape((n_samples, self.ndim_x)), Y.reshape((n_samples, self.ndim_y))
    return X, Y

  def simulate(self, n_samples=1000):
    """ Draws random samples from the joint distribution p(x,y)
    Args:
      n_samples: (int) number of samples to be drawn from the joint distribution

    Returns:
      (X,Y) - random samples drawn from p(x,y) - numpy arrays of shape (n_samples, ndim_x) and (n_samples, ndim_y)
    """
    assert n_samples > 0
    X = self.random_state.uniform(-1,1, size=(n_samples, self.ndim_x))
    Y = self._mean(X) + self._std(X) * self.random_state.normal(size=n_samples)
    X, Y = X.reshape((n_samples, self.ndim_x)), Y.reshape((n_samples, self.ndim_y))
    return X, Y

  def mean_(self, x_cond, n_samples=None):
    """ Conditional mean of the distribution
    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

    Returns:
      Means E[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y)
    """
    assert x_cond.ndim == 2 and x_cond.shape[1] == self.ndim_x
    x_cond = self._handle_input_dimensionality(x_cond)
    return self._mean(x_cond)

  def covariance(self, x_cond, n_samples=None):
    """ Covariance of the distribution conditioned on x_cond

      Args:
        x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)

      Returns:
        Covariances Cov[y|x] corresponding to x_cond - numpy array of shape (n_values, ndim_y, ndim_y)
    """
    assert x_cond.ndim == 2 and x_cond.shape[1] == self.ndim_x
    x_cond = self._handle_input_dimensionality(x_cond)
    covs = self._std(x_cond)**2
    return covs.reshape((covs.shape[0],self.ndim_y, self.ndim_y))

  def value_at_risk(self, x_cond, alpha=0.01, **kwargs):
    """ Computes the Value-at-Risk (VaR) of the fitted distribution. Only if ndim_y = 1

    Args:
      x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)
      alpha: quantile percentage of the distribution

    Returns:
       VaR values for each x to condition on - numpy array of shape (n_values)
    """
    assert self.ndim_y == 1, "Value at Risk can only be computed when ndim_y = 1"
    assert x_cond.ndim == 2

    VaR = norm.ppf(alpha, loc=self._mean(x_cond), scale=self._std(x_cond))[:,0]
    assert VaR.shape == (x_cond.shape[0],)
    return VaR

  def conditional_value_at_risk(self, x_cond, alpha=0.01, **kwargs):
    """ Computes the Conditional Value-at-Risk (CVaR) / Expected Shortfall of the fitted distribution. Only if ndim_y = 1

       Args:
         x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)
         alpha: quantile percentage of the distribution
         n_samples: number of samples for monte carlo model_fitting

       Returns:
         CVaR values for each x to condition on - numpy array of shape (n_values)
       """
    assert self.ndim_y == 1, "Value at Risk can only be computed when ndim_y = 1"
    x_cond = self._handle_input_dimensionality(x_cond)
    assert x_cond.ndim == 2

    mean = self._mean(x_cond)
    sigma = self._std(x_cond)
    CVaR = (mean - sigma * (1/alpha) * norm.pdf(norm.ppf(alpha)))[:,0]
    assert CVaR.shape == (x_cond.shape[0],)
    return CVaR

  def tail_risk_measures(self, x_cond, alpha=0.01, n_samples=10 ** 7):
    """ Computes the Value-at-Risk (VaR) and Conditional Value-at-Risk (CVaR)

        Args:
          x_cond: different x values to condition on - numpy array of shape (n_values, ndim_x)
          alpha: quantile percentage of the distribution
          n_samples: number of samples for monte carlo model_fitting

        Returns:
          - VaR values for each x to condition on - numpy array of shape (n_values)
          - CVaR values for each x to condition on - numpy array of shape (n_values)
        """
    assert self.ndim_y == 1, "Value at Risk can only be computed when ndim_y = 1"
    assert x_cond.ndim == 2

    VaRs = self.value_at_risk(x_cond, alpha=alpha, n_samples=n_samples)
    CVaRs = self.conditional_value_at_risk(x_cond, alpha=alpha, n_samples=n_samples)
    return VaRs, CVaRs

  def _mean(self, X):
    return np.expand_dims(self.mu + np.mean(self.mu_slope * X), axis=-1)

  def _std(self, X):
    return np.expand_dims(self.base_std + np.mean(self.std_slope * X), axis=-1)

  def __str__(self):
    return "\nProbabilistic model type: {}\n base_std: {}\n n_dim_x: {}\n n_dim_y: {}\n".format(self.__class__.__name__, self.base_std, self.ndim_x,
                                                                                                self.ndim_y)

  def __unicode__(self):
    return self.__str__()