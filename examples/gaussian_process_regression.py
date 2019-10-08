# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A basic example demonstrating using JAX to do Gaussian process regression.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import app
from absl import flags
from functools import partial
from jax import grad
from jax import jit
from jax import vmap
from jax.config import config
import jax.numpy as np
import jax.random as random
import jax.scipy as scipy
import matplotlib.pyplot as plt

FLAGS = flags.FLAGS

def main(unused_argv):

  numpts = 7
  key = random.PRNGKey(0)
  eye = np.eye(numpts)

  def cov_map(cov_func, xs, xs2=None):
    """Compute a covariance matrix from a covariance function and data points.

    Args:
      cov_func: callable function, maps pairs of data points to scalars.
      xs: array of data points, stacked along the leading dimension.
    Returns:
      A 2d array `a` such that `a[i, j] = cov_func(xs[i], xs[j])`.
    """
    if xs2 is None:
      return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs)
    else:
      return vmap(lambda x: vmap(lambda y: cov_func(x, y))(xs))(xs2).T

  def softplus(x):
    return np.logaddexp(x, 0.)

  # Note, writing out the vectorized form of the identity
  # ||x-y||^2 = <x-y,x-y> = ||x||^2 + ||y||^2 - 2<x,y>
  # for computing squared distances would be more efficient (but less succinct).
  def exp_quadratic(x1, x2):
    return np.exp(-np.sum((x1 - x2)**2))

  def gp(params, x, y, xtest=None, compute_marginal_likelihood=False):
    noise = softplus(params['noise'])
    amp = softplus(params['amplitude'])
    ls = softplus(params['lengthscale'])
    ymean = np.mean(y)
    y = y - ymean
    x = x / ls
    train_cov = amp * cov_map(exp_quadratic, x) + eye * (noise + 1e-6)
    chol = scipy.linalg.cholesky(train_cov, lower=True)
    kinvy = scipy.linalg.solve_triangular(chol.T,
                                          scipy.linalg.solve_triangular(chol, y, lower=True))
    if compute_marginal_likelihood:
      log2pi = np.log(2. * 3.1415)
      ml = np.sum(-0.5 * np.dot(y.T, kinvy) - np.sum(np.log(np.diag(chol))) -
                  (numpts / 2.) * log2pi)
      ml -= np.sum(-0.5 * np.log(2 * 3.1415) - np.log(amp)**2)  # lognormal prior
      return -ml

    if xtest is not None:
      xtest = xtest / ls
    cross_cov = amp * cov_map(exp_quadratic, x, xtest)
    mu = np.dot(cross_cov.T, kinvy) + ymean
    v = scipy.linalg.solve_triangular(chol, cross_cov, lower=True)
    var = (amp * cov_map(exp_quadratic, xtest) - np.dot(v.T, v))
    return mu, var

  marginal_likelihood = partial(gp, compute_marginal_likelihood=True)
  predict = partial(gp, compute_marginal_likelihood=False)
  grad_fun = jit(grad(marginal_likelihood))

  # Covariance hyperparameters to be learned
  params = {
      "amplitude": np.zeros((1, 1)),
      "noise": np.zeros((1, 1)) - 5.,
      "lengthscale": np.zeros((1, 1))
  }
  momentums = dict([(k, p * 0.) for k, p in params.items()])
  scales = dict([(k, p * 0. + 1.) for k, p in params.items()])

  lr = 0.01  # Learning rate

  def train_step(params, momentums, scales, x, y):
    grads = grad_fun(params, x, y)
    for k in (params):
      momentums[k] = 0.9 * momentums[k] + 0.1 * grads[k][0]
      scales[k] = 0.9 * scales[k] + 0.1 * grads[k][0]**2
      params[k] -= lr * momentums[k] / np.sqrt(scales[k] + 1e-5)
    return params, momentums, scales

  # Create a really simple toy 1D function
  y_fun = lambda x: np.sin(x) + 0.1 * random.normal(key, shape=(x.shape[0], 1))
  x = (random.uniform(key, shape=(numpts, 1)) * 4.) + 1
  y = y_fun(x)
  xtest = np.linspace(0, 6., 200)[:, None]
  ytest = y_fun(xtest)

  for i in range(1000):
    params, momentums, scales = train_step(params, momentums, scales, x, y)
    if i % 50 == 0:
      ml = marginal_likelihood(params, x, y)
      print("Step: %d, neg marginal likelihood: %f" % (i, ml))

  print(params)
  mu, var = predict(params, x, y, xtest)
  std = np.sqrt(np.diag(var))
  plt.plot(x, y, "k.")
  plt.plot(xtest, mu)
  plt.fill_between(xtest.flatten(), mu.flatten() - std * 2, mu.flatten() + std * 2)

if __name__ == "__main__":
  config.config_with_absl()
  app.run(main)
