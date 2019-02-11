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
from jax import grad
from jax import jit
from jax.config import config
import jax.numpy as np
import jax.random as random
import jax.scipy as scipy
import matplotlib.pyplot as plt


FLAGS = flags.FLAGS


def main(unused_argv):

  numpts = 25
  key = random.PRNGKey(0)
  eye = np.eye(numpts)

  def sqdist(x1, x2):
    return (-2. * np.dot(x1, x2.T) + np.sum(x2**2, axis=1) +
            np.sum(x1**2, axis=1)[:, None])

  def cov(params, x1, x2):
    x1 = x1/np.exp(params[2])
    x2 = x2/np.exp(params[2])
    return np.exp(params[0]) * np.exp(-sqdist(x1, x2)/(2. * np.exp(params[1])))

  def marginal_likelihood(params, x, y):
    train_cov = cov(params, x, x) + eye * 1e-6
    chol = np.linalg.cholesky(train_cov + eye * 1e-4).T
    inv_chol = scipy.linalg.solve_triangular(chol, eye, lower=True)
    inv_train_cov = np.dot(inv_chol.T, inv_chol)
    ml = np.sum(
        -0.5 * np.dot(y.T, np.dot(inv_train_cov, y)) -
        0.5 * np.sum(2.0 * np.log(np.dot(inv_chol * eye, np.ones(
            (numpts, 1))))) - (numpts / 2.) * np.log(2. * 3.1415))
    return ml
  grad_fun = jit(grad(marginal_likelihood))

  def predict(params, x, y, xtest):
    train_cov = cov(params, x, x) + eye * 1e-6
    chol = np.linalg.cholesky(train_cov + eye * 1e-4)
    inv_chol = scipy.linalg.solve_triangular(chol, eye, lower=True)
    inv_train_cov = np.dot(inv_chol.T, inv_chol)
    cross_cov = cov(params, x, xtest)
    mu = np.dot(cross_cov.T, np.dot(inv_train_cov, y))
    var = (cov(params, xtest, xtest) -
           np.dot(cross_cov.T, np.dot(inv_train_cov, cross_cov)))
    return mu, var

  # Covariance hyperparameters to be learned
  params = [np.zeros((1, 1)),  # Amplitude
            np.zeros((1, 1)),  # Bandwidth
            np.zeros((1, 1))]  # Length-scale
  momentums = [p * 0. for p in params]
  scales = [p * 0. + 1. for p in params]

  lr = 0.01  # Learning rate
  def train_step(params, momentums, scales, x, y):
    grads = grad_fun(params, x, y)
    for i in range(len(params)):
      momentums[i] = 0.9 * momentums[i] + 0.1 * grads[i][0]
      scales[i] = 0.9 * scales[i] + 0.1 * grads[i][0]**2
      params[i] -= lr * momentums[i]/np.sqrt(scales[i] + 1e-5)
    return params, momentums, scales

  # Create a really simple toy 1D function
  y_fun = lambda x: np.sin(x) + 0.01 * random.normal(key, shape=(x.shape[0], 1))
  x = np.linspace(1., 4., numpts)[:, None]
  y = y_fun(x)
  xtest = np.linspace(0, 5., 200)[:, None]
  ytest = y_fun(xtest)

  for i in range(1000):
    params, momentums, scales = train_step(params, momentums, scales, x, y)
    if i % 50 == 0:
      ml = marginal_likelihood(params, x, y)
      print("Step: %d, neg marginal likelihood: %f" % (i, ml))

  print([i.copy() for i in params])
  mu, var = predict(params, x, y, xtest)
  std = np.sqrt(np.diag(var))
  plt.plot(x, y, "k.")
  plt.plot(xtest, mu)
  plt.fill_between(xtest.flatten(),
                   mu.flatten() - std * 2, mu.flatten() + std * 2)


if __name__ == "__main__":
  config.config_with_absl()
  app.run(main)
