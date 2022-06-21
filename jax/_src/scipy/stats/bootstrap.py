# Copyright 2022 Google LLC
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

from typing import NamedTuple

import scipy.stats
import textwrap

import jax
from jax import random, vmap
import jax.numpy as jnp
from jax._src.scipy.special import ndtr, ndtri
from jax._src.random import _check_prng_key
from jax._src.numpy.util import _wraps

_replace_random_state_by_key_doc = textwrap.dedent("""
Does not support the Scipy argument ``random_state=None``.
Instead we add an argument ``key`` which is a :class:`jax.random.PRNGKey` random key.
"""
)
_replace_random_state_by_key_no_batch_jnp_statistic_doc = (
  _replace_random_state_by_key_doc
  + "\nDoes not support the Scipy argument ``batch``."
    "\nIn addition, statistic is suggested to implement in JAX. Numpy implementation may lead to error."
)


class ConfidenceInterval(NamedTuple):
  low: jnp.ndarray
  high: jnp.ndarray


class BootstrapResult(NamedTuple):
  confidence_interval: ConfidenceInterval
  standard_error: jnp.ndarray


def _bootstrap_resample_and_compute_statistic(sample, statistic, n_resamples, key):
  """
    differences against scipy's _bootstrap_resample:
    1. last arg `key` is jax.random.PRNGKey (v.s. int `random_state`)
    2. `batch` arg is not needed because lax.scan auto parallelize the resampling
    3. to save memory, it computes statistic right after resampling
  """
  n = sample[0].shape[-1]

  @vmap
  def _resample_and_compute_once(input_key):
    idxs = random.randint(input_key, shape=(n,), minval=0, maxval=n)
    # `sample` is a tuple of sample sets, we need to apply same indexing on each sample set
    resample = jax.tree_map(lambda data: data[..., idxs], sample)
    return statistic(*resample, axis=-1)

  theta_hat_b = _resample_and_compute_once(jax.random.split(key, n_resamples))
  # if statistics return a tuple of size n_tuple, then reshape it into (n_tuple, n_resamples)
  #return jnp.stack(theta_hat_b)
  return theta_hat_b.T


def _jackknife_resample_and_compute_statistic(sample, statistic):
  """
    differences against scipy's _jackknife_resample:
    1. `batch` arg is not needed because lax.scan auto parallelize the resampling
    2. to save memory, it computes statistic right after resampling
    """
  n = sample.shape[-1]
  # these assume arg `sample` has > 1 entries
  idxs = jnp.arange(n - 1)
  miss_first_sample = sample[1:]
  miss_last_sample = sample[:-1]

  @vmap
  def _jackknife_resample_and_compute(idx):
    resample = jnp.where(idxs >= idx, miss_first_sample, miss_last_sample)
    return statistic(resample, axis=-1)

  # TODO: check if it can handle `statistic` that return multiple scalars
  theta_hat_i = _jackknife_resample_and_compute(jnp.arange(n))
  return theta_hat_i.T


def _bca_interval(data, statistic, axis, alpha, theta_hat_b):
  # bca interval only works on 1-sample statistic
  sample = data[0]

  # calculate z0_hat
  resample_n = theta_hat_b.shape[-1]
  theta_hat = statistic(sample, axis=axis)[..., None]
  percentile = (theta_hat_b < theta_hat).sum(axis=-1) / resample_n
  z0_hat = ndtri(percentile)

  # calculate a_hat
  theta_hat_i = _jackknife_resample_and_compute_statistic(sample, statistic)
  theta_hat_dot = theta_hat_i.mean(axis=-1, keepdims=True)
  num = ((theta_hat_dot - theta_hat_i) ** 3).sum(axis=-1)
  den = 6 * ((theta_hat_dot - theta_hat_i) ** 2).sum(axis=-1) ** (3 / 2)
  a_hat = num / den

  # calculate alpha_1, alpha_2
  z_alpha = ndtri(alpha)
  z_1alpha = -z_alpha
  num1 = z0_hat + z_alpha
  alpha_1 = ndtr(z0_hat + num1 / (1 - a_hat * num1))
  num2 = z0_hat + z_1alpha
  alpha_2 = ndtr(z0_hat + num2 / (1 - a_hat * num2))
  return alpha_1, alpha_2


def _percentile_along_axis(theta_hat_b, alpha):
  shape = theta_hat_b.shape[:-1]
  alpha = jnp.broadcast_to(alpha, shape)
  vmap_percentile = jnp.percentile
  for i in range(theta_hat_b.ndim - 1):
    vmap_percentile = vmap(vmap_percentile)
  percentiles = vmap_percentile(theta_hat_b, alpha)
  return percentiles[()]


def bootstrap_iv(key, data, statistic, vectorized, paired, axis, confidence_level, n_resamples, method):
  # TODO: add input validations
  # TODO: handle the case for paired sample
  # TODO: handle the case for vectorized with arb axis
  # TODO: handle the case when axis = 0 is used for multi dim data
  # e.g. data = (np.random.randn(3, 10, 100), ); bootstrap(data, statistic = np.std, axis = 0, confidence_level = 0.95)
  # how axis argument is used
  # check alpha is jax array type


  if vectorized not in (True, False):
    raise ValueError("`vectorized` must be `True` or `False`.")

  n_samples = 0
  try:
    n_samples = len(data)
  except TypeError:
    raise ValueError("`data` must be a sequence of samples.")

  if n_samples == 0:
    raise ValueError("`data` must contain at least one sample.")

  # enforce data to be jax array type
  # else will get TracerArrayConversionError in jax.tree_map(lambda data: data[..., idxs], sample)
  # coz of indexing a numpy array using jax array
  data_iv = jax.tree_map(lambda data: jnp.asarray(data), data)

  if paired not in (True, False):
    raise ValueError(f"`paired` must be `True` or `False`.")

  methods = ('percentile', 'basic', 'bca')
  method = method.lower()
  if method not in methods:
    raise ValueError(f"`method` must be in {methods}")

  if not paired and n_samples > 1 and method == 'bca':
    raise ValueError("`method = 'BCa' is only available for one-sample statistics")

  key, _ = _check_prng_key(key)

  return (key, data_iv, statistic, vectorized, paired, axis,
          confidence_level, n_resamples, method)


@_wraps(
  scipy.stats.bootstrap,
  lax_description=_replace_random_state_by_key_no_batch_jnp_statistic_doc,
  skip_params=("batch",),
)
def bootstrap(
  key,
  data,
  statistic,
  *,
  vectorized=True,
  paired=False,
  axis=0,
  confidence_level=0.95,
  n_resamples=9999,
  method="BCa"
):
  """
    differences against scipy's bootstrap:
    1. arg `batch` is removed
    2. arg `vectorized` is removed (we should assume its always vectorized??)

    should test the following in unit test:
    1. handle `statistic` function that accept more than 1 set of samples
        (e.g. statistic(data1, data2, data3))
    2. handle `statistic` function that return > 1 scalars
    3. handle paired-sample correctly
    4. handle the case when sample in data are multi-dimensional (e.g. (n_trials, 100))
        (e.g. see scipy example: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html)

    Issue tracker:
    1. numerical difference between numpy and scipy
    2. speed is slower for JAX bootstrap
    3. some numpy statistic function doesn't work (e.g. np.std, scipy.stats.ttest_rel, scipy.pearsonr)
    4. should it support multi-output statistic (depend on whether scipy version supports it)
  """
  args = bootstrap_iv(key, data, statistic, vectorized, paired, axis,
                      confidence_level, n_resamples, method)
  key, data, statistic, vectorized, paired, axis = args[:6]
  confidence_level, n_resamples, method = args[6:]

  theta_hat_b = _bootstrap_resample_and_compute_statistic(
    data, statistic, n_resamples, key
  )

  alpha = jnp.array((1 - confidence_level) / 2)
  if method == "BCa":
    interval = _bca_interval(
      data, statistic, axis=-1, alpha=alpha, theta_hat_b=theta_hat_b
    )
    percentile_fun = _percentile_along_axis
  else:
    interval = alpha, 1 - alpha

    def percentile_fun(a, q):
      return jnp.percentile(a=a, q=q, axis=-1)

  ci_l = percentile_fun(theta_hat_b, interval[0] * 100)
  ci_u = percentile_fun(theta_hat_b, interval[1] * 100)
  if method == "basic":
    theta_hat = statistic(*data, axis=-1)
    ci_l = ci_u = 2 * theta_hat - ci_u, 2 * theta_hat - ci_l

  return BootstrapResult(
    confidence_interval=ConfidenceInterval(ci_l, ci_u),
    standard_error=jnp.std(theta_hat_b, ddof=1, axis=-1),
  )
