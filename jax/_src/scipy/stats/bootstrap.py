# Copyright 2020 Google LLC
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

import scipy.stats as osp_stats

import jax
from jax import lax, random, vmap
import jax.numpy as jnp
from jax._src.scipy.special import ndtr as ndtr, ndtri as ndtri
from jax._src.numpy.util import _wraps


class ConfidenceInterval(NamedTuple):
    ci_l: jnp.ndarray
    ci_u: jnp.ndarray


class BootstrapResult(NamedTuple):
    confidence_interval: ConfidenceInterval
    standard_error: jnp.ndarray


def _bootstrap_resample_and_compute_statistic(sample, statistic, n_resamples, rng):
    """
    differences against scipy's _bootstrap_resample:
    1. last arg `rng` is jax.random.PRNGKey (v.s. int `random_state`)
    2. `batch` arg is not needed because lax.scan auto parallelize the resampling
    3. to save memory, it computes statistic right after resampling
    """
    n = sample[0].shape[-1]

    def _resample_and_compute_once(rng, x):
        idxs = random.randint(rng, shape=(n,), minval=0, maxval=n)
        # `sample` is a tuple of sample sets, we need to apply same indexing on each sample set
        resample = jax.tree_map(lambda data: data[..., idxs], sample)
        next_rng = jax.random.split(rng, 1)[0]
        return next_rng, statistic(*resample)

    # xs is dummy simply for the sake of carrying loops
    _, theta_hat_b = lax.scan(_resample_and_compute_once, rng, jnp.ones(n))
    return theta_hat_b


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

    def _jackknife_resample_and_compute(idx, resample):
        resample = jnp.where(idxs >= idx, miss_first_sample, miss_last_sample)
        return idx + 1, statistic(resample)

    # TODO: check if it can handle `statistic` that return multiple scalars
    _, theta_hat_i = lax.scan(_jackknife_resample_and_compute, 0, jnp.ones(n))
    return theta_hat_i


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
    # QUESTION: is it good practice to use vmap here?
    # TODO: may need to handle nan
    # TODO: handle numeric discrepancy against scipy's _percentile_along_axis
    vmap_percentile = vmap(jnp.percentile)  # dispatch first axis of both inputs
    percentiles = vmap_percentile(theta_hat_b, alpha)
    return percentiles[()]


def bootstrap(
    data,
    statistic,
    *,
    paired=False,
    axis=0,
    confidence_level=0.95,
    n_resamples=9999,
    method="BCa",
    random_state=None
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
    """
    # TODO: add input validations
    pass

    # TODO: how to handle random_state = None?
    rng = jax.random.PRNGKey(random_state)
    theta_hat_b = _bootstrap_resample_and_compute_statistic(
        data, statistic, n_resamples, rng
    )

    alpha = (1 - confidence_level) / 2
    if method == "bca":
        interval = _bca_interval(
            data, statistic, axis=-1, alpha=alpha, theta_hat_b=theta_hat_b
        )
        percentile_fun = _percentile_along_axis
    else:
        interval = alpha, 1 - alpha

        def percentile_fun(a, q):
            return jnp.percentile(a=a, q=q, axis=-1)

    ci_l = percentile_fun(theta_hat_b, interval[0].repeat(100))
    ci_u = percentile_fun(theta_hat_b, interval[1].repeat(100))
    if method == "basic":
        theta_hat = statistic(*data, axis=-1)
        ci_l = ci_u = 2 * theta_hat - ci_u, 2 * theta_hat - ci_l

    return BootstrapResult(
        confidence_interval=ConfidenceInterval(ci_l, ci_u),
        standard_error=jnp.std(theta_hat_b, ddof=1, axis=-1),
    )
