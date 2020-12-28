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

"""Support for the k-means algorithm."""

import functools

import jax
from jax import jit
from jax import random
import jax.numpy as jnp
import numpy as np


def safe_mean(counts, sums):
  """A convenience function used in kmeans_single_iter."""
  return jnp.where(counts > 0, sums/counts, np.zeros(counts.shape))


@functools.partial(jit, static_argnums=(2,))
def py_vq(obs, code_book, square_root=False):
  """Jittable version of the vq algorithm from scipy.

  The original cython implementation can is located here:
  https://github.com/scipy/scipy/blob/v1.5.4/scipy/cluster/vq.py
  The algorithm computes the square of Euclidean distance between each
  observation and every frame in the code_book. Optionally it also computes
  the Euclidean distance.

  Args:
    obs: expects a rank 2 array. Each row is one observation.
    code_book : a code book to use; expects a rank 2 array. The array should
      have same number of features (i.e., columns) as obs.
    square_root: if set to true, then square root is computed inside of py_vq.

  Returns:
    A pair consisting of two jax arrays: codes[i] is the cluster index of
      the i-th obversation and min_dist[i] is the square of the distance
      (or just the distance if square_root set to True) between the i-th
      observation and its corresponding code.
  """

  assert obs.shape[1] == code_book.shape[1]
  assert len(code_book.shape) == 2
  assert len(obs.shape) == 2

  shape_final = (obs.shape[0], code_book.shape[0], obs.shape[1])
  obs = obs[:, None, :]
  obs = jnp.broadcast_to(obs, shape_final)
  code_book = jnp.broadcast_to(code_book, shape_final)

  if square_root:
    dist = jnp.linalg.norm(obs - code_book, axis=2)
  else:
    x = obs - code_book
    dist = jnp.sum(jnp.real(x * jnp.conj(x)), axis=2)
  codes = dist.argmin(axis=1)
  min_dist = jnp.amin(dist, axis=1)
  return codes, min_dist


@functools.partial(jit, static_argnums=(2,))
def kmeans_single_iter(obs, code_book, square_root=False):
  """A single step of k-means.

  In the original scipy implementation
  https://github.com/scipy/scipy/blob/v1.5.4/scipy/cluster/vq.py#L267
  kmeans_single_iter functions performs a full k-means loop. This would be
  harder to jit, hence in the implementation below we perform a single step
  of the loop and the rest is done in kmeans function which is not jitted.
  Args:
    obs: expects a rank 2 array. Each row is one observation.
    code_book : a code book to use; expects a rank 2 array. The array should
      have same number of features (e.g., columns) as obs.
    square_root: if set to true, then square root is computed inside of py_vq.
  Returns:
    A pair consisting of the new_code_book and an array min_dist of square
      distances (or just distances if square_root set to True) between
      the i-th observation and its corresponding code in code_book.
  """
  # compute membership and distances between obs and code_book
  obs_codes, min_dist = py_vq(obs, code_book, square_root)
  # First we calculate the sums per centroid...
  sums = jax.ops.index_add(jnp.zeros(code_book.shape), obs_codes, obs)
  # and then manually compute the mean using the safe_mean function.
  counts = jnp.bincount(obs_codes, length=code_book.shape[0])
  # Using safe_mean diverges from the original scipy implementation -
  # the original implementation would drop zero rows, but then the function
  # would not longer be jittable. The original implementation can be done
  # in the following (unjittable) way:
  #   pos = np.where(counts > 0)[0]
  #   code_book = code_book[pos,:]
  #   sums = sums[pos,:]
  #   counts = counts[pos]
  #   new_code_book = sums/counts
  new_code_book = safe_mean(counts[:, None], sums)
  return obs_codes, new_code_book, min_dist


def kmeans(obs, k, key, iterations=20, thresh=1e-5,
           square_root=False, absolute=True):
  """Performs k-means loop.

  The k-means algorithm adjusts the classification of the observations
  into clusters and updates the cluster centroids until the position of
  the centroids is stable over successive iterations. In this
  implementation of the algorithm, the stability of the centroids is
  determined by comparing the value of the change in the average
  Euclidean distance between the observations and their corresponding
  centroids against a threshold.

  Args:
    obs: expects a rank 2 array. Each row is one observation.
    k: the number of centroids to generate.
    key: a random key used for an initialization of the codebook.
    iterations: the number of times to run k-means until convergence.
    thresh: terminates the k-means algorithm if the change in
      distortion since the last k-means iteration is less than
      or equal to threshold.
    square_root: if set to true, then square root is computed inside of py_vq.
      Otherwise the square root is computed here. In practice this controls
      whether the square root is computed in python or in jax. The option is
      added here, because for some synthetic datasets performing the square
      root operation outside of jax improved numerical stability of kmeans
      alogithm.
    absolute: whether to compute absolute or relative distortion. The
      original scipy implementation computer a relative difference
      https://github.com/scipy/scipy/blob/v1.5.4/scipy/cluster/vq.py#L311.
  Returns:
    In every iteration until iter we construct a new code book
    and returns the best of them in in terms of the average distance to
    observations along with the best_dist scalar.
  """

  best_dist = np.inf
  keys = jax.random.split(key, iterations)
  for i in range(iterations):
    idx = random.choice(keys[i], obs.shape[0], shape=(k,), replace=False)
    # Randomly choose the initial code book and...
    book = obs[idx, ::-1]
    # set the difference to infinity. The difference will be used to
    # control when the k-means loop should be finished.
    diff = np.inf
    cur_avg_dist = np.inf
    # Beginning of the k-means loop.
    while diff > thresh:
      _, book, dist = kmeans_single_iter(obs, book, square_root)
      if not square_root:
        dist = np.sqrt(dist)
      prev_avg_dist, cur_avg_dist = cur_avg_dist, dist.mean(axis=-1)
      if absolute:
        diff = np.absolute(prev_avg_dist - cur_avg_dist)
      else:
        diff = prev_avg_dist - cur_avg_dist
    if cur_avg_dist < best_dist:
      best_book = book
      best_dist = cur_avg_dist
  return best_book, best_dist
