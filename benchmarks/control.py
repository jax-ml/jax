# Copyright 2019 Google LLC
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
"""
Benchmarks for model-predictive linear control
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import jax
from jax import lax
import jax.numpy as np
import jax.ops as jo


# Finite-horizon, time-varying LQR problem. Notation is:
#
#   cost(t, x, u) = sum(
#       dot(x.T, Q[t], x) + dot(q[t], x) +
#       dot(u.T, R[t], u) + dot(r[t], u) +
#       dot(x.T, M[t], u)
#
#   dynamics(t, x, u) = dot(A[t], x) + dot(B[t], u)
#
LqrSpec = collections.namedtuple('LqrSpec', 'Q q R r M A B')


dot = np.dot
mm = np.matmul


def mv(mat, vec):
  assert mat.ndim == 2
  assert vec.ndim == 1
  return dot(mat, vec)


def trajectory(dynamics, U, x0):
  '''Unrolls `X[t+1] = dynamics(t, X[t], U[t])`, where `X[0] = x0`.'''
  T, _ = U.shape
  d, = x0.shape

  X = np.zeros((T + 1, d))
  X = jo.index_update(X, jo.index[0], x0)

  def loop(t, X):
    x = dynamics(t, X[t], U[t])
    X = jo.index_update(X, jo.index[t + 1], x)
    return X

  return lax.fori_loop(0, T, loop, X)


def lqr_solve(spec):
  EPS = 1e-7
  T, control_dim, _ = spec.R.shape
  _, state_dim, _ = spec.Q.shape

  K = np.zeros((T, control_dim, state_dim))
  k = np.zeros((T, control_dim))

  def rev_loop(t_, (P, p, K, k)):
    t = T - t_ - 1

    Q, q = spec.Q[t], spec.q[t]
    R, r = spec.R[t], spec.r[t]
    M = spec.M[t]
    A, B = spec.A[t], spec.B[t]

    AtP = mm(A.T, P)
    BtP = mm(B.T, P)
    G = R + mm(BtP, B)
    H = mm(BtP, A) + M.T
    h = r + mv(B.T, p)
    K_ = -np.linalg.solve(G + EPS * np.eye(G.shape[0]), H)
    k_ = -np.linalg.solve(G + EPS * np.eye(G.shape[0]), h)
    P_ = Q + mm(AtP, A) + mm(K_.T, H)
    p_ = q + mv(A.T, p) + mv(K_.T, h)

    K = jo.index_update(K, jo.index[t], K_)
    k = jo.index_update(k, jo.index[t], k_)
    return P_, p_, K, k

  P, p, K, k = lax.fori_loop(
      0, T, rev_loop,
      (spec.Q[T + 1], spec.q[T + 1], K, k))

  return K, k


def lqr_predict(spec, x0):
  T, control_dim, _ = spec.R.shape
  _, state_dim, _ = spec.Q.shape

  K, k = lqr_solve(spec)

  def fwd_loop(t, (X, U)):
    A, B = spec.A[t], spec.B[t]
    u = mv(K[t], X[t]) + k[t]
    x = mv(A, X[t]) + mv(B, u)
    X = jo.index_update(X, jo.index[t + 1], x)
    U = jo.index_update(U, jo.index[t], u)
    return X, U

  U = np.zeros((T, control_dim))
  X = np.zeros((T + 1, state_dim))
  X = jo.index_update(X, jo.index[0], x0)
  return lax.fori_loop(0, T, fwd_loop, (X, U))
