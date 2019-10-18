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
from jax import grad, hessian, jacfwd, jacobian, jacrev, vmap


# Specifies a general finite-horizon, time-varying control problem. Given cost
# function `c`, transition function `f`, and initial state `x0`, the goal is to
# compute:
#
#   argmin(lambda U, U: c(T, X[T]) + sum(c(t, X[t], U[t]) for t in range(T)))
#
# subject to the constraints that `X[0] == x0` and that:
#
#   all(X[t + 1] == f(X[t], U[t]) for t in range(T)) .
#
# The special case in which `c` is quadratic and `f` is linear is the
# linear-quadratic regulator (LQR) problem, and can be specified explicity
# further below.
#
ControlSpec = collections.namedtuple(
    'ControlSpec', 'cost dynamics horizon state_dim control_dim')


# Specifies a finite-horizon, time-varying LQR problem. Notation:
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


def make_lqr_approx(p):
  T = p.horizon

  def approx_timestep(t, x, u):
    M = jacfwd(grad(p.cost, argnums=2), argnums=1)(t, x, u).T
    Q = jacfwd(grad(p.cost, argnums=1), argnums=1)(t, x, u)
    R = jacfwd(grad(p.cost, argnums=2), argnums=2)(t, x, u)
    q, r = grad(p.cost, argnums=(1, 2))(t, x, u)
    A, B = jacobian(p.dynamics, argnums=(1, 2))(t, x, u)
    return Q, q, R, r, M, A, B

  _approx = vmap(approx_timestep)

  def approx(X, U):
    assert X.shape[0] == T + 1 and U.shape[0] == T
    U_pad = np.vstack((U, np.zeros((1,) + U.shape[1:])))
    Q, q, R, r, M, A, B = _approx(np.arange(T + 1), X, U_pad)
    return LqrSpec(Q, q, R[:T], r[:T], M[:T], A[:T], B[:T])

  return approx


def lqr_solve(spec):
  EPS = 1e-7
  T, control_dim, _ = spec.R.shape
  _, state_dim, _ = spec.Q.shape

  K = np.zeros((T, control_dim, state_dim))
  k = np.zeros((T, control_dim))

  def rev_loop(t_, (spec, P, p, K, k)):
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
    return spec, P_, p_, K, k

  _, P, p, K, k = lax.fori_loop(
      0, T, rev_loop,
      (spec, spec.Q[T + 1], spec.q[T + 1], K, k))

  return K, k


def lqr_predict(spec, x0):
  T, control_dim, _ = spec.R.shape
  _, state_dim, _ = spec.Q.shape

  K, k = lqr_solve(spec)

  def fwd_loop(t, (spec, X, U)):
    A, B = spec.A[t], spec.B[t]
    u = mv(K[t], X[t]) + k[t]
    x = mv(A, X[t]) + mv(B, u)
    X = jo.index_update(X, jo.index[t + 1], x)
    U = jo.index_update(U, jo.index[t], u)
    return spec, X, U

  U = np.zeros((T, control_dim))
  X = np.zeros((T + 1, state_dim))
  X = jo.index_update(X, jo.index[0], x0)
  _, X, U = lax.fori_loop(0, T, fwd_loop, (spec, X, U))
  return X, U


def ilqr(p, iterations, x0, U):
  assert x0.ndim == 1 and x0.shape[0] == p.state_dim, x0.shape
  assert U.ndim > 0 and U.shape[0] == p.horizon, (U.shape, p.horizon)

  lqr_approx = make_lqr_approx(p)

  def loop(_, (X, U)):
    p_lqr = lqr_approx(X, U)
    dX, dU = lqr_predict(p_lqr, np.zeros_like(x0))
    U = U + dU
    X = trajectory(p.dynamics, U, X[0] + dX[0])
    return X, U

  X = trajectory(p.dynamics, U, x0)
  return lax.fori_loop(0, iterations, loop, (X, U))
