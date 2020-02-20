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
Model-predictive non-linear control example.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from jax import lax, grad, jacfwd, jacobian, vmap
import jax.numpy as np
import jax.ops as jo


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


LOOP_VIA_SCAN = False


def fori_loop(lo, hi, loop, init):
  if LOOP_VIA_SCAN:
    return scan_fori_loop(lo, hi, loop, init)
  else:
    return lax.fori_loop(lo, hi, loop, init)


def scan_fori_loop(lo, hi, loop, init):
  def scan_f(x, t):
    return loop(t, x), ()
  x, _ = lax.scan(scan_f, init, np.arange(lo, hi))
  return x


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

  return fori_loop(0, T, loop, X)


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

  def rev_loop(t_, state):
    t = T - t_ - 1
    spec, P, p, K, k = state

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

  _, P, p, K, k = fori_loop(
      0, T, rev_loop,
      (spec, spec.Q[T + 1], spec.q[T + 1], K, k))

  return K, k


def lqr_predict(spec, x0):
  T, control_dim, _ = spec.R.shape
  _, state_dim, _ = spec.Q.shape

  K, k = lqr_solve(spec)

  def fwd_loop(t, state):
    spec, X, U = state
    A, B = spec.A[t], spec.B[t]
    u = mv(K[t], X[t]) + k[t]
    x = mv(A, X[t]) + mv(B, u)
    X = jo.index_update(X, jo.index[t + 1], x)
    U = jo.index_update(U, jo.index[t], u)
    return spec, X, U

  U = np.zeros((T, control_dim))
  X = np.zeros((T + 1, state_dim))
  X = jo.index_update(X, jo.index[0], x0)
  _, X, U = fori_loop(0, T, fwd_loop, (spec, X, U))
  return X, U


def ilqr(iterations, p, x0, U):
  assert x0.ndim == 1 and x0.shape[0] == p.state_dim, x0.shape
  assert U.ndim > 0 and U.shape[0] == p.horizon, (U.shape, p.horizon)

  lqr_approx = make_lqr_approx(p)

  def loop(_, state):
    X, U = state
    p_lqr = lqr_approx(X, U)
    dX, dU = lqr_predict(p_lqr, np.zeros_like(x0))
    U = U + dU
    X = trajectory(p.dynamics, U, X[0] + dX[0])
    return X, U

  X = trajectory(p.dynamics, U, x0)
  return fori_loop(0, iterations, loop, (X, U))


def mpc_predict(solver, p, x0, U):
  assert x0.ndim == 1 and x0.shape[0] == p.state_dim
  T = p.horizon

  def zero_padded_controls_window(U, t):
    U_pad = np.vstack((U, np.zeros(U.shape)))
    return lax.dynamic_slice_in_dim(U_pad, t, T, axis=0)

  def loop(t, state):
    cost = lambda t_, x, u: p.cost(t + t_, x, u)
    dyns = lambda t_, x, u: p.dynamics(t + t_, x, u)

    X, U = state
    p_ = ControlSpec(cost, dyns, T, p.state_dim, p.control_dim)
    xt = X[t]
    U_rem = zero_padded_controls_window(U, t)
    _, U_ = solver(p_, xt, U_rem)
    ut = U_[0]
    x = p.dynamics(t, xt, ut)
    X = jo.index_update(X, jo.index[t + 1], x)
    U = jo.index_update(U, jo.index[t], ut)
    return X, U

  X = np.zeros((T + 1, p.state_dim))
  X = jo.index_update(X, jo.index[0], x0)
  return fori_loop(0, T, loop, (X, U))
