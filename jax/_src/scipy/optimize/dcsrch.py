# Copyright 2020 The JAX Authors.
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
"""Moré-Thuente line search satisfying strong Wolfe conditions."""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

from jax._src import api
from jax._src import lax
from jax._src import numpy as jnp
from jax._src.scipy.optimize.line_search import _LineSearchResults
from jax._src.typing import Array


_dot = partial(jnp.dot, precision=lax.Precision.HIGHEST)


class _DCStepResult(NamedTuple):
  stx: Array
  fx: Array
  dx: Array
  sty: Array
  fy: Array
  dy: Array
  stp: Array
  brackt: Array


def _dcstep(stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax):
  """Safeguarded step computation for the Moré-Thuente line search.

  Dispatches to one of four cases via lax.switch so only the selected
  case executes.
  """
  sgnd = jnp.sign(dp) * jnp.sign(dx)

  case1 = fp > fx
  case2 = (~case1) & (sgnd < 0)
  case3 = (~case1) & (~case2) & (jnp.abs(dp) < jnp.abs(dx))
  index = jnp.where(
    case1, 0, jnp.where(case2, 1, jnp.where(case3, 2, 3))
  ).astype(jnp.int32)

  operands = (stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax)

  def _case1(ops):
    # higher function value, minimum is bracketed
    stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax = ops
    theta = 3.0 * (fx - fp) / (stp - stx + 1e-30) + dx + dp
    s = jnp.maximum(jnp.maximum(jnp.abs(theta), jnp.abs(dx)), jnp.abs(dp))
    disc = (theta / s) ** 2 - (dx / s) * (dp / s)
    gamma = s * jnp.sqrt(jnp.maximum(disc, 0.0))
    gamma = jnp.where(stp < stx, -gamma, gamma)
    p = (gamma - dx) + theta
    q = ((gamma - dx) + gamma) + dp
    r = p / (q + 1e-30)
    stpc = stx + r * (stp - stx)
    denom = (fx - fp) / (stp - stx + 1e-30) + dx
    stpq = stx + (dx / (denom + jnp.sign(denom) * 1e-30)) / 2.0 * (stp - stx)
    stpf = jnp.where(
      jnp.abs(stpc - stx) <= jnp.abs(stpq - stx),
      stpc, stpc + (stpq - stpc) / 2.0,
    )
    stpf = jnp.where(disc < 0, stx + 0.5 * (stp - stx), stpf)
    stpf = jnp.where(jnp.isfinite(stpf), stpf, stx + 0.5 * (stp - stx))
    return stpf, jnp.array(True)

  def _case2(ops):
    # lower function value, derivatives of opposite sign, bracketed
    stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax = ops
    theta = 3.0 * (fx - fp) / (stp - stx + 1e-30) + dx + dp
    s = jnp.maximum(jnp.maximum(jnp.abs(theta), jnp.abs(dx)), jnp.abs(dp))
    disc = (theta / s) ** 2 - (dx / s) * (dp / s)
    gamma = s * jnp.sqrt(jnp.maximum(disc, 0.0))
    gamma = jnp.where(stp > stx, -gamma, gamma)
    p = (gamma - dp) + theta
    q = ((gamma - dp) + gamma) + dx
    r = p / (q + 1e-30)
    stpc = stp + r * (stx - stp)
    stpq = stp + (dp / (dp - dx + 1e-30)) * (stx - stp)
    stpf = jnp.where(jnp.abs(stpc - stp) > jnp.abs(stpq - stp), stpc, stpq)
    stpf = jnp.where(disc < 0, (stx + stp) / 2.0, stpf)
    stpf = jnp.where(jnp.isfinite(stpf), stpf, (stx + stp) / 2.0)
    return stpf, jnp.array(True)

  def _case3(ops):
    # lower function value, same sign derivatives, magnitude decreasing
    stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax = ops
    theta = 3.0 * (fx - fp) / (stp - stx + 1e-30) + dx + dp
    s = jnp.maximum(jnp.maximum(jnp.abs(theta), jnp.abs(dx)), jnp.abs(dp))
    gamma = s * jnp.sqrt(
      jnp.maximum(0.0, (theta / s) ** 2 - (dx / s) * (dp / s)),
    )
    gamma = jnp.where(stp > stx, -gamma, gamma)
    p = (gamma - dp) + theta
    q = (gamma + (dx - dp)) + gamma
    r = p / (q + 1e-30)
    stpc = jnp.where(
      (r < 0) & (gamma != 0),
      stp + r * (stx - stp),
      jnp.where(stp > stx, stpmax, stpmin),
    )
    stpq = stp + (dp / (dp - dx + 1e-30)) * (stx - stp)
    stpf_b = jnp.where(jnp.abs(stpc - stp) < jnp.abs(stpq - stp), stpc, stpq)
    stpf_b = jnp.where(
      stp > stx,
      jnp.minimum(stp + 0.66 * (sty - stp), stpf_b),
      jnp.maximum(stp + 0.66 * (sty - stp), stpf_b),
    )
    stpf_nb = jnp.where(jnp.abs(stpc - stp) > jnp.abs(stpq - stp), stpc, stpq)
    stpf_nb = jnp.clip(stpf_nb, stpmin, stpmax)
    stpf = jnp.where(brackt, stpf_b, stpf_nb)
    return stpf, brackt

  def _case4(ops):
    # lower function value, same sign derivatives, magnitude not decreasing
    stx, fx, dx, sty, fy, dy, stp, fp, dp, brackt, stpmin, stpmax = ops
    theta = 3.0 * (fp - fy) / (sty - stp + 1e-30) + dy + dp
    s = jnp.maximum(jnp.maximum(jnp.abs(theta), jnp.abs(dy)), jnp.abs(dp))
    disc = (theta / s) ** 2 - (dy / s) * (dp / s)
    gamma = s * jnp.sqrt(jnp.maximum(disc, 0.0))
    gamma = jnp.where(stp > sty, -gamma, gamma)
    p = (gamma - dp) + theta
    q = ((gamma - dp) + gamma) + dy
    r = p / (q + 1e-30)
    stpc = stp + r * (sty - stp)
    stpc = jnp.where(disc < 0, stx + 0.5 * (sty - stx), stpc)
    stpf = jnp.where(brackt, stpc, jnp.where(stp > stx, stpmax, stpmin))
    return stpf, brackt

  stpf, new_brackt = lax.switch(
    index, [_case1, _case2, _case3, _case4], operands,
  )

  stpf = jnp.where(jnp.isfinite(stpf), stpf, (stx + sty) / 2.0)

  # update the interval containing the minimizer
  fp_gt_fx = fp > fx
  new_sty = jnp.where(fp_gt_fx, stp, jnp.where(sgnd < 0, stx, sty))
  new_fy = jnp.where(fp_gt_fx, fp, jnp.where(sgnd < 0, fx, fy))
  new_dy = jnp.where(fp_gt_fx, dp, jnp.where(sgnd < 0, dx, dy))
  new_stx = jnp.where(fp_gt_fx, stx, stp)
  new_fx = jnp.where(fp_gt_fx, fx, fp)
  new_dx = jnp.where(fp_gt_fx, dx, dp)

  return _DCStepResult(
    stx=new_stx, fx=new_fx, dx=new_dx,
    sty=new_sty, fy=new_fy, dy=new_dy,
    stp=stpf, brackt=new_brackt,
  )


class _DCSRCHState(NamedTuple):
  """State for the Moré-Thuente line search."""
  stp: Array
  f: Array
  g: Array
  g_full: Array  # full gradient vector at stp
  stx: Array
  fx: Array
  gx: Array
  sty: Array
  fy: Array
  gy: Array
  brackt: Array
  stage: Array
  finit: Array
  ginit: Array
  gtest: Array
  width: Array
  width1: Array
  stmin: Array
  stmax: Array
  done: Array
  failed: Array
  nfev: Array


def _dcsrch_step_modified(state, stp, fval, gval, stpmin, stpmax):
  fm = fval - stp * state.gtest
  fxm = state.fx - state.stx * state.gtest
  fym = state.fy - state.sty * state.gtest
  gm = gval - state.gtest
  gxm = state.gx - state.gtest
  gym = state.gy - state.gtest
  r = _dcstep(
    state.stx, fxm, gxm, state.sty, fym, gym,
    stp, fm, gm, state.brackt, state.stmin, state.stmax,
  )
  return (
    r.stx, r.fx + r.stx * state.gtest, r.dx + state.gtest,
    r.sty, r.fy + r.sty * state.gtest, r.dy + state.gtest,
    r.stp, r.brackt,
  )


def _dcsrch_step_direct(state, stp, fval, gval, stpmin, stpmax):
  r = _dcstep(
    state.stx, state.fx, state.gx, state.sty, state.fy, state.gy,
    stp, fval, gval, state.brackt, state.stmin, state.stmax,
  )
  return (r.stx, r.fx, r.dx, r.sty, r.fy, r.dy, r.stp, r.brackt)


def line_search_dcsrch(f, xk, pk, old_fval=None, old_old_fval=None,
                       gfk=None, c1=1e-4, c2=0.9, maxiter=100,
                       alpha0=None):
  """Moré-Thuente line search satisfying strong Wolfe conditions.

  Args:
    f: scalar objective function.
    xk: current point.
    pk: search direction (must be a descent direction).
    old_fval: f(xk) if already known.
    old_old_fval: unused, for API compatibility.
    gfk: gradient at xk if already known.
    c1: sufficient decrease parameter.
    c2: curvature condition parameter.
    maxiter: maximum number of function evaluations.
    alpha0: initial step size guess.

  Returns:
    Line search results.
  """
  xk, pk = jnp.asarray(xk), jnp.asarray(pk)
  xtol = 1e-14
  stpmin = 1e-15
  stpmax = 1e15

  def full_eval(alpha):
    val, grad = api.value_and_grad(f)(xk + alpha * pk)
    dphi = jnp.real(_dot(grad, pk))
    return val, dphi, grad

  if old_fval is None or gfk is None:
    phi0, dphi0, gfk = full_eval(jnp.array(0.0))
  else:
    phi0 = old_fval
    dphi0 = jnp.real(_dot(gfk, pk))

  if alpha0 is None:
    alpha0 = jnp.array(1.0)

  xtrapu = 4.0
  init_nfev = jnp.where((old_fval is None) | (gfk is None), 1, 0)

  init_state = _DCSRCHState(
    stp=alpha0,
    f=phi0, g=dphi0, g_full=gfk,
    stx=jnp.array(0.0), fx=phi0, gx=dphi0,
    sty=jnp.array(0.0), fy=phi0, gy=dphi0,
    brackt=jnp.array(False),
    stage=jnp.array(1),
    finit=phi0, ginit=dphi0,
    gtest=c1 * dphi0,
    width=jnp.array(stpmax - stpmin),
    width1=jnp.array(2.0 * (stpmax - stpmin)),
    stmin=jnp.array(0.0),
    stmax=alpha0 + xtrapu * alpha0,
    done=jnp.array(False),
    failed=jnp.array(False),
    nfev=init_nfev,
  )

  def cond(state):
    return (~state.done) & (~state.failed) & (state.nfev < maxiter)

  def body(state):
    stp = state.stp
    fval, gval, gfull = full_eval(stp)
    nfev = state.nfev + 1

    ftest = state.finit + stp * state.gtest

    new_stage = jnp.where(
      (state.stage == 1) & (fval <= ftest) & (gval >= 0), 2, state.stage,
    )

    warn1 = state.brackt & ((stp <= state.stmin) | (stp >= state.stmax))
    warn2 = state.brackt & (state.stmax - state.stmin <= xtol * state.stmax)
    warn3 = (stp == stpmax) & (fval <= ftest) & (gval <= state.gtest)
    warn4 = (stp == stpmin) & ((fval > ftest) | (gval >= state.gtest))
    any_warn = warn1 | warn2 | warn3 | warn4

    converged = (fval <= ftest) & (jnp.abs(gval) <= c2 * jnp.abs(state.ginit))

    bad = ~jnp.isfinite(stp)
    done = any_warn | converged | bad
    failed = (any_warn & ~converged) | bad

    use_modified = (new_stage == 1) & (fval <= state.fx) & (fval > ftest)

    def _compute_next_step(args):
      state, stp, fval, gval, use_modified, new_stage = args
      new_stx, new_fx, new_gx, new_sty, new_fy, new_gy, new_stp, new_brackt = lax.cond(
        use_modified,
        lambda s: _dcsrch_step_modified(s, stp, fval, gval, stpmin, stpmax),
        lambda s: _dcsrch_step_direct(s, stp, fval, gval, stpmin, stpmax),
        state,
      )

      new_width = jnp.abs(new_sty - new_stx)
      need_bisect = new_brackt & (new_width >= 0.66 * state.width1)
      new_stp = jnp.where(
        need_bisect, new_stx + 0.5 * (new_sty - new_stx), new_stp,
      )
      new_width1 = jnp.where(new_brackt, state.width, state.width1)

      xtrapl = 1.1
      new_stmin = jnp.where(
        new_brackt,
        jnp.minimum(new_stx, new_sty),
        new_stp + xtrapl * (new_stp - new_stx),
      )
      new_stmax = jnp.where(
        new_brackt,
        jnp.maximum(new_stx, new_sty),
        new_stp + xtrapu * (new_stp - new_stx),
      )

      new_stp = jnp.clip(new_stp, stpmin, stpmax)

      stuck = new_brackt & (
        (new_stp <= new_stmin) | (new_stp >= new_stmax) |
        (new_stmax - new_stmin <= xtol * new_stmax)
      )
      new_stp = jnp.where(stuck, new_stx, new_stp)
      new_stp = jnp.where(jnp.isfinite(new_stp), new_stp, state.stx)

      return (new_stp, new_stx, new_fx, new_gx, new_sty, new_fy, new_gy,
              new_brackt, new_stage, new_width, new_width1, new_stmin, new_stmax)

    def _no_step(args):
      state, stp, fval, gval, use_modified, new_stage = args
      return (state.stp, state.stx, state.fx, state.gx, state.sty, state.fy,
              state.gy, state.brackt, state.stage, state.width, state.width1,
              state.stmin, state.stmax)

    (new_stp, new_stx, new_fx, new_gx, new_sty, new_fy, new_gy,
     new_brackt, upd_stage, new_width, new_width1, new_stmin, new_stmax
    ) = lax.cond(
      ~done,
      _compute_next_step,
      _no_step,
      (state, stp, fval, gval, use_modified, new_stage),
    )

    return _DCSRCHState(
      stp=jnp.where(done, stp, new_stp),
      f=fval, g=gval, g_full=gfull,
      stx=new_stx, fx=new_fx, gx=new_gx,
      sty=new_sty, fy=new_fy, gy=new_gy,
      brackt=new_brackt,
      stage=upd_stage,
      finit=state.finit, ginit=state.ginit, gtest=state.gtest,
      width=new_width, width1=new_width1,
      stmin=new_stmin, stmax=new_stmax,
      done=done, failed=failed, nfev=nfev,
    )

  final = lax.while_loop(cond, body, init_state)

  return _LineSearchResults(
    failed=final.failed,
    nit=final.nfev,
    nfev=final.nfev,
    ngev=final.nfev,
    k=final.nfev,
    a_k=final.stp,
    f_k=final.f,
    g_k=final.g_full,
    status=jnp.where(final.failed, jnp.array(1), jnp.array(0)),
  )
