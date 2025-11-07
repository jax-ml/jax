# Copyright 2025
"""Tests for ppf (percent point function) implementations in jax.scipy.stats

These tests check basic inversion properties and gradients for the Gamma and
Beta ppf implemented as Newton solvers.
"""
from absl.testing import absltest

import numpy as np

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu
import scipy.stats as osp_stats
from jax.scipy import stats as lsp_stats


class PpfTests(jtu.JaxTestCase):

  def test_gamma_ppf_inversion(self):
    # Use moderate quantiles to avoid extreme tail behavior of the pure-JAX
    # solver which is less accurate in the far tails.
    q = jnp.array([1e-3, 1e-2, 0.1, 0.5, 0.9])
    shapes = [0.5, 1.0, 2.0, 5.0]
    for a in shapes:
      x = lsp_stats.gamma.ppf(q, a)
      q2 = lsp_stats.gamma.cdf(x, a)
      # Allow a looser tolerance for this pure-JAX solver implementation.
      self.assertAllClose(q2, q, atol=1e-3, rtol=1e-3)

  def test_beta_ppf_inversion(self):
    q = jnp.array([1e-3, 1e-2, 0.1, 0.5, 0.9])
    params = [(0.5, 0.5), (1.0, 1.0), (2.0, 5.0), (5.0, 2.0)]
    for a, b in params:
      x = lsp_stats.beta.ppf(q, a, b)
      q2 = lsp_stats.beta.cdf(x, a, b)
      # Allow a looser tolerance for this pure-JAX solver implementation.
      self.assertAllClose(q2, q, atol=1e-3, rtol=1e-3)

  def test_gamma_ppf_derivative_wrt_q(self):
    # For invertible cdf F, d/dq F^{-1}(q) = 1 / f(F^{-1}(q))
    q0 = 0.3
    a = 2.5
    ppf_fun = lambda qq: lsp_stats.gamma.ppf(qq, a)
    dp_dq = jax.grad(ppf_fun)(q0)
    x = ppf_fun(q0)
    expected = 1.0 / lsp_stats.gamma.pdf(x, a)
    # Solver switching (Newton/bisection) can introduce small differences in
    # the autodiff gradients; use a relaxed tolerance here.
    self.assertAllClose(dp_dq, expected, atol=1e-2, rtol=1e-2)

  def test_beta_ppf_derivative_wrt_q(self):
    q0 = 0.4
    a, b = 2.0, 3.0
    ppf_fun = lambda qq: lsp_stats.beta.ppf(qq, a, b)
    dp_dq = jax.grad(ppf_fun)(q0)
    x = ppf_fun(q0)
    expected = 1.0 / lsp_stats.beta.pdf(x, a, b)
    self.assertAllClose(dp_dq, expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
  absltest.main()
