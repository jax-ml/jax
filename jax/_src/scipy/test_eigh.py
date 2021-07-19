"""Test for eigh.py."""
from jax._src.scipy import eigh

import jax.numpy as jnp
from jax import lax
import numpy as np
import pytest

Ns = [16, 256]
precisions = [lax.Precision.HIGHEST, ]

@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("precision", precisions)
def test_eigh(N, precision):
  np.random.seed(10)
  H = np.random.randn(N, N).astype(np.float32)
  H = jnp.array(0.5 * (H + H.conj().T))
  ev_exp, eV_exp = jnp.linalg.eigh(H)
  evs, V = eigh.eigh(H)
  HV = jnp.dot(H, V, precision=precision)
  vV = evs * V
  eps = np.finfo(H.dtype).eps
  atol = jnp.linalg.norm(H) * eps
  np.testing.assert_allclose(ev_exp, jnp.sort(evs), atol=20 * atol)
  np.testing.assert_allclose(HV, vV, atol=20 * atol)


@pytest.mark.parametrize("N", Ns)
@pytest.mark.parametrize("precision", precisions)
def test_svd(N, precision):
  np.random.seed(10)
  H = np.random.randn(N, N).astype(np.float32)
  S_expected = np.linalg.svd(H, compute_uv=False)
  U, S, V = eigh.svd(H, precision=precision)
  recon = jnp.dot((U * S), V, precision=precision)
  eps = np.finfo(H.dtype).eps
  eps = eps * jnp.linalg.norm(H) * 10
  np.testing.assert_allclose(np.sort(S), np.sort(S_expected), atol=eps)
  np.testing.assert_allclose(H, recon, atol=eps)

  # U is unitary.
  u_unitary_delta = jnp.dot(U.conj().T, U, precision=lax.Precision.HIGHEST)
  u_eye = np.eye(u_unitary_delta.shape[0])
  np.testing.assert_allclose(u_unitary_delta, u_eye, atol=eps)

  # V is unitary.
  v_unitary_delta = jnp.dot(V.conj().T, V, precision=lax.Precision.HIGHEST)
  v_eye = np.eye(v_unitary_delta.shape[0])
  np.testing.assert_allclose(v_unitary_delta, v_eye, atol=eps)
