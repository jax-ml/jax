# Copyright 2025 The JAX Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Tests for jax.nn.geglu
#
# Reference:
#   Shazeer, N. (2020). "GLU Variants Improve Transformer"
#   (https://arxiv.org/abs/2002.05202)

import pytest
import jax
import jax.numpy as jnp
from jax import grad
from jax.nn import geglu, gelu
jax.config.update("jax_enable_x64", True)




# Basic correctness and shape tests
def test_geglu_basic():
    """Sanity check: shape and value match GELU * gate."""
    x = jnp.array([1.0, 2.0, 3.0])
    gate = jnp.array([0.5, 1.0, 1.5])
    out = geglu(x, gate)

    assert out.shape == x.shape
    assert jnp.allclose(out, gelu(x) * gate, atol=1e-6)



# Approximation flag behavior
@pytest.mark.parametrize("approximate", [True, False])
def test_geglu_approximate_modes(approximate):
    """GEGLU runs for both approximate modes and outputs finite values."""
    x = jnp.linspace(-2, 2, 5)
    gate = jnp.ones_like(x)
    out = geglu(x, gate, approximate=approximate)

    assert jnp.all(jnp.isfinite(out))
    # Outputs shouldn't all be exactly zero
    assert not jnp.allclose(out, 0)


def test_geglu_approximate_consistency():
    """Ensure approximate vs. exact modes produce slightly different outputs."""
    x = jnp.linspace(-3, 3, 7)
    gate = jnp.ones_like(x)
    out_true = geglu(x, gate, approximate=False)
    out_approx = geglu(x, gate, approximate=True)

    assert jnp.any(jnp.abs(out_true - out_approx) > 1e-6)



# Broadcasting and dtype behavior
def test_geglu_broadcasting():
    """Broadcasting support for gate tensor."""
    x = jnp.ones((2, 3))
    gate = jnp.array([0.5, 1.0, 1.5])
    out = geglu(x, gate)

    assert out.shape == x.shape
    assert jnp.allclose(out, gelu(x) * gate, atol=1e-6)


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.float32, jnp.float64])
def test_geglu_dtypes(dtype):
    """Output preserves input dtype."""
    x = jnp.linspace(-2, 2, 5, dtype=dtype)
    gate = jnp.ones_like(x)
    out = geglu(x, gate)
    assert out.dtype == dtype


# Gradient and numerical stability
def test_geglu_gradients():
    """Gradients with respect to both x and gate are finite and shaped correctly."""
    x = jnp.array([1.0, 2.0, 3.0])
    gate = jnp.array([0.5, 1.0, 1.5])

    def fn(x, g):
        return jnp.sum(geglu(x, g))

    gx = grad(lambda x: fn(x, gate))(x)
    gg = grad(lambda g: fn(x, g))(gate)

    assert gx.shape == x.shape
    assert gg.shape == gate.shape
    assert jnp.all(jnp.isfinite(gx))
    assert jnp.all(jnp.isfinite(gg))


def test_geglu_jit_gradients():
    """GEGLU gradients remain valid under JIT compilation."""
    x = jnp.array([1.0, 2.0, 3.0])
    gate = jnp.array([0.5, 1.0, 1.5])

    @jax.jit
    def fn(x, g):
        return jnp.sum(geglu(x, g))

    gx = grad(lambda x: fn(x, gate))(x)
    gg = grad(lambda g: fn(x, g))(gate)

    assert jnp.all(jnp.isfinite(gx))
    assert jnp.all(jnp.isfinite(gg))


def test_geglu_large_inputs():
    """Check numerical stability for large-magnitude inputs."""
    x = jnp.linspace(-20, 20, 1000)
    gate = jnp.ones_like(x)
    out = geglu(x, gate)
    assert jnp.all(jnp.isfinite(out))