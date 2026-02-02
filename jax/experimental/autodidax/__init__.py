"""Autodidax with Rust core - a minimal JAX-like system.

This package provides a Python API backed by a Rust implementation of the
core tracing and JVP machinery from the autodidax tutorial.
"""

from jax.experimental.autodidax.primitives import add
from jax.experimental.autodidax.primitives import cos
from jax.experimental.autodidax.primitives import greater
from jax.experimental.autodidax.primitives import less
from jax.experimental.autodidax.primitives import mul
from jax.experimental.autodidax.primitives import neg
from jax.experimental.autodidax.primitives import sin
from jax.experimental.autodidax.core import jvp

__all__ = [
    "add",
    "mul",
    "neg",
    "sin",
    "cos",
    "greater",
    "less",
    "jvp",
]
