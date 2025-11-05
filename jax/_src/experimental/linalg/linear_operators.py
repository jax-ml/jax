# linear_operators.py 

"""
Purpose:
This module introduces a prototype abstraction for linear operators
in JAX. A LinearOperator represents a matrix-like transformation
without explicitly storing the full matrix. This is useful for
"matrix-free" computations such as large-scale optimization,
physics simulations, or implicit Jacobian-vector products.
"""


from abc import ABC, abstractmethod     # For defining an abstract base class
import jax.numpy as jnp                 # JAX's NumPy-compatible API for Array Ops 
from typing import Tuple                # For Shape Annotations 


class LinearOperator(ABC):
    """Abstract base class for matrix-free linear transformations."""

    def __init__(self, shape: Tuple[int, int]):
        self.shape: Tuple[int, int] = shape  # (m, n)
    
    @abstractmethod
    def matvec(self, x: jnp.ndarray) -> jnp.ndarray:
        "Apply the operator to a vector x (shape (n,)) -> (m,)."
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convenience: op(x) calls matvec(x)."""
        return self.matvec(x)
    

class IdentityOperator(LinearOperator):
    """An operator that returns its input unchanged: y = x."""

    def __init__(self, size: int):
        #  identity has shape (n, n)
        super().__init__((size, size))
    
    def matvec(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply I * x = x."""
        assert x.shape[0] == self.shape[1], f"Expected shape {(self.shape[1],)}, got {x.shape}"
        return x