"""
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
    """Abstract base class for matrix-free linear transformations"""

    def __init__(self, shape: Tuple[int, int]):
        self.shape: Tuple[int, int] = shape  # (m, n)
    
    @abstractmethod
    def matvec(self, x: jnp.ndarray) -> jnp.ndarray:
        "Apply the operator to a vector x (shape (n,)) -> (m,)"

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Convenience: op(x) calls matvec(x)."""
        return self.matvec(x)
    
    def __repr__(self) -> str:
        """Official string representation for debugging."""
        classname = self.__class__.__name__
        return f"{classname}(shape={self.shape})"
    
    def __str__(self) -> str:
        """Readable string version shown in print()."""
        return f"{self.__class__.__name__} with shape {self.shape}"

    def to_dense(self) -> jnp.ndarray:
        """Materialize the operator as a dense matrix (useful for debugging)"""
        m, n = self.shape
        I = jnp.eye(n) # Create an identity matrix of size (n, n)
        columns = [] # Start with an empty list to collect each column of the dense matrix

        for i in range(n):
            col = self(I[:, i]) # Apply the operator to the i-th basis vector
            columns.append(col) # Append the resulting column to the list
        
        # Stack all columns side-by-side to form a full dense matrix
        dense_matrix = jnp.stack(columns, axis=1)

        return dense_matrix
    
    def __add__(self, other: "LinearOperator") -> "LinearOperator":
        """Add two linear operators with the same shape: (A + B)"""
        assert isinstance(other, LinearOperator), "Can only add LinearOperator objects."
        assert self.shape == other.shape, f"Shape mismatch: {self.shape} vs {other.shape}"

        class SumOperator(LinearOperator):
            """Operator representing the sum of two linear operators"""

            def __init__(self, A, B):
                super().__init__(A.shape)
                self.A = A
                self.B = B
            
            def matvec(self, x: jnp.ndarray) -> jnp.ndarray:
                """Apply (A + B)x = A(x) + B(x)"""
                return self.A(x) + self.B(x)
            
        return SumOperator(self, other)
    
    def __mul__(self, scalar: float) -> "LinearOperator":
        """Scale the operator by a scalar: (a * A)"""
        assert isinstance(scalar, (int, float)), "Can only multiply by a scalar"

        class ScaledOperator(LinearOperator):
            """Operator representing a scalar multiple of another operator"""

            def __init__(self, A, scale):
                super().__init__(A.shape)
                self.A = A
                self.scale = scale 
            
            def matvec(self, x: jnp.ndarray) -> jnp.ndarray:
                """Apply (a * A)x = a * A(x)"""
                return self.scale * self.A(x)
            
        return ScaledOperator(self, scalar)
    
    # Support reversed multiplication
    __rmul__ = __mul__



class IdentityOperator(LinearOperator):
    """An operator that returns its input unchanged: y = x"""

    def __init__(self, size: int):
        #  identity has shape (n, n)
        super().__init__((size, size))
    
    def matvec(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply I * x = x."""
        assert x.shape[0] == self.shape[1], f"Expected shape {(self.shape[1],)}, got {x.shape}"
        return x




class ScalingOperator(LinearOperator):
    """A linear operator that scales its input vector by a scalar factor"""

    def __init__(self, scale: float, size: int):
        """Initialize a scaling operator"""
        super().__init__((size, size))
        self.scale = scale

    def matvec(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies the scaling transformation: y = a * x."""
        return self.scale * x




class MatrixOperator(LinearOperator):
    """A linear operator that wraps an explicit JAX matrix"""

    def __init__(self, matrix: jnp.ndarray):
        """Initialize the operator"""
        assert matrix.ndim == 2, f"Matrix must be 2D, got {matrix.ndim}D"
        m, n = matrix.shape
        super().__init__((m, n))
        self.matrix = matrix 

    
    def matvec(self, x: jnp.ndarray) -> jnp.ndarray:
        """Applies the linear transformation y = A @ x"""
        assert x.shape[0] == self.shape[1], (f"Expected input of shape {(self.shape[1],)}, got {x.shape}")
        return jnp.dot(self.matrix, x)




class CompositionOperator(LinearOperator):
    """An operator that represents the composition of two linear operators"""

    def __init__(self, A: LinearOperator, B: LinearOperator):
        """Initialize the composition A(B(x))"""
        
        assert A.shape[1] == B.shape[0], (f"Shape mismatch: A {A.shape} cannot follow B {B.shape}")
        super().__init__((A.shape[0], B.shape[1]))
        
        self.A = A
        self.B = B
    
    def matvec(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the composed operator: y = A(B(x))"""
        return self.A(self.B(x))




class DiagonalOperator(LinearOperator):
    """A linear operator representing a diagonal matrix (stores only the diagonal)"""

    def __init__(self, diag: jnp.ndarray):
        """Initialize with a vector of diagonal entries"""
        assert diag.ndim == 1, f"Diagonal must be 1D, got {diag.ndim}D"

        n = diag.shape[0]
        super().__init__((n, n))
        self.diag = diag

    def matvec(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the diagonal transformation: y_i = d_i * x_i"""
        assert x.shape[0] == self.shape[1], (f"Expected input shape {(self.shape[1],)}, got {x.shape}")
        return self.diag * x



class TransposeOperator(LinearOperator):
    """A linear operator representing the transpose of another operator"""

    def __init__(self, A: LinearOperator):
        """Initialize with another linear operator A; represents A^T"""

        # Swap shape dimensions
        super().__init__((A.shape[1], A.shape[0]))
        self.A = A

    def matvec(self, x: jnp.ndarray) -> jnp.ndarray:
        """Apply the transpose transformation: y = A^T x"""

        # If A is a MatrixOperator, use its matrix directly
        if isinstance(self.A, MatrixOperator):
            return jnp.dot(self.A.matrix.T, x)
        else:
            # Fall back to a dense conversion for other operator types
            return jnp.dot(self.A.to_dense().T, x)

