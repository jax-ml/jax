import pytest

import jax
from jax import grad, random
import jax.numpy as jnp
from jax._src.third_party.scipy.linalg import solve_sylvester

# float32 is not close enough
jax.config.update("jax_enable_x64", True)  # Ensure float64 for higher precision

class TestSolveSylvester:
  def test_solve_sylvester(self):
    n = 10
    m = 12

    A = random.normal(random.key(0), shape=(n, n))
    B = random.normal(random.key(1), shape=(m, m))
    X_true = random.normal(random.key(2), shape=(n, m))

    C = A @ X_true + X_true @ B

    X_computed = solve_sylvester(A, B, C)
    assert jnp.allclose(X_computed, X_true)

  def test_no_solution_sylvester(self):
    # Test no solution case to AX + XB = C
    # The following would generate (1)X + X(-1) = (1)
    # which equals X - X = (1) which is not possible because X - X = 0
    A = jnp.array([[1]])
    B = jnp.array([[-1]])
    C = jnp.array([[1]])
    with pytest.raises(RuntimeError) as rte_excpt:
      solve_sylvester(A, B, C)
    assert "Sylvester Equation has no solution!" in str(rte_excpt.value)
