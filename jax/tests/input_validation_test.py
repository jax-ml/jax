
import pytest
import jax.numpy as jnp
import warnings


def test_stacking_validation():
  # These should now raise TypeError instead of warning
  # Using nested lists as inputs, which are not JAX array-likes (scalars are, lists are not)
  with pytest.raises(TypeError, match="requires ndarray or scalar arguments"):
      jnp.vstack([[1], [2]])
  
  with pytest.raises(TypeError, match="requires ndarray or scalar arguments"):
      jnp.hstack([[1], [2]])

  with pytest.raises(TypeError, match="requires ndarray or scalar arguments"):
      jnp.dstack([[1], [2]])
      
  with pytest.raises(TypeError, match="requires ndarray or scalar arguments"):
      jnp.column_stack([[1], [2]])

  with pytest.raises(TypeError, match="requires ndarray or scalar arguments"):
      # atleast_2d takes *arys. If we pass a list [1, 2], it treats it as one arg [1, 2] (list).
      # check_arraylike("atleast_2d", [1, 2]) fails.
      jnp.atleast_2d([1, 2])

  with pytest.raises(TypeError, match="requires ndarray or scalar arguments"):
      jnp.atleast_3d([1, 2])


