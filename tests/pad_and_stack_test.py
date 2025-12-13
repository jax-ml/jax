"""Tests for jax.numpy.pad_and_stack."""

import numpy as np

from absl.testing import absltest
from absl.testing import parameterized

import jax
import jax.numpy as jnp
from jax._src import test_util as jtu

jax.config.parse_flags_with_absl()


class PadAndStackTest(jtu.JaxTestCase):
  """Tests for jnp.pad_and_stack function."""

  def _numpy_pad_and_stack(
    self, arrays, axis=-1, padding_value=0, min_size=None, stack_axis=0
  ):
    """NumPy reference implementation."""
    if not arrays:
      raise ValueError("Need at least one array to pad and stack.")

    arrays = [np.asarray(arr) for arr in arrays]
    first_array = arrays[0]

    # Normalize axis
    if axis < 0:
      axis = first_array.ndim + axis
    if stack_axis < 0:
      stack_axis = first_array.ndim + 1 + stack_axis

    # Determine target size
    current_sizes = [arr.shape[axis] for arr in arrays]
    max_size = max(current_sizes) if current_sizes else 0

    if min_size is not None:
      target_size = max(max_size, min_size)
    else:
      target_size = max_size

    # Pad each array
    padded_arrays = []
    for arr in arrays:
      current_size = arr.shape[axis]
      if current_size < target_size:
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (0, target_size - current_size)
        padded = np.pad(arr, pad_width, constant_values=padding_value)
      else:
        padded = arr
      padded_arrays.append(padded)

    return np.stack(padded_arrays, axis=stack_axis)

  @parameterized.named_parameters(
    dict(
      testcase_name="1d_basic",
      arrays=[np.array([1, 2, 3]), np.array([4, 5]), np.array([6])],
      expected_shape=(3, 3),
    ),
    dict(
      testcase_name="2d_axis0",
      arrays=[np.array([[1, 2]]), np.array([[3, 4], [5, 6]])],
      axis=0,
      expected_shape=(2, 2, 2),
    ),
    dict(
      testcase_name="with_min_size",
      arrays=[np.array([1, 2]), np.array([3])],
      min_size=5,
      expected_shape=(2, 5),
    ),
  )
  def test_basic_functionality(
    self, arrays, expected_shape, axis=-1, min_size=None
  ):
    """Test basic pad_and_stack functionality."""
    jax_arrays = [jnp.asarray(arr) for arr in arrays]
    result = jnp.pad_and_stack(jax_arrays, axis=axis, min_size=min_size)
    self.assertEqual(result.shape, expected_shape)

    # Compare with numpy reference
    expected = self._numpy_pad_and_stack(arrays, axis=axis, min_size=min_size)
    np.testing.assert_allclose(result, expected)

  def test_custom_padding_value(self):
    """Test padding with custom value."""
    arrays = [jnp.array([1, 2, 3]), jnp.array([4, 5])]
    result = jnp.pad_and_stack(arrays, padding_value=-1)
    expected = np.array([[1, 2, 3], [4, 5, -1]])
    np.testing.assert_allclose(result, expected)

  def test_empty_raises(self):
    """Test that empty list raises ValueError."""
    with self.assertRaises(ValueError):
      jnp.pad_and_stack([])

  def test_mismatched_dims_raises(self):
    """Test that arrays with different ndims raise ValueError."""
    arrays = [jnp.array([1, 2]), jnp.array([[1, 2]])]
    with self.assertRaises(ValueError):
      jnp.pad_and_stack(arrays)

  def test_single_array(self):
    """Test that single array works."""
    arrays = [jnp.array([1, 2, 3])]
    result = jnp.pad_and_stack(arrays)
    expected = np.array([[1, 2, 3]])
    np.testing.assert_allclose(result, expected)

  def test_jit_compatible(self):
    """Test JIT compilation with min_size."""

    @jax.jit
    def fn(arrays):
      return jnp.pad_and_stack(arrays, min_size=5)

    arrays = [jnp.array([1, 2]), jnp.array([3])]
    result = fn(arrays)
    self.assertEqual(result.shape, (2, 5))


if __name__ == "__main__":
  absltest.main()
