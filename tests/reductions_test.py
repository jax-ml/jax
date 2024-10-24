import jax
from jax._src import dtypes
from jax._src import test_util as jtu
from jax._src.numpy.reductions import _promote_integer_dtype
import jax.numpy as jnp


class PrngTest(jtu.JaxTestCase):
  """Test the compatibility of jnp.sum with low bitwidth integers."""

  def test__promote_integer_dtype(self) -> None:
    assert _promote_integer_dtype(jnp.int4) is dtypes.int_
    assert _promote_integer_dtype(jnp.uint4) is dtypes.int_
    assert _promote_integer_dtype(jnp.int2) is dtypes.int_
    assert _promote_integer_dtype(jnp.uint2) is dtypes.int_

  def test_low_bitwidth_sum(self) -> None:
    sum_jit = jax.jit(jnp.sum)
    arr = jax.random.randint(
        key=jax.random.PRNGKey(0),
        shape=(10, 10),
        minval=0,
        maxval=7,
        dtype=jnp.int32,
    )
    s_int4 = sum_jit(arr.astype(jnp.int4))
    s_uint4 = sum_jit(arr.astype(jnp.uint4))
    s_ref = sum_jit(arr.astype(jnp.int32))
    assert s_int4 == s_ref
    assert s_uint4 == s_ref

