import functools

from absl.testing import absltest
from absl.testing import parameterized
import haiku as hk # type: ignore
from haiku._src import test_utils # type: ignore
from haiku._src.integration import descriptors # type: ignore
import jax
import jax.numpy as jnp
import numpy as np

ModuleFn = descriptors.ModuleFn

class BatchPolymorphismTest(parameterized.TestCase):

  @test_utils.combined_named_parameters(descriptors.ALL_MODULES)
  def test_shapecheck(self, module_fn, shape, dtype):
    rng = jax.random.PRNGKey(42)
    if jnp.issubdtype(dtype, jnp.integer):
      x = jax.random.randint(rng, shape, 0, np.prod(shape), dtype)
    else:
      x = jax.random.uniform(rng, shape, dtype)

    def g(x):
      return module_fn()(x)

    f = hk.transform_with_state(g)
    params, state = f.init(rng, x)
    f_jax = functools.partial(f.apply, params, state, rng)

    def _make_shape_spec(shape, batched=False):
      # all the input functions are assumed to take a batched input
      if batched and len(shape) > 0:
        return '(b,' + ','.join(list(map(str, shape[1:]))) + ')'
      else:
        return '(' + ','.join(list(map(str, shape))) + ')'

    _make_batched_shape_spec = functools.partial(_make_shape_spec, batched=True)
    out, state = f_jax(x)
    debug_in_shape = _make_shape_spec(x.shape)
    debug_out_shapes = (
      tuple([ jax.tree_map(lambda e: _make_shape_spec(e.shape), out)
            , jax.tree_map(lambda e: _make_shape_spec(e.shape), state)
            ]))

    in_shape = _make_batched_shape_spec(x.shape)
    out_shapes = (
      tuple([ jax.tree_map(lambda e: _make_batched_shape_spec(e.shape), out)
            , jax.tree_map(lambda e: _make_shape_spec(e.shape), state)
            ]))

    print(in_shape, out_shapes)
    print(debug_in_shape, debug_out_shapes)

    jax.shapecheck([in_shape], out_shapes, dtypes=[dtype])(f_jax)

if __name__ == "__main__":
  absltest.main()
