from absl.testing import absltest
import numpy as np
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jax._src import test_util as jtu
from jax._src import config

config.parse_flags_with_absl()


class ReducedEltwiseMathTest(jtu.JaxTestCase):
  @jtu.with_explicit_mesh((2,), ("x",))
  def test_add_sharded_reduced(self, mesh):
    x_shape = (10, 4)
    y_shape = (1, 4)

    s_x = NamedSharding(mesh, P("x", None))
    s_y = NamedSharding(mesh, P(None, reduced=frozenset("x")))

    x = jnp.ones(x_shape, out_sharding=s_x)
    y = jnp.ones(y_shape, out_sharding=s_y)
    out, (dx, dy) = self.__run_op_with_grads(lambda u, v: u + v, x, y)

    self.assertEqual(out.sharding.spec, s_x.spec)
    self.assertArraysEqual(out, np.full(x_shape, fill_value=2, dtype=jnp.float32))

    self.assertEqual(dx.sharding.spec, P("x", None))
    self.assertArraysEqual(dx, np.ones(x_shape))

    self.assertEqual(dy.sharding.spec, P(None, None, unreduced=frozenset("x")))
    self.assertArraysEqual(
      dy, np.full(y_shape, fill_value=x_shape[0], dtype=jnp.float32)
    )

  @jtu.with_explicit_mesh((2,), ("x",))
  def test_mul_sharded_reduced(self, mesh):
    x_shape = (10, 4)
    y_shape = (1, 4)

    s_x = NamedSharding(mesh, P("x", None))
    s_y = NamedSharding(mesh, P(None, reduced=frozenset("x")))

    x = jnp.arange(np.prod(x_shape), dtype=jnp.float32).reshape(
      x_shape, out_sharding=s_x
    )
    y = jnp.ones(y_shape, dtype=jnp.float32, out_sharding=s_y)

    out, (dx, dy) = self.__run_op_with_grads(lambda u, v: u * v, x, y)

    self.assertEqual(out.sharding.spec, s_x.spec)
    self.assertArraysEqual(
      out,
      np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape)
      * np.ones(y_shape, dtype=np.float32),
    )

    self.assertEqual(dx.sharding.spec, P("x", None))
    self.assertArraysEqual(dx, np.ones(x_shape, dtype=np.float32))

    self.assertEqual(dy.sharding.spec, P(None, None, unreduced=frozenset("x")))
    self.assertArraysEqual(
      dy,
      np.sum(
        np.arange(np.prod(x_shape), dtype=np.float32).reshape(x_shape),
        axis=0,
        keepdims=True,
      ),
    )

  @staticmethod
  def __run_op_with_grads(op, a, b):
    @jax.jit
    def step(x, y):
      def loss_fn(u, v):
        out = op(u, v)
        return jnp.sum(out), out

      (loss, out), grads = jax.value_and_grad(
        loss_fn, argnums=(0, 1), has_aux=True
      )(x, y)
      return out, grads

    return step(a, b)


class ReducedReductionMathTest(jtu.JaxTestCase):
  @jtu.with_explicit_mesh((2,), ("x",))
  def test_reduce_sum_unreduced(self, mesh):
    s_x = NamedSharding(mesh, P("x", None))
    s_x_unr = NamedSharding(mesh, P(None, None, unreduced=frozenset("x")))

    x = jnp.arange(8, dtype=jnp.float32).reshape(4, 2, out_sharding=s_x)
    x_unr = jax.reshard(x, s_x_unr)

    out = jnp.sum(x_unr)

    self.assertEqual(out.shape, ())
    self.assertEqual(out.sharding.spec.unreduced, frozenset("x"))

    # check pre-reduction
    expected_sum = np.sum(np.arange(8, dtype=np.float32))  # 28
    self.assertLess(float(out), expected_sum)

    # trigger reduction and check
    out_reduced = jax.reshard(out, P())
    self.assertEqual(float(out_reduced), expected_sum)


class ReducedArrayManipulationOpsTest(jtu.JaxTestCase):
  @jtu.with_explicit_mesh((2,), ("x",))
  def test_reshape_preserves_unreduced(self, mesh):
    x = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    x = jax.device_put(x, NamedSharding(mesh, P("x", None)))
    x_unr = jax.reshard(x, P(None, None, unreduced=frozenset("x")))

    out = jnp.reshape(x_unr, (2, 4))

    self.assertEqual(out.sharding.spec.unreduced, frozenset("x"))
    self.assertEqual(out.shape, (2, 4))

  @jtu.with_explicit_mesh((2,), ("x",))
  def test_split_unreduced(self, mesh):
    x = jnp.arange(8, dtype=jnp.float32).reshape(4, 2)
    x = jax.device_put(x, NamedSharding(mesh, P("x", None)))
    x_unr = jax.reshard(x, P(None, None, unreduced=frozenset("x")))

    out = jnp.split(x_unr, 2, axis=0)

    self.assertEqual(len(out), 2)
    self.assertEqual(out[0].sharding.spec.unreduced, frozenset("x"))
    self.assertEqual(out[1].sharding.spec.unreduced, frozenset("x"))

  @jtu.with_explicit_mesh((2,), ("x",))
  def test_slice_reduced(self, mesh):
    x = jax.device_put(
      jnp.arange(10.0), NamedSharding(mesh, P(None, reduced=frozenset("x")))
    )

    @jax.jit
    def f(arr):
      return arr[0:5]

    y = f(x)

    self.assertEqual(y.sharding.spec, P(None, reduced=frozenset("x")))
    self.assertArraysEqual(y, jnp.arange(5.0))

  @jtu.with_explicit_mesh((2,), ("x",))
  def test_gather_reduced_replicated_index(self, mesh):
    x = jax.device_put(
      jnp.arange(10.0), NamedSharding(mesh, P(None, reduced=frozenset("x")))
    )

    @jax.jit
    def f(arr):
      return arr[2]

    y = f(x)

    self.assertEqual(y.sharding.spec, P(reduced=frozenset("x")))
    self.assertEqual(y, 2.0)


if __name__ == "__main__":
  absltest.main()
