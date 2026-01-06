# tests/issue_34083_test.py
import jax
import jax.numpy as jnp
from absl.testing import absltest
# You can remove 'from jax import test_util as jtu' if you want, strictly not needed here.

# CHANGE: Inherit directly from absltest.TestCase
class Issue34083Test(absltest.TestCase):
  
  def test_pure_callback_exception_cached(self):
    # Regression test for https://github.com/jax-ml/jax/issues/34083
    # ... (rest of the code remains exactly the same) ...

    def error_callback(x):
      raise RuntimeError("CRITICAL_FAILURE_MESSAGE")

    def conditional_error(x):
      def _cb(operand):
        shape = jax.ShapeDtypeStruct(operand.shape, operand.dtype)
        return jax.pure_callback(error_callback, shape, operand)

      return jax.lax.cond(
          x > 0,
          _cb,
          lambda o: o,
          x
      )

    jit_f = jax.jit(conditional_error)

    # 1. Prime the cache (Safe path)
    res = jit_f(jnp.array(-1.0))
    self.assertEqual(res, -1.0)

    # 2. Trigger the error (Error path)
    with self.assertRaisesRegex(Exception, "CRITICAL_FAILURE_MESSAGE"):
      jit_f(jnp.array(1.0))

if __name__ == "__main__":
  absltest.main()
