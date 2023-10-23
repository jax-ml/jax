from cloudpickle import cloudpickle
from copy import deepcopy

from jax._src import test_util as jtu

import jax
from jax.extend import core
import jax.extend.cloudpickle_support


class CopyingTest(jtu.JaxTestCase):

  def test_jaxpr_cloudpickle(self):
    def f(a, b):
      return a + b, b * a

    jxpr = jax.make_jaxpr(f)(1, 2)

    # Load then dump with cloudpickle
    encoded = cloudpickle.dumps(jxpr)
    loaded_jxpr = cloudpickle.loads(encoded)

    # Eval, jit and execute.
    fun = core.jaxpr_as_fun(loaded_jxpr)
    assert jax.jit(fun)(5, 3) == [8, 15]

  def test_jaxpr_deepcopy(self):
    def f(a, b):
      return b * a, a + b,

    jxpr = jax.make_jaxpr(f)(1, 2)

    # Deepcopy jaxpr
    copied_jaxpr = deepcopy(jxpr)

    # Eval, jit and execute.
    fun = core.jaxpr_as_fun(copied_jaxpr)
    assert jax.jit(fun)(5, 3) == [15, 8]
