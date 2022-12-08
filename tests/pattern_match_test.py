# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Test that pattern match optimisations occur."""

import timeit

from absl.testing import parameterized

import jax
import jax.lax as lax
import jax.numpy as jnp
from jax._src import test_util as jtu
from jax._src.lax.control_flow.pattern_match import pattern_match_inplace_select


class PatternMatchTest(jtu.JaxTestCase):
  @parameterized.named_parameters(
      {"inplace_op": inplace_op, "testcase_name": f"_{inplace_op}"}
      for inplace_op in ("scatter", "dynamic_update_slice")
  )
  @jtu.skip_on_devices("gpu", "tpu")  # cpu only, to get reliable timings
  def test_batch_inplace_while(self, inplace_op):
    @jax.jit
    @jax.vmap
    def f(init_step, init_xs):
      def cond(carry):
        step, xs = carry
        return step < xs.size

      def body(carry):
        step, xs = carry
        if inplace_op == "scatter":
          xs = xs.at[step].set(1)
        elif inplace_op == "dynamic_update_slice":
          # Actually batched into a scatter anyway?
          xs = lax.dynamic_update_index_in_dim(xs, 1., step, 0)
        else:
          assert False
        return step + 1, xs

      return lax.while_loop(cond, body, (init_step, init_xs))

    size = 100_000
    args = jnp.array([0]), jnp.zeros((1, size))
    f(*args)  # compile
    time = timeit.timeit(lambda: f(*args), number=1)
    # Check that runtime took less than a second.
    # This shouldn't be flaky as true runtime should be ~1e-3 seconds.
    # (Without the pattern-match it takes about 12 seconds.)
    self.assertLess(time, 1)


  @parameterized.named_parameters(
      {"inplace_op": inplace_op, "testcase_name": f"_{inplace_op}"}
      for inplace_op in ("scatter", "dynamic_update_slice")
  )
  def test_basic(self, inplace_op):
    def f(pred, x):
      if inplace_op == "scatter":
        new_x = x.at[0].set(0)
      elif inplace_op == "dynamic_update_slice":
        new_x = lax.dynamic_update_index_in_dim(x, 0, 0, 0)
      else:
        assert False
      return lax.select(pred, new_x, x)

    args = (True, jax.numpy.arange(4))
    jaxpr = jax.make_jaxpr(f)(*args)
    rewritten_jaxpr = pattern_match_inplace_select(jaxpr)

    if inplace_op == "scatter":
      set_prim = lax.scatter_p
      get_prim = lax.gather_p
    elif inplace_op == "dynamic_update_slice":
      set_prim = lax.dynamic_update_slice_p
      get_prim = lax.dynamic_slice_p
    else:
      assert False

    prims = [eqn.primitive for eqn in jaxpr.eqns]
    rewritten_prims = [eqn.primitive for eqn in rewritten_jaxpr.eqns]

    self.assertEqual(prims, [lax.broadcast_in_dim_p, set_prim, lax.select_n_p])
    self.assertEqual(rewritten_prims,
                     [lax.broadcast_in_dim_p, get_prim, lax.select_n_p, set_prim])
