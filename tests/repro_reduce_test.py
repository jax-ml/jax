# Copyright 2024 The JAX Authors.
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
from __future__ import annotations

import pathlib
from typing import Any, Callable, Sequence, Union

from absl.testing import absltest

from jax._src import config

from jax._src.repro import emitter
from jax._src.repro import tracker
from jax._src.repro import reducer

from jax._src import test_util as jtu
from jax._src import traceback_util
from jax._src import tree_util


config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


class MyRepro(reducer.DDRepro):
  def __init__(self, all_parts: Sequence[int], min_repro: set[int]):
    super().__init__(pathlib.Path("<here>"))
    # Still a repro as long as we have all the pieces of min_repro
    self.all_parts = all_parts
    self.min_repro = min_repro

  def get_all_candidates(self) -> Sequence[int]:
    return self.all_parts

  def is_reduced_repro(self, reduce_candidates: Sequence[reducer.Candidate]) -> Union["MyRepro" | None]:
    kept = list(i for i in self.all_parts if i not in reduce_candidates)
    if all(i in kept for i in self.min_repro):
      return MyRepro(kept, self.min_repro)
    else:
      return None


@jtu.with_config(jax_traceback_filtering="off")
class DDTest(jtu.JaxTestCase):

  def test_state_advance(self):
    state = reducer.DDState(all_candidates=list(range(16)), granularity=4, start=8)
    self.assertEqual(state.advance(0),
                     reducer.DDState(all_candidates=list(range(16)), granularity=4, start=8))
    self.assertEqual(state.advance(1),
                     reducer.DDState(all_candidates=list(range(16)), granularity=4, start=12))
    self.assertEqual(state.advance(2),
                     reducer.DDState(all_candidates=list(range(16)), granularity=8, start=0))
    self.assertEqual(state.advance(3),
                     reducer.DDState(all_candidates=list(range(16)), granularity=8, start=2))
    self.assertEqual(state.advance(9),
                     reducer.DDState(all_candidates=list(range(16)), granularity=8, start=14))
    self.assertEqual(state.advance(10),
                     reducer.DDState(all_candidates=list(range(16)), granularity=16, start=0))
    self.assertEqual(state.advance(10+15),
                     reducer.DDState(all_candidates=list(range(16)), granularity=16, start=15))
    self.assertEqual(state.advance(10+15+1), None)

    # Some odd divisions
    state = reducer.DDState(all_candidates=list(range(9)), granularity=2, start=8)
    self.assertEqual(state.advance(1),
                     reducer.DDState(all_candidates=list(range(9)), granularity=4, start=0))
    self.assertEqual(state.advance(2),
                     reducer.DDState(all_candidates=list(range(9)), granularity=4, start=3))

  def test_state_select_candidate(self):
    state = reducer.DDState(all_candidates=list(range(16)), granularity=4, start=8)
    self.assertEqual(state.select_candidates(),
                     [8, 9, 10, 11])

    state = reducer.DDState(all_candidates=list(range(9)), granularity=4, start=8)
    self.assertEqual(state.select_candidates(),
                     [8])

  def test_next_repro_1(self):
    r = MyRepro(list(range(10)), {5})

    state = reducer.DDState(all_candidates=r.all_parts, granularity=5, start=4)
    r1, new_state, stats = reducer.next_smaller_repro(r, state, reducer.DDStats())
    self.assertEqual(r1.all_parts, [0, 1, 2, 3, 4, 5, 8, 9])
    self.assertEqual(stats.total_steps, 2)

  def test_next_repro_2(self):
    r = MyRepro(list(range(5)), {0, 2, 4})  # The even ones make the repro

    state = reducer.DDState(all_candidates=r.all_parts, granularity=3, start=4)
    r1, new_state, stats = reducer.next_smaller_repro(r, state, reducer.DDStats())
    self.assertEqual(r1.all_parts, [0, 2, 3, 4])

  def test_ddmin_1(self):
    r = MyRepro(list(range(5)), {0, 2, 4})  # The even ones make the repro
    r1, stats = reducer.ddmin(r)
    self.assertEqual(r1.all_parts, [0, 2, 4])

  def test_drop_functions(self):
    src = """
import jax
from jax import numpy as jnp

def main_repro():
  def f(x, y):
    jnp.tan(y)  # to drop
    return x + y

  x = jnp.array([0, 1, 2], dtype=jnp.int32)
  y = jnp.array([10, 11, 12], dtype=jnp.int32)
  return jax.jit(f)(x, y)
"""
    col = emitter.collector(lambda: emitter.eval_repro(pathlib.Path("<here>"),
                                                       src))
    col()
    collect_funcs = reducer.DropFunctionCallsStrategy(None)
    source1 = col.to_source(strategy=collect_funcs)
    call_tans = [c
                 for c in collect_funcs.all_candidates
                 if not isinstance(c.func, tracker.Func) and c.func.name == "tan"]
    self.assertLen(call_tans, 1)
    call_tan, = call_tans
    self.assertIn("jax_primitive_bind(\"tan\")", source1)

    drop_funcs = reducer.DropFunctionCallsStrategy([call_tan])
    source2 = col.to_source(strategy=drop_funcs)

    self.assertNotIn("jax_primitive_bind(\"tan\")", source2)

  def test_reduce_functions(self):
    src = """
import jax
from jax import numpy as jnp

def main_repro():
  def g():
    return 5.
  def f(x, y):
    jnp.sin(x)  # to drop
    jnp.tan(y)  # to drop
    jax.jit(g)()  # to drop
    return x + y
    
  x = jnp.array([1, 2, 3], dtype=jnp.int32)
  y = jnp.array([11, 12], dtype=jnp.int32)
  return jax.jit(f)(x, y)
"""
    r = reducer.Repro.make(pathlib.Path("<here>"), src,
                           expect_error=(TypeError, "incompatible shapes for broadcasting"),
                           strategy=reducer.DropFunctionCallsStrategy)
    r2, _ = reducer.ddmin(r)
    self.assertNotIn("fun_g_", r2.repro_source)
    self.assertNotIn("primitive_bind(\"sin\")", r2.repro_source)
    self.assertNotIn("primitive_bind(\"tan\")", r2.repro_source)

  def test_drop_expressions(self):
      src = """
import jax
from jax import numpy as jnp

def main_repro():
  def my_f(x, y, d):
    v = x + d["x"]
    return x + d["y"]

  x = jnp.array([0, 1, 2], dtype=jnp.int32)
  y = jnp.array([10, 11, 12], dtype=jnp.int32)
  return jax.jit(my_f)(x, y, dict(x=x, y=y))
  """
      col = emitter.collector(lambda: emitter.eval_repro(pathlib.Path("<here>"),
                                                         src))
      col()
      collect_reductions = reducer.DropExpressionsStrategy(None)
      source1 = col.to_source(strategy=collect_reductions)
      args_to_jit_call_my_f = [
         c
         for c in collect_reductions.all_candidates
         if (isinstance(c[0].func, tracker.Func) and
             c[0].func.api_name == "jax_jit_call" and
             "my_f" in c[0].args[0].fun_name and
             c[1]  # for_args
             )]
      # 7 = jax_jit_call(fun_my_f_3, None, (), {}, v_0, v_1, {'x': v_0, 'y': v_1})
      # But we don't include callables
      self.assertLen(args_to_jit_call_my_f, 10)
      res_for_my_f = [
        c
        for c in collect_reductions.all_candidates
        if (isinstance(c[0].func, tracker.Func) and
            c[0].func.fun_name == "my_f" and
            not c[1] # not for_args
            )]
      self.assertLen(res_for_my_f, 1)
      drop_exprs = reducer.DropExpressionsStrategy(args_to_jit_call_my_f[3:] +
                                                   res_for_my_f)
      source2 = col.to_source(strategy=drop_exprs)

      # TODO: check that it worked
      self.assertNotIn("jax_primitive_bind(\"tan\")", source2)

  def test_reduce_expressions(self):
    src = """
import jax
from jax import numpy as jnp

def main_repro():
  def g():
    return 5.
  def f(x, y):
    jnp.sin(x)  # to drop
    jnp.tan(y)  # to drop
    jax.jit(g)()  # to drop
    return x + y

  x = jnp.array([1, 2, 3], dtype=jnp.int32)
  y = jnp.array([11, 12], dtype=jnp.int32)
  return jax.jit(f)(x, y)
"""
    r = reducer.Repro.make(pathlib.Path("<here>"), src,
                           expect_error=(TypeError, "incompatible shapes for broadcasting"),
                           strategy=reducer.DropFunctionCallsStrategy)
    r2, _ = reducer.ddmin(r)
    self.assertNotIn("fun_g_", r2.repro_source)
    self.assertNotIn("primitive_bind(\"sin\")", r2.repro_source)
    self.assertNotIn("primitive_bind(\"tan\")", r2.repro_source)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
