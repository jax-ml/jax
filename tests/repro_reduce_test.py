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
from typing import Sequence, Union

from absl.testing import absltest

from jax._src import config

from jax._src import repro
from jax._src.repro import emitter
from jax._src.repro import tracker
from jax._src.repro import reducer
from jax._src.repro import ddmin

from jax._src import test_util as jtu
from jax._src import traceback_util


config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


class MyRepro(ddmin.Repro):
  def __init__(self, all_parts: Sequence[int], min_repro: set[int]):
    # Still a repro as long as we have all the pieces of min_repro
    self.all_candidates = all_parts
    self.min_repro = min_repro

  def progress_msg(self, state: ddmin.State[ddmin.Candidate]) -> str:
    return f"at chunk_size {state.chunk_size} and start_index {state.start_index}"

  def get_reduced_repro(self, reduce_candidates: Sequence[ddmin.Candidate]) -> Union["MyRepro" | None]:
    kept = [i for i in self.all_candidates if i not in reduce_candidates]
    if all(i in kept for i in self.min_repro):
      return MyRepro(kept, self.min_repro)
    else:
      return None


@jtu.with_config(jax_traceback_filtering="off")
class DDTest(jtu.JaxTestCase):

  def test_state_advance(self):
    def advance_n(s: ddmin.State, n: int) -> ddmin.State | None:
      for _ in range(n):
        s = s.advance()
      return s
    state = ddmin.State(all_candidates=list(range(16)), chunk_size=4, start_index=8)
    self.assertEqual(advance_n(state, 1),
                     ddmin.State(all_candidates=list(range(16)), chunk_size=4, start_index=12))
    self.assertEqual(advance_n(state, 2),
                     ddmin.State(all_candidates=list(range(16)), chunk_size=1, start_index=0))
    self.assertEqual(advance_n(state, 3),
                     ddmin.State(all_candidates=list(range(16)), chunk_size=1, start_index=1))
    self.assertEqual(advance_n(state, 9),
                     ddmin.State(all_candidates=list(range(16)), chunk_size=1, start_index=7))
    self.assertEqual(advance_n(state, 10),
                     ddmin.State(all_candidates=list(range(16)), chunk_size=1, start_index=8))
    self.assertEqual(advance_n(state, 10+7),
                     ddmin.State(all_candidates=list(range(16)), chunk_size=1, start_index=15))
    self.assertEqual(advance_n(state, 10+7+1), None)

    # Some odd divisions
    state = ddmin.State(all_candidates=list(range(9)), chunk_size=5, start_index=8)
    self.assertEqual(advance_n(state, 1),
                     ddmin.State(all_candidates=list(range(9)), chunk_size=1, start_index=0))
    self.assertEqual(advance_n(state, 2),
                     ddmin.State(all_candidates=list(range(9)), chunk_size=1, start_index=1))

  def test_state_select_candidate(self):
    state = ddmin.State(all_candidates=list(range(16)), chunk_size=4, start_index=8)
    self.assertEqual(state.select_candidates(),
                     [8, 9, 10, 11])

    state = ddmin.State(all_candidates=list(range(9)), chunk_size=4, start_index=8)
    self.assertEqual(state.select_candidates(),
                     [8])

  def test_next_repro_1(self):
    r = MyRepro(list(range(10)), {5})

    state = ddmin.State(all_candidates=r.all_candidates, chunk_size=2, start_index=4)
    r1, new_state, stats = ddmin.next_smaller_repro(r, state, ddmin.Stats())
    self.assertEqual(r1.all_candidates, [0, 1, 2, 3, 4, 5, 8, 9])
    self.assertEqual(stats.total_steps, 2)

  def test_next_repro_2(self):
    r = MyRepro(list(range(5)), {0, 2, 4})  # The even ones make the repro

    state = ddmin.State(all_candidates=r.all_candidates, chunk_size=3, start_index=4)
    r1, new_state, stats = ddmin.next_smaller_repro(r, state, ddmin.Stats())
    self.assertEqual(r1.all_candidates, [0, 2, 3, 4])

  def test_ddmin_1(self):
    r = MyRepro(list(range(8)), {0, 2, 4})  # The even ones make the repro
    r1, _ = ddmin.ddmin(r, chunk_size=4)
    self.assertEqual(r1.all_candidates, [0, 2, 4])

  def test_ddmin_1_stats(self):
    r = MyRepro(list(range(8)), {0, 2, 4})  # The even ones make the repro
    r1, stats = ddmin.ddmin(r, chunk_size=4)
    # With chunk size of 4, the first 2 steps produce no improvement
    # Instead of switching to chunk_size 2, we skip to 1
    # On step 3, we try to remove 0, and we fail
    # On step 4, we try to remove 1, and we succeed; we rotate the candidates
    # so that they are [2, 3, 4, 5, 6, 7, 0]
    # On step 5, we try to remove 2, and we fail
    # On step 6, we try to remove 3, and we succeed; we rotate the candidates
    # so that they are [4, 5, 6, 7, 0, 2]
    # On step 7, we try to remove 4, and we succeed; we rotate the candidates
    # so that they are [6, 7, 0, 2, 4]
    # On step 8, we try to remove 6, and we fail
    # On step 9, we try to remove 7, and we succeed; we rotate the candidates
    # so that they are [0, 2, 4]
    self.assertEqual(r1.all_candidates, [0, 2, 4])

  def test_ddmin_no_reduction_possible(self):
    r = MyRepro(list(range(5)), list(range(5)))
    r1, stats = ddmin.ddmin(r, chunk_size=2)
    self.assertEqual(r1.all_candidates, list(range(5)))

  def test_ddmin_all_reductions_possible(self):
    r = MyRepro(list(range(5)), [])
    r1, stats = ddmin.ddmin(r, chunk_size=2)
    self.assertEqual(r1.all_candidates, [])


@jtu.with_config(jax_traceback_filtering="off")
class ReduceTest(jtu.JaxTestCase):
  def setUp(self):
    if not traceback_util.repro_is_enabled():
      self.skipTest("JAX_REPRO_DIR not set")

  def test_check_repro(self):
    src_pass = """
import jax
import numpy as np
from jax import numpy as jnp

def main_repro():
  @jax.jit
  def f(x):
    return x + jnp.concatenate([x, x])
  f(np.array([1, 2, 3]))
"""
    src_fail = """
def main_repro():
  pass
"""
    src_error = """
def main_repro():
  raise NotImplementedError("unexpected error")
"""
    def test_repro_fun(repro_fun):
      try:
        repro_fun()
      except TypeError as e:
        return "add got incompatible shapes" in str(e)
      return False

    # Test successful repro
    res = reducer.Repro.check_repro(src_pass, pathlib.Path("<test>"),
                                    test_repro_fun,
                                    reducer.DropFunctionCallsStrategy)
    self.assertIsInstance(res, tuple)
    col, candidates = res
    self.assertLen(candidates, 4)

    # Test failed repro (raises ValueError by default)
    with self.assertRaisesRegex(ValueError, "Not a reproducer"):
      reducer.Repro.check_repro(src_fail, pathlib.Path("<test>"),
                                test_repro_fun,
                                reducer.DropFunctionCallsStrategy)

    # Test failed repro with raise_on_failure=False
    msg = reducer.Repro.check_repro(src_fail, pathlib.Path("<test>"),
                                    test_repro_fun,
                                    reducer.DropFunctionCallsStrategy,
                                    raise_on_failure=False)
    self.assertIsInstance(msg, str)
    self.assertIn("Not a reproducer", msg)

    # Test repro that raises unexpected exception
    with self.assertRaisesRegex(ValueError, "Not a reproducer, it raises: unexpected error"):
      reducer.Repro.check_repro(src_error, pathlib.Path("<test>"),
                                test_repro_fun,
                                reducer.DropFunctionCallsStrategy)


  def test_drop_calls(self):
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
    col = emitter.collector(repro.load(src, pathlib.Path("<here>")))
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

  def test_reduce_calls(self):
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
    return x + y  # Error: incompatible shapes for broadcasting

  x = jnp.array([1, 2, 3], dtype=jnp.int32)
  y = jnp.array([11, 12], dtype=jnp.int32)
  return jax.jit(f)(x, y)

if __name__ == "__main__":
  main_repro()
"""
    def test_repro_fun(repro_fun):
      try:
        repro_fun()
      except TypeError as e:
        return "add got incompatible shapes for broadcasting" in str(e)
      return False

    r = reducer.Repro.make(src, pathlib.Path("<here>"),
                           test_repro_fun=test_repro_fun,
                           strategy=reducer.DropFunctionCallsStrategy)
    r2, _ = ddmin.ddmin(r, chunk_size=len(r.all_candidates) // 2)
    self.assertNotIn("jax_repro_collect()", r2.repro_source)
    self.assertNotIn("fun_g_", r2.repro_source)
    self.assertNotIn("primitive_bind(\"sin\")", r2.repro_source)
    self.assertNotIn("primitive_bind(\"tan\")", r2.repro_source)

  def test_inline_cond(self):
    src = """
import jax
from jax import lax
from jax import numpy as jnp

def main_repro():
  x = jnp.array([0, 1, 2], dtype=jnp.int32)
  y = jnp.array([10, 11, 12], dtype=jnp.int32)

  def true_branch(x):
    return jnp.cos(x)
  def false_branch(x):
    return jnp.sin(x)
  return lax.cond(x[0] >= 0, true_branch, false_branch, y)
"""
    col = emitter.collector(repro.load(src, pathlib.Path("<here>")))
    col()
    collect_inlining = reducer.FunctionInlineStrategy(None)
    source1 = col.to_source(strategy=collect_inlining)
    true_branches = [(c, *rest)
                     for c, *rest in collect_inlining.all_candidates
                     if tracker.func_api_name(c.func) == "jax_cond" and rest[0] == 0]
    self.assertLen(true_branches, 1)

    self.assertIn("jax_primitive_bind(\"sin\")", source1)
    inline_true_branch = reducer.FunctionInlineStrategy(true_branches)
    source2 = col.to_source(strategy=inline_true_branch)
    self.assertNotIn("jax_primitive_bind(\"sin\")", source2)

    # Now drop the false branch
    self.assertIn("jax_primitive_bind(\"cos\")", source1)
    false_branches = [(c, *rest)
                      for c, *rest in collect_inlining.all_candidates
                      if tracker.func_api_name(c.func) == "jax_cond" and rest[0] == 1]
    self.assertLen(false_branches, 1)
    inline_false_branch = reducer.FunctionInlineStrategy(false_branches)
    source3 = col.to_source(strategy=inline_false_branch)
    self.assertNotIn("jax_primitive_bind(\"cos\")", source3)

  def test_inline_jit(self):
    src = """
import jax
from jax import numpy as jnp

def main_repro():
  @jax.jit
  def my_f(x, y):
    jnp.tan(y)  # to drop
    return x + y

  x = jnp.array([0, 1, 2], dtype=jnp.int32)
  y = jnp.array([10, 11, 12], dtype=jnp.int32)
  return jax.jit(my_f)(x, y)
"""
    col = emitter.collector(repro.load(src, pathlib.Path("<here>")))
    col()
    collect_inlining = reducer.FunctionInlineStrategy(None)
    source1 = col.to_source(strategy=collect_inlining)

    self.assertIn("jax_jit_call(fun_my_f", source1)
    jit_my_f = [(c, *rest)
                 for c, *rest in collect_inlining.all_candidates
                 if tracker.func_api_name(c.func) == "jax_jit_call"
                    and c.args[0].fun_name.startswith("my_f")]
    self.assertLen(jit_my_f, 2)  # nested jax.jit for my_f
    inline_jit_my_f = reducer.FunctionInlineStrategy(jit_my_f)
    source4 = col.to_source(strategy=inline_jit_my_f)
    self.assertNotIn("jax_jit_call(fun_my_f", source4)

  def test_inline_checkpoint(self):
    src = """
import jax
from jax import numpy as jnp

def main_repro():
  def my_f(x, y):
    return jnp.sin(x) + y

  x = jnp.array([0, 1, 2], dtype=jnp.float32)
  y = jnp.array([10, 11, 12], dtype=jnp.float32)
  return jax.checkpoint(my_f)(x, y)
"""
    col = emitter.collector(repro.load(src, pathlib.Path("<here>")))
    col()
    collect_inlining = reducer.FunctionInlineStrategy(None)
    source1 = col.to_source(strategy=collect_inlining)

    self.assertIn("jax_checkpoint_call(fun_my_f", source1)
    checkpoint_my_f = [(c, *rest)
                 for c, *rest in collect_inlining.all_candidates
                 if tracker.func_api_name(c.func) == "jax_checkpoint_call"
                    and c.args[0].fun_name.startswith("my_f")]
    self.assertLen(checkpoint_my_f, 1)
    inline_checkpoint_my_f = reducer.FunctionInlineStrategy(checkpoint_my_f)
    source2 = col.to_source(strategy=inline_checkpoint_my_f)
    self.assertNotIn("jax_checkpoint_call(fun_my_f", source2)

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
      col = emitter.collector(repro.load(src, pathlib.Path("<here>")))
      col()
      collect_reductions = reducer.DropExpressionsStrategy(None)
      source1 = col.to_source(strategy=collect_reductions)
      args_to_jit_call_my_f = [
         c
         for c in collect_reductions.all_candidates
         if (tracker.func_api_name(c[0].func) == "jax_jit_call" and
             "my_f" in c[0].args[0].fun_name and
             c[1]  # for_args
             )]
      # 7 = jax_jit_call(fun_my_f_3, None, {}, v_0, v_1, {'x': v_0, 'y': v_1})
      # But we don't include callables
      self.assertLen(args_to_jit_call_my_f, 9)
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
    return x + y  # Error: incompatible shapes for broadcasting

  x = jnp.array([1, 2, 3], dtype=jnp.int32)
  y = jnp.array([11, 12], dtype=jnp.int32)
  return jax.jit(f)(x, y)

if __name__ == "__main__":
  main_repro()
"""
    def test_repro_fun(repro_fun):
      try:
        repro_fun()
      except TypeError as e:
        return "add got incompatible shapes for broadcasting" in str(e)
      return False

    r = reducer.Repro.make(src, pathlib.Path("<here>"),
                           test_repro_fun=test_repro_fun,
                           strategy=reducer.DropFunctionCallsStrategy)
    r2, _ = ddmin.ddmin(r, chunk_size=len(r.all_candidates) // 2)
    self.assertNotIn("fun_g_", r2.repro_source)
    self.assertNotIn("primitive_bind(\"sin\")", r2.repro_source)
    self.assertNotIn("primitive_bind(\"tan\")", r2.repro_source)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
