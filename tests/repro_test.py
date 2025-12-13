# Copyright 2025 The JAX Authors.
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

import contextlib
import dataclasses
import gc
import pathlib
from functools import partial

import concurrent
import logging
import math
import numpy as np
from typing import Any, Callable
import weakref

from absl.testing import absltest

import jax
from jax import export
from jax import lax
from jax import numpy as jnp
from jax import checkpoint as new_checkpoint
from jax.experimental import hijax
from jax.experimental import pallas as pl
from jax.experimental import pjit
from jax.experimental.pallas import fuser
from jax.experimental.pallas import tpu as pltpu
from jax.experimental import shard_map as exp_shard_map
from jax.sharding import PartitionSpec as P
from jax.sharding import AxisType

from jax._src import config
from jax._src import dtypes
from jax._src import hashable_array
from jax._src import literals
from jax._src import repro
from jax._src.repro import tracker
from jax._src.repro import emitter


from jax._src import test_util as jtu
from jax._src import traceback_util
from jax._src import tree_util


config.parse_flags_with_absl()
jtu.request_cpu_devices(8)

intx = dtypes.default_int_dtype()
floatx = dtypes.default_float_dtype()


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class Value:
  a: float
  b: float

@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class CallableValue:
  a: float
  b: float

  def __call__(self):
    return self.a + self.b


def flatten_custom_pytree(t) -> Any:
  # The repro results contain custom pytree nodes flattened to a tuple. We
  # do the same to the direct results so that we can compare them.
  if isinstance(t, tuple) and not hasattr(t, "_fields"):  # Not a NamedTuple
    return tuple(flatten_custom_pytree(e) for e in t)
  if isinstance(t, list):
    return [flatten_custom_pytree(e) for e in t]
  if isinstance(t, dict):
    return {k: flatten_custom_pytree(e) for k, e in t.items()}
  if isinstance(t, tree_util.Partial):
    return t
  if t is None:
    return t
  # We want to eliminate the custom pytrees, but not the ones that JAX uses
  # internally.
  leaves = tree_util.tree_leaves(t)
  return leaves[0] if len(leaves) == 1 else leaves


@contextlib.contextmanager
def missing_lax_Precision_emitter():
  tracker.lazy_init()
  # Pretend we don't have a lax.Precision emitter
  old_lax_Precision_emitter = emitter._operand_emitter_by_type[lax.Precision]
  try:
    del emitter._operand_emitter_by_type[lax.Precision]
    yield
  finally:
    emitter._operand_emitter_by_type[lax.Precision] = old_lax_Precision_emitter


@jtu.with_config(jax_traceback_filtering="off")
class EmitterTest(jtu.JaxTestCase):
  def setUp(self):
    if not traceback_util.repro_is_enabled():
      self.skipTest("JAX_REPRO_DIR not set")
    tracker._thread_local_state = tracker._ThreadLocalState()
    tracker._thread_local_state.initialize_state_if_needed()
    super().setUp()

  def test_traverse_0(self):
    ctx = emitter.EmitFunctionDefContext(emitter.EmitGlobalContext(),
                                  None)
    self.assertEqual("0", ctx.traverse_value(0))
    self.assertEqual("()", ctx.traverse_value(()))
    self.assertEqual("(0,)", ctx.traverse_value((0,)))
    self.assertEqual("{}", ctx.traverse_value({}))
    self.assertEqual("(0, 1, {'a': 3},)",
                     ctx.traverse_value((0, 1, dict(a=3))))


  def test_traverse_collect_expressions(self):
    self.skipTest("Broken")
    class MyCollect(emitter.EmitReductionStrategy):
      def __init__(self, to_drop: set[tuple[int, int]]):
        self.collection = []
        self.to_drop = to_drop

      def keep_expression(self, c: tracker.Call, for_args: bool, idx: int, v: Any) -> bool:
        self.collection.append((c, for_args, idx, v))
        return (c.id, idx) not in self.to_drop

    col = MyCollect(set())
    ctx = emitter.EmitFunctionDefContext(emitter.EmitGlobalContext(strategy=col),
                                  None)
    c = tracker.Call(tracker._thread_local_state.call_stack[-1],
                     lax.sin_p, (), {})
    value = (0, 1, dict(b=4, a=3), 5)
    ctx.global_ctx.set_current_traverse_value_context(c, True)
    self.assertEqual("(0, 1, {'a': 3, 'b': 4}, 5,)",
                     ctx.traverse_value(value))
    self.assertLen(col.collection, 7)

    drop_all = MyCollect({(c.id, 0)})
    ctx = emitter.EmitFunctionDefContext(emitter.EmitGlobalContext(strategy=drop_all),
                                  None)
    ctx.global_ctx.set_current_traverse_value_context(c, True)
    self.assertEqual(None, ctx.traverse_value(value))

    drop_1 = MyCollect({(c.id, 2)})
    ctx = emitter.EmitFunctionDefContext(emitter.EmitGlobalContext(strategy=drop_1),
                                  None)
    ctx.global_ctx.set_current_traverse_value_context(c, True)
    self.assertEqual("(0, {'a': 3, 'b': 4}, 5,)",
                     ctx.traverse_value(value))

    drop_3 = MyCollect({(c.id, 4)})
    ctx = emitter.EmitFunctionDefContext(emitter.EmitGlobalContext(strategy=drop_3),
                                  None)
    ctx.global_ctx.set_current_traverse_value_context(c, True)
    self.assertEqual("(0, 1, {'b': 4}, 5,)",
                     ctx.traverse_value(value))

    drop_3_and_4 = MyCollect({(c.id, 4), (c.id, 5)})
    ctx = emitter.EmitFunctionDefContext(emitter.EmitGlobalContext(strategy=drop_3_and_4),
                                  None)
    ctx.global_ctx.set_current_traverse_value_context(c, True)
    self.assertEqual("(0, 1, {}, 5,)",
                     ctx.traverse_value(value))



@jtu.with_config(jax_traceback_filtering="off",
                 jax_enable_checks=True)
class ReproTest(jtu.JaxTestCase):

  def setUp(self):
    if not traceback_util.repro_is_enabled():
      self.skipTest("JAX_REPRO_DIR not set")
    tracker._thread_local_state = tracker._ThreadLocalState()
    tracker._thread_local_state.initialize_state_if_needed()
    super().setUp()

  def assert_empty_call_stack(self):
    self.assertLen(tracker._thread_local_state.call_stack, 0)

  def collect_and_check(self, func: Callable, *args,
                        expect_exception: tuple[type[Exception], str] | None = None,
                        skip_repro_read: bool = False,
                        skip_repro_eval: bool = False,
                        atol=None, rtol=None,
                        **kwargs) -> str:
    direct_result = None
    if expect_exception:
      context = self.assertRaisesRegex(* expect_exception)
    else:
      context = contextlib.nullcontext()

    if not args and not kwargs:
      to_repro = func  # Preserves source info
    else:
      # TODO: move this to the individual tests
      to_repro = lambda: func(*args, **kwargs)
    with context:
      col = repro.collector(to_repro)
      try:
        direct_result = col()
      finally:
        col.to_source(repro_name_prefix=f"{self._testMethodName.removeprefix('test_')}_repro")

    if skip_repro_read:
      return None
    repro_path, repro_source = repro.last_saved_repro()
    if skip_repro_eval:
      return repro_source

    with tracker.enable(False):
      main_repro = repro.load(repro_source, repro_path)
      with context:
        repro_result = main_repro()


    if not expect_exception:
      self.assertAllClose(repro_result,
                          flatten_custom_pytree(direct_result),
                          atol=atol, rtol=rtol)
    return repro_source

  def test_basic(self):
    @jax.jit
    def f1(x, y1, y2):
      v = x + jnp.sin(y1)
      return v + jnp.cos(y2)

    @jax.jit
    def f2():
      x = np.ones((8,), dtype=np.float32)
      return jax.jit(f1)(x, x, y2=x)

    self.collect_and_check(f2)

  def test_normalize_0(self):
    emitter.initialize_operand_emitter()
    nctx = tracker.NormalizerContext()
    self.assertEqual(0, nctx.normalize_value(0, True))
    self.assertEqual((0, [1, 2]), nctx.normalize_value(Value(0, [1, 2]), True))
    self.assertEqual((0, [1, (3, 4)]), nctx.normalize_value(Value(0, [1, Value(3, 4)]), True))
    v = lax.DotDimensionNumbers(((0, 0), True))
    self.assertEqual(v, nctx.normalize_value(v, True))

    # from jax._src import core
    # from jax._src import api_util
    # from jax._src.interpreters import partial_eval as pe
    # trace = pe.DynamicJaxprTrace(api_util.debug_info("test", lambda: 0, (), {}))
    # t1 = pe.DynamicJaxprTracer(trace, core.ShapedArray((), dtype=np.float32), 1.)
    # t2 = pe.DynamicJaxprTracer(trace, core.ShapedArray((), dtype=np.float32), 2.)

    # with tracker.flags_override(error_mode="raise"):
    #   with self.assertRaisesRegex(repro.ReproError, "Normalizing"):
    #     _ = nctx.normalize_value(t1, True)

    #   # Define it now
    #   nt1 = nctx.normalize_value(t1, False)
    #   self.assertIsInstance(nt1, tracker.NormTracer)

    #   nt11 = nctx.normalize_value(t1, True)
    #   self.assertIs(nt1, nt11)

    #   nt2 = nctx.normalize_value(t2, False)
    #   nt21 = nctx.normalize_value(t2, True)
    #   self.assertIs(nt2, nt21)

    #   nt12 = nctx.normalize_value(t1, True)
    #   self.assertIs(nt1, nt12)


  def test_normalize_weakref_dict(self):
    k1 = np.array(5, dtype=np.int32)
    k2 = np.array(6, dtype=np.int32)
    d = tracker.WeakUnhashableKeyDictionary()
    d[k1] = 1
    d[k2] = 2

    v1 = d[k1]
    self.assertEqual(v1, 1)
    self.assertEqual(d.get(k1), 1)
    self.assertEqual(d.get(k2), 2)
    self.assertLen(d.keys, 2)
    del k1
    self.assertLen(d.keys, 1)

  def test_implicit_collect_success(self):
    def f1(x):

      @jax.jit
      def nested(x):
        self.assertLen(tracker._thread_local_state.call_stack, 2)  # jit(nested, x), nested
        self.assertFalse(tracker._thread_local_state.call_stack[0].func.is_user)
        self.assertTrue(tracker._thread_local_state.call_stack[1].func.is_user)
        self.assertEqual(tracker._thread_local_state.call_stack[1].func.fun_name, "nested")
        return jnp.sin(x)
      drop1 = nested(x)
      self.assertEmpty(tracker._thread_local_state.call_stack)
      drop2 = lax.cond(x[0] > 0., lambda: jnp.sin(x), lambda: jnp.cos(x))
      self.assertEmpty(tracker._thread_local_state.call_stack)
      drop3 = lax.sin(x)  # A primitive call
      self.assertEmpty(tracker._thread_local_state.call_stack)

    x = np.ones((4,), dtype=np.float32)
    f1(x)
    self.assertEmpty(tracker._thread_local_state.call_stack)
    self.assertIsNone(repro.last_saved_repro())

  def test_explicit_collect_success(self):
    y = jnp.cos(42.)  # This will be in main, but should be ignored: outside collect

    @jax.jit
    def f1(x):
      return jnp.sin(x) + y

    def f2():
      x = np.float32(1.)
      y = lax.tan(x)  # Direct primitive invocation, must be kept
      z = f1(y) + 1.
      return 2.

    repro_source = self.collect_and_check(f2)
    self.assertEmpty(tracker._thread_local_state.call_stack)
    self.assertIn("sin", repro_source)
    self.assertIn("tan", repro_source)
    self.assertNotIn("cos", repro_source)

  @jtu.parameterized_filterable(
      kwargs=[
        dict(mode=mode)
        for mode in ["implicit", "explicit"]
      ])
  def test_collect_on_primitive_error(self, mode: str):
    x = jnp.ones((4,), dtype=np.float32)  # Don't keep, outside collect
    y = np.ones((5,), dtype=np.float32)
    self.assertEmpty(tracker._thread_local_state.call_stack)
    def f1():
      z = lax.tan(x)  # keep if in explicit mode
      if mode == "explicit":
        self.assertLen(tracker._thread_local_state.call_stack, 2)
        self.assertLen(tracker._thread_local_state.call_stack[1].body, 1)
        self.assertEqual(tracker._thread_local_state.call_stack[1].body[0].func.name, "tan")
      else:
        self.assertEmpty(tracker._thread_local_state.call_stack)
      return x + y  # Cannot broadcast together

    if mode == "explicit":
      repro_source = self.collect_and_check(
          f1, expect_exception=(TypeError, "incompatible shapes"))
    else:
      with self.assertRaisesRegex(TypeError, "incompatible shapes"):
        f1()
      repro_path, repro_source = repro.last_saved_repro()
      main_func = repro.load(repro_source, repro_path)
      with self.assertRaisesRegex(TypeError, "incompatible shapes"):
        with repro.enable(False):
          main_func()

    self.assertEmpty(tracker._thread_local_state.call_stack)
    if mode == "implicit":
      # TODO: for explit mode, we do not include the traceback on error
      self.assertIn("got incompatible shapes", repro_source)
    self.assertNotIn("\"broadcast_in_dim\"", repro_source)
    if mode == "explicit":
      self.assertIn("\"tan\"", repro_source)
    else:
      self.assertNotIn("\"tan\"", repro_source)
    self.assertIn("\"add\"", repro_source)

  def test_collect_cleanup_main_body_mode_implicit(self):
    # TODO: perhaps we should actually keep the function-generating calls
    inp = np.arange(4)

    # A "main" call that produces some arrays
    jnp.sin(inp)
    # Now a call that produces a function
    y, cos_jvp = jax.linearize(jnp.cos, 5.)
    # A call with an error, triggers saving on error
    try:
      lax.scan(lambda x: 0., None, None, length=2)
    except TypeError:
      pass

    self.assertLen(tracker._thread_local_state.call_stack, 0)
    _, repro_source = repro.last_saved_repro()
    self.assertIn("jax.lax.scan", repro_source)

  @jtu.parameterized_filterable(
      kwargs=[
        dict(error_mode=error_mode, mode=mode)
        for error_mode in ["defer", "raise", "log"]
        for mode in ["implicit", "explicit"]
      ])
  def test_collect_error_forgotten_api_boundary(self, *, error_mode: str, mode: str):
    # Simulate forgetting to declare a repro_boundary. We use
    # _initial_style_jaxpr to set up tracing of a user function.
    def f(x):
      return x + x
    from jax._src import api_util
    from jax._src.interpreters import partial_eval as pe
    from jax._src import core
    aval = core.ShapedArray((), dtype=np.float32)
    avals, avals_tree = tree_util.tree_flatten(((aval,), {}))
    def doit():
      pe.trace_to_jaxpr(f, tree_util.FlatTree(avals, avals_tree),
                        api_util.debug_info("test", f, (5.,), {}))

    # with deferred errors
    with tracker.flags_override(log_traceback_frames=40, error_mode=error_mode):
      with self.assertLogs(level=logging.ERROR) as logs:
        if error_mode == "log":
          if mode == "implicit":
            doit()
          else:
            self.collect_and_check(doit, skip_repro_eval=True)
        else:
          if error_mode == "defer":
            expect_msg = "There were errors during repro source generation"
          else:
            expect_msg = "Expected that the call .* is to a USER function"
          if mode == "implicit":
            if error_mode == "raise":
              with self.assertRaisesRegex(repro.ReproError, expect_msg):
                doit()
            else:
              doit()  # In implicit mode, defer does not raise exception !!!
          else:
            self.collect_and_check(doit, skip_repro_eval=True,
                                   expect_exception=(repro.ReproError, expect_msg))
      log_output = "\n".join(logs.output)
      self.assertRegex(log_output, r"Repro error: Expected that the call .* is to a USER function")


  @jtu.parameterized_filterable(
      kwargs=[
        dict(error_mode=error_mode, mode=mode)
        for error_mode in ["defer", "raise", "log"]
        for mode in ["implicit", "explicit"]
      ])
  @jtu.thread_unsafe_test()  # We override the emitters
  def test_collect_error_missing_custom_emitter(self, *, error_mode: str, mode: str):
    x = jnp.ones((8, 8), dtype=np.float32)
    y = jnp.ones((2,), dtype=np.float32)
    @jax.jit
    def my_func_with_distinct_name():
      return jnp.dot(x, x, precision=lax.Precision.HIGHEST)

    if error_mode == "defer":
      expect_msg = "There were errors during repro source generation"
    else:
      expect_msg = "Undefined g_0 = HIGHEST of type .* without custom emitter"

    # Pretend we don't have a lax.Precision emitter. This results in ReproError
    # but only during repro source generation, not during collection.
    with (missing_lax_Precision_emitter()):
      with tracker.flags_override(log_traceback_frames=40,
                                  error_mode=error_mode):
        if mode == "implicit":
          my_func_with_distinct_name()  # No errors during implicit collection

          with tracker.flags_override(check_repro_emit=True):
            with self.assertLogs(level=logging.ERROR) as logs:
              if error_mode == "log":
                my_func_with_distinct_name()
              else:
                with self.assertRaisesRegex(repro.ReproError, expect_msg):
                  my_func_with_distinct_name()

        else:
          with self.assertLogs(level=logging.ERROR) as logs:
            if error_mode == "log":
              self.collect_and_check(my_func_with_distinct_name, skip_repro_eval=True)
            else:
              self.collect_and_check(my_func_with_distinct_name, skip_repro_eval=True,
                                     expect_exception=(repro.ReproError, expect_msg),
                                     skip_repro_read=(error_mode == "raise"))

        log_output = "\n".join(logs.output)
        self.assertRegex(log_output, r"Repro error: Undefined .* = HIGHEST")
        if error_mode != "raise":
          self.assertIn("<locals>.my_func_with_distinct_name", log_output)
          self.assertIsNotNone(repro.last_saved_repro())
          repro_path, repro_source = repro.last_saved_repro()
          self.assertIn("def fun_my_func_with_distinct_name", repro_source)
          self.assertRegex(repro_source, "# Undefined g_.* = HIGHEST of type")
        else:
          self.assertIsNone(repro.last_saved_repro())

  def test_repro_at_top_level_only(self):
    @jax.jit
    def f():
      pass

    @jax.jit
    def in_jax():
      repro.collector(f)()

    with self.assertRaisesRegex(ValueError, "only be used outside"):
      in_jax()

  def test_save_repro_reproducible(self):
    def f1():
      return jnp.ones((4,), dtype=np.float32)

    col1 = repro.collector(f1)
    col1()
    src1 = col1.to_source()

    col2 = repro.collector(f1)
    col2()
    src2 = col2.to_source()
    self.assertEqual(src1, src2)

  def test_save_repro_idempotent(self):
    def f1():
      return jnp.ones((4,), dtype=np.float32)

    col1 = repro.collector(f1)
    col1()
    src1 = col1.to_source()

    col2 = repro.collector(repro.load(src1, pathlib.Path("<memory>")))
    col2()
    src2 = col2.to_source()  # src2 will have different source info

    col3 = repro.collector(repro.load(src2, pathlib.Path("<memory>")))
    col3()
    src3 = col3.to_source()
    self.assertEqual(src2, src3)

  def test_multiple_save_repro(self):
    def f1():
      return jnp.ones((4,), dtype=np.float32)

    c1 = repro.collector(f1)
    c1()
    src1 = c1.to_source()
    self.assertIn("broadcast_in_dim", src1)

    def f2():
      return jnp.arange(4, dtype=np.float32)
    c2 = repro.collector(f2)
    c2()
    src2 = c2.to_source()
    self.assertNotIn("broadcast_in_dim", src2)
    self.assertIn("iota", src2)

  def test_function_ids(self):
    def f1():
      x = np.ones(2, dtype=np.float32)
      idx0 = next(tracker._thread_local_state.func_index)
      self.assertEqual(1, idx0)  # 0 is is f1
      branch = lambda: x
      lax.cond(True, branch, branch)
      # + 2 branch, + lax.cond
      idx1 = next(tracker._thread_local_state.func_index)
      self.assertEqual(idx0 + 3, idx1)
      # + 2 branch, + lax.cond
      lax.cond(True, branch, branch)
      idx2 = next(tracker._thread_local_state.func_index)
      self.assertEqual(idx1 + 3, idx2)

    self.collect_and_check(f1)

  def test_no_return(self):
    @jax.jit
    def f1(x):
      return jnp.sin(x)

    def f2(x):
      f1(x)  # No explicit return from the main function

    self.collect_and_check(f2, np.float32(1.))


  @jtu.thread_unsafe_test()  # Weakref destruction seems unpredictable with threads
  def test_no_leak_static(self):
    @jax.jit(static_argnums=(0,))
    def f(x_static):
      return x_static
    x = Value(1., 2.)
    x_wr = weakref.ref(x)
    self.assertIsNotNone(x_wr())
    f(x)
    del x
    del f
    gc.collect()
    self.assertIsNone(x_wr())

  @jtu.thread_unsafe_test()  # Weakref destruction seems unpredictable with threads
  def test_no_leak_array(self):
    x = [np.array(v, dtype=np.float32) for v in range(5)]
    x_wr = [weakref.ref(v) for v in x]

    @jax.jit(static_argnums=(1,))
    def f(y, y_static):  # Ensure that calls to f do not hold on to arrays
      def h():
        return x[2]  # noqa: F821
      return (y + jax.jit(h)() + lax.sin(x[3]), x[4])  # type: ignore # noqa: F821

    f(x[0], hashable_array.HashableArray(x[1]))
    del x
    gc.collect()
    for wr in x_wr:
      if wr() is not None:
        from jax._src import core
        raise ValueError(core._why_alive({id(wr)}, wr()))
      self.assertIsNone(wr())

  @jtu.thread_unsafe_test()  # Weakref destruction seems unpredictable with threads
  def test_no_leak_tracer(self):
    self.skipTest("disabled tracer normalization for now")
    with jax.checking_leaks():
      @jax.jit
      def f(x):
        return x
      @jax.jit
      def g(x):
        return f(x) + f(x)
      x = np.array(1.)
      g(x)

  def test_user_func_side_effects(self):
    op_list = [1., 2., 3., 4.]
    op_dict = dict(a=5.)

    @jax.jit
    def f(op_list: list[Any]):
      op1 = op_list.pop()  # We modify the arguments, ensure repro generation
                           # sees the original
      op2 = op_list.pop()
      op3 = op_dict["a"]
      return dict(c=op1 + op2 + op3)

    @jax.jit
    def g():
      r1 = f(op_list)
      r1["d"] = r1["c"] + 1.
      r2 = f(op_list)
      return r1["c"] * r2["c"]
    self.collect_and_check(g)

  def test_duplicate_arg(self):
    @jax.jit
    def f(x, y):
      return x + y
    @jax.jit
    def g(x):
      return f(x, x)

    self.collect_and_check(g, np.ones((4,), dtype=np.float32))

  def test_no_pos_args(self):
    @jax.jit
    def fun(*, a, b):
      return a + b

    self.collect_and_check(fun, a=1, b=2)

  def test_multiple_calls_0(self):
    # Same body multiple invocations with the same shapes
    @jax.jit
    def f1(x):
      return jnp.sin(x)

    @jax.jit
    def f2(x):
      a = f1(x)
      b = f1(x)
      return a + b

    self.collect_and_check(f2, np.float32(1.))

  def test_multiple_calls_1(self):
    # Body changes based on shape
    @jax.jit
    def f1(x):
      return jnp.sin(x) if x.shape else jnp.cos(x)

    @jax.jit
    def f2(x, xlarge):
      a = f1(x)
      alarge = f1(xlarge)
      b = f1(x)
      blarge = f1(xlarge)
      return (a + b, alarge + blarge)

    self.collect_and_check(f2, np.float32(1.), np.arange(4.))

  def test_multiple_calls_2(self):
    # We invoke a function multiple times, but it results in the same
    # body
    def f1(x):
      return jnp.sin(x)

    @jax.jit
    def f2(x):
      j_f1 = jax.jit(jax.jit(jax.jit(f1)))
      return (j_f1(x), j_f1(x))

    self.collect_and_check(f2, np.float32(1.))

  def test_multiple_calls_different_body(self):
    # We invoke a function multiple times, but it results in the same
    # body
    @jax.jit
    def f1(x):
      # called first with shape [] and then with shape [8]
      # In repro should be turned into two functions
      return jnp.sin(x) if x.shape else jnp.cos(x)

    @jax.jit
    def f2(x):
      return x + f1(x)

    @jax.jit
    def f3(x):
      # We call twice for each shape
      two_shapes = [x, jnp.full(x.shape + (8,), 5.)] * 2
      calls = [f2(v) for v in two_shapes]
      return sum(calls)

    self.collect_and_check(f3, np.float32(1.))

  def test_user_calls_user(self):
    @jax.jit(static_argnums=(1,))
    def f1(x, other_f: Callable):
      return other_f(x)
    def f2(x):
      return jnp.sin(x)
    self.collect_and_check(f1, 42., f2)

  def test_user_function_cache_hit(self):
    @jax.jit
    def f1(x):
      return jnp.sin(x)

    @jax.jit
    def f2(x):
      v1 = f1(x)
      v2 = f1(x + 1.)
      return v1 + v2

    self.collect_and_check(f2, 42.)

  def test_nested_10(self):
    @jax.jit
    def f1(x1):
      return x1 + x1
    @jax.jit
    def f2(x2):
      return f1(x2)  # closes over a JAX function
    self.collect_and_check(f2, 42.)

  def test_nested_11(self):
    def f1(x1):
      return x1 + x1
    @jax.jit
    def f2(x2):
      return jax.jit(f1)(x2)  # closes over a USER function
    self.collect_and_check(f2, 42.)

  def test_nested_20(self):
    @jax.jit
    def f1(x1):
      v1 = x1 + x1
      def f2(x2):  # goes under "f1" because it uses "v1"
        v3 = x2 + x2
        def f3(x3):  # goes under "main"
          return x3 + x3
        def f4(x4):  # goes under "f1" due to "v1"
          return x4 + v1  # goes under "f1"
        def f5(x5):  # goes under "f2" due to "v3"
          return x5 + v3
        return v1 + x2 + jax.jit(f3)(v1) + jax.jit(f4)(v1) + jax.jit(f5)(v1)
      return jax.jit(f2)(x1)
    self.collect_and_check(f1, 42.)

  def test_nested_30(self):
    @jax.jit
    def f(x0, x1, y):
      @jax.jit
      def g(x1, y):
        def body(c, z):
          # x0 and x1 are external, from different levels
          return c + z, c + x0 + x1
        def h(y1):
          return lax.scan(body, 42., y1)
        return jax.vmap(h)(y)
      return g(x1, y)

    x0 = np.float32(3.)
    x1 = np.float32(4.)
    y = np.ones((8, 4), dtype=np.float32)
    self.collect_and_check(f, x0, x1, y)

  def test_different_x64(self):
    @jax.jit
    def f(x):
      return x * 5.

    self.collect_and_check(f, np.ones((4,), dtype=np.float32))


  def test_partial_arg_to_function(self):
    # Partial used in user-space.
    def my_fun(x, y):
      return x + y
    add_one = tree_util.Partial(my_fun, 5)
    @jax.jit
    def call_func(f):
      return f(2) + 5
    self.collect_and_check(call_func, add_one)

  def test_mutable_args_results(self):
    @jax.jit
    def f(arg_dict):
      arg_dict["new_arg"] = arg_dict["a"] + 1.
      return dict(b=arg_dict["a"] + 2.)

    @jax.jit
    def f2():
      x = np.ones((4,), dtype=np.float32)
      res1 = f(dict(a=x))
      res1["c"] = x + 1.
      return res1
    self.collect_and_check(f2)

  def test_using_jax_tree(self):
    @jax.jit
    def fn(x):
      return jax.tree.map(lambda i: i * 2, x)

    self.collect_and_check(fn, {"A": np.array(1.0), "B": np.array(2.0)})

  def test_pytree(self):
    @jax.jit
    def f1(x_dict, y_pair):
      y1, y2 = y_pair
      v = x_dict["a"] + jnp.sin(y1)
      return dict(res=v + jnp.cos(y2))

    @jax.jit
    def f2(x):
      return f1({"a": x}, y_pair=(x , x))

    self.collect_and_check(f2, np.ones((8,), dtype=np.float32))

  def test_pytree_non_identifier_keys(self):
    @jax.jit
    def f1(x_dict, y_pair):
      y1, y2 = y_pair
      v = x_dict["a"] + jnp.sin(y1)
      return {"some/string": v + jnp.cos(y2), "with spaces": jnp.sin(y2)}

    @jax.jit
    def f2(x):
      return f1({"a": x}, y_pair=(x , x))

    self.collect_and_check(f2, np.ones((8,), dtype=np.float32))

  @jtu.parameterized_filterable(
      kwargs=[
        dict(new_consts=new_consts)
        for new_consts in [False, True]
      ])
  def test_with_constants(self, *, new_consts: bool):
    ct1 = jnp.arange(16., dtype=np.float32)
    @jax.jit
    def f1(x):
      ct2 = np.arange(16., dtype=np.float32)
      return x + ct1 + ct2

    with config.use_simplified_jaxpr_constants(new_consts):
      self.collect_and_check(f1, np.ones(shape=(16,), dtype=np.float32))

  def test_numpy_dtypes(self):
    @jax.jit
    def f():
      x = np.array([0.], dtype=np.float32)
      return lax.convert_element_type(x, new_dtype=np.int32)

    self.collect_and_check(f)

  def test_printing_constants(self):
    x = np.zeros((16, 16, 16), dtype=np.int32)
    @jax.jit
    def f(x):
      return jnp.sum(x)

    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(f, x)

  def test_shared_constants(self):

    def f():
      const = jnp.arange(4, dtype=np.float32)
      return (const, const)

    repro_source = self.collect_and_check(f)
    res = repro.load(repro_source, pathlib.Path(""))()
    self.assertIs(res[0], res[1])


  @jtu.parameterized_filterable(
      kwargs=[dict(dtype=dt,
                   testcase_name=jtu.dtype_str(dt))
              for dt in jtu.supported_dtypes()]
  )
  def test_constants_dtype(self, *, dtype):
    if "bool" in str(dtype): self.skipTest("n/a for bool")
    @jax.jit
    def f(x):
      return x + 1, lax.convert_element_type(1, new_dtype=dtype)

    self.collect_and_check(f, dtype(2))

  def test_literals(self):
    @jax.jit
    def f():
      return (literals.TypedInt(42, dtype=jnp.int32),
              literals.TypedFloat(42, dtype=jnp.float32),
              literals.TypedComplex(42, dtype=jnp.complex64),
              literals.TypedNdArray(np.array(42, dtype=np.int32), weak_type=True))


    self.collect_and_check(f)

  def test_partial_arg_to_jit(self):
    # Partial used in user-space.
    def my_fun(x, y):
      return x + y
    add_one = tree_util.Partial(my_fun, 5)
    self.collect_and_check(lambda: jax.jit(add_one)(2))

  def test_jit_with_callable_custom(self):
    arr = jnp.arange(16).reshape(8, 2)

    @jax.jit
    def g(value):
      return CallableValue(jnp.sin(value.a), jnp.cos(value()))

    self.collect_and_check(g, CallableValue(arr, arr))

  def test_jit_with_kwargs_0(self):
    # kwargs that are preserved for the user function
    @jax.jit
    def my_fun(x, y, k1, k2):
      return x + y + k1 + k2

    @jax.jit
    def f2(x):
      return my_fun(x, x, k1=x, k2=x)
    self.collect_and_check(f2, np.arange(6.))

  def test_jit_with_kwargs_1(self):
    # Pass as kwargs static arguments declared with static_argnums
    @partial(jax.jit, static_argnums=(1,))
    def _uniform(v, get_shape: Callable):
      shape = get_shape()
      return jnp.full(shape, v)

    @jax.jit
    def f2(x):
      return _uniform(x, get_shape=lambda: (2, 4))

    self.collect_and_check(f2, 5.)

  def test_jit_with_kwargs_2(self):
    # Pass as pos args static arguments declared with static_argnames
    @partial(jax.jit, static_argnames=("get_shape",))
    def _uniform(v, get_shape: Callable):
      shape = get_shape()
      return jnp.full(shape, v)

    @jax.jit
    def f2(x):
      return _uniform(x, lambda: (2, 4))

    self.collect_and_check(f2, 5.)

  def test_concurrent_jit(self):
    @jax.jit
    def f(x):
      return x + x - 3.

    xs = [self.rng().randn(i) for i in range(10)]
    with concurrent.futures.ThreadPoolExecutor() as executor:
      futures = [executor.submit(partial(f, x)) for x in xs]
      ys = [f.result() for f in futures]
    for x, y in zip(xs, ys):
      self.assertAllClose(x * 2 - 3., y)

  def test_jit_signature_fail(self):
    inp = np.arange(4)
    # inspect.get_signature fails for np.dot
    with self.assertRaisesRegex(jax.errors.TracerArrayConversionError,
                           "__array__"):
      jax.jit(np.dot)(inp, inp)

  def test_aot_trace_lower(self):
    @jax.jit
    def f1(x):
      return jnp.cos(x)
    x = np.ones((8, 8), dtype=np.float32)
    traced = f1.trace(x)
    self.assertIn("cos", str(traced.jaxpr))
    lowered1 = traced.lower()
    lowered2 = f1.lower(x)
    self.assertEqual(lowered1.as_text(), lowered2.as_text())

  def test_eval_shape(self):
    @jax.jit
    def f1(x, y1, y2):
      v = x + jnp.sin(y1)
      return v + jnp.cos(y2)

    def f2(x):
      return jax.jit(f1)(x, x, x)

    self.collect_and_check(lambda *args: jax.eval_shape(f2, *args),
                      np.ones((8,), dtype=np.float32))

  def test_export(self):
    @jax.jit
    def f(x):
      return jnp.sin(x)

    def run_export(x):
      exported = export.export(f)(x)
      return exported.call(x)

    self.collect_and_check(run_export,
                           np.ones((8,), dtype=np.float32))

  def test_pass_through_jit(self):
    def f(x, y):
      return x * y

    @jax.jit
    def g():
      primals = 2., 3.
      y, f_vjp = jax.experimental.saved_input_vjp(f, [True, True], *primals)
      return y, f_vjp

    @jax.jit
    def h(f_vjp):
      return f_vjp(1., 2., 3.)

    @jax.jit
    def top():
      y, f_vjp = g()
      return h(f_vjp)

    self.collect_and_check(top)

  @jtu.ignore_warning(category=DeprecationWarning,
                      message=".*pjit has been deprecated")
  def test_pjit(self):
    if jax.local_device_count() < 2:
      self.skipTest("Need at least 2 devices")

    mesh = jtu.create_mesh((2,), ("a",))
    s = jax.sharding.NamedSharding(mesh, P("a"))
    def top(x):
        @partial(pjit.pjit, in_shardings=(s,))
        def f(x):
          return x + x

        return f(x)

    x = np.ones((8,), dtype=np.float32)
    self.collect_and_check(top, x)

  @jtu.ignore_warning(category=DeprecationWarning,
                      message=".*pjit has been deprecated")
  def test_pjit_with_statics(self):
    if jax.local_device_count() < 2:
      self.skipTest("Need at least 2 devices")

    mesh = jtu.create_mesh((2,), ("a",))
    s = jax.sharding.NamedSharding(mesh, P("a"))
    def top(x):
      @partial(pjit.pjit, in_shardings=(s,), static_argnums=(1,))
      def f(x, get_shape):
        return x + x - jnp.ones(get_shape(), dtype=x.dtype)

      f.trace(x, lambda: x.shape)
      f.lower(x, lambda: x.shape)
      return f(x, lambda: x.shape)

    x = np.ones((8,), dtype=np.float32)
    self.collect_and_check(top, x)

  @jtu.ignore_warning(category=DeprecationWarning,
                      message=".*pjit has been deprecated")
  @jtu.ignore_warning(category=DeprecationWarning,
                      message="`with mesh:` context manager")
  def test_pjit_with_mesh(self):
    if jax.local_device_count() < 2:
      self.skipTest("Need at least 2 devices")

    mesh = jtu.create_mesh((2,), ("a",))

    def top(x):
      with mesh:
        @partial(pjit.pjit, in_shardings=(P("a"),))
        def f(x):
          return x + x

        return f(x)

    x = np.ones((8,), dtype=np.float32)
    self.collect_and_check(top, x)

  @jtu.ignore_warning(category=DeprecationWarning,
                      message=".*pjit has been deprecated")
  @jtu.ignore_warning(category=DeprecationWarning,
                      message="`with mesh:` context manager")
  def test_pjit_eval_shape(self):
    if jax.local_device_count() < 2:
      self.skipTest("Need at least 2 devices")

    mesh = jtu.create_mesh((2,), ("a",))

    @partial(pjit.pjit, in_shardings=(P("a"),))
    def f(x):
      return x + x

    x = np.ones((8,), dtype=np.float32)
    with jax.set_mesh(mesh):
      self.collect_and_check(lambda: jax.eval_shape(f, x))

  def test_jit_out_shardings_unconstrained(self):
    mesh = jtu.create_mesh((2, 2), ('x', 'y'))
    s = jax.sharding.NamedSharding(mesh, P('x', 'y'))
    np_inp = np.arange(16).reshape(8, 2)
    arr = jax.device_put(np_inp, s)

    out_s = jax.sharding.NamedSharding(mesh, P(P.UNCONSTRAINED,
                                               P.UNCONSTRAINED))
    @partial(jax.jit, out_shardings=out_s)
    def f(x):
      return x * 2

    self.collect_and_check(f, arr)

  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_jit_with_custom(self, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    s_x_y = jax.sharding.NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s_x_y)
    arr2 = jax.device_put(np_inp, s_x_y)

    @partial(jax.jit,
             # One sharding for the tuple passes as Value.a
             in_shardings=(Value(s_x_y, s_x_y),),
             out_shardings={"c": s_x_y})
    def g(xy_value):
      return dict(c=xy_value.a[0] * xy_value.a[1] * xy_value.b)

    self.collect_and_check(g, Value((arr, arr), arr2))

  def test_jit_with_mesh(self):
    np_inp = np.arange(16).reshape(8, 2)
    mesh = jtu.create_mesh((2, 2), ("x", "y"),
                           axis_types=(AxisType.Auto, AxisType.Auto))
    s_x_y = P("x", "y")

    @partial(jax.jit,
             in_shardings=(s_x_y, s_x_y),
             out_shardings=s_x_y)
    def g(x, y):
      return x + y

    def top(x, y):
      with jax.set_mesh(mesh):
        return g(x, y)

    self.collect_and_check(top, np_inp, np_inp)

  def test_fusion_with_jit_example(self):
    @fuser.fusible
    def fusible_sin(x_fn, y_fn):
      if y_fn is None:
        y_fn = lambda x: x
      @jax.jit
      def _impl():
        x = x_fn()
        out = jnp.sin(x)
        return y_fn(out)
      return _impl()

    @fuser.fuse
    def f(x):
      x = 2 * x
      out = fusible_sin(x)
      return out + 1.0

    x = np.ones( (128,), dtype=jnp.float32)
    self.collect_and_check(f, x)

  def test_jit_with_statics(self):
    @dataclasses.dataclass(frozen=True)
    class CustomType:  # This is a type that should not arise in the repro
      v: int

    @partial(jax.jit, static_argnums=(1, 3), static_argnames=("as2",))
    def f(x01, xs2, x3, xs4, a12, as2):
      x0, x1 = x01
      a1, a2 = a12
      self.assertEqual(xs2, CustomType(2))
      self.assertEqual(xs4, (40, 41, 42))
      self.assertEqual(as2, CustomType(22))
      return x0 + x1 + x3 + a1 + a2

    def jit_and_aot(*args, **kwargs):
      f.lower(*args, **kwargs)
      f.trace(*args, **kwargs)
      return f(*args, **kwargs)

    self.collect_and_check(jit_and_aot,(0., 1.), CustomType(2), 3., (40, 41, 42),
                           a12 = (5., 6.), as2 = CustomType(22))

  def test_jit_with_statics_and_donation(self):
    @partial(jax.jit, static_argnums=(0,), donate_argnums=(1,))
    def f(get_shape, x1_donated, x2):  # x1: f32[4], x2: f32[8]
      v = jnp.ones(get_shape(), dtype=x2.dtype)
      return v + x1_donated + x2[:4]

    x2 = np.arange(8, dtype=np.float32)
    x1 = np.full((4,), 42., dtype=np.float32)
    repro_source = self.collect_and_check(f, lambda: (4,), x1, x2)
    # We adjusted the donate_argnums by the static_argnums
    self.assertIn("{'donate_argnums': (0,)}", repro_source)

  def test_gather(self):
    operand = jnp.zeros((3, 3), dtype=jnp.int32)
    indices = jnp.zeros((2, 1), dtype=jnp.int32)

    dimension_numbers = jax.lax.GatherDimensionNumbers(
        offset_dims=(1,),
        collapsed_slice_dims=(0,),
        start_index_map=(0,),
    )

    f = jax.jit(lambda x, y: jax.lax.gather(
        x, y,
        dimension_numbers=dimension_numbers,
        slice_sizes=(1, 3),
        mode=lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    ))
    self.collect_and_check(f, operand, indices)

  @jtu.with_explicit_mesh((2,), ('x',))
  def test_scatter_gather(self, mesh):
    self.skipTest("TODO")
    x = np.random.uniform(size=(mesh.size * 2, 3))
    i = np.random.randint(0, x.shape[1], len(x))
    j = np.random.randint(0, x.shape[1], len(x))
    x = jax.device_put(x, P("x"))
    i = jax.device_put(i, P("x"))
    j = jax.device_put(j, P("x"))

    @jax.jit
    def f1(x, i, j):
      x_a_j = x.at[:, j].get(out_sharding=jax.typeof(i).sharding)
      return x.at[:, i].set(x_a_j, out_sharding=jax.typeof(x).sharding)
    f1(x,i,j)  # doesn't crash

    @jax.jit
    @jax.vmap
    def f2(x, i, j):
      x_j = x.at[j].get(out_sharding=jax.typeof(x).sharding)
      return x.at[i].set(x_j, out_sharding=jax.typeof(x).sharding)

    self.collect_and_check(f2, x, i, j)

  def test_scatter_add(self):
    def f(x, updates):
      return lax.scatter_add(
          x, np.array([[1], [2]], np.int32),  # indices: [2, 1]
          updates,  # updates: [7, 2]
          lax.ScatterDimensionNumbers((0,), (1,), (1,)),  # dimension_numbers
          indices_are_sorted=False, unique_indices = True)

    x = jnp.arange(28, dtype=np.float32).reshape((7, 4))
    updates = jnp.zeros((7, 2), dtype=np.float32)
    self.collect_and_check(f, x, updates)

  def test_prng_0(self):
    @jax.jit
    def f():
      key = jax.random.key(0)
      key, split = jax.random.split(key)
      return jax.random.uniform(split, (4,))
    self.collect_and_check(f)

  def test_prng_closed_over_key(self):
    key_0 = jax.random.key(0)
    @jax.jit
    def f():
      key, split = jax.random.split(key_0)
      return jax.random.uniform(split, (4,))
    self.collect_and_check(f)

  def test_primitive_with_array_params_split(self):
    def f():
      x = jnp.ones((8,), dtype=np.float32)
      sizes = (np.int32(2), np.int32(6))  # Primitives must have hashable params
      return lax.split(x, sizes)

    f()
    self.collect_and_check(f)

  def test_jnp_histogram(self):
    # Had issues with undefined variables in the repro, repro
    # also by test_nested_30
    def f(x):
      return jnp.histogram(x)
    self.collect_and_check(f, np.arange(8, dtype=np.float32))

  def test_cond_0(self):
    def true_branch(x):
      return jnp.sin(x)
    def f1(x, i: int):
      return lax.cond(x[i] >= 0., true_branch, jnp.cos, x)

    @jax.jit
    def f2(x):
      acc = x
      for i in range(3):
        acc = f1(acc, i)
      return acc

    self.collect_and_check(f2,
                      np.arange(8, dtype=np.float32) - 4.)

  def test_cond_with_per_branch_args(self):
    def true_branch(x):
      return jnp.sin(x)
    def f1(x, i: int):
      return lax.cond(x[i] >= 0., x, true_branch, x, jnp.cos)

    @jax.jit
    def f2(x):
      acc = x
      for i in range(3):
        acc = f1(acc, i)
      return acc

    self.collect_and_check(f2,
                      np.arange(8, dtype=np.float32) - 4.)

  def test_cond_with_callable_value(self):
    def true_branch(x):
      return CallableValue(jnp.sin(x.a), jnp.cos(x()))
    def f1(x):
      return lax.cond(x.a[0] >= 0., true_branch, lambda y:y, x)

    @jax.jit
    def f2():
      x = jnp.full((4,), 42., dtype=np.float32)
      acc = CallableValue(x, x)
      for i in range(3):
        acc = f1(acc)
      return acc

    self.collect_and_check(f2)

  def test_switch_0(self):
    def f():
      x = np.float32(.42)
      return lax.switch(0, [jnp.sin, jnp.cos], x)
    self.collect_and_check(jax.jit(f))

  def test_platform_dependent(self):
    # We add a different value to it: cpu=2., tpu=3., cuda=.4, rocm=5.
    _testing_multi_platform_to_add = dict(cpu=2., tpu=3., cuda=4., rocm=5.)

    def my_f():
      x = np.float32(.42)
      return x + lax.platform_dependent(
        tpu=lambda: _testing_multi_platform_to_add["tpu"],
        cuda=lambda: _testing_multi_platform_to_add["cuda"],
        rocm=lambda: _testing_multi_platform_to_add["rocm"],
        default=lambda: _testing_multi_platform_to_add["cpu"]
      )

    self.collect_and_check(jax.jit(my_f))

  def test_scan_0(self):
    def body(carry, x):
      c0, c1 = carry
      xa, xb = x["a"], x["b"]
      return (c0 + c1, c1), dict(c=xa + xb)
    @jax.jit
    def f1(x):
      return lax.scan(body, (0., 1.), x)

    self.collect_and_check(f1, dict(a=np.arange(8, dtype=np.float32),
                                          b=np.ones(8, dtype=np.float32)))

  def test_scan_custom_pytree(self):

    def body(carry: Value, x: Value):
      # c1 is being forwarded
      return Value(carry.a + carry.b, carry.b), dict(c=x.a + x.b)
    @jax.jit
    def f1(x):
      return lax.scan(body, Value(0., 1.), x)

    self.collect_and_check(f1, Value(a=np.arange(8, dtype=np.float32),
                                           b=np.ones(8, dtype=np.float32)))

  def test_scan_custom_pytree_no_y(self):

    def body(carry: Value, x: Value):
      return Value(carry.a + carry.b, carry.b), None
    @jax.jit
    def f1(x):
      return lax.scan(body, Value(0., 1.), x["d"])

    self.collect_and_check(f1,
                           {"d": Value(a=np.arange(8, dtype=np.float32),
                                             b=np.ones(8, dtype=np.float32))})

  def test_scan_custom_pytree_no_x(self):

    def body(carry: Value, x: Value):
      return Value(carry.a + carry.b, carry.b), dict(c=carry.b)
    @jax.jit
    def f1():
      return lax.scan(body, Value(0., 1.), length=8)

    self.collect_and_check(f1)

  def test_scan_custom_pytree_no_x_no_y(self):

    def body(carry: Value, x: Value):
      assert x is None
      return Value(carry.a + carry.b, carry.b), None
    @jax.jit
    def f1():
      return lax.scan(body, Value(0., 1.), length=8)

    self.collect_and_check(f1)

  @jtu.parameterized_filterable(
      kwargs=[
        dict(new_consts=new_consts)
        for new_consts in [False, True]
      ])
  def test_scan_with_constants(self, *, new_consts: bool):
    c1 = np.arange(4., dtype=np.float32)
    c2 = jnp.full((8,), 42., dtype=np.float32)  # distinctive shape
    def body(carry, _):
      return carry + c1 + c2[0:4], None
    @jax.jit
    def f1():
      return lax.scan(body, jnp.ones((4,), dtype=np.float32), length=8)

    with config.use_simplified_jaxpr_constants(new_consts):
      self.collect_and_check(f1)

  def test_scan_none(self):
    def f(_, __):
      return None, jnp.add(1, 1)
    # Run in eager mode to ensure we don't error when dealing with
    # xla_primitive_callable with higher-order primitives
    lax.scan(f, None, None, length=2)

  def test_scan_none_error(self):
    def f(_, __):
      return jnp.add(1, 1)

    with self.assertRaisesRegex(TypeError, "scan body output must be a pair"):
      lax.scan(f, None, None, length=2)

  def test_while_loop_0(self):
    def body(x):
      i, x1 = x
      return (i + 1, x1 * i)
    def cond(x):
      i, _ = x
      return i <= 10

    @jax.jit
    def f1():
      v1 = lax.while_loop(cond, body, (1., 1.))
      return v1

    self.collect_and_check(f1)

  def test_fori_loop_0(self):
    def body(i, x):
      return i + x

    @jax.jit
    def f1():
      v1 = lax.fori_loop(0, 5, body, 42)
      return v1
    self.collect_and_check(f1)

  def test_one_hot(self):
    from jax._src.nn import functions
    @jax.jit
    def f(x):
      return functions.one_hot(x, num_classes=8, dtype=jnp.bool_)

    self.collect_and_check(f, np.arange(16, dtype=np.int32))

  def test_convert_element_type(self):
    @jax.jit
    def f(x):
      l1 = [lax.convert_element_type(x, new_dtype=t)
           for t in jtu.supported_dtypes()]
      l2 = [lax.convert_element_type(x, new_dtype=t)
            for t in [jnp.int32, jnp.bool_, jnp.float32]]
      return l1 + l2

    self.collect_and_check(f, 1)

  def test_jvp(self):
    def f1(x):
      return jnp.sin(x)

    @jax.jit
    def f2(x):
      return jax.jvp(f1, (x,), (jnp.full_like(x, 0.2),))

    self.collect_and_check(f2, np.ones((8,), dtype=np.float32))

  def test_jvp_custom_pytree(self):
    def f1(x: Value):
      return Value(a=jnp.sin(x.a * x.b), b=5.)

    @jax.jit
    def f2(x):
      perturb_x = jnp.full_like(x, 0.2)
      return jax.jvp(f1, (Value(x, x),),
                     (Value(perturb_x, perturb_x),))

    self.collect_and_check(f2, np.ones((8,), dtype=np.float32))

  def test_custom_jvp_0(self):
    @jax.custom_jvp
    def f(x):
      return jax.numpy.sin(x)

    @f.defjvp
    def f_jvp_rule(primals, tangents):
      # 3 * x * x_t
      x, = primals
      x_dot, = tangents
      primal_out = f(x)
      tangent_out = 3. * x * x_dot
      return primal_out, tangent_out

    def compute_jvp(x):
      x_tan = jnp.full_like(x, .1)
      return jax.jvp(f, (x,), (x_tan,))

    self.collect_and_check(jax.jit(compute_jvp), np.arange(16.).reshape(4, 4))

  def test_custom_jvp_defjvps_0(self):
    @jax.custom_jvp
    def f(x, y):
      return jnp.sin(x) * y
    def f_jvp_0(x_dot, primal_out, x, y):
      return jnp.cos(x) * x_dot * y
    def f_jvp_1(y_dot, primal_out, x, y):
      return jnp.sin(x) * y_dot
    f.defjvps(f_jvp_0, f_jvp_1)

    @jax.jit
    def top(x):
      return jax.value_and_grad(f)(x, x)

    self.collect_and_check(top, 42.)

  def test_custom_jvp_defjvps_1(self):
    # A defjvps with just 1 arg
    @jax.custom_jvp
    def f(x):
      return jnp.sin(x)
    def f_jvp_0(x_dot, primal_out, x):
      return jnp.cos(x) * x_dot
    f.defjvps(f_jvp_0)

    @jax.jit
    def top(x):
      return jax.value_and_grad(f)(x)

    self.collect_and_check(top, 42.)

  def test_custom_jvp_defjvps_with_None(self):
    @jax.custom_jvp
    def f(x, y):
      return jnp.sin(x) * y
    def f_jvp_0(x_dot, primal_out, x, y):
      return jnp.cos(x) * x_dot * y
    f.defjvps(f_jvp_0, None)

    @jax.jit
    def top(x):
      return jax.value_and_grad(f)(x, x + 1.)

    self.collect_and_check(top, 42.)

  def test_custom_jvp_with_kwargs(self):
    @jax.custom_jvp
    def f(x, other):  # other will be passed as kwarg
      return jax.numpy.sin(x) + other

    @f.defjvp
    def f_jvp_rule(primals, tangents):
      # 3 * x * x_t
      x, other = primals
      x_dot, other_dot = tangents
      primal_out = f(x, other=other)
      tangent_out = 3. * x * x_dot + other_dot
      return primal_out, tangent_out

    def uses_f(x):
      return f(x, other=x)

    @jax.jit
    def compute_jvp():
      x = np.arange(16.).reshape(4, 4)
      x_tan = jnp.full_like(x, .1)
      return jax.jvp(uses_f, (x,), (x_tan,))

    self.collect_and_check(compute_jvp)

  def test_custom_jvp_nondiff_argnums_argnames(self):
    @partial(jax.custom_jvp, nondiff_argnums=(0,), nondiff_argnames=("g",))
    def app(f, x, g):
      return f(x) + g(x)
    def app_jvp(f, g, primals, tangents):
      (x,), (t,) = primals, tangents
      return app(f, x, g), 3 * t
    app.defjvp(app_jvp)

    f = lambda x: 2 * x
    g = lambda x: x * x
    @jax.jit
    def top():
      return jax.jvp(lambda x: app(f, x, g), (1.,), (1.,))

    self.collect_and_check(top)

  def test_linearize_0(self):
    def f1(x):
      return jnp.sin(x)

    @jax.jit
    def f2(x):
      y, f_jvp = jax.linearize(f1, x)
      x_tan = jnp.full_like(x, 0.2)
      return f_jvp(x_tan)
    self.collect_and_check(f2, np.ones((8,), dtype=np.float32))

  def test_grad_0(self):
    @jax.jit
    def f1(x):
      return jnp.sin(x)

    def f2(x):
      return jnp.sum(f1(x))

    self.collect_and_check(jax.jit(jax.grad(f2)),
                           np.ones((8,), dtype=np.float32))

  def test_grad_with_aux(self):
    @jax.jit
    def f1(x):
      return jnp.sin(x)

    metrics = {}
    def f2(x):
      return jnp.sum(f1(x)), metrics

    @jax.jit
    def f3():
      x = jnp.ones((8,), dtype=np.float32)
      g, aux = jax.grad(f2, has_aux=True)(x)
      # Mutate the return of `jax.grad` with a new value.
      aux["counter"] = x + 1.
      return g, aux

    self.collect_and_check(f3)

  def test_value_and_grad_0(self):
    @jax.jit
    def f1(x):
      return jnp.sin(x)

    def f2(x):
      return jnp.sum(f1(x))

    self.collect_and_check(jax.jit(jax.value_and_grad(f2)),
                           np.ones((8,), dtype=np.float32))

  def test_vjp(self):
    @jax.jit
    def f1(x):
      return jnp.sin(jnp.sin(x))

    @jax.jit
    def f2(x):
      out, vjpfun = jax.vjp(f1, x)
      return vjpfun(jnp.full_like(x, 1.))

    self.collect_and_check(f2, np.ones((8,), dtype=np.float32))

  def test_vjp_with_aux(self):
    @jax.jit
    def f1(x):
      return jnp.sin(jnp.sin(x)), x

    @jax.jit
    def f2(x):
      out, vjpfun, aux = jax.vjp(f1, x, has_aux=True)
      return vjpfun(jnp.full_like(x, 1.))

    self.collect_and_check(f2, np.ones((8,), dtype=np.float32))

  def test_custom_vjp_example(self):
    @jax.custom_vjp
    def my_sin(x):
      return jax.numpy.sin(x)
    def sin_fwd(x):
      return my_sin(x), x  # This is the "sin" with the custom_vjp
    def sin_bwd(x, y_bar):
      v = 2. * jax.numpy.cos(x)
      return (v * y_bar,)
    my_sin.defvjp(sin_fwd, sin_bwd)

    def f1(x):
      x = my_sin(x * 1e-3)
      x = jnp.dot(x, x)
      return x

    @jax.jit
    def f2(x):
      out, f2_vjp = jax.vjp(f1, x)
      return f2_vjp(np.ones((4, 4)))

    self.collect_and_check(f2, np.arange(16.).reshape(4, 4))

  def test_custom_vjp_with_kwargs(self):
    @jax.custom_vjp
    def my_sin(x, other):
      return jax.numpy.sin(x) + other
    def sin_fwd(x, other):
      return my_sin(x, other=other), x  # This is the "sin" with the custom_vjp
    def sin_bwd(x, y_bar):
      v = 2. * jax.numpy.cos(x)
      return (v * y_bar, y_bar)
    my_sin.defvjp(sin_fwd, sin_bwd)

    def f1(x):
      x = my_sin(x * 1e-3, other=x)
      x = jnp.dot(x, x)
      return x

    @jax.jit
    def f2():
      x = np.arange(16.).reshape(4, 4)
      out, f2_vjp = jax.vjp(f1, x)
      return f2_vjp(np.ones((4, 4)))

    self.collect_and_check(f2)

  def test_custom_vjp_nondiff_argnums_argnames(self):
    @partial(jax.custom_vjp, nondiff_argnums=(0,), nondiff_argnames=("g",))
    def app(f, x, g):
      return f(x) + g(x)
    def app_fwd(f, x, g):
      return app(f, x, g), jnp.cos(x)
    def app_rev(f, g, cos_x, v):
      return (cos_x * v,)
    app.defvjp(app_fwd, app_rev)

    f = lambda x: 2 * x
    g = lambda x: x * x
    @jax.jit
    def top():
      return jax.value_and_grad(lambda x: app(f, x, g))(1.)

    self.collect_and_check(top)

  def test_custom_gradient(self):
    @jax.custom_gradient
    def my_f(x):
      z = x ** 2
      def my_f_vjp(g):
        return (g * z,)
      return z * x, my_f_vjp

    def top():
      return jax.jit(jax.value_and_grad(my_f))(3.)

    #res = top()
    self.collect_and_check(top)

  def test_remat_custom_jvp_policy(self):
    @jax.custom_jvp
    def sin(x):
      return jnp.sin(x)
    def sin_jvp(primals, tangents):
      x, = primals
      g, = tangents
      return sin(x), jnp.cos(x) * g
    sin.defjvp(sin_jvp)

    @partial(jax.remat, policy=jax.checkpoint_policies.checkpoint_dots)
    def f(x):
      x = jnp.dot(x, x, precision=lax.Precision.HIGHEST)
      x = sin(x * 1e-3)
      x = jnp.dot(x, x, precision=lax.Precision.HIGHEST)
      x = sin(x * 1e-3)
      x = jnp.dot(x, x, precision=lax.Precision.HIGHEST)
      x = sin(x * 1e-3)
      return x

    def g(x):
      return lax.scan(lambda x, _: (f(x), None), x, None, length=2)[0]

    jtu.check_grads(f, (3.,), order=2, modes=['fwd', 'rev'])
    jtu.check_grads(g, (3.,), order=2, modes=['fwd', 'rev'])

  def test_remat_example_0(self):
    @jax.remat
    def f1(x):
      x = jnp.dot(x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x * 1e-3)
      return x

    @jax.jit
    def f2(x):
      out, f2_vjp = jax.vjp(f1, x)
      return f2_vjp(np.ones((4, 4)))

    self.collect_and_check(f2, np.arange(16.).reshape(4, 4))

  def test_remat_example_1(self):
    @jax.custom_vjp
    def sin(x):
      return jax.numpy.sin(x)
    def sin_fwd(x):
      return sin(x), x
    def sin_bwd(x, y_bar):
      v = 2. * jax.numpy.cos(x)
      return (v * y_bar,)

    sin.defvjp(sin_fwd, sin_bwd)

    def f(x):
      x = jnp.dot(x, x, precision=lax.Precision.HIGHEST)
      x = sin(x * 1e-3)
      x = jnp.dot(x, x)
      return x

    f2 = jax.remat(f)

    @jax.jit
    def f3(x):
      out, f2_vjp = jax.vjp(f2, x)
      return f2_vjp(np.ones((4, 4)))

    self.collect_and_check(f3, np.arange(16.).reshape(4, 4))

  def test_remat_custom_vjp_policy(self):
    @jax.custom_vjp
    def sin(x):
      return jnp.sin(x)
    def sin_fwd(x):
      return sin(x), x
    def sin_bwd(x, y_bar):
      return (jnp.cos(x) * y_bar,)
    sin.defvjp(sin_fwd, sin_bwd)

    @partial(jax.remat, policy=jax.checkpoint_policies.checkpoint_dots)
    def f(x):
      @partial(jax.named_call, name="dot")
      def dot2(y, z):
        return jnp.dot(x, jnp.dot(y, z, precision=lax.Precision.HIGHEST),
                       precision=lax.Precision.HIGHEST)

      x = dot2(x, x)
      x = sin(x * 1e-3)
      x = dot2(x, x)
      x = sin(x * 1e-3)
      x = dot2(x, x)
      x = sin(x * 1e-3)
      return x

    with tracker.flags_override(error_mode="defer"):
      # We know that for higher order differentiation we'll invoke the
      # bwd function multiple times
      jtu.check_grads(f, (3.,), order=2, modes=['rev'])

    def g(x):
      return lax.scan(lambda x, _: (f(x), None), x, None, length=2)[0]
    with tracker.flags_override(error_mode="log"):
      jtu.check_grads(g, (3.,), order=2, modes=['rev'])

  @jtu.parameterized_filterable(
      kwargs=[
        dict(testcase_name=f"_{remat_name}", remat=remat)
          for remat_name, remat in [
              ("old_remat", jax.remat),
              ("new_remat", new_checkpoint),
          ]
      ])
  def test_remat_checkpoint_dots(self, remat=jax.remat):
    @partial(remat, policy=jax.checkpoint_policies.checkpoint_dots)
    def f(x):
      x = jnp.dot(x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x)
      x = jnp.dot(x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x)
      x = jnp.dot(x, x, precision=lax.Precision.HIGHEST)
      x = jnp.sin(x)
      return x

    def my_grad(f, x):
      y1, f_vjp1 = jax.vjp(f, x)
      return f_vjp1(y1)

    self.collect_and_check(partial(my_grad, partial(my_grad, f)),
                           jnp.ones((2, 2)),
                           atol=1e-6)

  def test_linear_transpose_complex(self):
    f = lambda x: (1 + 2j) * x
    @jax.jit
    def transpose_fun():
      transpose = jax.linear_transpose(f, 1j)
      actual, = transpose(3 + 4j)
      return actual

    self.collect_and_check(transpose_fun)

  def test_vmap_0(self):
    def f1(x):
      update = jnp.zeros((4,), dtype=x.dtype)
      inserted = jax.lax.dynamic_update_slice_in_dim(
          x, update, start_index=0, axis=0)
      sliced = jax.lax.dynamic_slice_in_dim(
          inserted, start_index=2, slice_size=4, axis=0)
      return sliced

    self.collect_and_check(jax.jit(jax.vmap(f1, in_axes=1)),
                           np.ones((8, 8), dtype=np.float32))

  def test_vmap_with_custom(self):
    def f1(v: Value):
      a, b = v.a, v.b
      update = jnp.zeros((4,), dtype=a.dtype)
      inserted = jax.lax.dynamic_update_slice_in_dim(
          a, update, start_index=0, axis=0) + b
      sliced = jax.lax.dynamic_slice_in_dim(
          inserted, start_index=2, slice_size=4, axis=0)
      return sliced

    a = np.arange(64.).reshape((8, 8))
    self.collect_and_check(
        jax.jit(jax.vmap(f1, in_axes=(Value(0, 1),))),
        Value(a, a))

  def test_vmap_with_custom_in_axis(self):
    def f1(v: Value):
      (a0, a1), b = v.a, v.b
      update = jnp.zeros((4,), dtype=a0.dtype)
      inserted = jax.lax.dynamic_update_slice_in_dim(
          a0, update, start_index=0, axis=0) + b
      sliced = jax.lax.dynamic_slice_in_dim(
          inserted, start_index=2, slice_size=4, axis=0)
      return sliced

    a = np.arange(64.).reshape((8, 8))
    vmap_arg = Value((a, a), a)
    # The in_axes contains the value 0 for the tuple (a, a)
    vmap_func = jax.jit(jax.vmap(f1, in_axes=(Value(0, 1),)))
    self.collect_and_check(vmap_func, vmap_arg)


  def test_jacfwd_jacrev(self):
    R = self.rng().randn
    A = R(4, 3)
    x = R(3)

    @jax.jit
    def f(x):
      return jnp.tanh(jnp.dot(A, x))

    @jax.jit
    def g(x):
      fwd = jax.jacfwd(f)(x)
      rev = jax.jacrev(f)(x)
      return fwd - rev

    self.collect_and_check(g, x)

  @jax.default_matmul_precision("float32")
  def test_hessian(self):
    R = self.rng().randn
    A = R(4, 4)
    x = R(4)

    f = lambda x: jnp.dot(x, jnp.dot(A, x))

    self.collect_and_check(lambda: jax.hessian(f)(x),
                           atol=1e-7)

  @jtu.parameterized_filterable(
      kwargs=[dict(exper=exper)
              for exper in (True, False)]
  )
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  # We want to ensure repros work even for the deprecated exp.shard_map
  @jtu.ignore_warning(category=DeprecationWarning, message=".*shard_map.*")
  def test_shard_map_0(self, mesh, exper: bool = False):
    np_inp = np.arange(16).reshape(8, 2)
    s_x_y = jax.sharding.NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s_x_y)
    arr2 = jax.device_put(np_inp, s_x_y)

    def g(x, y):
      return x * y

    shard_map = exp_shard_map.shard_map if not exper else jax.shard_map
    @jax.jit
    def f(x, y):
      z = shard_map(g, mesh=mesh,
                    in_specs=(x.aval.sharding.spec, y.aval.sharding.spec),
                    out_specs=P('x', 'y'))(x, y)
      self.assertEqual(z.aval.sharding.spec, P('x', 'y'))
      out = z * 2
      self.assertEqual(out.aval.sharding.spec, P('x', 'y'))
      return out

    self.collect_and_check(f, arr, arr2)

  @jtu.ignore_warning(category=DeprecationWarning,
                      message=".*pjit has been deprecated")
  def test_sharding_abstract_mesh(self):
    if jax.local_device_count() < 2:
      self.skipTest("Need at least 2 devices")

    abs_mesh = jax.sharding.AbstractMesh((2,), 'x')
    output_sharding = jax.sharding.NamedSharding(abs_mesh, P(None, "x"))
    @jax.jit
    def f(a):
      b = a @ a.T
      return jax.lax.with_sharding_constraint(b, output_sharding)

    a = jnp.arange(4 * 4, dtype=np.float32).reshape((4, 4))
    self.collect_and_check(f, a)



  def test_pmap(self):
    @jax.pmap
    def f(x):
      return jnp.sin(x) + 1.

    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

    with jtu.ignore_warning(category=UserWarning, message=".* includes a pmap"):
      self.collect_and_check(f, x)

  def test_pmap_linearize(self):
    @partial(jax.pmap, axis_name='i')
    def f(x):
      def branch_true(x):
        return jnp.sin(x) + x
      def branch_false(x):
        return jnp.sin(x)
      return lax.cond(x[0] > 0., jax.jit(branch_true), branch_false, jnp.sin(x))

    @jax.jit
    def splitjvp(x):
      _, jvp = jax.linearize(f, x)
      return jvp(jnp.ones_like(x))

    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=np.float32).reshape(shape)

    with jtu.ignore_warning(category=UserWarning, message=".* includes a pmap"):
      self.collect_and_check(splitjvp, x)


  def test_named_call_0(self):
    @jax.jit
    def f(x):
      return jnp.dot(x, x)

    named_f = jax.named_call(f, name="my_f")
    x = jnp.ones([4, 4])

    self.collect_and_check(jax.jit(named_f), x)

  def test_named_call_statics(self):
    @partial(jax.jit, static_argnums=(1,))
    def f(x, get_shape):
      return jnp.broadcast_to(x, get_shape())

    named_f = jax.named_call(f, name="my_f")
    x = jnp.ones([1, 4])

    self.collect_and_check(named_f, x, lambda: (4, 4))

  def test_einsum(self):
    # One issue with einsum is that it modifies its args
    @jax.jit
    def f(w, x):
      a = jnp.dot(x, w)
      b = jnp.einsum("btd,bTd->btT", a, a)
      return b

    w = jnp.ones([1, 1])
    x = jnp.ones([1, 1, 1])

    self.collect_and_check(f, w, x)

  def test_pallas_call_0(self):
    m, n = 32, 4
    out_shape = jax.ShapeDtypeStruct((4, n), floatx)
    @jax.jit
    @partial(
        pl.pallas_call,
        interpret=True,
        out_shape=out_shape,
    )
    def slice_kernel(x_ref, y_ref):
      y_ref[:4, :4] = x_ref[:4, :4]
    x = np.arange(m * n, dtype=floatx).reshape((m, n))
    y = slice_kernel(x)
    self.collect_and_check(slice_kernel, y)

  def test_pallas_call_1(self):
    @partial(jax.jit, static_argnames=["bm", "bn", "bk",
                                       "interpret", "debug"])
    def matmul_block_spec(x, y, *, bm, bn, bk, interpret, debug=False):
      m, n, k = x.shape[0], y.shape[1], x.shape[1]

      @partial(
          pl.pallas_call,
          out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
          interpret=interpret,
          debug=debug,
          in_specs=[
              pl.BlockSpec((bm, x.shape[1]), lambda i, _: (i, 0),
                           memory_space=pltpu.MemorySpace.SMEM),
              pl.BlockSpec((y.shape[0], bn), lambda _, j: (0, j)),
          ],
          out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
          grid=(pl.cdiv(m, bm), pl.cdiv(n, bn)),
      )
      def matmul_kernel(x_ref, y_ref, o_ref):
        acc = jnp.zeros(o_ref.shape, dtype=jnp.float32)

        def body(i, acc):
          x_block = x_ref[:, pl.ds(i * bk, bk)]
          y_block = y_ref[pl.ds(i * bk, bk), :]
          return acc + pl.dot(x_block, y_block)

        acc = lax.fori_loop(0, k // bk, body, acc).astype(o_ref.dtype)
        o_ref[:, :] = acc

      return matmul_kernel(x, y)

    x = y = np.ones((16, 16), dtype=np.float32)

    with tracker.flags_override(fake_array_threshold=1026):
      self.collect_and_check(matmul_block_spec,
                             x, y, bm=2, bn=4, bk=8, interpret=True)

  def test_pallas_ref_transforms(self):

    def kernel(x_ref, y_ref):
      ref = (
          x_ref.at[:16, :256]  # i32(16, 256)
          .bitcast(jnp.int16)  # i16(32, 256)
          .reshape((2, 16, 256))  # i16(2, 16, 256)
          .bitcast(jnp.float16)  # bf16(2, 16, 256)
          .at[1:, :, :]  # bf16(1, 16, 256)
          .reshape((16, 256))  # bf16(16, 256)
          .at[:, :128]  # bf16(16, 128)
          .bitcast(jnp.int32)  # i32(8, 128)
      )
      y_ref[...] = ref[...]

    x = jnp.arange(32 * 256, dtype=jnp.int32).reshape((32, 256))
    def myf():
      return pl.pallas_call(
          kernel,
          out_shape=jax.ShapeDtypeStruct((8, 128), jnp.int32),
          interpret=True)(x)

    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(myf)

  def test_pallas_cost_analysis(self):
    def kernel(x, y):
      y[:] = x[:]
    x = jnp.arange(1024.).reshape(8, 128)
    f = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct((8, 128), jnp.float32),
        cost_estimate=pl.CostEstimate(
            flops=1234, transcendentals=21, bytes_accessed=12345
        ),
        interpret=True,
    )
    with tracker.flags_override(fake_array_threshold=x.size + 1):
      self.collect_and_check(f, x)


  def test_pallas_call_scalar_prefetch(self):
    def body(_, x_ref, o_ref):
      o_ref[...] = x_ref[...]

    s = jnp.array([4, 3, 2, 5, 3, 5, 2, 7], jnp.int32)
    x = jnp.arange(2 * 8 * 8 * 4, dtype=jnp.int32).reshape((2, 8 * 8, 4))

    def _x_transform(i, s_ref):
      s = s_ref[i]
      return (s, 0)

    s = jnp.tile(s[None], [2, 1])

    @jax.jit
    @jax.vmap
    def kernel(s, x):
      return pl.pallas_call(
          body,
          out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=1,
              in_specs=[
                  pl.BlockSpec((x.shape[0] // 8, x.shape[1]), _x_transform),
              ],
              out_specs=pl.BlockSpec(
                  (x.shape[0] // 8, x.shape[1]), lambda i, _: (i, 0)
              ),
              grid=8,
          ),
          compiler_params=pltpu.CompilerParams(
              allow_input_fusion=[False, True]
          ),
          interpret=True,
      )(s, x)

    with tracker.flags_override(fake_array_threshold=1024):
      self.collect_and_check(kernel, s, x)

  def test_pallas_call_fusions_trivial(self):
    @fuser.fusible(output_fusion_prefix=(True, True))
    def f(x_fn, y_fn, z_fns):
      x = x_fn()
      y = y_fn()
      if z_fns is None:
        z_fns = lambda x: x, lambda x: x
      z_fn1, z_fn2 = z_fns
      return z_fn1(x), z_fn2(y)

    @jax.jit
    @fuser.fuse
    def g(x, y):
      x, y = f(x, y)
      return x, y * 2

    x = np.ones((128, 128), dtype=np.float32)
    y = np.ones((1, 128), dtype=np.float32)
    # g(x, y)
    self.collect_and_check(g, x, y)


  def test_pallas_call_fusible_matmul(self):
    self.skipTest("TODO")
    # Copied from tpu_fusible_matmult_test
    def matmul_kernel(
        x_scalar_prefetch,
        y_scalar_prefetch,
        z_scalar_prefetch,
        x_value_refs,
        y_value_refs,
        z_value_refs,
        o_ref,
        acc_ref,
        *,
        x_fn: Any,
        y_fn: Any,
        z_fn: Any,
        out_dtype: jnp.dtype,
    ):
      @pl.when(pl.program_id(2) == 0)
      def _():
        acc_ref[...] = jnp.zeros_like(acc_ref)

      pids = pl.program_id(0), pl.program_id(1), pl.program_id(2)
      scalar_prefetch = (x_scalar_prefetch, y_scalar_prefetch, z_scalar_prefetch)

      x_values = jax.tree.map(lambda ref: ref.get(), x_value_refs)
      x = x_fn(pids, scalar_prefetch, x_values)
      y_values = jax.tree.map(lambda ref: ref.get(), y_value_refs)
      y = y_fn(pids, scalar_prefetch, y_values)
      acc_ref[...] += jnp.dot(x, y, preferred_element_type=jnp.float32)

      @pl.when(pl.program_id(2) == pl.num_programs(2) - 1)
      def _():
        acc = acc_ref[...].astype(out_dtype)
        z_values = jax.tree.map(lambda ref: ref.get(), z_value_refs)
        out = z_fn(pids, scalar_prefetch, z_values, acc)
        jax.tree.map(lambda ref, x: ref.set(x), o_ref, out)

    def _fusible_matmul(
        x: fuser.Fusion[[], jax.Array],  # pytype: disable=invalid-annotation
        y: fuser.Fusion[[], jax.Array],  # pytype: disable=invalid-annotation
        z: fuser.Fusion[[jax.Array], jax.Array] | None,  # pytype: disable=invalid-annotation
        *,
        bm: int,
        bk: int,
        bn: int,
        interpret: bool,
        debug: bool,
    ) -> jax.Array:
      m, k = x.shape
      k_, n = y.shape
      out_dtype = jnp.float32
      z_type = jax.ShapeDtypeStruct((m, n), dtype=out_dtype)
      if not z:
        z = lambda x: x
      if k != k_:
        raise ValueError(f'X and Y shapes must be compatible. Got {k} != {k_}')

      assert m % bm == 0
      assert k % bk == 0
      assert n % bn == 0
      grid = (m // bm, n // bn, k // bk)

      def x_index_map(i, j, k, *_):
        del j
        return i, k

      x_block_spec = pl.BlockSpec(block_shape=(bm, bk), index_map=x_index_map)

      def y_index_map(i, j, k, *_):
        del i
        return k, j

      y_block_spec = pl.BlockSpec(block_shape=(bk, bn), index_map=y_index_map)

      def z_index_map(i, j, k, *_):
        del k
        return i, j

      z_block_spec = pl.BlockSpec(block_shape=(bm, bn), index_map=z_index_map)
      dimension_semantics = (pltpu.PARALLEL, pltpu.PARALLEL, pltpu.ARBITRARY)

      z_out_type = jax.eval_shape(z, z_type)

      # First thing we do is extract the values from the fusions. These will be
      # values that are passed in directly and values that are passed in via
      # scalar prefetch.
      x_fn, x_values, x_scalar_prefetch = fuser.get_fusion_values(x)
      y_fn, y_values, y_scalar_prefetch = fuser.get_fusion_values(y)
      z_fn, z_values, z_scalar_prefetch = fuser.get_fusion_values(z, z_type)

      # We construct the set of scalar prefetch arguments that will be passed to
      # the kernel.
      scalar_prefetch = (x_scalar_prefetch, y_scalar_prefetch, z_scalar_prefetch)

      x_fn, (x_value_block_specs,), _ = fuser.pull_block_spec(
          x_fn,
          x_block_spec,
          scalar_prefetch_handler=fuser.make_scalar_prefetch_handler(0),
          grid=grid,
      )(x_values)

      y_fn, (y_value_block_specs,), _ = fuser.pull_block_spec(
          y_fn,
          y_block_spec,
          scalar_prefetch_handler=fuser.make_scalar_prefetch_handler(1),
          grid=grid,
      )(y_values)

      z_out_block_spec = fuser.push_block_spec(z, z_block_spec)(z_type)
      z_fn, (z_value_block_specs, _), _ = fuser.pull_block_spec(
          z_fn,
          z_out_block_spec,
          scalar_prefetch_handler=fuser.make_scalar_prefetch_handler(2),
          grid=grid,
      )(z_values, z_type)

      # TODO(sharadmv): This is a hack. We should be able to pass in the scalar
      # prefetch arguments directly to the kernel but don't have Mosaic support atm.
      scalar_prefetch = jax.tree.map(lambda x: x[None], scalar_prefetch)

      return pl.pallas_call(
          partial(
              matmul_kernel,
              x_fn=x_fn,
              y_fn=y_fn,
              z_fn=z_fn,
              out_dtype=out_dtype,
          ),
          grid_spec=pltpu.PrefetchScalarGridSpec(
              num_scalar_prefetch=len(scalar_prefetch),
              grid=grid,
              scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
              in_specs=[
                  x_value_block_specs,
                  y_value_block_specs,
                  z_value_block_specs,
              ],
              out_specs=[z_out_block_spec],
          ),
          compiler_params=pltpu.CompilerParams(
              dimension_semantics=dimension_semantics,
          ),
          out_shape=[z_out_type],
          interpret=interpret,
          debug=debug,
      )(
          *scalar_prefetch,
          x_values,
          y_values,
          z_values,
      )[0]

    def fusible_matmul(
        x: jax.Array,
        y: jax.Array,
        *,
        bm: int = 128,
        bk: int = 128,
        bn: int = 128,
        debug: bool = False,
        interpret: bool = True,  # TODO: False on a TPU backend
    ) -> jax.Array:
      return fuser.fusible(
          partial(
              _fusible_matmul,
              bm=bm,
              bk=bk,
              bn=bn,
              interpret=interpret,
              debug=debug,
          )
      )(x, y)

    dtype = np.float32
    k0, k1 = jax.random.split(jax.random.key(0))
    x = jax.random.normal(k0, (512, 512), dtype)
    y = jax.random.normal(k1, (512, 512), dtype)

    @jax.jit
    @fuser.fuse
    def matmul_relu(x, y):
      x = fusible_matmul(x, y)
      x = jnp.maximum(x, 0.0)
      return x

    def mm_ref(x, y):
      return jnp.dot(x, y, preferred_element_type=jnp.float32)

    @partial(jax.jit, compiler_options={'xla_allow_excess_precision': False})
    def matmul_relu_ref(x, y):
      return jax.nn.relu(mm_ref(x, y))

    logging.warning(matmul_relu.trace(x, y).jaxpr)

    self.collect_and_check(matmul_relu, x, y)

  def test_pallas_call_custom_fusion_0(self):
    @fuser.custom_fusion
    def c(x, y):
      return x + y

    c.def_pull_block_spec(lambda bss: (bss[0], bss[0]))
    c.def_push_block_spec(lambda bss: (bss[0],))
    c.def_eval_rule(lambda _, x, y: (c(x, y),))

    @fuser.fusible(output_fusion_prefix=(True, True))
    def f(x_fn, y_fn, z_fns):
      x = x_fn()
      y = y_fn()
      if z_fns is None:
        z_fns = lambda x: x, lambda x: x
      z_fn1, z_fn2 = z_fns
      return z_fn1(x), z_fn2(y)

    def g(x, y, z):
      x, y = f(x, c(y, z))
      return c(x, z), y * 2

    x = jax.random.normal(jax.random.key(0), (4, 4), dtype=jnp.float32)
    y = jax.random.normal(jax.random.key(1), (1, 4), dtype=jnp.float32)
    z = jax.random.normal(jax.random.key(2), (1, 4), dtype=jnp.float32)

    self.collect_and_check(jax.jit(fuser.fuse(g)), x, y, z)

  def test_pallas_call_custom_fusion_with_kwargs(self):
    @fuser.custom_fusion
    def c(x, y):  # y is passed as kwarg
      return x + y

    c.def_pull_block_spec(lambda bss: (bss[0], bss[0]))
    c.def_push_block_spec(lambda bss: (bss[0],))
    c.def_eval_rule(lambda _, x, y: (c(x, y=y),))

    @fuser.fusible(output_fusion_prefix=(True, True))
    def f(x_fn, y_fn, z_fns):
      x = x_fn()
      y = y_fn()
      if z_fns is None:
        z_fns = lambda x: x, lambda x: x
      z_fn1, z_fn2 = z_fns
      return z_fn1(x), z_fn2(y)

    def g(x, y1, y2):
      x, y3 = f(x, c(y1, y=y2))
      return c(x, y=y2), y3 * 2

    x = jax.random.normal(jax.random.key(0), (4, 4), dtype=jnp.float32)
    y = jax.random.normal(jax.random.key(1), (1, 4), dtype=jnp.float32)
    z = jax.random.normal(jax.random.key(2), (1, 4), dtype=jnp.float32)

    self.collect_and_check(jax.jit(fuser.fuse(g)), x, y, z)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
