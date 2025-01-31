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

from __future__ import annotations

import contextlib
import functools
import math
import operator
import re
from typing import Any
import unittest

from absl.testing import absltest
import jax
from jax import ad_checkpoint
from jax import lax

import jax.custom_batching
import jax.custom_derivatives
import jax.custom_transpose
from jax.experimental import checkify
import jax.experimental.custom_dce
from jax.experimental import pallas as pl
import jax.numpy as jnp
import jax.scipy as jsp

from jax._src import api_util
from jax._src.ad_checkpoint import saved_residuals
from jax._src import config
from jax._src import core
from jax._src import test_util as jtu
from jax._src.compilation_cache import is_persistent_cache_enabled
from jax._src.lax.control_flow import for_loop
from jax._src.interpreters import mlir


import numpy as np

config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


def _collect_jaxprs(jaxpr: core.Jaxpr,
                    acc: list[core.Jaxpr] | None = None) -> list[core.Jaxpr]:
  """Collect all Jaxprs in a depth-first order."""
  if acc is None: acc = []
  acc.append(jaxpr)
  for e in jaxpr.eqns:
    # Take first the block mapping Jaxprs
    if e.primitive.name == "pallas_call":
      # For pallas_call, extract also jaxprs inside the grid_mapping
      mapping = e.params["grid_mapping"]
      for bm in mapping.block_mappings:
        _collect_jaxprs(bm.index_map_jaxpr.jaxpr, acc)
    for sj in core.jaxprs_in_params(e.params):
      _collect_jaxprs(sj, acc)
  return acc


def _debug_info_to_string(dbg: api_util.TracingDebugInfo | core.JaxprDebugInfo | None) -> list[str]:
  if dbg is None: return "None"
  # Strip the absolute path and the line number but check that it references
  # this file (to catch errors when the source info points in JAX internals)
  fun_src_info = re.sub(r"^(\S+)( at .*/debug_info_test.py:.*)?", "\\1", dbg.func_src_info)
  res = f"traced_for={dbg.traced_for}, fun={fun_src_info}, arg_names={','.join(dbg.arg_names)}"
  if isinstance(dbg, core.JaxprDebugInfo):
    if dbg.result_paths:
      res += f", result_paths={','.join(dbg.result_paths)}"
    else:
      res += ", result_paths=<empty>"
  return res


class TracerSpy:
  """Use to inspect tracers.

  We can `append` tracers from tracing contexts to this object. We collect
  the tracer, along with the error message we get when we try to concretize
  it. This is meant to simulate errors like concretization or leaking.
  """
  tracers: list[tuple[core.Tracer, Exception]]
  def __init__(self):
    self.tracers = []

  def append(self, t: core.Tracer):
    try:
      if t:
        pass
      assert False, t
    except Exception as e:
      self.tracers.append((t, e))


@jtu.with_config(jax_mutable_array_checks=True)
class DebugInfoTest(jtu.JaxTestCase):

  def _check_tracers_and_jaxprs(self, traceable: Any,
                                *args,
                                expected_jaxpr_debug_infos: list[str | re.Pattern],
                                tracer_spy: TracerSpy | None = None,
                                expected_tracer_debug_infos: list[str | re.Pattern] = [],
                                check_lowering: bool = True,
                                check_tracer_arg_name: bool = False,
                                expected_lowering_lines: list[str | re.Pattern] = [],
                                **kwargs):
    """Checks for expected debug info in all jaxprs, and in inspected tracers.

    `traceable` will be traced as `traceable.trace(*args, **kwargs)` if it has
    a `trace` attribute, or will be called as `traceable(*args, **kwargs)`
    otherwise. The debug infos in the nested Jaxprs are first converted to
    strings using `_debug_info_to_string` and then
    compared against `expected_jaxpr_debug_infos`.
    An element of `expected_jaxpr_debug_infos` can be a string, in which case
    it is looked up by equality, or a `re.Pattern` (the result of `re.compile`)
    in which case it is looked up by `.match()`. All elements of
    `expected_jaxpr_debug_infos` must appear, and all Jaxprs must be matched.

    Optionally, we can pass a TracerSpy object into which we have
    appended tracers from the execution of `traceable`. Those
    tracers must have debugging info matching `expected_tracer_debug_infos`.
    If the `check_tracer_arg_name` is True, then we append
    ", from <arg_name>" to the inspected tracer's debug info (to test
    provenance information).
    """
    if hasattr(traceable, "trace"):
      traced = traceable.trace(*args, **kwargs)
      all_jaxprs = _collect_jaxprs(traced.jaxpr.jaxpr)
    else:
      # Just run the function and collect the Jaxprs and modules that are
      # lowered
      traced = None
      with jtu.collect_lowered_jaxprs() as collection:
        traceable(*args, **kwargs)

      all_jaxprs = []
      for jaxpr, _ in collection:
        all_jaxprs.extend(_collect_jaxprs(jaxpr.jaxpr))

    found_jaxprs_debug_infos = [_debug_info_to_string(j.debug_info)
                                for j in all_jaxprs]

    self._check_matches(expected_jaxpr_debug_infos, found_jaxprs_debug_infos)  # JAXPRS

    found_tracer_debug_infos = []
    if tracer_spy is not None:
      for t, exc in tracer_spy.tracers:
        if hasattr(t, "_debug_info"):
          t_debug_info = _debug_info_to_string(t._debug_info)
          if check_tracer_arg_name:
            msg = str(exc)
            m = re.match(r".* while tracing the function (.+) for (.+)\. .* depends on the value of the argument ([^\n]+)\.",
                         msg,
                         re.DOTALL)
            self.assertIsNotNone(m, msg)
            self.assertEqual(t._debug_info.func_src_info, m.group(1))
            self.assertEqual(t._debug_info.traced_for, m.group(2))
            found_tracer_debug_infos.append(
                f"{t_debug_info}, from {m.group(3)}")
          else:
            found_tracer_debug_infos.append(t_debug_info)
        else:
          found_tracer_debug_infos.append("None")

      self._check_matches(expected_tracer_debug_infos, found_tracer_debug_infos)  # INSPECTED TRACERS

    if not check_lowering: return
    # Collect all the lines in all the MLIR modules
    mlir_modules_lines = []
    if traced is not None:
      mlir_modules_lines.extend(
          traced.lower().as_text("stablehlo", debug_info=True).split("\n"))
    else:
      for _, mod in collection:
        mlir_modules_lines.extend(
            mlir.module_to_string(mod, enable_debug_info=True).split("\n"))

    expected_and_found = set()
    expected_and_not_found = set()
    for exp in expected_lowering_lines:
      for l in mlir_modules_lines:
        ok = exp.match(l) if isinstance(exp, re.Pattern) else exp == l
        if ok:
          expected_and_found.add(exp)
          break
      else:
        expected_and_not_found.add(exp)

    if expected_and_not_found:
      msg = "\n".join(mlir_modules_lines)
      self.assertEmpty(expected_and_not_found, "\nNot found in the MLIR module lines:\n" + msg)

  def _check_matches(self,
                     expected: list[str | re.Pattern],
                     found: list[str]):
    expected_and_found = set()
    unexpected: set[str] = set()
    for debug_info in found:
      for exp_re in expected:
        ok = exp_re.match(debug_info) if isinstance(exp_re, re.Pattern) else exp_re == debug_info
        if ok:
          expected_and_found.add(exp_re)
          break
      else:
        unexpected.add(debug_info)
    self.assertEmpty(unexpected)  # found unexpected debug_info
    self.assertEmpty([e for e in expected if e not in expected_and_found])  # expected element that was not found

  def test_debug_info_basic(self):
    def my_f(x, y, z, w):
      pass

    dbg = api_util.tracing_debug_info("jit", my_f, (1, 2), dict(z=3, w=4))
    self.assertRegex(dbg.func_src_info, r"^my_f at .*debug_info_test.py:\d+")
    self.assertEqual(dbg.arg_names, ("x", "y", "z", "w"))
    self.assertIsNone(dbg.result_paths_thunk)

  def test_debug_info_arg_passed_as_kwarg(self):
    def my_f(x, y, z):
      pass

    dbg = api_util.tracing_debug_info("jit", my_f, (1, 2), dict(z=3))
    self.assertEqual(dbg.arg_names, ("x", "y", "z"))

  def test_debug_info_pytrees(self):
    def my_f(x_tree, *, y_tree):
      pass

    dbg = api_util.tracing_debug_info("jit", my_f, ((1, 2),),
                                      dict(y_tree=dict(z=3, w=4)))
    self.assertEqual(dbg.arg_names, ("x_tree[0]", "x_tree[1]",
                                     "y_tree['w']", "y_tree['z']"))

  def test_debug_info_with_statics(self):
    def my_f(x, y, *, z, w):
      pass

    dbg = api_util.tracing_debug_info("jit", my_f, (1, 2), dict(z=3, w=4),
                                      static_argnums=(1,),
                                      static_argnames=("w",))
    self.assertEqual(dbg.arg_names, ("x", "z"))

  def test_debug_info_with_pytrees_and_statics(self):
    def my_f(x, y, *, z, w):
      pass

    dbg = api_util.tracing_debug_info("jit", my_f, ((1, 2), (2, 3)),
                                      dict(z=(3, 4), w=(5, 6)),
                                      static_argnums=(1,),
                                      static_argnames=("w",))
    self.assertEqual(dbg.arg_names, ("x[0]", "x[1]", "z[0]", "z[1]"))

  def test_debug_info_too_many_args(self):
    def my_f(x):
      pass

    dbg = api_util.tracing_debug_info("jit", my_f, (1, 2, 3), dict(z=3))
    self.assertEqual(dbg.arg_names, ('args[0]', 'args[1]', 'args[2]', "kwargs['z']"))

  def test_debug_info_no_source_info_built_in(self):
    # built-in function "int" does not have an inspect.Signature
    dbg = api_util.tracing_debug_info("jit", max, (1,), {})
    self.assertEqual(dbg.func_src_info, "max")
    self.assertEqual(dbg.arg_names, ("args[0]",))

  def test_debug_info_lambda(self):
    # built-in function "int" does not have an inspect.Signature
    dbg = api_util.tracing_debug_info("jit", lambda my_arg: False, (1,), {})
    self.assertRegex(dbg.func_src_info, r"^<lambda> at .*debug_info_test.py:\d+")
    self.assertEqual(dbg.arg_names, ("my_arg",))

  def test_debug_info_no_source_info_not_callable(self):
    # built-in function "int" does not have an inspect.Signature
    dbg = api_util.tracing_debug_info("jit", False, (1,), {})
    self.assertEqual(dbg.func_src_info, "<unknown>")
    self.assertEqual(dbg.arg_names, ("args[0]",))

  def test_debug_info_no_source_info_callable(self):
    class Foo:
      x: int

      def __call__(self, y):
        return self.x + y

    dbg = api_util.tracing_debug_info("jit", Foo(), (1,), {})
    self.assertRegex(dbg.func_src_info, "<unknown>")
    self.assertEqual(dbg.arg_names, ("y",))

  def test_debug_info_no_source_info_callable_with_repr_errors(self):
    class Foo:
      x: int

      def __call__(self, y):
        return self.x + y

      def __repr__(self):
        raise NotImplementedError

    dbg = api_util.tracing_debug_info("jit", Foo(), (1,), {})
    self.assertRegex(dbg.func_src_info, "<unknown>")
    self.assertEqual(dbg.arg_names, ("y",))

  def helper_save_tracer(self, x):
    self._saved_tracer = x
    return x

  def test_jit_lower_arg_names_with_error1(self):
    def f(x):
      return jnp.sqrt(x ** 2) + 1.

    x = jnp.array(1, dtype=int)
    x_f32 = x.astype(jnp.float32)
    x_i32 = x.astype(jnp.int32)
    f_exe = jax.jit(f).lower(x_f32).compile()
    self.assertRaisesRegex(
        TypeError,
        r"Argument types differ .*"
        r"The mismatches are:\n"
        r"Argument 'x' compiled with.*float32.*and called with.*int32.*",
        lambda: f_exe(x_i32))

  def test_jit_lower_arg_names_with_error2(self):
    def f(x):
      return x

    err_str = ("Error interpreting argument to .* as an abstract array. The problematic "
               "value is of type .* and was passed to the function at path x.")
    with self.assertRaisesRegex(TypeError, err_str):
      jax.jit(f)("foo")

    # Jax type objects aren't valid data arguments.
    with self.assertRaisesRegex(TypeError, err_str):
      jax.jit(f)(jnp.int32)

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  def test_arg_names_cache_miss_explanations(self):
    @jax.jit
    def f(x, y):
      return jnp.sin(x) * y['hi']

    x = jnp.float32(1.)
    y = {'hi': jnp.arange(3., dtype='float32')}

    expected_log_len = 1 if not is_persistent_cache_enabled() else 3

    # print on first miss, not on hit
    with config.explain_cache_misses(True):
      with self.assertLogs(level='WARNING') as cm:
        f(x, y)
        f(x, y)
    self.assertLen(cm.output, expected_log_len)
    msg = cm.output[0]
    self.assertIn('TRACING CACHE MISS', msg)
    self.assertIn('never seen function', msg)

    # shape change
    y_ = {'hi': jnp.arange(4, dtype='float32')}
    with config.explain_cache_misses(True):
      with self.assertLogs(level='WARNING') as cm:
        f(x, y_)
    self.assertLen(cm.output, expected_log_len)
    msg = cm.output[0]
    self.assertIn('never seen input type signature', msg)
    self.assertIn('closest seen input type signature has 1 mismatches', msg)
    self.assertIn('seen f32[3], but now given f32[4]', msg)

    # weak type change (assuming no x64)
    if not config.enable_x64.value:
      with config.explain_cache_misses(True):
        with self.assertLogs(level='WARNING') as cm:
          f(1., y)
      self.assertLen(cm.output, expected_log_len)
      msg = cm.output[0]
      self.assertIn('weak_type=True', msg)
      self.assertIn('https://jax.readthedocs.io/en/latest/type_promotion.html#weak-types', msg)

    # kwarg change
    with config.explain_cache_misses(True):
      with self.assertLogs(level='WARNING') as cm:
        f(1, y=y)
    self.assertLen(cm.output, expected_log_len)
    msg = cm.output[0]
    self.assertIn('never seen passing 1 positional args and 1 keyword args', msg)

    # tracing config change
    with config.explain_cache_misses(True):
      with self.assertLogs(level='WARNING') as cm:
        with jax.numpy_rank_promotion('warn'):
          f(x, y)
    # depending on the backend, we may or may not get persistent cache warnings
    self.assertTrue(1 <= len(cm.output) <= expected_log_len)
    msg = cm.output[0]
    self.assertIn("tracing context doesn't match", msg)

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  def test_arg_names_cache_miss_explanations_new_function_in_loop(self):
    @jax.jit
    def f(x, y):
      return jnp.sin(x) * y['hi']

    x = jnp.float32(1.)

    with config.explain_cache_misses(True):
      with self.assertLogs(level='WARNING') as cm:
        for _ in range(2):
          jax.jit(lambda x: 2 * x)(3)
    if is_persistent_cache_enabled():
      # number of warnings depends on the backend
      self.assertTrue(4 <= len(cm.output) <= 6)
      msg = cm.output[3]
      self.assertIn('another function defined on the same line', msg)
    else:
      self.assertLen(cm.output, 2)
      _, msg = cm.output
      self.assertIn('another function defined on the same line', msg)

  @jtu.thread_unsafe_test()  # logging is not thread-safe
  @unittest.skip("Test fails, probably due to caching")
  def test_arg_names_cache_miss_explanations_unpacks_transforms(self):
    # Tests that the explain_tracing_cache_miss() function does not throw an
    # error when unpacking `transforms` with a length greater than 3.
    @jax.jit
    def f(key):
      return jax.random.truncated_normal(key, 1, 1, dtype=jax.numpy.float32)

    with config.explain_cache_misses(True):
      with self.assertLogs(level="WARNING") as cm:
        f(jax.random.key(seed=123))

    if is_persistent_cache_enabled():
      # 5 warnings from tracing cache, 5-10 from persistent cache depending on
      # the backend
      self.assertTrue(10 <= len(cm.output) <= 15)
      self.assertTrue(any("TRACING CACHE MISS" in msg for msg in cm.output))
    else:
      self.assertLen(cm.output, 5)
      for msg in cm.output:
        self.assertIn("TRACING CACHE MISS", msg)

  def test_arg_names_cache_miss_explanations_no_source_info(self):
    # ``operator.add`` is a built-in function and does not have source info.
    with config.explain_cache_misses(True):
      jax.jit(operator.add)(42, 24)

  def test_concrete_error_because_arg_unary(self):
    @jax.jit
    def f(x):
      if x > 0:
        return x
      else:
        return 0

    msg = r"on the value of the argument x"
    with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
      f(1)

  def test_concrete_error_because_arg_binary(self):
    @jax.jit
    def f(x, y):
      if x > y:
        return x
      else:
        return y

    msg = r"on the values of the arguments x and y"
    with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
      f(1, 2)

  def test_concrete_error_because_arg_ternary(self):
    @jax.jit
    def f(x, y, z):
      if x > z:
        return x
      else:
        return y

    msg = r"on the values of the arguments x and z"
    with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
      f(1, 2, 3)

    with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
      f(1, 2, z=3)

    with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
      f(1, y=2, z=3)

  def test_concrete_error_because_arg_varargs(self):
    @jax.jit
    def f(*args):
      x, y, z = args
      if x > z:
        return x
      else:
        return y

    msg = r"on the values of the arguments args"
    with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
      f(1, 2, 3)

  def test_concrete_error_because_arg_kwargs(self):
    @jax.jit
    def f(**kwargs):
      x, y, z = kwargs['x'], kwargs['y'], kwargs['z']
      if x > z:
        return x
      else:
        return y

    msg = r"on the values of the arguments kwargs"
    with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
      f(x=1, y=2, z=3)

  def test_concrete_error_because_arg_pytree(self):
    @jax.jit
    def f(xy, z):
      x, y = xy
      if x > 0:
        return x
      else:
        return y

    msg = r"on the value of the argument xy"
    with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
      f((1, 2), z=3)

  def test_concrete_error_because_const(self):
    @jax.jit
    def f():
      assert jnp.add(1, 1) > 0

    msg = "on these lines"
    with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
      f()

  def test_concrete_error_because_const_2(self):
    @jax.jit
    def f():
      result = sum(jnp.add(1, 1) for _ in range(6))
      assert result > 0

    msg = "Additional originating lines are not shown."
    with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
      f()

  def test_concrete_error_with_nested_call(self):
    @jax.jit
    def f(x, y):
      if y:
        return x

    @jax.jit
    def g(x):
      return f(x, True)

    msg = r"on the value of the argument y"
    with self.assertRaisesRegex(core.ConcretizationTypeError, msg):
      g(1)

  def test_remat_concrete_error(self):
    @jax.remat  # no static_argnums or concrete
    def g(x):
      if x > 0:
        return lax.sin(x)
      else:
        return lax.cos(x)

    with self.assertRaisesRegex(core.ConcretizationTypeError, "static_argnums"):
      g(3.)

    @functools.partial(jax.remat, static_argnums=(0,))  # using static_argnums but...
    def g(x):
      if x > 0:  # jnp operations still get staged!
        return lax.sin(x)
      else:
        return lax.cos(x)

    with self.assertRaisesRegex(core.ConcretizationTypeError, "static_argnums"):
      g(jnp.array(3.))

    # But don't raise an error mentioning static_argnums here:
    @jax.remat
    def g(x):
      jax.jit(lambda: 0 if jnp.add(1, 1) else 0)()
      return lax.sin(x)

    try:
      g(jnp.array(3.))
    except core.ConcretizationTypeError as e:
      msg = str(e)
    self.assertNotIn('static_argnums', msg)

  def test_simple_jit(self):
    tracer_spy = TracerSpy()
    def my_f(x_dict, y):
      tracer_spy.append(x_dict["a"])
      return dict(c=x_dict["a"] + x_dict["b"], d=y)

    self._check_tracers_and_jaxprs(
        jax.jit(my_f),
        dict(a=1, b=2), 3,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=x_dict['a'],x_dict['b'],y, result_paths=['c'],['d']"
        ],
        check_tracer_arg_name=True,
        expected_tracer_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=x_dict['a'],x_dict['b'],y, from x_dict['a']",
        ])

  def test_jit_with_static_argnums(self):
    tracer_spy = TracerSpy()
    @functools.partial(jax.jit, static_argnums=(1,))
    def my_f(a, d):
      tracer_spy.append(a)
      return a

    def my_g(b, d=1):
      tracer_spy.append(b)
      return my_f(b, d)

    self._check_tracers_and_jaxprs(
        jax.jit(my_g),
        3,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            # TODO(necula): bad result names
            'traced_for=jit, fun=my_f, arg_names=a, result_paths=<empty>',
            'traced_for=jit, fun=my_g, arg_names=b, result_paths=',
        ],
        check_tracer_arg_name=True,
        expected_tracer_debug_infos=[
            "traced_for=jit, fun=my_g, arg_names=b, from b",
            "traced_for=jit, fun=my_f, arg_names=a, from a",
        ])

  def test_jit_arg_names(self):
    tracer_spy = TracerSpy()
    def f(x, y, *args, **kwargs):
      # args[0] is dead
      tracer_spy.append(kwargs["w"])
      tracer_spy.append(args[0])
      return y['hi'] + args[1] + sum(kwargs.values())

    self._check_tracers_and_jaxprs(
        jax.jit(f),
        {"ho": 1.}, {"hi": 2.}, 3., 4., z=5., w=6.,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=f, arg_names=x['ho'],y['hi'],args[0],args[1],kwargs['w'],kwargs['z'], result_paths=",
        ],
        tracer_spy=tracer_spy,
        check_tracer_arg_name=True,
        expected_tracer_debug_infos=[
            "traced_for=jit, fun=f, arg_names=x['ho'],y['hi'],args[0],args[1],kwargs['w'],kwargs['z'], from kwargs['w']",
            "traced_for=jit, fun=f, arg_names=x['ho'],y['hi'],args[0],args[1],kwargs['w'],kwargs['z'], from args[0]",
        ],
        expected_lowering_lines=[
            re.compile(r".*func.func public @main\(%arg0: tensor<f..> loc\(\"y\['hi'\]\"\)"),
            re.compile(r".*func.func public @main\(.*%arg1: tensor<f..> loc\(\"args\[1\]\"\)"),
            re.compile(r".*func.func public @main\(.*%arg2: tensor<f..> loc\(\"kwargs\['w'\]\"\)"),
            re.compile(r".*func.func public @main\(.*%arg3: tensor<f..> loc\(\"kwargs\['z'\]\"\)"),
            re.compile(r".*func.func public @main\(.*\{jax.result_info = \"\"\}"),
        ]
    )

  def test_jit_arg_names_static_argnums(self):
    tracer_spy = TracerSpy()
    def my_f(x, y, z, *args, **kwargs):
      # z and args[0] and kwargs["w"] are dead
      # x and args[2] are static
      tracer_spy.append(kwargs["w"])
      tracer_spy.append(x[0])
      return x[0] + y["hi"] + args[1] + args[2] + kwargs["t"]

    self._check_tracers_and_jaxprs(
        jax.jit(my_f, static_argnums=(0, 5)),
        (1.,), {"hi": 2.}, 3., 4., 5., 6.,  # x, y, z, args[0], args[1], args[2]
        t=11., w=12.,  # kwargs
        expected_jaxpr_debug_infos=[
            # TODO(necula): the arg_names include the dead ones
            "traced_for=jit, fun=my_f, arg_names=y['hi'],z,args[0],args[1],kwargs['t'],kwargs['w'], result_paths=",
        ],
        tracer_spy=tracer_spy,
        check_tracer_arg_name=True,
        expected_tracer_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=y['hi'],z,args[0],args[1],kwargs['t'],kwargs['w'], from kwargs['w']",
            "None",  # TODO(necula)
        ],
        expected_lowering_lines=[
            re.compile(r".*func.func public @main\(%arg0: tensor<f..> loc\(\"y\['hi'\]\"\)"),
            re.compile(r".*func.func public @main\(.*%arg1: tensor<f..> loc\(\"args\[1\]\"\)"),
            re.compile(r".*func.func public @main\(.*%arg2: tensor<f..> loc\(\"kwargs\['t'\]\"\)"),
            re.compile(r".*func.func public @main\(.*\{jax.result_info = \"\"\}"),
        ])

  def test_jit_arg_names_static_argnames(self):
    tracer_spy = TracerSpy()
    def f(x, y, *args, **kwargs):
      # x and args[0] and kwargs["z"] are dead
      # kwargs[a] is static
      tracer_spy.append(x[0])
      return y['hi'] + args[1] + kwargs['a'] + kwargs['b'] + kwargs['w']

    self._check_tracers_and_jaxprs(
        jax.jit(f, static_argnames=("a",)),
        (1.,), {'hi': 2.}, 3., 4.,  # x, y, args[0], args[1]
        z=5., w=6., a=7., b=8.,  # kwargs
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=f, arg_names=x[0],y['hi'],args[0],args[1],kwargs['b'],kwargs['w'],kwargs['z'], result_paths=",
        ],
        tracer_spy=tracer_spy,
        check_tracer_arg_name=True,
        expected_tracer_debug_infos=[
            "traced_for=jit, fun=f, arg_names=x[0],y['hi'],args[0],args[1],kwargs['b'],kwargs['w'],kwargs['z'], from x[0]",
        ],
        expected_lowering_lines=[
            re.compile(r".*func.func public @main\(%arg0: tensor<f..> loc\(\"y\['hi'\]\"\)"),
            re.compile(r".*func.func public @main\(.*%arg1: tensor<f..> loc\(\"args\[1\]\"\)"),
            re.compile(r".*func.func public @main\(.*%arg2: tensor<f..> loc\(\"kwargs\['b'\]\"\)"),
            re.compile(r".*func.func public @main\(.*%arg3: tensor<f..> loc\(\"kwargs\['w'\]\"\)"),
            re.compile(r".*func.func public @main\(.*\{jax.result_info = \"\"\}"),
        ])

  def test_jit_result_info(self):
    def f(x, y, z):
      return {'a': x, 'b': [y]}
    self._check_tracers_and_jaxprs(
        jax.jit(f),
        1., (2.,), [3.],
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=f, arg_names=x,y[0],z[0], result_paths=['a'],['b'][0][0]",
        ],
        expected_lowering_lines=[
            re.compile(r".*func.func public @main\(%arg0: tensor<f..> loc\(\"x\"\)"),
            re.compile(r".*func.func public @main\(.*%arg1: tensor<f..> loc\(\"y\[0\]\"\)"),
            re.compile(r".*func.func public @main\(.*\{jax.result_info = \"\['a'\]\"\}"),
            re.compile(r".*func.func public @main\(.*\{jax.result_info = \"\['b'\]\[0\]\[0\]\"\}"),
        ])

  def test_nested_jit(self):
    tracer_spy = TracerSpy()
    def my_f(x, y):
      tracer_spy.append(x)

      def my_g(u, v):
        tracer_spy.append(u)
        return dict(c=u * v, d=v)

      return jax.jit(my_g)(y, x)["c"]

    self._check_tracers_and_jaxprs(
        jax.jit(my_f),
        2, 3,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=x,y, result_paths=",
            "traced_for=jit, fun=my_g, arg_names=u,v, result_paths=[\'c\']"
        ],
        expected_tracer_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=x,y",
            "traced_for=jit, fun=my_g, arg_names=u,v"
        ])

  def test_jvp_of_jit(self):
    tracer_spy = TracerSpy()
    def f(x, y, z):
      tracer_spy.append(x)
      return {'a': x, 'b': [y]}
    self._check_tracers_and_jaxprs(
        lambda x, y, z: jax.jvp(jax.jit(f), (x, y, z), (x, y, z)),
        jnp.float32(1.), (jnp.float32(2.),), [jnp.float32(3.)],
        expected_jaxpr_debug_infos=[
            "None",  # TODO(necula): missing debug info
        ],
        tracer_spy=tracer_spy,
        expected_tracer_debug_infos=[
            "traced_for=jit, fun=f, arg_names=x,y[0],z[0]",
        ],
        expected_lowering_lines=[
            # TODO(necula): missing arg_names
            re.compile(r".*func.func public @main\(%arg0: tensor<f..> loc\(unknown\)"),
            re.compile(r".*func.func public @main\(.*%arg1: tensor<f..> loc\(unknown\)"),
            re.compile(r".*func.func public @main\(.*%arg2: tensor<f..> loc\(unknown\)"),
            re.compile(r".*func.func public @main\(.*%arg3: tensor<f..> loc\(unknown\)"),
            # TODO(necula): missing result names
            re.compile(r".*func.func public @main\(.*-> \(tensor<f..>, tensor<f..>, tensor<f..>, tensor<f..>\) {"),
        ])

  def test_vjp_of_jit(self):
    tracer_spy = TracerSpy()
    def my_f(x, y, z):
      tracer_spy.append(y[0])
      return {'a': x * y[0], 'b': [y]}
    self._check_tracers_and_jaxprs(
        lambda x, y, z: jax.vjp(jax.jit(my_f), x, y, z)[1](dict(a=x, b=[y])),
        jnp.float32(1.), (jnp.float32(2.),), [jnp.float32(3.)],
        expected_jaxpr_debug_infos=[
            "None",  # TODO(necula): missing debug info
        ],
        tracer_spy=tracer_spy,
        check_tracer_arg_name=True,
        expected_tracer_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=x,y[0],z[0], from y[0]",
        ],
        expected_lowering_lines=[
            # TODO(necula): missing arg_names
            re.compile(r".*func.func public @main\(%arg0: tensor<f..> loc\(unknown\)"),
            re.compile(r".*func.func public @main\(.*%arg1: tensor<f..> loc\(unknown\)"),
            # TODO(necula): missing result names
            re.compile(r".*func.func public @main\(.*-> tensor<f..> {"),
        ])

  def test_vjp_of_nested_jit(self):
    tracer_spy = TracerSpy()
    def my_f(x, y):
      tracer_spy.append(x)

      def my_g(u, v):
        tracer_spy.append(u)
        return dict(c=u * v, d=v)

      return jax.jit(my_g)(y, x)["c"]

    self._check_tracers_and_jaxprs(
        jax.jit(lambda x, y, res_ct: jax.vjp(my_f, x, y)[1](res_ct)),
        2., 3., 0.3,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=<lambda>, arg_names=x,y,res_ct, result_paths=[0],[1]",
            # TODO(necula): missing debug info
            "None",
        ],
        check_tracer_arg_name=True,
        expected_tracer_debug_infos=[
            # TODO(necula): missing debug info
            "None",
            "traced_for=jit, fun=my_g, arg_names=u,v, from u"
        ],
        expected_lowering_lines=[
            re.compile(r".*func.func public @main\(%arg0: tensor<f..> loc\(\"x\"\)"),
            re.compile(r".*func.func public @main\(.*%arg1: tensor<f..> loc\(\"y\"\)"),
            re.compile(r".*func.func public @main\(.*%arg2: tensor<f..> loc\(\"res_ct\"\)"),
            re.compile(r".*func.func public @main\(.*jax.result_info = \"\[0\]\"}"),
            re.compile(r".*func.func public @main\(.*jax.result_info = \"\[1\]\"}"),
        ])


  def test_vmap_of_nested_jit(self):
    tracer_spy = TracerSpy()
    def my_f(x, y):
      tracer_spy.append(x)

      def my_g(u, v):
        tracer_spy.append(u)
        return dict(c=u * v, d=v)

      return jax.jit(my_g)(y, x)["c"]

    self._check_tracers_and_jaxprs(
        jax.jit(jax.vmap(my_f)),
        np.ones((8,), dtype=np.float32), np.zeros((8,), dtype=np.float32),
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=x,y, result_paths=",
            # TODO(necula): missing debug info
            'None',
        ],
        expected_tracer_debug_infos=[
            # TODO(necula): missing debug info
            "None",
            "traced_for=jit, fun=my_g, arg_names=u,v"
        ])

  def test_custom_vmap(self):
    tracer_spy = TracerSpy()
    @jax.custom_batching.custom_vmap
    def my_f(xdict):
      x = xdict["x"]
      tracer_spy.append(x)
      return dict(a=jnp.sin(x))

    @my_f.def_vmap
    def my_rule(axis_size, in_batched, xys):
      xs = xys["x"]
      tracer_spy.append(xs)
      xs_batched, = in_batched
      self.assertEqual(xs_batched["x"], True)
      self.assertEqual(axis_size, xs.shape[0])
      return dict(a=jnp.cos(xs)), dict(a=xs_batched["x"])

    xy = dict(x=np.ones((8,), dtype=np.float32), y=np.zeros((8,), dtype=np.float32))
    self._check_tracers_and_jaxprs(
        jax.jit(jax.vmap(my_f)),
        xy,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=xdict['x'],xdict['y'], result_paths=['a']",
        ],
        expected_tracer_debug_infos=[
            "traced_for=custom_vmap, fun=my_f, arg_names=xdict['x'],xdict['y']",
            "traced_for=jit, fun=my_f, arg_names=xdict['x'],xdict['y']"
        ])

  def test_cond(self):
    tracer_spy = TracerSpy()
    def my_f(x):
      def my_true_branch(a, b):
        tracer_spy.append(a)
        return a + b

      def my_false_branch(c, d):
        tracer_spy.append(c)
        return c - d

      return lax.cond(x >= 0, my_true_branch, my_false_branch, x, x)

    self._check_tracers_and_jaxprs(
        jax.jit(my_f),
        0,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=x, result_paths=",
            # TODO(necula): some Jaxprs without debug info
            "None"],
        expected_tracer_debug_infos=[
            "traced_for=cond, fun=my_true_branch, arg_names=a,b",
            "traced_for=cond, fun=my_false_branch, arg_names=c,d"
        ])

  def test_switch(self):
    tracer_spy = TracerSpy()
    def my_f(x):
      def my_branch0(x0):
        tracer_spy.append(x0)
        return x0
      def my_branch1(x1):
        tracer_spy.append(x1)
        return x1 + 1
      def my_branch2(x2):
        tracer_spy.append(x2)
        return x2 + 2
      return lax.switch(x, [my_branch0, my_branch1, my_branch2], x)

    self._check_tracers_and_jaxprs(
        jax.jit(my_f),
        2,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=x, result_paths=",
            # TODO(necula): some Jaxprs without debug info
            "None"],
        expected_tracer_debug_infos=[
            "traced_for=switch, fun=my_branch0, arg_names=x0",
            "traced_for=switch, fun=my_branch1, arg_names=x1",
            "traced_for=switch, fun=my_branch2, arg_names=x2"
        ])

  def test_grad_cond_with_remat(self):
    tracer_spy = TracerSpy()
    def my_f(x, y):
      # The cond branches return two things, and only the first is needed
      # in the residuals.
      def my_true_branch(a, b):
        tracer_spy.append(a)
        return (a + 1, a + b)

      def my_false_branch(c, d):
        tracer_spy.append(c)
        return (c - 1, c - d)

      def my_g(x, y):
        # x1 does not depend on y
        x1, y1 = lax.cond(x >= 0, my_true_branch, my_false_branch, x, y)
        tracer_spy.append(x1)
        return x1, y1

      x2, y2 = jax.remat(my_g)(x, y)
      return y2 + lax.sin(x2)

    self._check_tracers_and_jaxprs(
        jax.jit(jax.grad(my_f)),
        1., 2.,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            'traced_for=jit, fun=my_f, arg_names=x,y, result_paths=',
            # TODO(necula): some Jaxprs without debug info
            'None',
        ],
        expected_tracer_debug_infos=[
            'traced_for=cond, fun=my_true_branch, arg_names=a,b',
            'traced_for=cond, fun=my_false_branch, arg_names=c,d',
            'traced_for=checkpoint / remat, fun=my_g, arg_names=x,y',
        ])

  def test_grad_scan(self):
      # Based on control_flow_test:testScanHigherOrderDifferentiation
    tracer_spy = TracerSpy()
    def f(c, a):
      tracer_spy.append(c)
      d = 0.75
      b = jnp.sin(c * jnp.sum(jnp.cos(d * a)))
      c = 0.9 * jnp.cos(d * jnp.sum(jnp.sin(c * a)))
      return c, b

    as_ = jnp.arange(6.).reshape((3, 2))
    c = jnp.array(1, dtype=as_.dtype)

    @jax.jit
    def my_f(x, as_):
      tracer_spy.append(x)
      return jax.remat(lambda *args: for_loop.scan(f, *args))(c, as_)

    def the_grad(c, as_):
      tracer_spy.append(c)
      _, pullback = jax.vjp(my_f, c, as_)
      return pullback((c, np.arange(3, dtype=c.dtype)))

    self._check_tracers_and_jaxprs(
        jax.jit(the_grad),
        c, as_,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=the_grad, arg_names=c,as_, result_paths=[0],[1]",
            'None',  # TODO(necula): some Jaxprs without debug info
        ],
        expected_tracer_debug_infos=[
            "traced_for=jit, fun=the_grad, arg_names=c,as_",
            "traced_for=scan, fun=f, arg_names=c,a",
            "traced_for=jit, fun=my_f, arg_names=x,as_",
            'None',  # TODO(necula): some missing debug info
        ])

  def test_while_loop(self):
    tracer_spy = TracerSpy()
    def my_f(x):
      def my_cond(a):
        tracer_spy.append(a)
        return a <= 8

      def my_body(b):
        tracer_spy.append(b)
        return b + 1

      return lax.while_loop(my_cond, my_body, x)

    self._check_tracers_and_jaxprs(
        jax.jit(my_f),
        0,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=x, result_paths=",
            'None',  # TODO(necula): some missing debug info
        ],
        check_tracer_arg_name=True,
        expected_tracer_debug_infos=[
            "traced_for=while_cond, fun=my_cond, arg_names=a, from a",
            "traced_for=while_body, fun=my_body, arg_names=b, from b",
        ])

  def test_fori(self):
    # See https://github.com/jax-ml/jax/issues/23637
    tracer_spy = TracerSpy()
    def my_body(_, c):
      tracer_spy.append(c)
      return 0.

    self._check_tracers_and_jaxprs(
        jax.jit(lambda x: jax.lax.fori_loop(0, 5, my_body, x)),
        3.,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=<lambda>, arg_names=x, result_paths=",
            'None',  # TODO(necula): some missing debug info
        ],
        expected_tracer_debug_infos=[
            # TODO(necula): the arg_names are not right
            "traced_for=scan, fun=my_body, arg_names=loop_carry[0],loop_carry[1]",
        ]
    )

    tracer_spy = TracerSpy()

    # When the ubound is not a constant, we use a while_loop
    self._check_tracers_and_jaxprs(
        jax.jit(lambda ub, x: jax.lax.fori_loop(0, ub, my_body, x)),
        5, 3.,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=<lambda>, arg_names=ub,x, result_paths=",
            'None',  # TODO(necula): some missing debug info
        ],
        expected_tracer_debug_infos=[
            # TODO(necula): the arg_names are not right
            "traced_for=while_body, fun=my_body, arg_names=loop_carry[0],loop_carry[1],loop_carry[2]",
        ])

  def test_scan(self):
    tracer_spy = TracerSpy()
    def my_f(x):
      def my_scan_body(carry, inp):
        tracer_spy.append(carry)
        return (carry + inp, carry)

      return lax.scan(my_scan_body, 0, x)

    self._check_tracers_and_jaxprs(
        jax.jit(my_f),
        np.arange(8, dtype=np.int32),
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=x, result_paths=[0],[1]",
            # TODO(necula): some Jaxprs without debug info
            'None'],
        expected_tracer_debug_infos=[
            "traced_for=scan, fun=my_scan_body, arg_names=carry,inp"
        ])

  def test_eval_shape(self):
    tracer_spy = TracerSpy()
    def my_f(x):
      tracer_spy.append(x)
      return x

    _ = jax.eval_shape(my_f, 0)
    self._check_tracers_and_jaxprs(
        lambda: jax.eval_shape(my_f, 0),
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[],
        expected_tracer_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=x"],
    )

  def test_pmap(self):
    tracer_spy = TracerSpy()
    def my_f(x):
      tracer_spy.append(x)
      return jnp.sin(x)

    self._check_tracers_and_jaxprs(
        jax.pmap(my_f),
        np.ones((jax.device_count(),), dtype=np.float32),
        expected_jaxpr_debug_infos=[
            "traced_for=pmap, fun=my_f, arg_names=x, result_paths="
        ],
        tracer_spy=tracer_spy,
        expected_tracer_debug_infos=[
            "traced_for=pmap, fun=my_f, arg_names=x"
        ],
    )

  def test_pmap_with_arg_and_result_names(self):
    tracer_spy = TracerSpy()
    x = np.ones((jax.device_count(),), dtype=np.float32)
    def my_f(x, y, *args, a, **kwargs):
      # y and kwargs[c] is dead
      tracer_spy.append(args[1])
      s = x + a + args[1] + kwargs["d"]
      return dict(u=s, v=x)

    self._check_tracers_and_jaxprs(
        jax.pmap(my_f, static_broadcasted_argnums=(0,)),
        1., x, x, x,  # x, y, args[0], args[1]
        d=x, a=x, b=x,  # kwargs
        expected_jaxpr_debug_infos=[
            "traced_for=pmap, fun=my_f, arg_names=y,args[0],args[1],a,kwargs['b'],kwargs['d'], result_paths=['u'],['v']",
        ],
        tracer_spy=tracer_spy,
        check_tracer_arg_name=True,
        expected_tracer_debug_infos=[
            "traced_for=pmap, fun=my_f, arg_names=y,args[0],args[1],a,kwargs['b'],kwargs['d'], from args[1]",
        ],
    )

  def test_pmap_of_grad(self):
    tracer_spy = TracerSpy()
    def my_f(x):
      tracer_spy.append(x)
      return jnp.sin(x)

    self._check_tracers_and_jaxprs(
        jax.pmap(jax.grad(my_f)),
        np.ones((jax.device_count(),), dtype=np.float32),
        expected_jaxpr_debug_infos=[
            "traced_for=pmap, fun=my_f, arg_names=x, result_paths=",
        ],
        tracer_spy=tracer_spy,
        expected_tracer_debug_infos=[
            # TODO(necula): missing debug_info
            'None'
        ],
    )

  def test_jvp_pmap_eager(self):
    tracer_spy = TracerSpy()
    def my_f(x, y, *args):
      # y is dead, x is static broadcasted
      tracer_spy.append(args[1])
      s = x + args[1]
      return dict(u=s, v=x)

    x = jnp.ones((jax.device_count(), 1), dtype=np.float32)
    x_tan = jnp.full_like(x, .1)

    self._check_tracers_and_jaxprs(
        lambda x, x_tan: jax.jvp(jax.pmap(my_f),
                                 (x, x, x, x), (x_tan, x_tan, x_tan, x_tan)),
        x, x_tan,
        expected_jaxpr_debug_infos=[
            # TODO(necula): why this?
            re.compile(r'traced_for=jit, fun=_multi_slice at .*/array_methods.py:.*, arg_names=self, result_paths=.*'),
            "traced_for=pmap, fun=my_f, arg_names=x,y,args[0],args[1], result_paths=['u'],['v']",
        ],
        tracer_spy=tracer_spy,
        expected_tracer_debug_infos=[
            # TODO(necula): missing debug_info
            'None'
        ],
    )

  @jtu.ignore_warning(category=UserWarning,
                      message=".* jitted function .* includes a pmap")
  def test_jvp_pmap(self):
    tracer_spy = TracerSpy()
    def my_f(x, y):
      tracer_spy.append(x)
      return jnp.sin(x) + y

    x = np.ones((jax.device_count(), 1), dtype=np.float32)
    x_tan = np.full_like(x, .1)

    self._check_tracers_and_jaxprs(
        jax.jit(lambda x, x_tan: jax.jvp(jax.pmap(my_f), (x, x), (x_tan, x_tan))),
        x, x_tan,
        expected_jaxpr_debug_infos=[
            'traced_for=jit, fun=<lambda>, arg_names=x,x_tan, result_paths=[0],[1]',
            "None",  # TODO(necula): missing debug info
        ],
        tracer_spy=tracer_spy,
        expected_tracer_debug_infos=[
            # TODO(necula): missing debug_info
            'None'
        ],
    )

  def test_remat(self):
    tracer_spy = TracerSpy()
    def my_f(x):
      @jax.remat
      def my_g(y):
        tracer_spy.append(y)
        return lax.sin(y)

      return my_g(x)

    self._check_tracers_and_jaxprs(
        jax.jit(my_f),
        0.,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=x, result_paths=",
            # TODO(necula): some Jaxprs without debug info
            'None'],
        expected_tracer_debug_infos=[
            "traced_for=checkpoint / remat, fun=my_g, arg_names=y"
        ])

  def test_grad_remat(self):
    tracer_spy = TracerSpy()
    def my_f(x):
      @jax.remat
      def my_g(y):
        tracer_spy.append(y)
        return lax.sin(lax.sin(y))

      return my_g(my_g(x))

    self._check_tracers_and_jaxprs(
        jax.jit(jax.grad(my_f)),
        0.,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=x, result_paths=",
            # TODO(necula): some Jaxprs without debug info
            "None"],
        expected_tracer_debug_infos=[
            "traced_for=checkpoint / remat, fun=my_g, arg_names=y"
        ])

  def test_remat_saved_residuals(self):
    @functools.partial(jax.remat,
                       static_argnums=(1,),
                       policy=lambda p, *_, **__: "mul" in str(p))
    def my_f(x, y):
      x = ad_checkpoint.checkpoint_name(x * x, "foo")
      x = x * x
      return x + y

    res = saved_residuals(my_f, 3., 4.)
    self.assertEqual(res[0][1], "from the argument x")
    self.assertRegex(res[1][1], r"named 'foo' from .*debug_info_test.py:.*my_f")

  def test_checkify_pmap_basic(self):
    if len(jax.devices()) < 2:
      self.skipTest("requires at least 2 devices")
    tracer_spy = TracerSpy()
    @jax.pmap
    def my_f(my_x):
      tracer_spy.append(my_x)
      y1 = jnp.sin(1./my_x)
      y2 = jnp.sin(my_x)
      return (y1 + y2,)

    self._check_tracers_and_jaxprs(
        jax.jit(checkify.checkify(my_f, errors=checkify.nan_checks)),
        np.arange(len(jax.devices()), dtype=np.float32),
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            # TODO(necula): this should not be pointing into the JAX internals
            re.compile(r"traced_for=jit, fun=checked_fun at .*jax/_src/checkify.py:.*, arg_names=args\[0\]"),
            re.compile(r"traced_for=jit, fun=argsort at .*numpy/lax_numpy.py:.*, arg_names=a, result_paths="),
            "None",  # TODO(necula): missing tracer debug info
        ],
        expected_tracer_debug_infos=[
            "traced_for=pmap, fun=my_f, arg_names=my_x",
        ],
        check_lowering=False,  # TODO(necula): warning during lowering
    )

  def test_custom_dce_static_argnums(self):
    tracer_spy = TracerSpy()
    @functools.partial(jax.experimental.custom_dce.custom_dce,
                       static_argnums=(0,))
    def my_g(f, x):
      tracer_spy.append(x)
      return f(x), 10 * f(x)

    @my_g.def_dce
    def my_g_dce(f, used_outs, x):  # note: static_argnums are always passed first
      tracer_spy.append(x)
      self.assertTrue(callable(f))
      return [2 * v if used else None
              for used, v in zip(used_outs, my_g(f, x))]

    def my_f(x):
      return jnp.exp(x)

    self._check_tracers_and_jaxprs(
        jax.jit(lambda x: my_g(my_f, x)[0]),
        0.,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=<lambda>, arg_names=x, result_paths=",
            # TODO(necula): some Jaxprs without debug info
            'None'],
        expected_tracer_debug_infos=[
            # TODO(necula): no leaked tracer from my_g_dce?
            "traced_for=custom_dce, fun=my_g, arg_names=x",
        ])

  def test_custom_dce_consts(self):
    tracer_spy = TracerSpy()
    @jax.experimental.custom_dce.custom_dce
    def my_f(x):
      tracer_spy.append(x)
      return np.eye(1) * jnp.sin(x), jnp.cos(x)

    @my_f.def_dce
    def my_rule(used_outs, y):
      tracer_spy.append(y)
      return (
          np.full((1, 1), 2.0) * jnp.exp(y) if used_outs[0] else None,
          jnp.sqrt(y) if used_outs[1] else None,
      )

    self._check_tracers_and_jaxprs(
        jax.jit(lambda x: my_f(x)[0]),
        np.array(1.1234),
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=<lambda>, arg_names=x, result_paths=",
            # TODO(necula): some Jaxprs without debug info
            'None'],
        check_tracer_arg_name=True,
        expected_tracer_debug_infos=[
            # TODO(necula): no leaked tracer from my_rule?
            "traced_for=custom_dce, fun=my_f, arg_names=x, from x",
        ])

  def test_custom_linear_solve_complex(self):
    tracer_spy = TracerSpy()
    def solve(a, b):
      tracer_spy.append(a)
      def my_solve(matvec, x):
        tracer_spy.append(x)
        return jsp.linalg.solve(a, x)

      def my_high_precision_dot(a, b):
        tracer_spy.append(a)
        return lax.dot(a, b, precision=lax.Precision.HIGHEST)

      def my_tr_solve(matvec, x):
        tracer_spy.append(x)
        return jsp.linalg.solve(a.T, x)
      matvec = functools.partial(my_high_precision_dot, a)
      return lax.custom_linear_solve(matvec, b, my_solve, my_tr_solve)

    rng = self.rng()
    a = 0.5 * rng.randn(2, 2) + 0.5j * rng.randn(2, 2)
    b = 0.5 * rng.randn(2) + 0.5j * rng.randn(2)

    self._check_tracers_and_jaxprs(
        jax.jit(lambda a, b: jax.jvp(solve, (a, b), (a, b))),
        a, b,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=<lambda>, arg_names=a,b, result_paths=[0],[1]",
            re.compile(r"traced_for=jit, fun=_solve at .*scipy/linalg.py:.*, arg_names=a,b, result_paths="),
            re.compile(r"traced_for=jit, fun=solve at .*/linalg.py:.*, arg_names=a,b, result_paths="),
            re.compile(r"traced_for=jit, fun=_lu_solve at .*/linalg.py:.*, arg_names=lu,permutation,b, result_paths="),
            "None",  # TODO(necula): there are missing jaxpr debug info
        ],
        expected_tracer_debug_infos=[
            "traced_for=custom_linear_solve solve, fun=my_solve, arg_names=x",
            "traced_for=custom_linear_solve transpose_solve, fun=my_tr_solve, arg_names=x",
            "None",  # TODO(necula): there are missing debug info
        ])

  def test_custom_root_errors(self):
    tracer_spy = TracerSpy()
    def dummy_root_usage(x):
      tracer_spy.append(x)
      def my_f(x):
        tracer_spy.append(x)
        return x - 3.
      def my_solve(f, x):
        tracer_spy.append(x)
        return x
      def my_transpose_solve(f, x):
        tracer_spy.append(x)
        return x
      return lax.custom_root(my_f, 0., my_solve, my_transpose_solve)

    self._check_tracers_and_jaxprs(
        jax.jit(lambda x: jax.jvp(dummy_root_usage, (x,), (0.0,))),
        0.,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=<lambda>, arg_names=x, result_paths=[0],[1]",
            "None",  # TODO(necula): there are missing Jaxpr debug info
        ],
        expected_tracer_debug_infos=[
            "traced_for=custom_root, fun=my_f, arg_names=x",
            "traced_for=custom_root solve, fun=my_solve, arg_names=x",
            "traced_for=custom_root tangent_solve, fun=my_transpose_solve, arg_names=x",
            "None",  # TODO(necula): there are missing debug info
        ])

  def test_pallas_call(self):
    tracer_spy = TracerSpy()
    def my_kernel(x_ref, y_ref, o_ref):
      tracer_spy.append(x_ref)
      o_ref[...] = x_ref[...] + y_ref[...]

    x = np.arange(256 * 16, dtype=np.float32).reshape((256, 16))

    def my_f(x):
      def my_index_map(i, j):
        tracer_spy.append(i)
        return (i, j)

      return pl.pallas_call(my_kernel,
                            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                            in_specs=(pl.BlockSpec((128, 8), my_index_map),
                                      pl.BlockSpec((128, 8), my_index_map)),
                            out_specs=pl.BlockSpec((128, 8), my_index_map),
                            grid=(pl.cdiv(x.shape[0], 128), pl.cdiv(x.shape[1], 8)))(x, x)

    self._check_tracers_and_jaxprs(
        jax.jit(my_f), x,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=x, result_paths=",
            # TODO(necula): missing Jaxpr debug info
            "None"],
        expected_tracer_debug_infos=[
            "traced_for=pallas_call index_map, fun=my_index_map, arg_names=i,j",
            "traced_for=pallas_call, fun=my_kernel, arg_names=x_ref,y_ref,o_ref",
        ],
        check_lowering=False,  # We need interpret mode on CPU. TODO(necula)
    )

  def test_checkify_pallas_call(self):
    tracer_spy = TracerSpy()
    def kernel(x_ref, y_ref):
      tracer_spy.append(x_ref)
      y_ref[...] = jnp.log(x_ref[...])

    def my_f(input):
      out_shape = jax.ShapeDtypeStruct(input.shape, input.dtype)
      pallas_call = pl.pallas_call(kernel,
                                   out_shape=out_shape)
      checked_call = checkify.checkify(pallas_call,
                                       errors=checkify.nan_checks)
      return checked_call(input)[1]

    self._check_tracers_and_jaxprs(
        jax.jit(my_f),
        jnp.arange(4, dtype=jnp.float32) - 2,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=input, result_paths=",
            "None",  # TODO(necula): missing tracer debug info
        ],
        expected_tracer_debug_infos=[
            "traced_for=pallas_call, fun=kernel, arg_names=x_ref,y_ref",
        ],
        check_lowering=False,  # We need interpret mode on CPU. TODO(necula)
    )

  def test_composite(self):
    tracer_spy = TracerSpy()
    scale = np.array([0.5, 0.4, 0.3], dtype=np.float32)
    @functools.partial(lax.composite, name="my.consts")
    def my_consts(x):
      tracer_spy.append(x)
      return x / scale


    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)

    self._check_tracers_and_jaxprs(
        jax.jit(my_consts), x,
        tracer_spy=tracer_spy,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_consts, arg_names=x, result_paths=",
            "None"
        ],
        expected_tracer_debug_infos=[
            "traced_for=composite, fun=my_consts, arg_names=x"])


class EagerPmapMixin:

  def setUp(self):
    super().setUp()
    stack = contextlib.ExitStack()
    stack.enter_context(jtu.thread_local_config_context(jax_disable_jit=True, jax_eager_pmap=True))
    stack.enter_context(jtu.ignore_warning(
        message="Some donated buffers were not usable", category=UserWarning))
    self.addCleanup(stack.close)


@jtu.pytest_mark_if_available('multiaccelerator')
class PythonPmapEagerTest(EagerPmapMixin, jtu.JaxTestCase):
  def test_pmap_lower_arg_info(self):
    def f(x, y, *args, **kwargs):
      return y['hi'] + args[1] + sum(kwargs.values())

    lowered = jax.pmap(f).lower(
        {'hi': jnp.array([1.])}, {'hi': jnp.array([2.])}, jnp.array([3.]),
        jnp.array([4.]), z=jnp.array([5.]), w=jnp.array([6.]))
    hlo_str = lowered.as_text("stablehlo", debug_info=True)
    self.assertNotIn("\"x\"", hlo_str)
    self.assertIn("y['hi']", hlo_str)
    self.assertIn("args[0]", hlo_str)
    self.assertIn("args[1]", hlo_str)
    self.assertIn("kwargs['z']", hlo_str)
    self.assertIn("kwargs['w']", hlo_str)

  def test_pmap_lower_result_info(self):
    def f(x, y, z):
      return {'a': x, 'b': [y]}

    lowered = jax.pmap(f).lower(jnp.array([1.]), (jnp.array([2]),),
                                [jnp.array([3])])
    hlo_str = lowered.as_text("stablehlo", debug_info=True)
    self.assertIn("jax.result_info = \"['a']\"", hlo_str)
    self.assertIn("jax.result_info = \"['b'][0][0]\"", hlo_str)

  def testLowerCompileArgTypeMismatch(self):
    f = jax.pmap(lambda x: x - lax.pmean(x, 'i'), axis_name='i')
    shape = (jax.device_count(), 4)
    x = np.arange(math.prod(shape), dtype=int).reshape(shape)
    x_f32 = x.astype(jnp.float32)
    x_i32 = x.astype(jnp.int32)
    f_exe = f.lower(x_f32).compile()
    self.assertRaisesRegex(
        TypeError,
        r"Argument types differ .*"
        r"The mismatches are:\n"
        r"Argument 'x' compiled with.*float32.*and called with.*int32.*",
        lambda: f_exe(x_i32))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
