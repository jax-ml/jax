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

from absl.testing import absltest, parameterized
import jax
from jax import lax
from jax._src import api_util
from jax._src import config
from jax._src import core
from jax._src import test_util as jtu
from jax._src.compilation_cache import is_persistent_cache_enabled
import jax.custom_batching
import jax.custom_derivatives
import jax.custom_transpose
from jax.experimental import pallas as pl
import jax.numpy as jnp
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
  res = f"traced_for={dbg.traced_for}, fun={fun_src_info}, arg_names={dbg.arg_names}"
  if isinstance(dbg, core.JaxprDebugInfo):
    res += f", result_paths={dbg.result_paths}"
  return res


class DebugInfoTest(jtu.JaxTestCase):

  def _check_tracers_and_jaxprs(self, traceable: Any,
                                *args,
                                expected_jaxpr_debug_infos: list[str],
                                leaked_tracers: list[core.Tracer] = [],
                                expected_tracer_debug_infos: list[str] = [],
                                **kwargs):
    """Checks for expected debug info in all jaxprs, and in leaked tracers.

    The `traceable.trace(*args, **kwargs)` is traced to a Jaxpr, and the
    debug infos in the nested Jaxprs are first converted to strings using
    `_debug_info_to_string` and then compared against `expected_jaxpr_debug_infos`.

    One way in which the debug info is used in JAX is for leaked tracer
    description, or for ConcretizationErrors. Optionally,
    the tracing of `fun` can append tracers to `leaked_tracers`, and those
    tracers must have debugging info matching `expected_tracer_debug_infos`.
    """
    self.assertTrue(hasattr(traceable, "trace"))
    traced = traceable.trace(*args, **kwargs)
    all_jaxprs = _collect_jaxprs(traced.jaxpr.jaxpr)

    jaxprs_debug_infos = [_debug_info_to_string(j.debug_info)
                          for j in all_jaxprs]
    self.assertEqual(expected_jaxpr_debug_infos, jaxprs_debug_infos)
    self._check_tracers(leaked_tracers, expected_tracer_debug_infos)

  def _check_tracers(self,
                     leaked_tracers: list[core.Tracer],
                     expected_tracer_debug_infos: list[str]):
    leaked_tracer_debug_infos = [_debug_info_to_string(t._debug_info) if hasattr(t, "_debug_info") else "None"
                                 for t in leaked_tracers]
    self.assertEqual(expected_tracer_debug_infos, leaked_tracer_debug_infos)

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

  def test_jit_lower_arg_info(self):
    def f(x, y, *args, **kwargs):
      return y['hi'] + args[1] + sum(kwargs.values())

    lowered = jax.jit(f).lower({'hi': 1.}, {'hi': 2.}, 3., 4., z=5., w=6.)
    hlo_str = lowered.as_text("stablehlo", debug_info=True)
    self.assertNotIn("\"x\"", hlo_str)
    self.assertIn("y['hi']", hlo_str)
    self.assertNotIn("args[0]", hlo_str)
    self.assertIn("args[1]", hlo_str)
    self.assertIn("kwargs['z']", hlo_str)
    self.assertIn("kwargs['w']", hlo_str)

    hlo_str = lowered.as_text("stablehlo", debug_info=False)
    for s in ("\"x\"", "y['hi']", "args[0]", "args[1]", "kwargs['z']", "kwargs['w']"):
      self.assertNotIn(s, hlo_str)

  @parameterized.parameters([0, 2, [(0, 2)]])
  def test_jit_lower_arg_info_static_argnums(self, static_argnums):
    def f(x, y, *args, **kwargs):
      return y['hi'] + args[1] + sum(kwargs.values())

    lowered = jax.jit(f, static_argnums=static_argnums).lower(
        (1.,), {'hi': 2.}, 3., 4., z=5., w=6.)

    hlo_str = lowered.as_text("stablehlo", debug_info=True)
    self.assertNotIn("\"x\"", hlo_str)
    self.assertIn("y['hi']", hlo_str)
    self.assertNotIn("args[0]", hlo_str)
    self.assertIn("args[1]", hlo_str)
    self.assertIn("kwargs['z']", hlo_str)
    self.assertIn("kwargs['w']", hlo_str)

    hlo_str = lowered.as_text("stablehlo", debug_info=False)
    for s in ("\"x\"", "y['hi']", "args[0]", "args[1]", "kwargs['z']", "kwargs['w']"):
      self.assertNotIn(s, hlo_str)

  @parameterized.parameters(['a', 'b', [('a', 'b')]])
  def test_jit_lower_arg_info_static_argnames(self, static_argnames):
    def f(x, y, *args, **kwargs):
      return y['hi'] + args[1] + kwargs['z'] + kwargs['w']

    lowered = jax.jit(f, static_argnames=static_argnames).lower(
        (1.,), {'hi': 2.}, 3., 4., z=5., w=6., a=7., b=8.)
    hlo_str = lowered.as_text("stablehlo", debug_info=True)
    self.assertNotIn("\"x\"", hlo_str)
    self.assertIn("y['hi']", hlo_str)
    self.assertNotIn("args[0]", hlo_str)
    self.assertIn("args[1]", hlo_str)
    self.assertIn("kwargs['z']", hlo_str)
    self.assertIn("kwargs['w']", hlo_str)
    self.assertNotIn("kwargs['a']", hlo_str)
    self.assertNotIn("kwargs['b']", hlo_str)

    hlo_str = lowered.as_text("stablehlo", debug_info=False)
    for s in (
        "\"x\"", "y['hi']", "args[0]", "args[1]", "kwargs['z']",
        "kwargs['w']", "kwargs['a']", "kwargs['b']"
    ):
      self.assertNotIn(s, hlo_str)

  def test_jit_lower_result_info(self):
    def f(x, y, z):
      return {'a': x, 'b': [y]}

    hlo_str = jax.jit(f).lower(1., (2,), [3]).as_text("stablehlo", debug_info=True)
    self.assertIn("jax.result_info = \"['a']\"", hlo_str)
    self.assertIn("jax.result_info = \"['b'][0][0]\"", hlo_str)

  def test_jit_lower_compile_arg_type_mismatch(self):
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

  def test_jit_bad_input(self):
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
  def test_cache_miss_explanations(self):
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
  def test_cache_miss_explanations_new_function_in_loop(self):
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
  def test_cache_miss_explanations_unpacks_transforms(self):
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

  def test_cache_miss_explanations_no_source_info(self):
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
    leaked_tracers: list[core.Tracer] = []

    def my_f(x_dict, y):
      leaked_tracers.append(x_dict["a"])
      return dict(c=x_dict["a"] + x_dict["b"], d=y)

    self._check_tracers_and_jaxprs(
        jax.jit(my_f),
        dict(a=1, b=2), 3,
        leaked_tracers=leaked_tracers,
        expected_jaxpr_debug_infos=[
            'traced_for=jit, fun=my_f, arg_names=(\"x_dict[\'a\']", "x_dict[\'b\']", \'y\'), result_paths=("[\'c\']", "[\'d\']")'
        ],
        expected_tracer_debug_infos=[
            'traced_for=jit, fun=my_f, arg_names=("x_dict[\'a\']", "x_dict[\'b\']", \'y\')',
        ])

  def test_nested_jit(self):
    leaked_tracers: list[core.Tracer] = []

    def my_f(x, y):
      leaked_tracers.append(x)

      def my_g(u, v):
        leaked_tracers.append(u)
        return dict(c=u * v, d=v)

      return jax.jit(my_g)(y, x)["c"]

    self._check_tracers_and_jaxprs(
        jax.jit(my_f),
        2, 3,
        leaked_tracers=leaked_tracers,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=('x', 'y'), result_paths=('',)",
            "traced_for=jit, fun=my_g, arg_names=('u', 'v'), result_paths=(\"[\'c\']\",)"
        ],
        expected_tracer_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=('x', 'y')",
            "traced_for=jit, fun=my_g, arg_names=('u', 'v')"
        ])

  def test_vjp_of_nested_jit(self):
    leaked_tracers: list[core.Tracer] = []

    def my_f(x, y):
      leaked_tracers.append(x)

      def my_g(u, v):
        leaked_tracers.append(u)
        return dict(c=u * v, d=v)

      return jax.jit(my_g)(y, x)["c"]

    self._check_tracers_and_jaxprs(
        jax.jit(lambda x, y, res_ct: jax.vjp(my_f, x, y)[1](res_ct)),
        2., 3., 0.3,
        leaked_tracers=leaked_tracers,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=<lambda>, arg_names=('x', 'y', 'res_ct'), result_paths=('[0]', '[1]')",
            # TODO(necula): missing debug info
            'None',
            'None'
        ],
        expected_tracer_debug_infos=[
            # TODO(necula): missing debug info
            "None",
            "traced_for=jit, fun=my_g, arg_names=('u', 'v')"
        ])

  def test_vmap_of_nested_jit(self):
    leaked_tracers: list[core.Tracer] = []

    def my_f(x, y):
      leaked_tracers.append(x)

      def my_g(u, v):
        leaked_tracers.append(u)
        return dict(c=u * v, d=v)

      return jax.jit(my_g)(y, x)["c"]

    self._check_tracers_and_jaxprs(
        jax.jit(jax.vmap(my_f)),
        np.ones((8,), dtype=np.float32), np.zeros((8,), dtype=np.float32),
        leaked_tracers=leaked_tracers,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=('x', 'y'), result_paths=('',)",
            # TODO(necula): missing debug info
            'None',
        ],
        expected_tracer_debug_infos=[
            # TODO(necula): missing debug info
            "None",
            "traced_for=jit, fun=my_g, arg_names=('u', 'v')"
        ])

  def test_cond(self):
    leaked_tracers: list[core.Tracer] = []

    def my_f(x):
      def my_true_branch(a, b):
        leaked_tracers.append(a)
        return a + b

      def my_false_branch(c, d):
        leaked_tracers.append(c)
        return c - d

      return lax.cond(x >= 0, my_true_branch, my_false_branch, x, x)

    self._check_tracers_and_jaxprs(
        jax.jit(my_f),
        0,
        leaked_tracers=leaked_tracers,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=('x',), result_paths=('',)",
            # TODO(necula): some Jaxprs without debug info
            'None',
            'None'],
        expected_tracer_debug_infos=[
            "traced_for=cond, fun=my_true_branch, arg_names=('a', 'b')",
            "traced_for=cond, fun=my_false_branch, arg_names=('c', 'd')"
        ])

  def test_grad_cond_with_remat(self):
    leaked_tracers: list[core.Tracer] = []

    def my_f(x, y):
      # The cond branches return two things, and only the first is needed
      # in the residuals.
      def my_true_branch(a, b):
        leaked_tracers.append(a)
        return (a + 1, a + b)

      def my_false_branch(c, d):
        leaked_tracers.append(c)
        return (c - 1, c - d)

      def my_g(x, y):
        # x1 does not depend on y
        x1, y1 = lax.cond(x >= 0, my_true_branch, my_false_branch, x, y)
        leaked_tracers.append(x1)
        return x1, y1

      x2, y2 = jax.remat(my_g)(x, y)
      return y2 + lax.sin(x2)

    self._check_tracers_and_jaxprs(
        jax.jit(jax.grad(my_f)),
        1., 2.,
        leaked_tracers=leaked_tracers,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=('x', 'y'), result_paths=('',)",
            # TODO(necula): some Jaxprs without debug info
            'None',
            'None',
            'None',
            'None',
            'None',
            'None',
            'None'
        ],
        expected_tracer_debug_infos=[
            "traced_for=cond, fun=my_true_branch, arg_names=('a', 'b')",
            "traced_for=cond, fun=my_false_branch, arg_names=('c', 'd')",
            "traced_for=checkpoint / remat, fun=my_g, arg_names=('x', 'y')"
        ])

  def test_while_loop(self):
    leaked_tracers: list[core.Tracer] = []

    def my_f(x):
      def my_cond(a):
        leaked_tracers.append(a)
        return a <= 8

      def my_body(b):
        leaked_tracers.append(b)
        return b + 1

      return lax.while_loop(my_cond, my_body, x)

    self._check_tracers_and_jaxprs(
        jax.jit(my_f),
        0,
        leaked_tracers=leaked_tracers,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=('x',), result_paths=('',)",
            # TODO(necula): some Jaxprs without debug info
            'None',
            'None'],
        expected_tracer_debug_infos=[
            "traced_for=while_cond, fun=my_cond, arg_names=('a',)",
            "traced_for=while_loop, fun=my_body, arg_names=('b',)"
        ])

  def test_eval_shape(self):
    leaked_tracers: list[core.Tracer] = []

    def my_f(x):
      leaked_tracers.append(x)
      return x

    _ = jax.eval_shape(my_f, 0)
    self._check_tracers(leaked_tracers, [
        "traced_for=jit, fun=my_f, arg_names=('x',)"
    ])

  def test_pmap(self):
    leaked_tracers: list[core.Tracer] = []

    def my_f(x):
      leaked_tracers.append(x)
      return jnp.sin(x)

    self._check_tracers_and_jaxprs(
        jax.pmap(my_f),
        np.ones((jax.device_count(),), dtype=np.float32),
        expected_jaxpr_debug_infos=[
            "traced_for=pmap, fun=my_f, arg_names=('x',), result_paths=('',)"
        ],
        leaked_tracers=leaked_tracers,
        expected_tracer_debug_infos=[
            "traced_for=pmap, fun=my_f, arg_names=('x',)"
        ],
    )

  def test_pmap_of_grad(self):
    leaked_tracers: list[core.Tracer] = []

    def my_f(x):
      leaked_tracers.append(x)
      return jnp.sin(x)

    self._check_tracers_and_jaxprs(
        jax.jit(jax.pmap(jax.grad(my_f))),
        np.ones((jax.device_count(),), dtype=np.float32),
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=('x',), result_paths=('',)",
            # TODO(necula): missing debug_info
            'None'
        ],
        leaked_tracers=leaked_tracers,
        expected_tracer_debug_infos=[
            # TODO(necula): missing debug_info
            'None'
        ],
    )

  def test_remat(self):
    leaked_tracers: list[core.Tracer] = []

    def my_f(x):
      @jax.remat
      def my_g(y):
        leaked_tracers.append(y)
        return lax.sin(y)

      return my_g(x)

    self._check_tracers_and_jaxprs(
        jax.jit(my_f),
        0.,
        leaked_tracers=leaked_tracers,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=('x',), result_paths=('',)",
            # TODO(necula): some Jaxprs without debug info
            'None'],
        expected_tracer_debug_infos=[
            "traced_for=checkpoint / remat, fun=my_g, arg_names=('y',)"
        ])

  def test_grad_remat(self):
    leaked_tracers: list[core.Tracer] = []

    def my_f(x):
      @jax.remat
      def my_g(y):
        leaked_tracers.append(y)
        return lax.sin(lax.sin(y))

      return my_g(my_g(x))

    self._check_tracers_and_jaxprs(
        jax.jit(jax.grad(my_f)),
        0.,
        leaked_tracers=leaked_tracers,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=('x',), result_paths=('',)",
            # TODO(necula): some Jaxprs without debug info
            'None',
            'None'],
        expected_tracer_debug_infos=[
            "traced_for=checkpoint / remat, fun=my_g, arg_names=('y',)"
        ])

  def test_pallas_call(self):
    leaked_tracers: list[core.Tracer] = []

    def my_kernel(x_ref, y_ref, o_ref):
      leaked_tracers.append(x_ref)
      o_ref[...] = x_ref[...] + y_ref[...]

    x = np.arange(256 * 16, dtype=np.float32).reshape((256, 16))

    def my_f(x):
      def my_index_map(i, j):
        leaked_tracers.append(i)
        return (i, j)

      return pl.pallas_call(my_kernel,
                            out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
                            in_specs=(pl.BlockSpec((128, 8), my_index_map),
                                      pl.BlockSpec((128, 8), my_index_map)),
                            out_specs=pl.BlockSpec((128, 8), my_index_map),
                            grid=(pl.cdiv(x.shape[0], 128), pl.cdiv(x.shape[1], 8)))(x, x)

    self._check_tracers_and_jaxprs(
        jax.jit(my_f), x,
        leaked_tracers=leaked_tracers,
        expected_jaxpr_debug_infos=[
            "traced_for=jit, fun=my_f, arg_names=('x',), result_paths=('',)",
            # TODO(necula): missing Jaxpr debug info
            "None", "None", "None", "None"],
        expected_tracer_debug_infos=[
            # TODO(necula): arg_names seem to be wrong
            # One tracer from every index map
            "traced_for=pallas_call index_map, fun=my_index_map, arg_names=('i[0]', 'i[1]')",
            "traced_for=pallas_call index_map, fun=my_index_map, arg_names=('i[0]', 'i[1]')",
            "traced_for=pallas_call index_map, fun=my_index_map, arg_names=('i[0]', 'i[1]')",
            "traced_for=pallas_call, fun=my_kernel, arg_names=('x_ref', 'y_ref', 'o_ref')",
        ])


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
