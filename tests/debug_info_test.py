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

from absl.testing import absltest, parameterized
import jax
from jax import lax
from jax._src import config
from jax._src import core
from jax._src import test_util as jtu
from jax._src.compilation_cache import is_persistent_cache_enabled
import jax.custom_batching
import jax.custom_derivatives
import jax.custom_transpose
from jax.errors import UnexpectedTracerError
import jax.numpy as jnp
import numpy as np


config.parse_flags_with_absl()
jtu.request_cpu_devices(8)


class DebugInfoTest(jtu.JaxTestCase):

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

  def test_escaped_tracer_transform_name(self):
    with self.assertRaisesRegex(UnexpectedTracerError,
                                "for jit"):
      jax.jit(self.helper_save_tracer)(1)
      _ = self._saved_tracer+1

    with self.assertRaisesRegex(UnexpectedTracerError,
                                "for pmap"):
      jax.pmap(self.helper_save_tracer)(jnp.ones((1, 2)))
      _ = self._saved_tracer+1

    with self.assertRaisesRegex(UnexpectedTracerError,
                                "for jit"):
      jax.eval_shape(self.helper_save_tracer, 1)
      _ = self._saved_tracer+1

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
