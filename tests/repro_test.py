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

import dataclasses
from functools import partial

import enum
import math
import numpy as np
from typing import Any, Callable

from absl.testing import absltest

import jax
from jax import lax
from jax import numpy as jnp

from jax._src import config
from jax._src import core
from jax._src.interpreters import ad
from jax._src.interpreters import mlir
from jax._src import repro
from jax._src import test_util as jtu
from jax._src import tree_util
# from jax.sharding import NamedSharding
# from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

config.parse_flags_with_absl()
jtu.request_cpu_devices(8)

class ErrorOn(enum.Enum):
  TRACING = 0
  LOWERING = 1
  JVP = 2
  TRANSPOSE = 3

raise_error_p = core.Primitive("raise_error")
def raise_error(x, *, on: ErrorOn):
  if on == ErrorOn.TRACING:
    raise ValueError("raise_error on TRACING")
  return raise_error_p.bind(x, on=on)

raise_error_p.def_abstract_eval(lambda xval, **_: xval)
def raise_error_lowering(ctx, x, *, on: ErrorOn):
  if on == ErrorOn.LOWERING:
    raise ValueError("raise_error on LOWERING")
  return [x]
mlir.register_lowering(raise_error_p, raise_error_lowering)

def raise_error_jvp_rule(primals, tangents, *,
                         on: ErrorOn):
  if on == ErrorOn.JVP:
    raise ValueError("raise_error in JVP")
  return primals[0], tangents[0]

ad.primitive_jvps[raise_error_p] = raise_error_jvp_rule


@tree_util.register_dataclass
@dataclasses.dataclass
class Value:
  a: float
  b: float


_test_counter = 1

@jtu.with_config(jax_traceback_filtering="off")
class ReproTest(jtu.JaxTestCase):

  def setUp(self):
    self.repro_state_snapshot = repro._thread_local_state.snapshot_state()
    super().setUp()

  def tearDown(self) -> None:
    repro._thread_local_state.restore_state(self.repro_state_snapshot)
    super().tearDown()

  def collect_repro(self, func: Callable, *args, **kwargs):
    assert config.repro_dir.value

    try:
      repro._thread_local_state.emitted_repros = []
      prev = repro._thread_local_state.emit_repro_on_success
      repro._thread_local_state.emit_repro_on_success = True
      res = func(*args, **kwargs)
      repro_stack_entry = repro._thread_local_state.call_stack[-1]
      assert repro_stack_entry.func.is_main
      repro._emit_repro(repro_stack_entry, "collect_repro")
      return res
    finally:
      repro._thread_local_state.emit_repro_on_success = prev

  def collect_and_check(self, func: Callable, *args,
                        expect_exception: tuple[Any, str] | None = None,
                        **kwargs) -> str:
    # TODO: keep this in the test file to make the check_traceback work
    assert config.repro_dir.value

    if expect_exception is None:
      result = self.collect_repro(func, *args, **kwargs)
    else:
      with self.assertRaisesRegex(*expect_exception):
        self.collect_repro(func, *args, **kwargs)
      result = None

    if len(repro._thread_local_state.emitted_repros) == 0:
      assert False, "No repros have been collected"
    elif len(repro._thread_local_state.emitted_repros) > 1:
      assert False, (
          "Multiple repros have been collected, perhaps you are running the "
          "function in eager mode? Seen: " +
          ", ".join(str(p) for p, _ in repro._thread_local_state.emitted_repros))
    repro_path, repro_source = repro._thread_local_state.emitted_repros[0]

    # Now try to evaluate the source
    compiled = compile(repro_source, repro_path, "exec")
    custom_namespace = {}
    custom_namespace['__builtins__'] = __builtins__
    try:  # Disable repros for this exec
      prev = repro._thread_local_state.emit_repro_enabled
      repro._thread_local_state.emit_repro_enabled = False
      if expect_exception is None:
        exec(compiled, custom_namespace, custom_namespace)
        self.assertAllClose(custom_namespace["_repro_result"],
                            repro.EmitFuncContext.flatten_custom_pytree(result))
      else:
        with self.assertRaisesRegex(*expect_exception):
          exec(compiled, custom_namespace, custom_namespace)
    finally:
      repro._thread_local_state.emit_repro_enabled = prev
    return repro_source

  def test_basic(self):
    @jax.jit
    def f1(x, y1, y2):
      v = x + jnp.sin(y1)
      return v + jnp.cos(y2)

    @jax.jit
    def f2(x):
      return jax.jit(f1)(x, x , x)

    self.collect_and_check(f2, np.ones((8,), dtype=np.float32))

  def test_eval_shape(self):
    @jax.jit
    def f1(x, y1, y2):
      v = x + jnp.sin(y1)
      return v + jnp.cos(y2)

    def f2(x):
      return jax.jit(f1)(x, x, x)

    self.collect_and_check(lambda *args: jax.eval_shape(f2, *args),
                      np.ones((8,), dtype=np.float32))

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

  def test_static_argnums_argnames(self):
    @dataclasses.dataclass(frozen=1)
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

    self.collect_and_check(f,(0., 1.), CustomType(2), 3., (40, 41, 42),
                           a12 = (5., 6.), as2 = CustomType(22))

  def test_partial(self):
    # Partial used in user-space.
    def my_fun(x, y):
      return x + y
    add_one = tree_util.Partial(my_fun, 1)
    @jax.jit
    def call_func(f, *args):
      return f(*args) + 5

    self.collect_and_check(call_func, add_one, 2)

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

  def test_cond_without_jax_boundary(self):
    def f1(x, i: int):
      # Bypass the jax_boundary, to test what happens
      from jax._src.lax.control_flow.conditionals import _cond
      return _cond(x[i] >= 0., jnp.sin, jnp.cos, x)

    @jax.jit
    def f2(x):
      acc = x
      for i in range(3):
        acc = f1(acc, i)
      return acc

    with self.assertRaisesRegex(ValueError,
                                "Seen primitive cond containing Jaxprs"):
      f2(np.arange(8, dtype=np.float32))

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

  def test_with_fake_arrays(self):
    pass

  def test_aot_trace_lower(self):
    @jax.jit
    def f1(x):
      return jnp.cos(x)

    traced = f1.trace(np.ones((8, 8), dtype=np.float32))
    self.assertIn("cos", str(traced.jaxpr))
    lowered1 = traced.lower()
    lowered2 = f1.lower(np.ones((8, 8), dtype=np.float32))
    self.assertEqual(lowered1.as_text(), lowered2.as_text())

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

  def test_linearize(self):
    def f1(x):
      return jnp.sin(x)

    @jax.jit
    def f2(x):
      y, f_jvp = jax.linearize(f1, x)
      return f_jvp(jnp.full_like(x, 0.2))
    self.collect_and_check(f2, np.ones((8,), dtype=np.float32))

  def test_grad(self):
    @jax.jit
    def f1(x):
      return jnp.sin(x)

    def f2(x):
      return jnp.sum(f1(x))

    self.collect_and_check(jax.jit(jax.grad(f2)),
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

  def test_remat_example(self):
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
    # TODO: this test does something strange to the test state
    self.skipTest("Very slow: 12s")
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

    jtu.check_grads(f, (3.,), order=2, modes=['rev'])

    def g(x):
      return lax.scan(lambda x, _: (f(x), None), x, None, length=2)[0]
    jtu.check_grads(g, (3.,), order=2, modes=['rev'])

  def test_vmap(self):
    def f1(x):
      return jnp.sin(x) + raise_error(x, on=ErrorOn.TRACING)

    jax.vmap(f1)(np.ones((8,), dtype=np.float32))

  def test_basic_pass_through_jit(self):
    self.skipTest("Fix passing Partial")
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

  def test_prng(self):
    # Error _unstack() missing 1 required positional arg, unless Func.__get__
    # is implemented
    @jax.jit
    def f():
      key = jax.random.key(0)
      key, split = jax.random.split(key)
      return jax.random.uniform(split, (4,))
    self.collect_and_check(f)

  def test_repro_2(self):
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
    self.collect_and_check(splitjvp, x)

  def test_repro_multiple_calls_same_body(self):
    # We invoke a function multiple times, but it results in the same
    # body
    @jax.jit
    def f1(x):
      return jnp.sin(x)

    @jax.jit
    def f2(x):
      return (f1(x),
              f1(jnp.full(x.shape + (8,), 5.)))

    self.collect_and_check(f2, np.float32(1.))

  def test_multiple_calls_0(self):
    # Same body multiple invocations0
    @jax.jit
    def f1(x):
      a = jnp.sin(x)
      return a + jnp.sin(x)

    self.collect_and_check(f1, np.float32(1.))

  def test_multiple_calls_1(self):
    # We invoke a function multiple times, but it results in the same
    # body
    @jax.jit
    def f1(x):
      # We call twice for each shape
      a1 = jnp.sin(x)
      a2 = jnp.sin(x)
      large = jnp.full(x.shape + (8,), 5., dtype=x.dtype)
      b1 = jnp.sin(large)
      b2 = jnp.sin(large)
      return (a1 + a2, b1 + b2)

    self.collect_and_check(f1, np.float32(1.))


  def test_repro_multiple_calls_different_body(self):
    # We invoke a function multiple times, but it results in the same
    # body
    @jax.jit
    def f1(x):
      # called first with shape [] and then with shape [8]
      # In repro should be turned into two functions
      return jnp.sin(x) if x.shape else jnp.cos(x)

    @jax.jit
    def f2(x):
      return x + f1(x) + jax.jvp(f1, (x,), (jnp.full_like(x, .1),))[0]

    @jax.jit
    def f3(x):
      # We call twice for each shape
      two_shapes = [x, jnp.full(x.shape + (8,), 5.)] * 2
      calls = [f2(v) for v in two_shapes]
      return sum(calls)

    self.collect_and_check(f3, np.float32(1.))

  def test_user_calls_user(self):
    @partial(jax.jit, static_argnums=(1,))
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
      def f2(x2):  # goes under "f1"
        v3 = x2 + x2
        def f3(x3):  # goes under "main"
          return x3 + x3
        def f4(x4):  # goes under "f1"
          return x4 + v1  # goes under "f1"
        def f5(x5):  # goes under "f2"
          return x5 + v3
        return v1 + x2 + jax.jit(f3)(v1) + jax.jit(f4)(v1) + jax.jit(f5)(v1)
      return jax.jit(f2)(x1)
    self.collect_and_check(f1, 42.)

  def test_sharding_abstract_mesh(self):
    if jax.local_device_count() < 2:
      self.skipTest("Need at least 2 devices")

    abs_mesh = jax.sharding.AbstractMesh((2,), 'x')
    input_sharding = jax.sharding.NamedSharding(abs_mesh, P("x", None))
    output_sharding = jax.sharding.NamedSharding(abs_mesh, P(None, "x"))
    @jax.jit
    def f(a):
      b = a @ a.T
      return jax.lax.with_sharding_constraint(b, output_sharding)

    a = jnp.arange(16 * 16, dtype=np.float32).reshape((16, 16))
    self.collect_and_check(f, a)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
