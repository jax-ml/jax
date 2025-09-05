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

from absl.testing import absltest, parameterized

import jax
from jax import lax
from jax import numpy as jnp

from jax._src import config
from jax._src import core
from jax._src.frozen_dict import FrozenDict
from jax._src.interpreters import ad
from jax._src.interpreters import mlir
from jax._src import mesh as mesh_lib
from jax._src import repro
from jax._src import test_util as jtu
from jax._src import tree_util
from jax.sharding import PartitionSpec as P
from jax.experimental import shard_map as exp_shard_map

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
@dataclasses.dataclass(frozen=True)
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
    assert config.repro_dir.value, "You must set JAX_REPRO_DIR=something"

    repro._thread_local_state.emitted_repros = []
    res = func(*args, **kwargs)
    repro_stack_entry = repro._thread_local_state.call_stack[-1]
    assert repro_stack_entry.func.is_main
    repro._emit_repro(repro_stack_entry, "collect_repro")
    return res


  def collect_and_check(self, func: Callable, *args,
                        expect_exception: tuple[Any, str] | None = None,
                        **kwargs) -> str:
    # TODO: keep this in the test file to make the check_traceback work
    assert config.repro_dir.value, "You must set JAX_REPRO_DIR=something"

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

  def test_with_constants(self):
    ct1 = jnp.arange(16.)
    @jax.jit
    def f1(x):
      ct2 = np.arange(16.)
      return x + ct1 + ct2

    self.collect_and_check(f1, np.ones(shape=(16,), dtype=np.float32))

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

  def test_duplicate_arg(self):
    @jax.jit
    def f(x, y):
      return x + y
    @jax.jit
    def g(x):
      return f(x, x)

    self.collect_and_check(g, np.ones((4,), dtype=np.float32))


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

  def test_partial(self):
    # Partial used in user-space.
    def my_fun(x, y):
      return x + y
    add_one = tree_util.Partial(my_fun, 1)
    @jax.jit
    def call_func(f, *args):
      return f(*args) + 5

    self.collect_and_check(call_func, add_one, 2)

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

  def test_remat_custom_jvp_policy(self):
    self.skipTest("Very slow: 12s")
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
    #@jax.remat
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

  def test_basic_pass_through_jit(self):
    # self.skipTest("Fix passing Partial")
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

    with jtu.ignore_warning(category=UserWarning, message=".* includes a pmap"):
      self.collect_and_check(splitjvp, x)

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
    output_sharding = jax.sharding.NamedSharding(abs_mesh, P(None, "x"))
    @jax.jit
    def f(a):
      b = a @ a.T
      return jax.lax.with_sharding_constraint(b, output_sharding)

    a = jnp.arange(4 * 4, dtype=np.float32).reshape((4, 4))
    self.collect_and_check(f, a)


  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_jit_with_custom(self, mesh):
    np_inp = np.arange(16).reshape(8, 2)
    s_x_y = jax.sharding.NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s_x_y)
    arr2 = jax.device_put(np_inp, s_x_y)

    @partial(jax.jit,
             in_shardings=(Value(s_x_y, s_x_y),),
             out_shardings={"c": s_x_y})
    def g(xy_value):
      return dict(c=xy_value.a * xy_value.b)

    self.collect_and_check(g, Value(arr, arr2))

  @parameterized.named_parameters(
      dict(testcase_name=f"_exp_{exp}", exp=exp)
      for exp in (True, False))
  @jtu.with_explicit_mesh((2, 2), ('x', 'y'))
  def test_shard_map_0(self, mesh, exp: bool = False):
    np_inp = np.arange(16).reshape(8, 2)
    s_x_y = jax.sharding.NamedSharding(mesh, P('x', 'y'))
    arr = jax.device_put(np_inp, s_x_y)
    arr2 = jax.device_put(np_inp, s_x_y)

    def g(x, y):
      return x * y

    shard_map = exp_shard_map.shard_map if not exp else jax.shard_map
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

  def test_user_func_modifies_args(self):
    @jax.jit
    def f(op_list: list[Any], op_dict: dict[str, Any]):
      op1 = op_list.pop()
      op2 = op_list.pop()
      op3 = op_dict["a"]
      del op_dict["a"]
      return op1 + op2 + op3

    self.collect_and_check(f, [np.ones((4,)), np.ones((4,))], dict(a=np.ones((4,))))

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


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
