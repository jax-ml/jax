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

from dataclasses import dataclass
from functools import partial
import itertools as it
import unittest

from absl.testing import absltest, parameterized

import jax
import jax.numpy as jnp
from jax import typeof
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src import state
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
from jax._src.interpreters import ad
from jax._src import test_util as jtu
from jax._src.util import safe_zip, safe_map
from jax._src.state.discharge import run_state

from jax._src.hijax import (HiPrimitive, HiType, Box, new_box, box_set, box_get,
                            box_effect, register_hitype, ShapedArray, Ty)
from jax.experimental.hijax import VJPHiPrimitive

config.parse_flags_with_absl()

if not jax._src.xla_bridge.backends_are_initialized():
  jax.config.update('jax_num_cpu_devices', 8)

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


@dataclass(frozen=True)
class QArray:
  arr: jax.Array  # int8[m, k]
  scale: jax.Array  # f32[m]

# Define a type
@dataclass(frozen=True)
class QArrayTy(HiType):
  shape: tuple[int, int]

  # how to lower to (lo)jax types
  def lo_ty(self) -> list[ShapedArray]:
    m, k = self.shape
    return [ShapedArray((m, k), jnp.dtype('int8')),
            ShapedArray((m,  ), jnp.dtype('float32'))]
  # these next two are essentially the pytree interface
  def lower_val(self, hi_val: QArray) -> list[jax.Array]:
    return [hi_val.arr, hi_val.scale]
  def raise_val(self, arr, scale) -> QArray:
    return QArray(arr, scale)  # alternative: LowerTrace

  def ref_get_abstract_eval(self, ref_aval, *args, tree):
    arr_aval = core.ShapedArray(self.shape, jnp.dtype('float32'))
    updated_ref = ref_aval.update(inner_aval=arr_aval)
    out, effects = state_primitives.get_p.abstract_eval(
        updated_ref, *args, tree=tree
    )
    assert isinstance(out, core.ShapedArray)
    return QArrayTy(out.shape), effects

  def ref_swap_abstract_eval(self, ref_aval, val_aval, *args, tree):
    arr_aval = core.ShapedArray(self.shape, jnp.dtype('float32'))
    val_arr_aval = core.ShapedArray(val_aval.shape, jnp.dtype('float32'))
    updated_ref = ref_aval.update(inner_aval=arr_aval)
    out_aval, effects = state_primitives.swap_p.abstract_eval(
        updated_ref, val_arr_aval,*args, tree=tree
    )
    assert isinstance(out_aval, core.ShapedArray)
    return QArrayTy(out_aval.shape), effects

  def ref_get_to_lojax(self, ref: state.TransformedRef | jax.Ref,
                       idx: indexing.NDIndexer):
    if isinstance(ref, state.TransformedRef):
      if ref.transforms: raise NotImplementedError(ref)
      ref = ref.ref
    # Unpack Ref type
    ref = ref._refs
    if not all(i.start == 0 and i.size == s
               for i, s in zip(idx.indices, ref.arr.shape)):
      raise NotImplementedError
    outs = [out.get() for out in self.lower_val(ref)]
    return self.raise_val(*outs)

  def ref_swap_to_lojax(self, ref: state.TransformedRef | jax.Ref,
                        val: jax.Array, idx: indexing.NDIndexer):
    if isinstance(ref, state.TransformedRef):
      if ref.transforms: raise NotImplementedError(ref)
      ref = ref.ref
    # Unpack Ref type
    ref = ref._refs
    if not all(i.start == 0 and i.size == s
               for i, s in zip(idx.indices, ref.arr.shape)):
      raise NotImplementedError
    outs = [out.swap(val) for out, val
            in zip(self.lower_val(ref), self.lower_val(val))]
    return self.raise_val(*outs)

  # autodiff
  def to_tangent_aval(self):
    return self  # different from what a pytree would do!
  def vspace_zero(self):
    m, k = self.shape
    return QArray(jnp.zeros((m, k), jnp.dtype('int8')),
                  jnp.ones ((m,  ), jnp.dtype('float32')))

register_hitype(QArray, lambda q: QArrayTy(q.arr.shape))

def to_qarray(x):
  return to_qarray_p.bind(x)

def from_qarray(x):
  return from_qarray_p.bind(x)

class ToQ(HiPrimitive):
  def abstract_eval(_, lo_aval):
    return QArrayTy(lo_aval.shape), set()

  def to_lojax(_, lo_val):
    m, _ = lo_val.shape
    scale = lo_val.max(1) / 32.
    return QArray((lo_val / scale[:, None]).astype('int8'), scale)

  def jvp(_, primals, tangents):
    (x,), (xdot,) = primals, tangents
    return to_qarray(x), to_qarray(xdot)

  def transpose(_, out_bar, __):
    return [from_qarray(out_bar)]
to_qarray_p = ToQ('to_q')

class FromQ(HiPrimitive):
  def abstract_eval(_, hi_aval):
    return ShapedArray(hi_aval.shape, jnp.dtype('float32')), set()

  def to_lojax(_, hi_val):
    return hi_val.arr.astype('float32') * hi_val.scale[:, None]

  def jvp(_, primals, tangents):
    (x,), (xdot,) = primals, tangents
    return from_qarray(x), from_qarray(xdot)

  def transpose(_, out_bar, __):
    return [to_qarray(out_bar)]
from_qarray_p = FromQ('from_q')


@dataclass
class HiTup:
  elts: tuple
  def __repr__(self):
    return 'Tup{' + ','.join(map(repr, self.elts)) + '}'

@dataclass(frozen=True)
class TupTy(HiType):
  tys: tuple[Ty]
  def __repr__(self):
    return 'Tup{' + ','.join(a.str_short() for a in self.tys) + '}'

  def lo_ty(self):
    return list(self.tys)

  def lower_val(self, hi_val: HiTup):
    return [lo for ty, elt in zip(self.tys, hi_val.elts)
            for lo in ty.lower_val(elt)]

  def raise_val(self, *elts_flat):
    elts_iter = iter(elts_flat)
    return HiTup(tuple(ty.raise_val(*it.islice(elts_iter, len(ty.lo_ty())))
                       for ty in self.tys))

register_hitype(HiTup, lambda t: TupTy(tuple(map(typeof, t.elts))))

class MakeTup(HiPrimitive):
  def abstract_eval(_, *in_avals):
    return TupTy(in_avals), set()

  def to_lojax(self, *elts):
    return HiTup(elts)
make_tup_p = MakeTup('make_tup')

class GetTupElt(HiPrimitive):
  def abstract_eval(_, tup, *, idx):
    return tup.tys[idx], set()

  def to_lojax(self, tup, *, idx):
    return tup.elts[idx]
get_tup_elt_p = GetTupElt('get_tup_elt')

def make_tup(*elts):
  return make_tup_p.bind(*elts)

def get_tuple_element(tup, idx):
  return get_tup_elt_p.bind(tup, idx=idx)


class HijaxTest(jtu.JaxTestCase):
  def test_basic_register(self):
    # older test that defines a slightly different QArray internally
    @dataclass(frozen=True)
    class QArray:
      arr: jax.Array
      scale: jax.Array
      axis: int

    @dataclass(frozen=True)
    class QArrayTy(HiType):
      shape: tuple[int, int]
      axis: int

      ndim = property(lambda self: len(self.shape))

      # how to lower to (lo)jax types
      def lo_ty(self) -> list[ShapedArray]:
        m, k = self.shape
        return [ShapedArray((m, k), jnp.dtype('int8')),
                ShapedArray((m,  ), jnp.dtype('float32'))]

      # these next two are essentially the pytree interface
      def lower_val(self, hi_val: QArray) -> list[jax.Array]:
        return [hi_val.arr, hi_val.scale]
      def raise_val(self, arr, scale) -> QArray:
        return QArray(arr, scale, self.axis)

    register_hitype(QArray, lambda q: QArrayTy(q.arr.shape, q.axis))

    q = QArray(jnp.zeros((4, 4), 'int8'), jnp.ones(4, 'float32'), axis=1)
    jax.jit(lambda x: x)(q)  # don't crash

  def test_custom_types_and_primitive(self):
    if config.enable_x64.value: raise unittest.SkipTest("no x64")

    @dataclass(frozen=True)
    class MyArray:
      arr: jax.Array  # always f32

    @dataclass(frozen=True)
    class MyTy(HiType):
      def to_tangent_aval(self):
        return MyTy()
      def str_short(self, short_dtypes=False):
        return 'MyTy'
      def lo_ty(self):
        return [core.ShapedArray((), jnp.dtype('float32'))]
      def lower_val(self, hi_val: MyArray) -> list[jax.Array]:
        return [hi_val.arr]
      def raise_val(self, val) -> MyArray:
        return MyArray(val)

      def __eq__(self, other): return isinstance(other, MyTy)

      def vspace_zero(self):
        return MyArray(jnp.zeros((), 'float32'))
      def vspace_add(self, x, y):
        return add(x, y)
    core.pytype_aval_mappings[MyArray] = lambda _: MyTy()
    dtypes.canonicalize_value_handlers[MyArray] = lambda x: x

    class ToMy(HiPrimitive):
      def is_high(self, _): return True

      def abstract_eval(_, lo_aval):
        return MyTy(), set()

      def to_lojax(_, lo):
        return MyArray(lo)

      def jvp(_, primals, tangents):
        x, x_dot = *primals, *tangents
        return to(x), to(x_dot)

      def transpose(self, out_bar, _):
        return from_(out_bar),

    class FromMy(HiPrimitive):
      def is_high(self, _): return True

      def abstract_eval(_, hi_aval):
        return hi_aval.lo_ty()[0], set()

      def to_lojax(_, hi):
        return hi.arr

      def jvp(_, primals, tangents):
        x, x_dot = *primals, *tangents
        return from_(x), from_(x_dot)

      def transpose(self, out_bar, _):
        return to(out_bar),

    def to(x): return to_p.bind(x)
    to_p = ToMy('to_my')

    def from_(x): return from_p.bind(x)
    from_p = FromMy('from_my')

    def mul(x, y): return mul_p.bind(x, y)
    def add(x, y): return add_p.bind(x, y)

    class MyMul(HiPrimitive):
      def is_high(self, *_): return True

      def abstract_eval(_, hi_x, hi_y):
        if hi_x != hi_y: raise Exception
        return hi_x, set()

      def to_lojax(_, hi_x, hi_y):
        return MyArray(hi_x.arr * hi_y.arr)

      def jvp(_, primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        return mul(x, y), add(mul(x, y_dot), mul(x_dot, y))

      def transpose(self, out_bar, x, y):
        assert ad.is_undefined_primal(x) ^ ad.is_undefined_primal(y)
        if ad.is_undefined_primal(x):
          return mul(out_bar, y), None
        else:
          return None, mul(x, out_bar)

    class MyAdd(HiPrimitive):
      def is_high(self, *_): return True

      def abstract_eval(_, hi_x, hi_y):
        if hi_x != hi_y: raise Exception
        return hi_x, set()

      def to_lojax(_, hi_x, hi_y):
        return MyArray(hi_x.arr + hi_y.arr)

      def jvp(_, primals, tangents):
        assert False  # TODO

      def transpose(self, out_bar, x, y):
        return out_bar, out_bar

    mul_p = MyMul('my_mul')
    add_p = MyAdd('my_add')


    @jax.jit
    def f(x):
      return to(from_(x))

    # test basic to/from jit
    a = MyArray(jnp.ones(()))
    b = f(a)  # don't crash
    self.assertIsInstance(b, MyArray)
    self.assertAllClose(b.arr, jnp.ones(()))

    # test basic to/from autodiff
    b, b_dot = jax.jvp(f, (a,), (a,))
    self.assertIsInstance(b, MyArray)
    self.assertIsInstance(b_dot, MyArray)

    # test mul jit and backward pass

    @jax.jit
    def f(x):
      return mul(x, x)

    b, f_vjp = jax.vjp(f, a)
    self.assertIn('MyTy', str(f_vjp))
    a_grad, = f_vjp(b)
    self.assertIsInstance(a_grad, MyArray)
    self.assertAllClose(a_grad.arr, 2.0, check_dtypes=False)

  def test_stages(self):
    @dataclass(frozen=True)
    class ArrayTuple:
      x0: jax.Array
      x1: jax.Array

    @dataclass(frozen=True)
    class ShapedArrayTuple(HiType):
      x0: ShapedArray
      x1: ShapedArray
      # sharding=None

      # how to lower to (lo)jax types
      def lo_ty(self) -> list[ShapedArray]:
        return [self.x0, self.x1]

      # these next two are essentially the pytree interface
      def lower_val(self, hi_val: ArrayTuple) -> list[jax.Array]:
        return [hi_val.x0, hi_val.x1]
      def raise_val(self, x0, x1) -> ArrayTuple:
        return ArrayTuple(x0, x1)

    register_hitype(ArrayTuple, lambda q: ShapedArrayTuple(
      jax.typeof(q.x0), jax.typeof(q.x1)))

    q = ArrayTuple(jnp.zeros((4, 4), 'int8'), jnp.ones(4, 'float32'))
    jax.jit(lambda x: x).lower(q).as_text()  # don't crash

    compiled = jax.jit(lambda x: x).lower(q).compile()
    compiled(q)  # don't crash

  @parameterized.parameters([False, True])
  def test_while_loop(self, jit):
    q = to_qarray(jnp.ones((2, 2), 'float32'))

    def f(q1, q2):
      def cond_fun(i_carry):
        i, _, __ = i_carry
        return i < 1
      def body_fun(i_carry):
        i, q_carry, _ = i_carry
        q_carry = to_qarray(from_qarray(q_carry))
        return i + 1, q_carry, q
      n, q_out, _ = jax.lax.while_loop(cond_fun, body_fun, (0, q1, q2))
      return n, q_out

    if jit:
      f = jax.jit(f)

    jax.make_jaxpr(f)(q, q)  # doesn't crash
    n, q_out = f(q, q)
    self.assertEqual(n, 1)
    expected = from_qarray(to_qarray(from_qarray(q)))
    self.assertAllClose(from_qarray(q_out), expected, check_dtypes=False)

  @parameterized.parameters([False, True])
  def test_tuple_basic(self, jit):
    def f():
      tup = make_tup(1, 2)
      return get_tuple_element(tup, 1)

    if jit:
      f = jax.jit(f)

    self.assertEqual(f(), 2)

  @parameterized.parameters([False, True])
  def test_ref_to_tuple(self, jit):
    def f():
      tup = make_tup(1, 2)
      ref = jax.new_ref(tup)
      tup_ = ref[...]
      return get_tuple_element(tup_, 1)

    if jit:
      f = jax.jit(f)

    self.assertEqual(f(), 2)

  @parameterized.parameters([False, True])
  def test_run_state(self, jit):
    def f():
      @run_state
      def g(ref_args):
        tup_ref, x_ref = ref_args
        tup = tup_ref[...]
        x_ref[...] = get_tuple_element(tup, 1)

      tup = make_tup(1, 2)
      _, ans =  g((tup, 3))
      return ans

    if jit:
      f = jax.jit(f)

    ans = f()
    self.assertEqual(ans, 2)

  @parameterized.parameters([False, True])
  def test_newstyle_hiprimitive(self, jit):

    class RaiseToStaticPower(VJPHiPrimitive):
      def __init__(self, in_aval, *, power):
        self.in_avals = (in_aval,)
        self.out_aval = in_aval
        self.params = dict(power=power)
        super().__init__()

      def expand(self, x):
        return x ** self.power

      def vjp_fwd(self, x):
        ans = self(x)
        return (ans, x)

      def vjp_bwd(self, res, t, xbar_accum):
        xbar = t * self.power * raise_to_static_power(res, self.power-1)
        xbar_accum.accum(xbar)

      def batch(self, _axis_data, args, in_dims):
        in_dim, = in_dims
        x, = args
        return raise_to_static_power(x, self.power), in_dim

    def raise_to_static_power(x, power):
      x_aval = jax.typeof(x)
      return RaiseToStaticPower(x_aval, power=power)(x)

    def f(x):
      return raise_to_static_power(x, power=3)

    if jit:
      f = jax.jit(f)

    self.assertEqual(f(2.0), 8.0)
    xs = jnp.arange(3.0)
    self.assertAllClose(jax.vmap(f)(xs), xs**3)
    self.assertEqual(jax.grad(f)(2.0), 12.0)


  @parameterized.parameters([False, True])
  def test_newstyle_hiprimitive_retval(self, jit):

    class RaiseToStaticPower(VJPHiPrimitive):
      def __init__(self, in_aval, *, power):
        self.in_avals = (in_aval,)
        self.out_aval = in_aval
        self.params = dict(power=power)
        super().__init__()

      def expand(self, x):
        return x ** self.power

      def vjp_fwd(self, x):
        ans = self(x)
        return (ans, x)

      def vjp_bwd_retval(self, res, t):
        return (t * self.power * raise_to_static_power(res, self.power-1),)

      def batch(self, _axis_data, args, in_dims):
        in_dim, = in_dims
        x, = args
        return raise_to_static_power(x, self.power), in_dim

    def raise_to_static_power(x, power):
      x_aval = jax.typeof(x)
      return RaiseToStaticPower(x_aval, power=power)(x)

    def f(x):
      return raise_to_static_power(x, power=3)

    if jit:
      f = jax.jit(f)

    self.assertEqual(f(2.0), 8.0)
    xs = jnp.arange(3.0)
    self.assertAllClose(jax.vmap(f)(xs), xs**3)
    self.assertEqual(jax.grad(f)(2.0), 12.0)

  def test_newstyle_hiprimitive_defines_both_types_of_vjp_error(self):
    class RaiseToStaticPower(VJPHiPrimitive):
      def __init__(self, in_aval, *, power):
        self.in_avals = (in_aval,)
        self.out_aval = in_aval
        self.params = dict(power=power)
        super().__init__()

      def expand(self, x):
        return x ** self.power

      def vjp_fwd(self, x):
        ans = self(x)
        return (ans, x)

      def vjp_bwd(self, res, t, xbar_accum):
        xbar = t * self.power * raise_to_static_power(res, self.power-1)
        xbar_accum.accum(xbar)

      def vjp_bwd_retval(self, res, t):
        return (t * self.power * raise_to_static_power(res, self.power-1),)

      def batch(self, _axis_data, args, in_dims):
        in_dim, = in_dims
        x, = args
        return raise_to_static_power(x, self.power), in_dim

    def raise_to_static_power(x, power):
      x_aval = jax.typeof(x)
      return RaiseToStaticPower(x_aval, power=power)(x)

    def f(x):
      return raise_to_static_power(x, power=3)

    with self.assertRaises(AttributeError):
      f(2.0)

  @config.numpy_dtype_promotion('standard')
  def test_newstyle_hiprimitive_qarray(self):

    @dataclass(frozen=True)  # not NamedTuple, which is a pytree
    class QArray:
      qvalue: jax.Array
      scale: jax.Array

    @dataclass(frozen=True)
    class QArrayTy(HiType):
      shape: tuple[int, int]

      def to_tangent_aval(self):
        return ShapedArray(self.shape, jnp.dtype('float32'))

    register_hitype(QArray, lambda q: QArrayTy(q.qvalue.shape))

    def q(x):
      return Q(jax.typeof(x))(x)

    def dq(qx):
      return DQ(jax.typeof(qx))(qx)

    class Q(VJPHiPrimitive):
      def __init__(self, unquantized_aval):
        if unquantized_aval.dtype != jnp.dtype('float32'): raise TypeError
        quantized_aval = QArrayTy(unquantized_aval.shape)
        self.in_avals = (unquantized_aval,)
        self.out_aval = quantized_aval
        self.params = {}
        super().__init__()

      def expand(self, x):
        scale = jnp.max(jnp.abs(x)) / 127
        qvalue = jnp.round(x / scale).astype(jnp.int8)
        return QArray(qvalue, scale)

      def vjp_fwd(self, x):
        return self(x), None

      def vjp_bwd_retval(self, _, g):
        return g,

    class DQ(VJPHiPrimitive):
      def __init__(self, quantized_aval):
        unquantized_aval = ShapedArray(quantized_aval.shape, jnp.dtype('float32'))
        self.in_avals = (quantized_aval,)
        self.out_aval = unquantized_aval
        self.params = {}
        super().__init__()

      def expand(self, qx):
        return qx.qvalue * qx.scale

      def vjp_fwd(self, qx):
        return self(qx), None

      def vjp_bwd_retval(self, _, g):
        return g,

    def f(x):
      return jnp.sum(dq(q(x)))

    x = jax.random.normal(jax.random.key(0), (3, 3), dtype='float32')
    g = jax.grad(f)(x)


class BoxTest(jtu.JaxTestCase):

  @parameterized.parameters([False, True])
  def test_qdd(self, jit):

    val1 = 1.0
    val2 = jnp.arange(3)

    box1 = Box(val1)

    def f(box2):
      assert core.cur_qdd(box2).leaf_avals == (core.typeof(val1),)
      box2.set(val2)
      assert core.cur_qdd(box2).leaf_avals == (core.typeof(val2),)

      box3 = new_box()
      box3.set(val2)
      assert core.cur_qdd(box3).leaf_avals == (core.typeof(val2),)
      box3.set(val1)
      assert core.cur_qdd(box3).leaf_avals == (core.typeof(val1),)

      assert core.cur_qdd(box1).leaf_avals == (core.typeof(val1),)
      box1.set(val2)
      assert core.cur_qdd(box1).leaf_avals == (core.typeof(val2),)

      return

    if jit:
      f = jax.jit(f)

    f(Box(val1))

  def test_jit_internal(self):
    @jax.jit
    def f(x):
      box = new_box()  # TODO not Box
      box.set(x)
      box.set(box.get() + box.get())
      return box.get()

    f(1)

  def test_jit_internal_box_constructor(self):
    @jax.jit
    def f(x):
      box = Box(x)
      box.set(box.get() + box.get())
      return box.get()

    f(1)

  @parameterized.parameters([False, True])
  def test_isinstance(self, jit):
    def f():
      box = Box()
      self.assertIsInstance(box, Box)
    if jit:
      f = jax.jit(f)
    f()

  def test_jit_arg(self):
    @jax.jit
    def f(box, x):
      assert tracing_ok
      box.set(box.get() + x)

    tracing_ok = True
    box1 = Box(1.0)
    f(box1, 1.)
    self.assertAllClose(box1.get(), 2.0)

    tracing_ok = False
    box2 = Box(2.0)
    f(box2, 2.)
    self.assertAllClose(box2.get(), 4.0)

  def test_jit_arg2(self):
    # set without get

    @jax.jit
    def f(box, x):
      box_set(box, x)

    box = Box(0.0)
    f(box, 1.)
    self.assertAllClose(box_get(box), 1.0, check_dtypes=False)

  def test_jit_arg_in_pytree(self):
    @jax.jit
    def f(dct, x):
      assert tracing_ok
      box = dct['box']
      box.set(box.get() + x)

    tracing_ok = True
    box1 = Box(1.0)
    f({'box': box1, 'a': 1.0}, 1.)
    self.assertAllClose(box1.get(), 2.0)

    tracing_ok = False
    box2 = Box(2.0)
    f({'box': box2, 'a': 2.0}, 2.)
    self.assertAllClose(box2.get(), 4.0)

    tracing_ok = True
    box3 = Box(3)  # int, dtype changed
    f({'box': box3, 'a': 2.0}, 2.)
    self.assertAllClose(box3.get(), 5.0)

  def test_jit_closure(self):
    box = Box(1.0)

    @jax.jit
    def f(x):
      assert tracing_ok
      box.set(box.get() + x)

    tracing_ok = True
    f(2.0)
    self.assertAllClose(box.get(), 3.0)
    tracing_ok = False
    f(5.0)
    self.assertAllClose(box.get(), 8.0)

  def test_jit_closure_nested(self):
    box = Box(5.0)

    @jax.jit
    def f(x):
      box.set(box.get() + x)

    @jax.jit
    def g(x):
      f(x)

    g(3.0)
    self.assertAllClose(box.get(), 8.0)

  def test_jit_closure_nested2(self):
    @jax.jit
    def h(x):
      box = new_box()
      box.set(x)

      @jax.jit
      def k(x):
        box.set(box.get() + x)

      k(1.0)
      k(1.0)
      return box.get()

    ans = h(2.0)
    self.assertAllClose(ans, 4.0)

  def test_jit_closure_nested3(self):
    box = new_box()

    @jax.jit
    def h(x):
      box.set(x)

      @jax.jit
      def k(x):
        box.set(box.get() + x)

      k(1.0)
      k(1.0)
      return box.get()

    ans = h(2.0)
    self.assertAllClose(ans, 4.0)

  @parameterized.parameters([False, True])
  def test_jvp_closure_stop_gradient(self, jit):
    box = Box(1.0)

    def f(x):
      y = 2 * x
      box.set(box.get() + jax.lax.stop_gradient(y))
      return y

    if jit:
      f = jax.jit(f)

    y, y_dot = jax.jvp(f, (1.0,), (1.0,))
    self.assertAllClose(y, 2.0)
    self.assertAllClose(y_dot, 2.0)
    self.assertAllClose(box.get(), 3.0)

  @parameterized.parameters([False, True])
  def test_jvp_arg(self, jit):
    def f(box, x):
      box.set(box.get() + x)
      return x

    if jit:
      f = jax.jit(f)

    box = Box(5.0)
    box_dot = Box(1.0)
    y, y_dot = jax.jvp(f, (box, 2.), (box_dot, 1.))
    self.assertAllClose(y, 2.0)
    self.assertAllClose(y_dot, 1.0)
    self.assertAllClose(box.get(), 7.0)
    self.assertAllClose(box_dot.get(), 2.0)

  @parameterized.parameters([False, True])
  def test_custom_vjp_plumbing(self, jit):
    box = Box(0.0)

    @jax.custom_vjp
    def foo(x):
      return x
    def foo_fwd(x):
      return foo(x), None
    def foo_bwd(_, g):
      box.set(g)
      return g,
    foo.defvjp(foo_fwd, foo_bwd)

    def f(x):
      x = 2 * x
      x = foo(x)
      x = 2 * x
      return x

    if jit:
      f = jax.jit(f)

    jax.grad(f)(1.0)

    self.assertAllClose(box.get(), 2.0)

  @parameterized.parameters([False, True])
  def test_custom_vjp_plumbing_abstracted(self, jit):
    box = Box(0.0)

    @jax.custom_vjp
    def foo(box, x):
      return x
    def foo_fwd(box, x):
      return x, box
    def foo_bwd(box, g):
      box.set(g)
      return None, g
    foo.defvjp(foo_fwd, foo_bwd)

    def f(box, x):
      x = 2 * x
      x = foo(box, x)
      x = 2 * x
      return x

    if jit:
      f = jax.jit(f)

    jax.grad(partial(f, box))(1.0)
    self.assertAllClose(box.get(), 2.0)

  @parameterized.parameters([False, True])
  def test_grad_closure_stop_gradient(self, jit):
    box = Box(0.0)

    def f(x):
      y = x * 2
      box.set(box.get() + jax.lax.stop_gradient(y))
      return y

    if jit:
      f = jax.jit(f)

    g = jax.grad(f)(1.0)
    self.assertAllClose(g, 2.0)
    self.assertAllClose(box.get(), 2.0)

  @parameterized.parameters([False, True])
  def test_scan_basic(self, jit):
    box = Box(1.0)

    def double_it_10():
      def body(_, __):
        box.set(box.get() * 2)
        return None, None
      _, _ = jax.lax.scan(body, None, None, length=10)

    if jit:
      double_it_10 = jax.jit(double_it_10)

    double_it_10()

    self.assertAllClose(box.get(), 1024., check_dtypes=False)

  def test_cond_box_internally_pure(self):
    @jax.jit
    def doubleit(x):
      b = new_box()
      b.set(x)
      b.set(b.get() + b.get())
      return b.get()

    def identity(x): return x

    @jax.jit
    def f(x):
      return jax.lax.cond(x > 0, doubleit, identity, x)

    self.assertAllClose(f(1.0), 2.0)

  def test_cond_box_arg(self):
    @jax.jit
    def f(x):
      b = new_box()
      b.set(x)
      jax.lax.cond(x > 0, lambda box: box.set(box.get() + 1), lambda _: None, b)
      return b.get()

    self.assertAllClose(f(1.0), 2.0)

  def test_cond_closed_over_box(self):
    # TODO: good error messages in the case that qdd changes differently in each branch
    def f(x):
      b = new_box()
      b.set(1.0)
      jax.lax.cond(x > 0., lambda _: b.set(b.get() + 1.0), lambda _: None, 1.0)
      return b.get()

    self.assertAllClose(f(1.0), 2.0)


  # TODO error-checking tests from attrs_test.py

  ###

  def test_box_autodiff(self):
    if config.enable_x64.value: raise unittest.SkipTest("no x64")

    class StashTangents(HiPrimitive):
      def is_high(self, *_):
        return True

      def abstract_eval(_, box_aval, x_aval):
        del box_aval
        return x_aval, {box_effect}

      def to_lojax(_, box, x):
        return x

      def jvp(_, primals, tangents):
        box, x = primals
        _, x_dot = tangents
        box_set(box, x_dot)
        return x, x_dot

      def transpose(self, *args):
        assert False  # TODO
    stash_tangents_p = StashTangents('stash_tangents')

    def stash_tangents(box, x):
      return stash_tangents_p.bind(box, x)

    @jax.jit
    def f(box, x):
      x = stash_tangents(box, x)
      return x

    box = Box(0.0)
    jax.jvp(partial(f, box), (3.,), (5.,))
    self.assertAllClose(box_get(box), 5.0, check_dtypes=False)

  def test_type_changing_box(self):
    box = Box(jnp.arange(1))
    box_set(box, jnp.arange(2))
    self.assertLen(box._val, 2)

    @jax.jit
    def f(box, x):
      box_set(box, x)

    f(box, jnp.arange(3))
    self.assertLen(box._val, 3)
    f(box, jnp.arange(4))
    self.assertLen(box._val, 4)

  def test_pytree_box(self):
    box = Box(None)

    @jax.jit
    def f(box, x):
      assert tracing_ok
      val = box_get(box)
      if val is None:
        box_set(box, x)
      else:
        box_set(box, [x, x])

    tracing_ok = True
    f(box, 1.0)
    self.assertAllClose(box_get(box), 1.0, check_dtypes=False)
    f(box, 2.0)
    self.assertAllClose(box_get(box), [2.0, 2.0], check_dtypes=False)
    f(box, 3.0)
    self.assertAllClose(box_get(box), [3.0, 3.0], check_dtypes=False)
    tracing_ok = False
    f(box, 4.0)
    self.assertAllClose(box_get(box), [4.0, 4.0], check_dtypes=False)

  def test_pytree_of_hijaxtypes_box(self):

    @dataclass(frozen=True)
    class MyArray:
      arr: jax.Array  # always f32

    @dataclass(frozen=True)
    class MyTy(HiType):
      has_qdd = False

      def to_tangent_aval(self):
        return MyTy()
      def str_short(self, short_dtypes=False):
        return 'MyTy'
      def lo_ty(self):
        return [core.ShapedArray((), jnp.dtype('float32'))]
      def lower_val(self, hi_val: MyArray) -> list[jax.Array]:
        return [hi_val.arr]
      def raise_val(self, val) -> MyArray:
        return MyArray(val)

      def __eq__(self, other): return isinstance(other, MyTy)

    core.pytype_aval_mappings[MyArray] = lambda _: MyTy()

    box = Box([MyArray(jnp.float32(1)),
               MyArray(jnp.float32(2))])

    @jax.jit
    def f(box):
      a, b = box_get(box)
      box_set(box, [b, a])

    f(box)
    val = box_get(box)
    self.assertIsInstance(val, list)
    self.assertLen(val, 2)
    b_, a_ = val
    self.assertIsInstance(a_, MyArray)
    self.assertIsInstance(b_, MyArray)
    self.assertAllClose(a_.arr, 1, check_dtypes=False)
    self.assertAllClose(b_.arr, 2, check_dtypes=False)

  def test_closed_over_type_changing_box(self):

    box = Box(None)
    box2 = Box(None)

    @jax.jit
    def f():
      assert tracing_ok
      x = box.get()
      if x is None:
        box.set(0)
      elif type(x) is dict:
        box.set(dict(x, a=5))
        box2.set(3)
      else:
        box.set(x + 1)

    tracing_ok = True
    f()  # tracing okay because first time
    f()  # tracing okay because first time with box as not None
    tracing_ok = False
    f()
    self.assertEqual(box.get(), 2)
    self.assertEqual(box2.get(), None)
    box.set(None)
    f()
    f()
    f()
    f()
    self.assertEqual(box.get(), 3)
    self.assertEqual(box2.get(), None)
    box.set({'b': 3})
    tracing_ok = True
    f()
    self.assertEqual(box.get(), dict(a=5, b=3))
    self.assertEqual(box2.get(), 3)

  @parameterized.parameters([False, True])
  def test_while_loop(self, jit):
    box = Box(1.)

    def f():
      zero = jnp.zeros((), 'int32')

      def cond_fun(i):
        return i + zero < 5
      def body_fun(i):
        box.set(box.get() * 2.)
        return i + 1
      _ = jax.lax.while_loop(cond_fun, body_fun, 0)

    if jit:
      f = jax.jit(f)

    f()
    self.assertAllClose(box.get(), 32, check_dtypes=False)

  def test_while_loop_typechange_error(self):
    box = Box([1.])
    def cond_fun(i):
      return i < 5
    def body_fun(i):
      box.set(box.get() * 2)
      return i + 1
    with self.assertRaisesRegex(TypeError, "type-changing mutations not allowed"):
      _ = jax.lax.while_loop(cond_fun, body_fun, 0)

  def test_eval_shape(self):
    qarray = QArray(jnp.ones((2, 2)), jnp.ones(2))

    @jax.jit
    def f():
      return qarray

    out_type = jax.eval_shape(f)
    self.assertEqual(out_type, QArrayTy((2, 2)))

  def test_stages_mutable(self):
    box = Box(1.0)

    @jax.jit
    def f(box):
      box.set(box.get() + 1.)

    f.lower(box).as_text()  # don't crash
    compiled = f.lower(box).compile()
    compiled(box)
    compiled(box)
    compiled(box)
    self.assertAllClose(box.get(), 4.)

  def test_jit_box_with_sharded_array(self):
    # Create a mesh and sharded array
    mesh = jax.make_mesh((4, 2), ('x', 'y'))
    sharding = NamedSharding(mesh, P('x', 'y'))

    # Create a sharded array
    x = jnp.arange(16.).reshape(8, 2)
    sharded_x = jax.device_put(x, sharding)

    # Put sharded array in a Box
    box = Box(sharded_x)

    @jax.jit
    def f(box, y):
      val = box.get()
      result = val + y
      box.set(result)
      return box.get()

    # Test with Box argument containing sharded array
    y = jnp.ones((8, 2))
    result = f(box, y)

    self.assertAllClose(result, x + y)
    self.assertAllClose(box.get(), x + y)


class RefTest(jtu.JaxTestCase):

  def test_get_ref_hitype(self):

    @jax.jit
    def f(q):
      ref = jax.new_ref(q)
      return ref[:, 0:2]

    qarray = QArray(jnp.ones((2, 2), dtype='int8'), jnp.ones(2, 'float32'))
    o = f(qarray)
    self.assertArraysEqual(o.arr, qarray.arr)
    self.assertArraysEqual(o.scale, qarray.scale)

  def test_swap_ref_hitype(self):

    @jax.jit
    def f(q1, q2):
      ref = jax.new_ref(q1)
      ref[:, :] = q2
      return ref.get()

    q1 = QArray(jnp.zeros((2, 2), dtype='int8'), jnp.zeros(2, 'float32'))
    q2 = QArray(jnp.ones((2, 2), dtype='int8'), jnp.ones(2, 'float32'))
    o = f(q1, q2)
    self.assertArraysEqual(o.arr, q2.arr)
    self.assertArraysEqual(o.scale, q2.scale)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
