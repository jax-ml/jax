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

from absl.testing import absltest

import jax
import jax.numpy as jnp

from jax._src import config
from jax._src import core
from jax._src import dtypes
from jax._src.interpreters import ad
from jax._src.interpreters import partial_eval as pe
from jax._src import test_util as jtu
from jax._src.util import safe_zip, safe_map

config.parse_flags_with_absl()

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip


# TODO(mattjj,dougalm): move HiPrimitive, Box, etc out of tests and into library
class HiPrimitive(core.Primitive):
  def __init__(self, name):
    self.name = name
    ad.primitive_jvps[self] = self.jvp
    ad.primitive_transposes[self] = self.transpose
    pe.custom_staging_rules[self] = self.staging

  def staging(self, trace, *args, **kwargs):
    trace.frame.is_high = True
    return trace.default_process_primitive(self, args, kwargs)

  def is_high(self, **params):
    return True

  def abstract_eval(self, *arg_avals, **params):
    assert False, "must override"

  def to_lojax(self, *lotypes_wrapped_in_hitypes, **params):
    assert False, "must override"

  def jvp(self, primals, tangents, **params):
    assert False, "must override"

  def transpose(self, *args, **params):
    assert False  # TODO


class BoxTy(core.AbstractValue):
  mutable = True

  def __init__(self, leaf_avals, treedef):
    self._leaf_avals = leaf_avals  # hijax avals
    self._treedef = treedef

  # aval interface: hashability and str_short
  def __hash__(self):
    return hash((self._leaf_avals, self._treedef))

  def __eq__(self, other):
    return (isinstance(other, BoxTy) and self._leaf_avals == other._leaf_avals
            and self._treedef == other._treedef)

  def str_short(self, short_dtypes=False):
    return 'BoxTy'

  # hijax interface: lower val, raise val, and low type
  def lo_ty(self):
    return [lo_aval for hi_aval in self._leaf_avals for lo_aval in hi_aval.lo_ty()]

  def lower_val(self, box):
    leaf_vals, treedef = jax.tree.flatten(box._val)
    assert treedef == self._treedef
    return [lo_val for hi_aval, hi_val in zip(self._leaf_avals, leaf_vals)
            for lo_val in hi_aval.lower_val(hi_val)]

  def raise_val(self, *lo_vals):
    lo_vals_ = iter(lo_vals)
    hi_vals = [hi_ty.raise_val(*it.islice(lo_vals_, len(hi_ty.lo_ty())))
               for hi_ty in self._leaf_avals]
    assert next(lo_vals_, None) is None
    return Box(jax.tree.unflatten(self._treedef, hi_vals))  # will be mutated

  # mutable interface: get/set
  def get(self, box):
    leaf_vals, treedef = jax.tree.flatten(box._val)
    assert treedef == self._treedef
    return [lo_val for hi_ty, hi_val in zip(self._leaf_avals, leaf_vals)
            for lo_val in hi_ty.lower_val(hi_val)]

  def set(self, box, *lo_vals):
    lo_vals_ = iter(lo_vals)
    hi_vals = [hi_ty.raise_val(*it.islice(lo_vals_, len(hi_ty.lo_ty())))
               for hi_ty in self._leaf_avals]
    assert next(lo_vals_, None) is None
    box._val = jax.tree.unflatten(self._treedef, hi_vals)

  # TODO placeholder thing
  def to_tangent_aval(self):
    return core.ShapedArray((), dtypes.float0)  # TODO revise placeholder

class Box:  # noqa: F811
  def __init__(self, val):
    self._val = val

  @property
  def ty(self):
    leaves, treedef = jax.tree.flatten(self._val)
    leaf_avals = tuple(map(core.typeof, leaves))
    return BoxTy(leaf_avals, treedef)
core.pytype_aval_mappings[Box] = lambda b: b.ty


class BoxSet(HiPrimitive):
  multiple_results = True

  def is_high(self, *, treedef) -> bool: return True

  def staging(self, trace, box, *leaves, treedef):
    super().staging(trace, box, *leaves, treedef=treedef)
    avals = tuple(t.aval for t in leaves)
    trace.frame.final_typechange_env[trace.getvar(box)] = BoxTy(avals, treedef)

  def abstract_eval(self, box_ty, *leaf_avals, treedef):
    return [], set()  # TODO better typechecking...

  def to_lojax(_, box, *leaves, treedef):
    box._val = jax.tree.unflatten(treedef, leaves)
    return []

  def jvp(_, primals, tangents, *, treedef):
    assert False  # TODO

  def transpose(_, *args, treedef):
    assert False  # TODO
box_set_p = BoxSet('box_set')

def box_set(box, val):
  leaves, treedef = jax.tree.flatten(val)
  box_set_p.bind(box, *leaves, treedef=treedef)


class BoxGet(HiPrimitive):
  multiple_results = True

  def is_high(self) -> bool: return True

  def abstract_eval(self, box_ty):
    return box_ty._leaf_avals, set()

  def to_lojax(_, box):
    return jax.tree.leaves(box._val)

  def jvp(_, primals, tangents):
    assert False  # TODO

  def transpose(_, *args):
    assert False  # TODO
box_get_p = BoxGet('box_get')

def box_get(box):
  leaf_vals = box_get_p.bind(box)
  return jax.tree.unflatten(core.typeof(box)._treedef, leaf_vals)


class HijaxTest(jtu.JaxTestCase):

  def test_custom_types_and_primitive(self):
    if config.enable_x64.value: raise unittest.SkipTest("no x64")

    @dataclass(frozen=True)
    class MyArray:
      arr: jax.Array  # always f32

    @dataclass(frozen=True)
    class MyTy(core.AbstractValue):
      mutable = False

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

    class ToMy(HiPrimitive):
      def is_high(self): return True

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
      def is_high(self): return True

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
      def is_high(self): return True

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
      def is_high(self): return True

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

  def test_box_autodiff(self):
    if config.enable_x64.value: raise unittest.SkipTest("no x64")

    class StashTangents(HiPrimitive):
      def is_high(self):
        return True

      def abstract_eval(_, box_aval, x_aval):
        del box_aval
        return x_aval, set()

      def to_lojax(_, box, x):
        assert False  # TODO

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
      box_set(box, x)

    box = Box(0.0)
    f(box, 1.)
    self.assertAllClose(box_get(box), 1.0, check_dtypes=False)

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
    class MyTy(core.AbstractValue):
      mutable = False

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


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
