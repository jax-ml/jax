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

from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
import itertools as it
from typing import Any
import unittest
import numpy as np

from absl.testing import absltest, parameterized

import jax
import jax.numpy as jnp
from jax import typeof

from jax._src import config
from jax._src import core
from jax._src import state
from jax._src.state import indexing
from jax._src.state import primitives as state_primitives
from jax._src.custom_derivatives import custom_jvp_call_p
from jax._src.custom_derivatives import custom_vjp_call_p
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src import test_util as jtu
from jax._src.util import safe_zip, safe_map
from jax._src.state.discharge import run_state

from jax._src.hijax import (
    HiType, register_hitype, ShapedArray, Ty, MappingSpec,
    HiPspec)
from jax.experimental.hijax import (
    VJPHiPrimitive, Zero, instantiate_zeros, jvp_from_lin, linearize_from_jvp,
    vjp_from_jvp, vjp_from_lin)

jtu.request_cpu_devices(8)

config.parse_flags_with_absl()

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

class ToQ(VJPHiPrimitive):
  def __init__(self, lo_aval):
    self.in_avals = (lo_aval,)
    self.out_aval = QArrayTy(lo_aval.shape)
    self.params = {}
    super().__init__()

  def expand(self, lo_val):
    m, _ = lo_val.shape
    scale = lo_val.max(1) / 32.
    return QArray((lo_val / scale[:, None]).astype('int8'), scale)

  def jvp(self, primals, tangents):
    (x,), (xdot,) = primals, tangents
    xdot = ad.instantiate_zeros(xdot)
    return to_qarray(x), to_qarray(xdot)

  vjp_fwd, vjp_bwd_retval = vjp_from_jvp

  def transpose(self, out_bar, accum):
    if isinstance(accum, ad.GradAccum):
      accum.accum(from_qarray(out_bar))


class FromQ(VJPHiPrimitive):
  def __init__(self, hi_aval):
    self.in_avals = (hi_aval,)
    self.out_aval = ShapedArray(hi_aval.shape, jnp.dtype('float32'))
    self.params = {}
    super().__init__()

  def expand(self, hi_val):
    return hi_val.arr.astype('float32') * hi_val.scale[:, None]

  def jvp(self, primals, tangents):
    (x,), (xdot,) = primals, tangents
    return from_qarray(x), from_qarray(xdot)

  vjp_fwd, vjp_bwd_retval = vjp_from_jvp

  def transpose(self, out_bar, accum):
    if isinstance(accum, ad.GradAccum):
      accum.accum(to_qarray(out_bar))


def to_qarray(x):
  return ToQ(core.typeof(x))(x)

def from_qarray(x):
  return FromQ(core.typeof(x))(x)


@dataclass
class HiTup:
  elts: tuple
  def __repr__(self):
    return 'Tup{' + ','.join(map(repr, self.elts)) + '}'

@dataclass(frozen=True)
class TupTy(HiType):
  tys: tuple[Ty, ...]

  def __repr__(self):
    return 'Tup{' + ','.join(a.str_short() for a in self.tys) + '}'

  def __hash__(self):
    return hash(self.tys)

  def __eq__(self, other):
    return isinstance(other, TupTy) and self.tys == other.tys

  def lo_ty(self):
    return list(self.tys)

  def lower_val(self, hi_val: HiTup):
    return [lo for ty, elt in zip(self.tys, hi_val.elts)
            for lo in ty.lower_val(elt)]

  def raise_val(self, *elts_flat):
    elts_iter = iter(elts_flat)
    return HiTup(tuple(ty.raise_val(*it.islice(elts_iter, len(ty.lo_ty())))
                       for ty in self.tys))

  def to_tangent_aval(self):
    return TupTy(tuple(ty.to_tangent_aval() for ty in self.tys))

  def normalize(self):
    return TupTy(tuple(ty.normalize() for ty in self.tys))

  def dec_rank(self, size, spec):
    return TupTy(tuple(ty.dec_rank(size, s) for ty, s in zip(self.tys, spec.val)))

  def inc_rank(self, size, spec):
    return TupTy(tuple(ty.inc_rank(size, 0) for ty in self.tys))

  def leading_axis_spec(self):
    return TupSpec(tuple(ty.leading_axis_spec() for ty in self.tys))

  def shard(self, mesh, manual_axes, check_vma, spec):
    return TupTy(tuple(ty.shard(mesh, manual_axes, check_vma, s)
                       for ty, s in zip(self.tys, spec.val)))

  def unshard(self, mesh, check_vma, spec):
    return TupTy(tuple(ty.unshard(mesh, check_vma, s)
                       for ty, s in zip(self.tys, spec.val)))

  def vspace_add(self, x_tup, y_tup):
    n = len(self.tys)
    x_elts = [get_tuple_element(x_tup, i) for i in range(n)]
    y_elts = [get_tuple_element(y_tup, i) for i in range(n)]
    return make_tup(*(ty.vspace_add(x, y) for ty, x, y
                      in zip(self.tys, x_elts, y_elts)))

register_hitype(HiTup, lambda t: TupTy(tuple(map(typeof, t.elts))))

@dataclass(frozen=True)
class TupSpec(MappingSpec):
  val: tuple

@dataclass(frozen=True)
class TupP(HiPspec):
  val: tuple

  def to_lo(self) -> tuple[jax.PartitionSpec, ...]:
    return self.val

class MakeTup(VJPHiPrimitive):
  def __init__(self, in_avals):
    in_avals = tuple(in_avals)
    self.in_avals = in_avals
    self.out_aval = TupTy(in_avals)
    self.params = {}
    super().__init__()

  def expand(self, *elts):
    return HiTup(elts)

  def jvp(self, primals, tangents):
    tangents = map(ad.instantiate_zeros, tangents)
    return make_tup(*primals), make_tup(*tangents)

  vjp_fwd, vjp_bwd_retval = vjp_from_jvp

  def transpose(self, ct, *maybe_accums):
    cts = [get_tuple_element(ct, i) for i in range(len(self.out_aval.tys))]
    for ct_, accum in zip(cts, maybe_accums):
      if isinstance(accum, ad.GradAccum):
        accum.accum(ct_)

  def batch(self, _axis_data, args, in_dims):
    return make_tup(*args), TupSpec(in_dims)

class GetTupElt(VJPHiPrimitive):
  def __init__(self, in_aval, idx):
    self.in_avals = in_aval,
    self.out_aval = in_aval.tys[idx]
    self.params = dict(idx=idx)
    super().__init__()

  def expand(self, tup):
    return tup.elts[self.idx]

  def jvp(self, primals, tangents):
    (tup,), (tup_dot,) = primals, tangents
    return get_tuple_element(tup, self.idx), get_tuple_element(tup_dot, self.idx)

  def transpose(self, g, tup_accum):
    tup_ty, = self.in_avals
    elts = map(ad.zeros_like_aval, tup_ty.tys)
    elts[self.idx] = g
    tup_accum.accum(make_tup(*elts))

  def vjp_fwd(self, nzs_in, tup):
    return get_tuple_element(tup, self.idx), None

  def vjp_bwd_retval(self, _res, g):
    tup_ty, = self.in_avals
    elts = map(ad.zeros_like_aval, tup_ty.tys)
    elts[self.idx] = g
    return make_tup(*elts),

  def batch(self, _axis_data, args, in_dims):
    (x,), (d,) = args, in_dims
    return get_tuple_element(x, self.idx), d.val[self.idx]

def make_tup(*elts):
  return MakeTup(map(typeof, elts))(*elts)

def get_tuple_element(tup, idx):
  return GetTupElt(typeof(tup), idx)(tup)

@dataclass(frozen=True)
class ImmutBox:
  _val: Any

  @property
  def shape(self):
    if hasattr(self._val, 'shape'):
      return self._val.shape
    leaves = jax.tree.leaves(self._val)
    if leaves and hasattr(leaves[0], 'shape'):
      return leaves[0].shape
    raise AttributeError(f"ImmutBox with value {self._val} has no shape")

  @property
  def ndim(self):
    return len(self.shape)

def _is_zero(x):
  return isinstance(x, ad.Zero)

def _get_aval(x):
  return x.aval if _is_zero(x) else core.typeof(x)

def immutbox_to_aval(box: ImmutBox) -> ImmutBoxTy:
  leaves, treedef = jax.tree.flatten(box._val, is_leaf=_is_zero)
  leaf_avals = tuple(map(_get_aval, leaves))
  return ImmutBoxTy(leaf_avals, treedef)

@dataclass(frozen=True)
class ImmutBoxTy(HiType):
  leaf_avals: tuple[core.AbstractValue, ...]
  treedef: Any

  @property
  def shape(self):
    reconstructed = jax.tree.unflatten(self.treedef, self.leaf_avals)
    if hasattr(reconstructed, 'shape'):
      return reconstructed.shape
    if self.leaf_avals and hasattr(self.leaf_avals[0], 'shape'):
      return self.leaf_avals[0].shape
    raise AttributeError(f"ImmutBoxTy with treedef {self.treedef} has no shape")

  @property
  def ndim(self):
    return len(self.shape)

  @property
  def sharding(self):
    reconstructed = jax.tree.unflatten(self.treedef, self.leaf_avals)
    if hasattr(reconstructed, 'sharding'):
      return reconstructed.sharding
    if self.leaf_avals and hasattr(self.leaf_avals[0], 'sharding'):
      return self.leaf_avals[0].sharding
    return None

  def lo_ty(self):
    return list(self.leaf_avals)

  def lower_val(self, hi_val: ImmutBox):
    leaves, treedef = jax.tree.flatten(hi_val._val, is_leaf=_is_zero)
    assert treedef == self.treedef
    return leaves

  def raise_val(self, *lo_vals):
    return ImmutBox(jax.tree.unflatten(self.treedef, lo_vals))

  def to_tangent_aval(self):
    tangent_leaf_avals = tuple(aval.to_tangent_aval() for aval in self.leaf_avals)
    return ImmutBoxTy(tangent_leaf_avals, self.treedef)

  def vspace_zero(self):
    zero_leaves = [ad.zeros_like_aval(a) for a in self.leaf_avals]
    return immutbox_new(jax.tree.unflatten(self.treedef, zero_leaves))

  def vspace_add(self, x, y):
    x_leaves = jax.tree.leaves(immutbox_get(x))
    y_leaves = jax.tree.leaves(immutbox_get(y))
    add_leaves = [a.vspace_add(i, j) for a, i, j in zip(self.leaf_avals, x_leaves, y_leaves)]
    return immutbox_new(jax.tree.unflatten(self.treedef, add_leaves))

def _map_immutbox_ty(size: int, axis: int | None, aval: ImmutBoxTy) -> ImmutBoxTy:
  if axis is None:
    return aval
  mapped_leaf_avals = tuple(core.mapped_aval(size, axis, leaf_aval)
                            for leaf_aval in aval.leaf_avals)
  return ImmutBoxTy(mapped_leaf_avals, aval.treedef)

def _unmap_immutbox_ty(size: int, axis: int | None, explicit_mesh_axis,
                       aval: ImmutBoxTy) -> ImmutBoxTy:
  if axis is None:
    return aval
  elif isinstance(axis, int):
    unmapped_leaf_avals = tuple(core.unmapped_aval(size, axis, explicit_mesh_axis, leaf_aval)
                                for leaf_aval in aval.leaf_avals)
    return ImmutBoxTy(unmapped_leaf_avals, aval.treedef)
  else:
    raise TypeError(axis)

core.aval_mapping_handlers[ImmutBoxTy] = (_map_immutbox_ty, _unmap_immutbox_ty)

class ImmutBoxNew(VJPHiPrimitive):
  def __init__(self, leaf_avals, treedef):
    self.in_avals = tuple(leaf_avals)
    self.out_aval = ImmutBoxTy(tuple(leaf_avals), treedef)
    self.params = dict(leaf_avals=tuple(leaf_avals), treedef=treedef)
    super().__init__()

  def expand(self, *leaves):
    val = jax.tree.unflatten(self.treedef, leaves)
    return ImmutBox(val)

  def jvp(self, primals, tangents):
    tangents = [ad.instantiate_zeros(t) for t in tangents]
    prim = ImmutBoxNew(self.leaf_avals, self.treedef)
    return prim(*primals), prim(*tangents)

  def vjp_fwd(self, nzs_in, *leaves):
    return self(*leaves), None

  def vjp_bwd_retval(self, _res, g):
    leaves, _ = jax.tree.flatten(immutbox_get(g), is_leaf=_is_zero)
    return tuple(leaves)

  def transpose(self, out_bar, *accums):
    val = out_bar._val
    leaves, _ = jax.tree.flatten(val, is_leaf=_is_zero)
    for leaf, accum in zip(leaves, accums):
      if isinstance(accum, ad.GradAccum):
        accum.accum(leaf)


def immutbox_new(val):
  leaves, treedef = jax.tree.flatten(val, is_leaf=_is_zero)
  leaf_avals = tuple(map(_get_aval, leaves))
  leaves = [ad.instantiate_zeros(leaf) for leaf in leaves]
  return ImmutBoxNew(leaf_avals, treedef)(*leaves)


class ImmutBoxGet(VJPHiPrimitive):
  def __init__(self, box_aval):
    self.in_avals = (box_aval,)
    self.out_aval = jax.tree.unflatten(box_aval.treedef, box_aval.leaf_avals)
    self.params = dict(box_aval=box_aval)
    super().__init__()

  def expand(self, box):
    return box._val

  def jvp(self, primals, tangents):
    (box,), (box_dot,) = primals, tangents
    box_dot = ad.instantiate_zeros(box_dot)
    return immutbox_get(box), immutbox_get(box_dot)

  def vjp_fwd(self, nzs_in, box):
    return self(box), None

  def vjp_bwd_retval(self, _res, g):
    return (immutbox_new(g),)

  def transpose(self, out_bar_tree, box_accum):
    if isinstance(box_accum, ad.GradAccum):
      box_accum.accum(immutbox_new(out_bar_tree))


def immutbox_get(box):
  return ImmutBoxGet(core.typeof(box))(box)

register_hitype(ImmutBox, immutbox_to_aval)


class Square(VJPHiPrimitive):
  """Simple parameterless hijax primitive for use in tests."""
  _jvp_execution_count = 0

  def __init__(self, in_aval):
    self.in_avals = (in_aval,)
    self.out_aval = in_aval
    self.params = {}
    super().__init__()

  @classmethod
  @contextmanager
  def assert_jvp_rule_called_once(cls):
    initial_count = cls._jvp_execution_count
    yield
    assert cls._jvp_execution_count == initial_count + 1

  def expand(self, x):
    return x ** 2

  def jvp(self, primals, tangents):
    self.__class__._jvp_execution_count += 1
    (x,), (t,) = primals, tangents
    return self(x), t * 2.0 * x

  def vjp_fwd(self, nzs_in, x):
    return (self(x), x)

  def vjp_bwd_retval(self, res, t):
    return (t * 2.0 * res,)

def square(x):
  """Bind a hijax primtive that returns the square of x."""
  return Square(jax.typeof(x))(x)


class NonDiffPrim(VJPHiPrimitive):
  def __init__(self, in_aval):
    self.in_avals = (in_aval,)
    self.out_aval = in_aval
    self.params = {}
    super().__init__()

  def expand(self, x):
    return x

  def jvp(self, primals, tangents):
    (x,), _ = primals, tangents
    import numpy as np
    return x, np.empty(x.shape, dtype=jax.dtypes.float0)

  lin, linearized = linearize_from_jvp


class HijaxTest(jtu.JaxTestCase):

  def test_closed_call(self):
    from jax._src import api_util
    from jax._src import linear_util as lu

    qx = QArray(
        arr=jnp.ones((2, 3), dtype=jnp.int8),
        scale=jnp.array([1.5, 2.5], dtype=jnp.float32),
    )

    def f(q):
      return q

    @jax.jit
    def test_fn(x):
      flat_x, in_tree = jax.tree.flatten((x,))
      dbg = api_util.debug_info('test_closed_call', f, flat_x, {})
      flat_f, out_tree = api_util.flatten_fun_nokwargs(
          lu.wrap_init(f, debug_info=dbg), in_tree
      )
      from jax._src.interpreters import partial_eval as pe
      jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
          flat_f, [core.typeof(v) for v in flat_x])
      out = core.closed_call_p.bind(*consts, *flat_x, call_jaxpr=jaxpr)
      return jax.tree.unflatten(out_tree(), out)

    res = test_fn(qx)
    self.assertIsInstance(res, QArray)
    self.assertArraysEqual(res.arr, qx.arr)
    self.assertArraysEqual(res.scale, qx.scale)

    traced = test_fn.trace(qx)
    self.assertTrue(
        traced.jaxpr.is_high, 'Initial jaxpr should contain hi-primitives'
    )
    lojaxpr = traced.lojax.jaxpr
    self.assertFalse(
        lojaxpr.is_high, 'Lowered jaxpr should not contain hi-primitives'
    )

  def test_closed_call_low_io(self):
    from jax._src import api_util
    from jax._src import linear_util as lu

    x = jnp.ones((2, 3), dtype=jnp.float32)

    def f(arr):
      q = to_qarray(arr)
      arr2 = from_qarray(q)
      return (arr2,)

    @jax.jit
    def test_fn(arr):
      dbg = api_util.debug_info('test_closed_call_low_io', f, [arr], {})
      f_wrapped = lu.wrap_init(f, debug_info=dbg)
      from jax._src.interpreters import partial_eval as pe
      jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
          f_wrapped, [core.typeof(arr)])
      (out,) = core.closed_call_p.bind(*consts, arr, call_jaxpr=jaxpr)
      return out

    res = test_fn(x)
    self.assertArraysEqual(res, x)

    traced = test_fn.trace(x)
    self.assertTrue(
        traced.jaxpr.is_high, 'Initial jaxpr should contain hi-primitives'
    )
    lojaxpr = traced.lojax.jaxpr
    self.assertFalse(
        lojaxpr.is_high, 'Lowered jaxpr should not contain hi-primitives'
    )

  def test_empty_ref_and_freeze(self):
    qx = QArray(arr=jnp.ones((2, 3), dtype=jnp.int8),
                scale=jnp.array([1.5, 2.5], dtype=jnp.float32))

    def f():
      q_ref = jax.empty_ref(jax.typeof(qx))
      return jax.freeze(q_ref)

    q_out = jax.jit(f)()
    self.assertIsInstance(q_out, QArray)
    self.assertEqual(q_out.arr.shape, (2, 3))
    self.assertEqual(q_out.scale.shape, (2,))
    self.assertEqual(q_out.arr.dtype, jnp.int8)
    self.assertEqual(q_out.scale.dtype, jnp.float32)

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
    register_hitype(MyArray, lambda _: MyTy())

    class ToMy(VJPHiPrimitive):
      def __init__(self, lo_aval):
        self.in_avals = (lo_aval,)
        self.out_aval = MyTy()
        self.params = {}
        super().__init__()

      def expand(self, lo):
        return MyArray(lo)

      def jvp(self, primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return to(x), to(x_dot)

      vjp_fwd, vjp_bwd_retval = vjp_from_jvp

      def transpose(self, out_bar, accum):
        if isinstance(accum, ad.GradAccum):
          accum.accum(from_(out_bar))

    class FromMy(VJPHiPrimitive):
      def __init__(self, hi_aval):
        self.in_avals = (hi_aval,)
        self.out_aval = hi_aval.lo_ty()[0]
        self.params = {}
        super().__init__()

      def expand(self, hi):
        return hi.arr

      def jvp(self, primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return from_(x), from_(x_dot)

      vjp_fwd, vjp_bwd_retval = vjp_from_jvp

      def transpose(self, out_bar, accum):
        if isinstance(accum, ad.GradAccum):
          accum.accum(to(out_bar))

    def to(x): return ToMy(core.typeof(x))(x)

    def from_(x): return FromMy(core.typeof(x))(x)

    def mul(x, y): return MyMul(core.typeof(x), core.typeof(y))(x, y)
    def add(x, y): return MyAdd(core.typeof(x), core.typeof(y))(x, y)

    class MyMul(VJPHiPrimitive):
      def __init__(self, hi_x, hi_y):
        if hi_x != hi_y: raise Exception
        self.in_avals = (hi_x, hi_y)
        self.out_aval = hi_x
        self.params = {}
        super().__init__()

      def expand(self, hi_x, hi_y):
        return MyArray(hi_x.arr * hi_y.arr)

      def jvp(self, primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        x_dot, y_dot = ad.instantiate_zeros(x_dot), ad.instantiate_zeros(y_dot)
        return mul(x, y), add(mul(x, y_dot), mul(x_dot, y))

      vjp_fwd, vjp_bwd_retval = vjp_from_jvp

      def transpose(self, out_bar, x, y):
        x_is_accum = isinstance(x, ad.GradAccum)
        y_is_accum = isinstance(y, ad.GradAccum)
        assert x_is_accum ^ y_is_accum
        if x_is_accum:
          x.accum(mul(out_bar, y))
        else:
          y.accum(mul(x, out_bar))

    class MyAdd(VJPHiPrimitive):
      def __init__(self, hi_x, hi_y):
        if hi_x != hi_y: raise Exception
        self.in_avals = (hi_x, hi_y)
        self.out_aval = hi_x
        self.params = {}
        super().__init__()

      def expand(self, hi_x, hi_y):
        return MyArray(hi_x.arr + hi_y.arr)

      def jvp(self, primals, tangents):
        assert False  # TODO

      vjp_fwd, vjp_bwd_retval = vjp_from_jvp

      def transpose(self, out_bar, x_accum, y_accum):
        if isinstance(x_accum, ad.GradAccum): x_accum.accum(out_bar)
        if isinstance(y_accum, ad.GradAccum): y_accum.accum(out_bar)

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

  def test_hijax_infer_params_cache_hit(self):
    x = np.arange(4)

    @jax.jit
    def f(x):
      return square(x)

    with jtu.count_infer_params_cache_miss() as count:
      f(x)
      f(x)
    self.assertEqual(count(), 1)

  def test_scan_mat(self):
    @dataclass(frozen=True)
    class Box:
      a: jax.Array

    @dataclass(frozen=True)
    class BoxTy(HiType):
      shape: tuple
      def lo_ty(self): return [ShapedArray(self.shape, jnp.dtype('float32'))]
      def lower_val(self, b): return [b.a]
      def raise_val(self, a): return Box(a)
      def to_tangent_aval(self): return ShapedArray(self.shape, jnp.dtype('float32'))
      def str_short(self, short_dtypes=False, **_): return f"box{list(self.shape)}"

    register_hitype(Box, lambda b: BoxTy(b.a.shape))

    class Wrap(VJPHiPrimitive):
      def __init__(self, av):
        self.in_avals, self.out_aval, self.params = (av,), BoxTy(av.shape), {}
        super().__init__()
      def expand(self, a): return Box(a)

    class Scale(VJPHiPrimitive):
      def __init__(self, av):
        self.in_avals, self.out_aval, self.params = (av,), av, {}
        super().__init__()
      def expand(self, b): return Box(b.a * 2.0)

    wrap  = lambda a: Wrap(jax.typeof(a))(a)
    scale = lambda b: Scale(jax.typeof(b))(b)

    b = wrap(jnp.arange(3, dtype='float32'))
    f = lambda b: jax.lax.scan(lambda c, _: (scale(c), None), b, None, length=2)[0]

    jax.typeof(f(b))  # doesn't crash
    jax.typeof(jax.jit(f)(b))  # doesn't crash

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

  def test_tuple_vmap(self):
    tup = make_tup(jnp.arange(3.), jnp.arange(3.))
    out = jax.vmap(lambda x: x, in_axes=TupSpec((0, 0)),
                   out_axes=TupSpec((0, 0)), axis_size=3)(tup)
    self.assertAllClose(out.elts, tup.elts)

  def test_tuple_vmap_of_jit(self):
    # https://github.com/jax-ml/jax/issues/38125
    tup = make_tup(jnp.arange(3.), jnp.arange(3.))
    out = jax.vmap(jax.jit(lambda x: x), in_axes=TupSpec((0, 0)),
                   out_axes=TupSpec((0, 0)), axis_size=3)(tup)
    self.assertAllClose(out.elts, tup.elts)

  def test_tuple_vmap_int_in_axes_error(self):
    tup = make_tup(jnp.arange(3.), jnp.arange(3.))
    with self.assertRaisesRegex(ValueError, "non-array type"):
      jax.vmap(lambda x: x, axis_size=3)(tup)

  def test_tuple_device_put_error(self):
    tup = make_tup(jnp.arange(3.), jnp.arange(3.))
    with self.assertRaisesRegex(NotImplementedError,
                                "device_put does not yet support"):
      jax.device_put(tup, jax.devices()[0])

  def test_missing_hitype_method_error(self):
    @dataclass(frozen=True)
    class Opaque:
      val: Any

    @dataclass(frozen=True)
    class OpaqueTy(HiType):
      pass

    @dataclass(frozen=True)
    class OpaqueSpec(MappingSpec):
      pass

    register_hitype(Opaque, lambda _: OpaqueTy())

    with self.assertRaisesRegex(
        NotImplementedError, r"vmap requires .*OpaqueTy.* to implement the "
        r"`dec_rank` method"):
      jax.vmap(lambda x: x, in_axes=OpaqueSpec(), out_axes=OpaqueSpec(),
               axis_size=3)(Opaque(jnp.arange(3.)))

  def test_tuple_vmap_internal(self):
    @jax.vmap
    def f(x):
      tup = make_tup(x, 2 * x)
      return get_tuple_element(tup, 0)
    x = jnp.arange(3.)
    self.assertAllClose(f(x), x)

  def test_tuple_vmap_custom_vjp(self):
    tup = make_tup(jnp.arange(3.), jnp.arange(3.) + 1)

    @jax.custom_vjp
    def inner(tup):
      return get_tuple_element(tup, 1)
    def fwd(tup):
      assert False  # unused under vmap-of-primal
    def bwd(*_):
      assert False
    inner.defvjp(fwd, bwd)

    f = jax.jit(jax.vmap(inner, in_axes=TupSpec((0, 0)), axis_size=3))
    self.assertAllClose(f(tup), jnp.arange(3.) + 1)

  def test_tuple_vmap_infer(self):
    tup = make_tup(jnp.arange(3.), jnp.arange(3.))
    jax.vmap(lambda _: make_tup(jnp.ones(3), jnp.ones(3)),
             in_axes=TupSpec((0, 0)), out_axes=batching.infer, axis_size=3)(tup)

  def test_tuple_nested_vmap(self):
    tup = make_tup(jnp.arange(12.).reshape((3, 4)), jnp.arange(12.).reshape((3, 4)))
    map1 = jax.vmap(lambda x: x, in_axes=TupSpec((0, 0)), out_axes=TupSpec((0, 0)),
                    axis_size=3)
    map2 = jax.vmap(map1, in_axes=TupSpec((1, 1)), out_axes=TupSpec((1, 1)),
                    axis_size=4)
    out = map2(tup)
    self.assertAllClose(out.elts, tup.elts)

  # def test_tuple_vmap_match(self):
  #   tup = make_tup(jnp.arange(3.), jnp.arange(3.))
  #   jax.vmap(lambda _: make_tup(jnp.ones(3), jnp.ones(3)),
  #            in_axes=TupSpec((0, 0)), out_axes=TupSpec((0, 0)), axis_size=3)(tup)

  def test_tuple_vmap_primitive(self):
    tup = make_tup(jnp.arange(3.), 5.)
    def f(tup):
      a, b = get_tuple_element(tup, 0), get_tuple_element(tup, 1)
      return make_tup(b, a)
    jax.vmap(f, in_axes=TupSpec((0, None)), out_axes=TupSpec((None, 0)), axis_size=3)(tup)

  def test_tuple_scan_mixed_length_inference(self):
    # length is inferred from array xs even when hi-type xs are present
    tup = make_tup(jnp.arange(3.), jnp.arange(3.))
    def body(c, arr_and_tup):
      arr, tup = arr_and_tup
      return c + arr + get_tuple_element(tup, 0), ()
    c, () = jax.lax.scan(body, 0., (jnp.arange(3.), tup))
    self.assertAllClose(c, 6.)

  def test_tuple_scan_length_required_error(self):
    tup = make_tup(jnp.arange(3.), jnp.arange(3.))
    with self.assertRaisesRegex(ValueError, "must provide `length`"):
      jax.lax.scan(lambda c, x: (c, ()), 0., tup)

  @parameterized.parameters([False, True])
  def test_tuple_scan(self, jit):
    tup = make_tup(jnp.arange(3.), jnp.arange(3. * 4).reshape(3, 4))
    def body(_, x):
      self.assertEqual(typeof(x), TupTy((typeof(jnp.zeros(())), typeof(jnp.arange(4.)))))
      a = get_tuple_element(x, 0)
      b = get_tuple_element(x, 1)
      return (), make_tup(a + 1, b * 2)
    def f(): return jax.lax.scan(body, (), tup, length=3)
    if jit:
      f = jax.jit(f)
    (), tup2 = f()
    a = get_tuple_element(tup2, 0)
    b = get_tuple_element(tup2, 1)
    self.assertAllClose(a, jnp.arange(3.) + 1)
    self.assertAllClose(b, jnp.arange(3. * 4).reshape(3, 4) * 2)

  def test_tuple_jit_shardings_error(self):
    # jit in_shardings/out_shardings must be unspecified for hi-type
    # args/outputs; anything else raises rather than crashing or silently
    # broadcasting one sharding across the lojax components
    mesh = jtu.create_mesh((2,), ('i',))
    tup = make_tup(jnp.arange(8., dtype='float32').reshape(4, 2),
                   jnp.arange(4., dtype='float32'))
    s = jax.NamedSharding(mesh, jax.P('i'))
    with jax.set_mesh(mesh):
      with self.assertRaisesRegex(NotImplementedError, "open an issue"):
        jax.jit(lambda t: t, in_shardings=s)(tup)
      with self.assertRaisesRegex(NotImplementedError, "open an issue"):
        jax.jit(lambda t: t, out_shardings=s)(tup)

  @jtu.with_explicit_mesh((2, 2), ('i', 'j'))
  def test_tuple_shit(self, mesh):
    x = jax.device_put(jnp.arange(4.), jax.P('i'))
    y = jax.device_put(jnp.arange(3.), jax.P(None))
    tup = make_tup(x, y)
    x_ = get_tuple_element(tup, 0)
    y_ = get_tuple_element(tup, 1)
    self.assertEqual(jax.typeof(x_).sharding.spec, jax.P('i'))
    self.assertEqual(jax.typeof(y_).sharding.spec, jax.P(None))

  @jtu.with_explicit_mesh((2, 2), ('i', 'j'))
  def test_tuple_shmap(self, mesh):
    x = jax.device_put(jnp.arange(4.), jax.P('i'))
    y = jax.device_put(jnp.arange(3.), jax.P(None))
    tup = make_tup(x, y)

    @jax.jit
    @jax.shard_map(in_specs=TupP((jax.P('i'), jax.P(None))),
                   out_specs=TupP((jax.P(None), jax.P('i'))))
    def fun(tup):
      a, b = get_tuple_element(tup, 0), get_tuple_element(tup, 1)
      return make_tup(b, a)
    out = fun(tup)
    x_ = get_tuple_element(out, 1)
    y_ = get_tuple_element(out, 0)
    self.assertAllClose(x, x_)
    self.assertAllClose(y, y_)
    self.assertEqual(x.sharding, x_.sharding)
    self.assertEqual(y.sharding, y_.sharding)

  # @jtu.with_explicit_mesh((2, 2), ('i', 'j'))
  # def test_tuple_shmap_out_specs_error(self, mesh):
  #   x = jax.device_put(jnp.arange(4.), jax.P('i'))
  #   y = jax.device_put(jnp.arange(3.), jax.P(None))
  #   tup = make_tup(x, y)

  #   # TODO(mattjj,yashkatariya): this errors too late, make shmap checks work
  #   @jax.jit
  #   @jax.shard_map(in_specs=TupP((jax.P('i'), jax.P(None))),
  #                  out_specs=TupP((jax.P('i'), jax.P('i'))))  # NOTE!!!!
  #   def fun(tup):
  #     a, b = get_tuple_element(tup, 0), get_tuple_element(tup, 1)
  #     return make_tup(b, a)
  #   out = fun(tup)
  #   x_ = get_tuple_element(out, 1)
  #   y_ = get_tuple_element(out, 0)
  #   self.assertAllClose(x, x_)
  #   self.assertAllClose(y, y_)
  #   self.assertEqual(x.sharding, x_.sharding)
  #   self.assertEqual(y.sharding, y_.sharding)

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

      def vjp_fwd(self, nzs_in, x):
        ans = self(x)
        return (ans, x)

      def vjp_bwd(self, res, t, xbar_accum):
        xbar = t * self.power * raise_to_static_power(res, self.power-1)
        xbar_accum.accum(xbar)

      def batch(self, _axis_data, args, in_dims):
        in_dim, = in_dims
        x, = args
        return raise_to_static_power(x, self.power), in_dim

      def jvp(self, primals, tangents):
        (x,), (t,) = primals, tangents
        return self(x), t * self.power * raise_to_static_power(x, self.power-1)

    def raise_to_static_power(x, power):
      x_aval = jax.typeof(x)
      return RaiseToStaticPower(x_aval, power=power)(x)

    def f(x):
      return raise_to_static_power(x, power=3)

    if jit:
      f = jax.jit(f)
      self.assertEqual(f.lower(2.0).compile()(2.0), 8.0)

    self.assertEqual(f(2.0), 8.0)
    xs = jnp.arange(3.0)
    self.assertAllClose(jax.vmap(f)(xs), xs**3)
    self.assertEqual(jax.grad(f)(2.0), 12.0)
    self.assertEqual(jax.jvp(f, (2.0,), (1.0,)),
                     (8.0, 12.0))

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

      def vjp_fwd(self, nzs_in, x):
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

  def test_newstyle_hiprimitive_vmap(self):

    class Mul(VJPHiPrimitive):

      def __init__(self, aval):
        self.in_avals = (aval, aval)
        self.out_aval = aval
        self.params = {}
        super().__init__()

      def expand(self, x, y):
        return x * y

      def batch_dim_rule(self, axis_data, in_dims):
        return in_dims[1] if in_dims[0] is None else in_dims[0]

    def mul(x, y):
      return Mul(typeof(x))(x, y)

    self.assertAllClose(mul(2.0, 3.0), 6.0)
    x = jnp.arange(3.0)
    y = jnp.arange(3.0) + 1.0
    self.assertAllClose(jax.vmap(mul)(x, y), x * y)
    x = jnp.arange(6.0).reshape(2, 3)
    self.assertAllClose(jax.vmap(mul, in_axes=(0, None))(x, y), x * y[None, :])
    self.assertAllClose(jax.vmap(mul, in_axes=(None, 0))(y, x), x * y[None, :])
    x = jnp.arange(24.0).reshape(2, 3, 4)
    f = jax.vmap(mul, in_axes=(0, None))
    f = jax.vmap(f, in_axes=(2, None), out_axes=2)
    self.assertAllClose(f(x, y), x * y[None, :, None])

  def test_newstyle_hiprimitive_nested_vmap_unmapped_axis(self):
    class Id(VJPHiPrimitive):
      def __init__(self, aval):
        self.in_avals = aval,
        self.out_aval = aval
        self.params = {}
        super().__init__()

      def expand(self, x):
        return x

      def batch_dim_rule(self, axis_data, in_dims):
        return in_dims[0]

    def ident(x): return Id(typeof(x))(x)

    x = jnp.arange(3.0, dtype='float32')
    f = jax.vmap(jax.vmap(ident), in_axes=None, axis_size=2)
    self.assertAllClose(f(x), jnp.tile(x, (2, 1)))

    g = jax.vmap(jax.vmap(ident, in_axes=None, axis_size=2))
    self.assertAllClose(g(x), jnp.tile(x[:, None], (1, 2)))

    # multiple args and a tuple output, so that None dims appear inside the
    # in_dims/out_dim pytrees (mixed with ints) at each level of nesting
    class AddSnd(VJPHiPrimitive):
      def __init__(self, x_aval, y_aval):
        self.in_avals = (x_aval, y_aval)
        self.out_aval = (x_aval, y_aval)
        self.params = {}
        super().__init__()

      def expand(self, x, y):
        return x + y, y

      def batch_dim_rule(self, axis_data, in_dims):
        d = in_dims[0] if in_dims[0] is not None else in_dims[1]
        return (d, in_dims[1])

    def addsnd(x, y):
      return AddSnd(typeof(x), typeof(y))(x, y)

    y = jnp.arange(2.0, dtype='float32')

    # outer maps x only, inner maps y only
    f = jax.vmap(jax.vmap(addsnd, in_axes=(None, 0)), in_axes=(0, None))
    s, t = f(x, y)
    self.assertAllClose(s, x[:, None] + y[None, :])
    self.assertAllClose(t, jnp.tile(y, (3, 1)))

    # outer maps y only, inner maps x only
    g = jax.vmap(jax.vmap(addsnd, in_axes=(0, None)), in_axes=(None, 0))
    s, t = g(x, y)
    self.assertAllClose(s, y[:, None] + x[None, :])
    self.assertAllClose(t, jnp.tile(y[:, None], (1, 3)))

    # inner maps nothing (axis_size only), outer maps x only
    h = jax.vmap(jax.vmap(addsnd, in_axes=(None, None), axis_size=2),
                 in_axes=(0, None))
    s, t = h(x, jnp.float32(5.0))
    self.assertAllClose(s, jnp.tile((x + 5.0)[:, None], (1, 2)))
    self.assertAllClose(t, jnp.full((3, 2), jnp.float32(5.0)))

  def test_newstyle_hiprimitive_vmap_jvp_symbolic_zero_tangent(self):

    class Mul(VJPHiPrimitive):

      def __init__(self, aval):
        self.in_avals = (aval, aval)
        self.out_aval = aval
        self.params = {}
        super().__init__()

      def expand(self, x, y):
        return x * y

      def jvp(self, primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        x_dot, y_dot = map(instantiate_zeros, (x_dot, y_dot))
        return mul(x, y), mul(x_dot, y) + mul(x, y_dot)

      def batch_dim_rule(self, axis_data, in_dims):
        return in_dims[1] if in_dims[0] is None else in_dims[0]

    def mul(x, y):
      return Mul(typeof(x))(x, y)

    # ys is closed over, so its tangent is a symbolic zero whose aval must be
    # mapped before reaching the jvp rule under vmap
    xs, ys = jnp.arange(3.0), jnp.full(3, 2.0)
    primals_out, tangents_out = jax.jvp(
        lambda x: jax.vmap(mul)(x, ys), (xs,), (jnp.ones(3),))
    self.assertAllClose(primals_out, xs * ys)
    self.assertAllClose(tangents_out, ys)

    primals_out, tangents_out = jax.jvp(
        lambda x: jax.vmap(mul, in_axes=(0, None))(x, 2.0), (xs,), (jnp.ones(3),))
    self.assertAllClose(primals_out, 2.0 * xs)
    self.assertAllClose(tangents_out, jnp.full(3, 2.0))
    x = jnp.arange(12.0).reshape(3, 4)
    y = jnp.arange(6.0).reshape(2, 3)
    f = jax.vmap(mul, in_axes=(None, 0))
    f = jax.vmap(f, in_axes=(1, None), out_axes=2)
    self.assertAllClose(f(x, y), x[None, :, :] * y[:, :, None])

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

      def vjp_fwd(self, nzs_in, x):
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

      def vjp_fwd(self, nzs_in, qx):
        return self(qx), None

      def vjp_bwd_retval(self, _, g):
        return g,

    def f(x):
      return jnp.sum(dq(q(x)))

    x = jax.random.normal(jax.random.key(0), (3, 3), dtype='float32')
    jax.grad(f)(x)

  def test_symbolic_zeros(self):

    class Mul(VJPHiPrimitive):
      def __init__(self, aval):
        self.in_avals = (aval, aval)
        self.out_aval = aval
        self.params = {}
        super().__init__()

      def expand(self, x, y):
        return x * y

      def vjp_fwd(self, nzs_in, x, y):
        assert list(nzs_in) == list(nzs_in_)  # defined below
        ans = self(x, y)
        return ans, (x, y)

      def vjp_bwd(self, res, g, x_acc, y_acc):
        assert list(nzs_in_) == [not isinstance(x_acc, ad.NullAccum),
                                 not isinstance(y_acc, ad.NullAccum)]
        x, y = res
        x_acc.accum(g * y)
        y_acc.accum(x * g)

    def mul(x, y):
      return Mul(typeof(x))(x, y)

    nzs_in_ = (True, False)
    self.assertAllClose(jax.grad(mul)(2., 3.), 3., check_dtypes=False)

    nzs_in_ = (False, True)
    self.assertAllClose(jax.grad(mul, 1)(2., 3.), 2., check_dtypes=False)

  def test_symbolic_zeros_retval(self):

    class Mul(VJPHiPrimitive):
      def __init__(self, aval):
        self.in_avals = (aval, aval)
        self.out_aval = aval
        self.params = {}
        super().__init__()

      def expand(self, x, y):
        return x * y

      def vjp_fwd(self, nzs_in, x, y):
        assert list(nzs_in) == list(nzs_in_)  # defined below
        ans = self(x, y)
        return ans, (x, y)

      def vjp_bwd_retval(self, res, g):
        x, y = res
        return (g * y, x * g)

    def mul(x, y):
      return Mul(typeof(x))(x, y)

    nzs_in_ = (True, False)
    self.assertAllClose(jax.grad(mul)(2., 3.), 3., check_dtypes=False)

    nzs_in_ = (False, True)
    self.assertAllClose(jax.grad(mul, 1)(2., 3.), 2., check_dtypes=False)

  @jtu.with_explicit_mesh((2,), ('data',))
  def test_hijax_primitive_under_shard_map(self, mesh):
    x = jax.device_put(jnp.arange(10), jax.P('data'))
    g = jax.shard_map(square, in_specs=(jax.P('data'),), out_specs=jax.P('data'))
    g(x)
    jax.jit(g)(x)

  def test_hijax_cond_platform_dependent(self):
    x = jnp.arange(10)
    result = jax.jit(partial(jax.lax.platform_dependent, cpu=square, default=square))(x)
    self.assertArraysAllClose(result, x ** 2)

  def test_hijax_primitive_under_remat(self):
    x = jnp.arange(10)
    expected = x ** 2
    with self.subTest("no jit"):
      self.assertArraysAllClose(jax.remat(square)(x), expected)
    with self.subTest("jit"):
      self.assertArraysAllClose(jax.jit(jax.remat(square))(x), expected)

    x = jnp.float32(2.0)
    expected_grad = jnp.float32(4.0)
    with self.subTest("jit-of-grad"):
      if config.remat3.value:
        # remat3 differentiates via the vjp rules; the jvp rule is unused
        count = Square._jvp_execution_count
        actual_grad = jax.jit(jax.grad(jax.remat(square)))(x)
        self.assertEqual(Square._jvp_execution_count, count)
      else:
        with Square.assert_jvp_rule_called_once():
          actual_grad = jax.jit(jax.grad(jax.remat(square)))(x)
      self.assertArraysAllClose(actual_grad, expected_grad)

  @parameterized.parameters([False, True])
  def test_linearize_rule(self, jit):
    class RaiseToStaticPower(VJPHiPrimitive):
      def __init__(self, in_aval, *, power):
        self.in_avals = (in_aval,)
        self.out_aval = in_aval
        self.params = dict(power=power)
        super().__init__()

      def expand(self, x):
        return x ** self.power

      def lin(self, nzs_in, x):
        nz, = nzs_in
        assert nz
        return self(x), x

      def linearized(self, x, t):
        return t * self.power * raise_to_static_power(x, self.power-1)

    def raise_to_static_power(x, power):
      x_aval = jax.typeof(x)
      return RaiseToStaticPower(x_aval, power=power)(x)

    def f(x):
      return raise_to_static_power(x, 3)

    if jit:
      f = jax.jit(f)

    self.assertEqual(f(2.0), 8.0)
    self.assertEqual(jax.linearize(f, 2.0)[1](1.0), 12.0)

  @parameterized.parameters([False, True])
  def test_rules_derived_from_jvp(self, jit):
    class Sin(VJPHiPrimitive):
      def __init__(self, x_aval):
        self.in_avals = (x_aval,)
        self.out_aval = x_aval
        self.params = {}
        super().__init__()

      def expand(self, x):
        return jnp.sin(x)

      def jvp(self, primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return self(x), jnp.cos(x) * x_dot

      lin, linearized = linearize_from_jvp
      vjp_fwd, vjp_bwd_retval = vjp_from_jvp

      def batch_dim_rule(self, _axis_data, in_dims):
        return in_dims[0]

    def sin(x):
      return Sin(jax.typeof(x))(x)

    f = jax.jit(sin) if jit else sin

    self.assertAllClose(f(2.0), jnp.sin(2.0))
    _, y_dot = jax.jvp(f, (2.0,), (1.0,))
    self.assertAllClose(y_dot, jnp.cos(2.0))
    y, f_lin = jax.linearize(f, 2.0)
    self.assertAllClose(y, jnp.sin(2.0))
    self.assertAllClose(f_lin(1.0), jnp.cos(2.0))
    self.assertAllClose(jax.grad(f)(2.0), jnp.cos(2.0))
    self.assertAllClose(jax.grad(jax.grad(f))(2.0), -jnp.sin(2.0))
    xs = jnp.arange(3.0)
    self.assertAllClose(jax.vmap(jax.grad(f))(xs), jnp.cos(xs))
    # forward-over-reverse and one-pass jacobians, via the batch rule
    self.assertAllClose(jax.hessian(f)(2.0), -jnp.sin(2.0))
    self.assertAllClose(jax.jacfwd(f)(2.0), jnp.cos(2.0))
    self.assertAllClose(jax.jacrev(f)(2.0), jnp.cos(2.0))

  def test_rules_derived_from_jvp_multiple_args(self):
    zero_tangents_seen = []

    class Mul(VJPHiPrimitive):
      def __init__(self, x_aval, y_aval):
        self.in_avals = (x_aval, y_aval)
        self.out_aval = x_aval
        self.params = {}
        super().__init__()

      def expand(self, x, y):
        return x * y

      def jvp(self, primals, tangents):
        (x, y), (x_dot, y_dot) = primals, tangents
        zero_tangents_seen.append((isinstance(x_dot, Zero),
                                   isinstance(y_dot, Zero)))
        x_dot, y_dot = instantiate_zeros(x_dot), instantiate_zeros(y_dot)
        return self(x, y), x_dot * y + x * y_dot

      lin, linearized = linearize_from_jvp
      vjp_fwd, vjp_bwd_retval = vjp_from_jvp

    def mul(x, y):
      return Mul(jax.typeof(x), jax.typeof(y))(x, y)

    gx, gy = jax.grad(mul, (0, 1))(2.0, 3.0)
    self.assertAllClose(gx, 3.0, check_dtypes=False)
    self.assertAllClose(gy, 2.0, check_dtypes=False)
    # symbolically-zero tangent for one argument
    self.assertAllClose(jax.grad(mul, 1)(2.0, 3.0), 2.0, check_dtypes=False)
    _, f_lin = jax.linearize(mul, 2.0, 3.0)
    self.assertAllClose(f_lin(1.0, 0.0), 3.0, check_dtypes=False)
    self.assertAllClose(f_lin(0.0, 1.0), 2.0, check_dtypes=False)
    # the jvp rule sees a symbolic Zero tangent for the constant argument
    zero_tangents_seen.clear()
    _, f_lin = jax.linearize(lambda x: mul(x, 3.0), 2.0)
    self.assertAllClose(f_lin(1.0), 3.0, check_dtypes=False)
    self.assertIn((False, True), zero_tangents_seen)

  @parameterized.parameters([False, True])
  def test_vjp_derived_from_user_lin(self, jit):
    class RaiseToStaticPower(VJPHiPrimitive):
      def __init__(self, in_aval, *, power):
        self.in_avals = (in_aval,)
        self.out_aval = in_aval
        self.params = dict(power=power)
        super().__init__()

      def expand(self, x):
        return x ** self.power

      def lin(self, nzs_in, x):
        return self(x), x

      def linearized(self, x, t):
        return t * self.power * raise_to_static_power(x, self.power-1)

      vjp_fwd, vjp_bwd_retval = vjp_from_lin

    def raise_to_static_power(x, power):
      return RaiseToStaticPower(jax.typeof(x), power=power)(x)

    def f(x):
      return raise_to_static_power(x, 3)

    if jit:
      f = jax.jit(f)

    self.assertEqual(f(2.0), 8.0)
    self.assertEqual(jax.linearize(f, 2.0)[1](1.0), 12.0)
    self.assertEqual(jax.grad(f)(2.0), 12.0)
    self.assertEqual(jax.grad(jax.grad(f))(2.0), 12.0)

  def test_vjp_derived_from_derived_lin(self):
    # the whole chain: jvp -> derived lin -> derived vjp
    class Sin(VJPHiPrimitive):
      def __init__(self, x_aval):
        self.in_avals = (x_aval,)
        self.out_aval = x_aval
        self.params = {}
        super().__init__()

      def expand(self, x):
        return jnp.sin(x)

      def jvp(self, primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return self(x), jnp.cos(x) * x_dot

      lin, linearized = linearize_from_jvp
      vjp_fwd, vjp_bwd_retval = vjp_from_lin

    def sin(x):
      return Sin(jax.typeof(x))(x)

    self.assertAllClose(jax.grad(sin)(2.0), jnp.cos(2.0))
    self.assertAllClose(jax.jit(jax.grad(sin))(2.0), jnp.cos(2.0))
    self.assertAllClose(jax.grad(jax.grad(sin))(2.0), -jnp.sin(2.0))

  def test_structured_residuals(self):
    # `lin` and `vjp_fwd` may return a fourth element, structured residuals,
    # in which case `linearized` and `vjp_bwd` receive them as an extra
    # argument after the (unstructured) residuals.
    class Square(VJPHiPrimitive):
      def __init__(self, in_aval):
        self.in_avals = (in_aval,)
        self.out_aval = in_aval
        self.params = {}
        super().__init__()

      def expand(self, x):
        return x ** 2

      def lin(self, nzs_in, x):
        return self(x), (), True, {'x1': x, 'x2': x}

      def linearized(self, res, sres, t):
        return t * 2.0 * sres['x2']

      def vjp_fwd(self, nzs_in, x):
        return self(x), (), True, {'x1': x, 'x2': x}

      def vjp_bwd(self, res, sres, t, x_accum):
        if isinstance(x_accum, ad.GradAccum):
          x_accum.accum(t * 2.0 * sres['x1'])

    def square(x):
      return Square(jax.typeof(x))(x)

    self.assertAllClose(jax.grad(square)(3.0), 6.0)
    y, f_lin = jax.linearize(square, 3.0)
    self.assertAllClose(y, 9.0)
    self.assertAllClose(f_lin(1.0), 6.0)

    # the sres tree is user-visible, with duplicate positions re-duplicated
    _, f_vjp = jax.vjp(square, 3.0)
    leaves = jax.tree.leaves(f_vjp.structured_residuals)
    self.assertLen(leaves, 2)
    self.assertIs(leaves[0], leaves[1])

    # under jit, both sres leaves are the input, so input forwarding plus
    # de-duplication leave the fwd call returning only the primal
    fj = jax.jit(square)
    self.assertAllClose(jax.grad(fj)(3.0), 6.0)
    jaxpr = jax.make_jaxpr(lambda x: jax.vjp(fj, x)[1](1.0))(3.0)
    fwd_eqn = next(e for e in jaxpr.eqns if e.primitive.name == 'jit')
    self.assertLen(fwd_eqn.outvars, 1)

  def test_structured_residuals_require_vjp_bwd_override(self):
    class Bad(VJPHiPrimitive):
      def __init__(self, in_aval):
        self.in_avals = (in_aval,)
        self.out_aval = in_aval
        self.params = {}
        super().__init__()

      def expand(self, x):
        return x ** 2

      def vjp_fwd(self, nzs_in, x):
        return self(x), (), True, {'x': x}

      def vjp_bwd_retval(self, res, t):
        return (t,)

    with self.assertRaisesRegex(TypeError, "structured residuals"):
      jax.grad(lambda x: Bad(jax.typeof(x))(x))(3.0)

  def test_backward_pass_logging(self):
    # A vjp_bwd rule can return a dict of pytrees to log out of the backward
    # pass; f_vjp.with_logs(out_ct) returns (arg_cts, logs), where logs merges
    # the rules' dicts with clobber semantics. Plain f_vjp(out_ct) drops them.
    class Square(VJPHiPrimitive):
      def __init__(self, in_aval, tag):
        self.in_avals = (in_aval,)
        self.out_aval = in_aval
        self.params = dict(tag=tag)
        super().__init__()

      def expand(self, x):
        return x ** 2

      def vjp_fwd(self, nzs_in, x):
        return self(x), x

      def vjp_bwd(self, res, t, x_accum):
        if isinstance(x_accum, ad.GradAccum):
          x_accum.accum(t * 2.0 * res)
        return {self.tag: {'x': res, 'ct_in': t}}

    def square(x, tag='sq'):
      return Square(jax.typeof(x), tag)(x)

    _, f_vjp = jax.vjp(square, 3.0)
    cts, logs = f_vjp.with_logs(1.0)
    self.assertAllClose(cts[0], 6.0)
    self.assertAllClose(logs, {'sq': {'x': 3.0, 'ct_in': 1.0}},
                        check_dtypes=False)
    self.assertAllClose(f_vjp(1.0)[0], 6.0)  # plain call drops the logs
    self.assertAllClose(jax.grad(square)(3.0), 6.0)

    # distinct keys are both present; a repeated key is clobbered, with the
    # earlier-in-forward-order rule winning
    f2 = lambda x: square(square(x, 'inner'), 'outer')
    _, f2_vjp = jax.vjp(f2, 2.0)
    _, logs2 = f2_vjp.with_logs(1.0)
    self.assertEqual(set(logs2), {'inner', 'outer'})
    self.assertAllClose(logs2['inner']['x'], 2.0)
    self.assertAllClose(logs2['outer']['x'], 4.0)
    _, f3_vjp = jax.vjp(lambda x: square(square(x)), 2.0)
    _, logs3 = f3_vjp.with_logs(1.0)
    self.assertAllClose(logs3['sq']['x'], 2.0)

    # logs flow out of a transposed jit, and with_logs itself can be traced
    fj = jax.jit(lambda x: square(x))
    _, fj_vjp = jax.vjp(fj, 3.0)
    ctsj, logsj = fj_vjp.with_logs(1.0)
    self.assertAllClose(ctsj[0], 6.0)
    self.assertAllClose(logsj['sq']['x'], 3.0)
    ctsT, logsT = jax.jit(
        lambda x, ct: jax.vjp(fj, x)[1].with_logs(ct))(3.0, 1.0)
    self.assertAllClose(logsT['sq']['x'], 3.0)

    # logs from a scan body are stacked leaf-wise, index-aligned with the
    # forward iterations
    def f_scan(xs):
      c_out, ys = jax.lax.scan(lambda c, x: (c + square(x), square(x)), 0., xs)
      return c_out + ys.sum()
    xs = jnp.array([1., 2., 3.])
    _, fs_vjp = jax.vjp(f_scan, xs)
    ctss, logss = fs_vjp.with_logs(1.0)
    self.assertAllClose(ctss[0], 4.0 * xs)
    self.assertArraysEqual(logss['sq']['x'], xs)
    self.assertEqual(logss['sq']['ct_in'].shape, xs.shape)

  def test_backward_pass_logging_cond(self):
    # A transposed cond logs a sum represented as a tagged product: each key
    # logged by any branch maps to a CondSum holding the taken-branch index
    # and one slot per branch (live value / zeros if untaken / None if the
    # branch doesn't log the key). Branches needn't agree on keys or types.
    from jax._src.lax.control_flow.conditionals import CondSum

    class Square(VJPHiPrimitive):
      def __init__(self, in_aval, tag):
        self.in_avals = (in_aval,)
        self.out_aval = in_aval
        self.params = dict(tag=tag)
        super().__init__()

      def expand(self, x):
        return x ** 2

      def vjp_fwd(self, nzs_in, x):
        return self(x), x

      def vjp_bwd(self, res, t, x_accum):
        if isinstance(x_accum, ad.GradAccum):
          x_accum.accum(t * 2.0 * res)
        return {self.tag: res}

    def square(x, tag):
      return Square(jax.typeof(x), tag)(x)

    def f(x):
      return jax.lax.cond(x > 0,
                          lambda x: square(x, 'pos') * 1.0,
                          lambda x: square(x, 'neg') * 2.0, x)

    _, f_vjp = jax.vjp(f, 3.0)
    cts, logs = f_vjp.with_logs(1.0)
    self.assertAllClose(cts[0], 6.0)
    self.assertEqual(set(logs), {'pos', 'neg'})
    self.assertIsInstance(logs['pos'], CondSum)
    i = int(logs['pos'].index)
    self.assertAllClose(logs['pos'].branches[i], 3.0)      # taken, live
    self.assertIsNone(logs['pos'].branches[1 - i])         # doesn't log 'pos'
    self.assertAllClose(logs['neg'].branches[1 - i], 0.0)  # untaken, zeros
    self.assertIsNone(logs['neg'].branches[i])

    _, f_vjp2 = jax.vjp(f, -3.0)
    cts2, logs2 = f_vjp2.with_logs(1.0)
    self.assertAllClose(cts2[0], -12.0)
    i2 = int(logs2['neg'].index)
    self.assertAllClose(logs2['neg'].branches[i2], -3.0)
    self.assertAllClose(logs2['pos'].branches[1 - i2], 0.0)

    def g(x):
      return jax.lax.cond(
          x > 0,
          lambda x: square(x, 'both') * 1.0,
          lambda x: square(jnp.stack([x, x]), 'both').sum() * 2.0, x)

    _, g_vjp = jax.vjp(g, 3.0)
    _, glogs = g_vjp.with_logs(1.0)
    self.assertEqual(set(glogs), {'both'})
    j = int(glogs['both'].index)
    self.assertAllClose(glogs['both'].branches[j], 3.0)     # taken, live
    self.assertAllClose(glogs['both'].branches[1 - j],
                        jnp.zeros(2))                       # untaken, zeros

    _, g_vjp2 = jax.vjp(g, -3.0)
    _, glogs2 = g_vjp2.with_logs(1.0)
    j2 = int(glogs2['both'].index)
    self.assertAllClose(glogs2['both'].branches[j2], jnp.array([-3., -3.]))
    self.assertAllClose(glogs2['both'].branches[1 - j2], 0.0)

    # nested conds nest their CondSums
    def nested(x):
      inner = lambda x: jax.lax.cond(x > 1, lambda x: square(x, 'deep'),
                                     lambda x: x * 5.0, x)
      return jax.lax.cond(x > 0, inner, lambda x: x * 7.0, x)
    _, n_vjp = jax.vjp(nested, 2.0)
    _, nlogs = n_vjp.with_logs(1.0)
    outer = nlogs['deep']
    self.assertIsInstance(outer, CondSum)
    inner_log = outer.branches[int(outer.index)]
    self.assertIsInstance(inner_log, CondSum)
    self.assertAllClose(inner_log.branches[int(inner_log.index)], 2.0)

    # under jit, and plain grad unaffected
    _, fj_vjp = jax.vjp(jax.jit(f), 3.0)
    _, logsj = fj_vjp.with_logs(1.0)
    self.assertAllClose(logsj['pos'].branches[int(logsj['pos'].index)], 3.0)
    self.assertAllClose(jax.grad(f)(3.0), 6.0)

  def test_backward_pass_logging_shard_map(self):
    # Logs from inside a transposed shard_map come out mesh-stacked along
    # their leading axis (per-shard scalars come out with shape (num_shards,)).
    class Square(VJPHiPrimitive):
      def __init__(self, in_aval):
        self.in_avals = (in_aval,)
        self.out_aval = in_aval
        self.params = {}
        super().__init__()

      def expand(self, x):
        return x ** 2

      def vjp_fwd(self, nzs_in, x):
        return self(x), x

      def vjp_bwd(self, res, t, x_accum):
        if isinstance(x_accum, ad.GradAccum):
          x_accum.accum(t * 2.0 * res)
        return {'x': res, 'norm2': (res ** 2).sum()}

    def square(x):
      return Square(jax.typeof(x))(x)

    mesh = jax.make_mesh((1,), ('i',), axis_types=(jax.sharding.AxisType.Auto,))
    spec = jax.sharding.PartitionSpec('i')
    sm = jax.shard_map(lambda x: square(x) * 2.0, mesh=mesh, in_specs=spec,
                       out_specs=spec)
    xs = jnp.arange(1., 5.)
    f = lambda x: sm(x).sum()
    _, f_vjp = jax.vjp(f, xs)
    cts, logs = f_vjp.with_logs(1.0)
    self.assertAllClose(cts[0], 4.0 * xs)
    self.assertArraysEqual(logs['x'], xs)
    self.assertEqual(logs['norm2'].shape, (1,))  # one shard
    self.assertAllClose(logs['norm2'][0], (xs ** 2).sum())
    self.assertAllClose(jax.grad(f)(xs), 4.0 * xs)

  def test_backward_pass_logging_bad_return(self):
    class Bad(VJPHiPrimitive):
      def __init__(self, in_aval):
        self.in_avals = (in_aval,)
        self.out_aval = in_aval
        self.params = {}
        super().__init__()

      def expand(self, x):
        return x ** 2

      def vjp_fwd(self, nzs_in, x):
        return self(x), x

      def vjp_bwd(self, res, t, x_accum):
        if isinstance(x_accum, ad.GradAccum):
          x_accum.accum(t * 2.0 * res)
        return [t]  # not a dict

    with self.assertRaisesRegex(TypeError, "backward-pass log"):
      jax.vjp(lambda x: Bad(jax.typeof(x))(x), 3.0)[1](1.0)

    bad_id_p = core.Primitive('bad_id')
    bad_id_p.def_impl(lambda x: x)
    bad_id_p.def_abstract_eval(lambda a: a)
    ad.defjvp(bad_id_p, lambda g, x: bad_id_p.bind(g))
    def bad_transpose(ct, x):
      if isinstance(x, ad.ValAccum):
        x.accum(ct)
      return [ct]  # not a dict
    ad.fancy_transposes[bad_id_p] = bad_transpose

    with self.assertRaisesRegex(TypeError, "backward-pass log"):
      jax.vjp(lambda x: bad_id_p.bind(x) * 2., 3.0)[1](1.0)

  def test_backward_pass_logging_with_refs(self):
    class Square(VJPHiPrimitive):
      def __init__(self, in_aval):
        self.in_avals = (in_aval,)
        self.out_aval = in_aval
        self.params = {}
        super().__init__()

      def expand(self, x):
        return x ** 2

      def vjp_fwd(self, nzs_in, x):
        return self(x), x

      def vjp_bwd(self, res, t, x_accum):
        if isinstance(x_accum, ad.GradAccum):
          x_accum.accum(t * 2.0 * res)
        return {'sq': {'x': res, 'ct_in': t}}

    def f(x, y):
      return Square(jax.typeof(x))(x) * y

    _, f_vjp = jax.vjp(f, 3.0, 2.0)
    arg_cts, logs = f_vjp.with_logs.with_refs(
        jax.ad.GradValue(), jax.ad.DontWant())(1.0)
    x_ct, y_ct = arg_cts
    self.assertAllClose(x_ct, 12.0)  # d(x**2 * y)/dx = 2xy
    self.assertIsInstance(y_ct, jax.ad.DidntWant)
    self.assertAllClose(logs, {'sq': {'x': 3.0, 'ct_in': 2.0}},
                        check_dtypes=False)

    x_ct, y_ct = f_vjp.with_refs(jax.ad.GradValue(), jax.ad.DontWant())(1.0)
    self.assertAllClose(x_ct, 12.0)
    (x_ct, y_ct), logs = f_vjp.with_logs(1.0)
    self.assertAllClose(x_ct, 12.0)
    self.assertAllClose(y_ct, 9.0)  # d(x**2 * y)/dy = x**2
    self.assertAllClose(logs, {'sq': {'x': 3.0, 'ct_in': 2.0}},
                        check_dtypes=False)

  def test_backward_pass_logging_vjp_pytree_roundtrip(self):
    _, f_vjp = jax.vjp(jnp.sin, 1.0)

    leaves, treedef = jax.tree.flatten(f_vjp)
    (ct,) = jax.tree.unflatten(treedef, leaves)(1.0)
    self.assertAllClose(ct, jnp.cos(1.0))

    leaves, treedef = jax.tree.flatten(f_vjp.with_logs)
    (ct,), logs = jax.tree.unflatten(treedef, leaves)(1.0)
    self.assertAllClose(ct, jnp.cos(1.0))
    self.assertEqual(logs, {})

  @jtu.run_on_devices("cpu")  # TODO(mattjj): debug xla failures
  def test_hijax_inside_call_primitives(self):
    from jax._src.compute_on import compute_on
    from jax.experimental.fused import fused
    from jax.experimental.scheduling_groups import scheduling_group

    class Square(VJPHiPrimitive):
      def __init__(self, in_aval):
        self.in_avals = (in_aval,)
        self.out_aval = in_aval
        self.params = {}
        super().__init__()

      def expand(self, x):
        return x ** 2

      def lin(self, nzs_in, x):
        c = x * 2.0
        return self(x), (), True, {'x1': x, 'x2': x, 'c1': c, 'c2': c}

      def linearized(self, res, sres, t):
        return t * sres['c2']

      def vjp_fwd(self, nzs_in, x):
        c = x * 2.0
        return self(x), (), True, {'x1': x, 'x2': x, 'c1': c, 'c2': c}

      def vjp_bwd(self, res, sres, t, x_accum):
        if isinstance(x_accum, ad.GradAccum):
          x_accum.accum(t * sres['c1'])
        return {'sq': sres['x1']}

    def square(x):
      return Square(jax.typeof(x))(x)

    co = lambda f: compute_on(f, compute_type='device_host',
                               out_memory_spaces=jax.memory.Space.Device)
    for wrap, prim_name in [(scheduling_group('g'), 'xla_metadata_call'),
                            (co, 'compute_on')]:
      f = wrap(square)
      self.assertAllClose(f(3.0), 9.0)
      self.assertAllClose(jax.grad(f)(3.0), 6.0)
      self.assertAllClose(jax.grad(jax.jit(f))(3.0), 6.0)
      _, f_vjp = jax.vjp(f, 3.0)
      cts, logs = f_vjp.with_logs(1.0)
      self.assertAllClose(cts[0], 6.0)
      self.assertAllClose(logs['sq'], 3.0)
      leaves = jax.tree.leaves(f_vjp.structured_residuals)
      self.assertLen(leaves, 4)
      self.assertIs(leaves[0], leaves[1])  # c pair, deduped
      self.assertIs(leaves[2], leaves[3])  # x pair, input-forwarded
      jaxpr = jax.make_jaxpr(lambda x: jax.vjp(f, x)[1](1.0))(3.0)
      fwd_eqn = next(e for e in jaxpr.eqns if e.primitive.name == prim_name)
      self.assertLen(fwd_eqn.outvars, 2)

    ff = fused(out_spaces=(jax.memory.Space.Device,))(square)
    lo = jax.jit(ff).trace(3.0).lojax.jaxpr
    eqn = next(e for e in lo.jaxpr.eqns if e.primitive.name == 'fused_call')
    self.assertFalse(eqn.params['jaxpr'].is_high)
    self.assertEqual(eqn.params['out_spaces'], (jax.memory.Space.Device,))

  def test_backward_pass_logging_call_primitives(self):
    from jax._src import api_util
    from jax._src import flattree as ft
    from jax._src.compute_on import compute_on
    from jax._src.interpreters import mlir, partial_eval as pe
    from jax._src.lax.eval_jaxpr import eval_jaxpr_p
    from jax.experimental.scheduling_groups import scheduling_group

    log_id_p = core.Primitive('log_id')
    log_id_p.def_impl(lambda x: x)
    log_id_p.def_abstract_eval(lambda a: a)
    ad.defjvp(log_id_p, lambda g, x: log_id_p.bind(g))
    mlir.register_lowering(log_id_p, lambda ctx, x: [x])
    def _log_id_transpose(ct, x):
      if isinstance(x, ad.ValAccum):
        x.accum(ct)
      return {'canary': ct}
    ad.fancy_transposes[log_id_p] = _log_id_transpose

    def f(x):
      return log_id_p.bind(jnp.sin(x)) * 2.

    dbg = api_util.debug_info('test', f, (1.0,), {})
    args_ft = ft.flatten(((1.0,), {}))
    jaxpr, _ = pe.trace_to_jaxpr(f, args_ft.map(core.shaped_abstractify), dbg)

    fns = [scheduling_group('g')(f),
           compute_on(f, compute_type='device_host',
                       out_memory_spaces=jax.memory.Space.Device),
           lambda x: eval_jaxpr_p.bind(x, call_jaxpr=jaxpr)[0]]
    for fn in fns:
      _, f_vjp = jax.vjp(fn, 1.0)
      cts, logs = f_vjp.with_logs(1.0)
      self.assertAllClose(cts[0], 2 * jnp.cos(1.0))
      self.assertAllClose(logs, {'canary': jnp.float32(2.0)},
                          check_dtypes=False)
      (ct,) = f_vjp(1.0)  # plain call drops the logs without error
      self.assertAllClose(ct, 2 * jnp.cos(1.0))

  def test_backward_pass_logging_remat3(self):
    # Under remat3, a vjp_bwd rule's logs flow out of a rematted computation's
    # backward pass.
    class Square(VJPHiPrimitive):
      def __init__(self, in_aval, tag):
        self.in_avals = (in_aval,)
        self.out_aval = in_aval
        self.params = dict(tag=tag)
        super().__init__()

      def expand(self, x):
        return x ** 2

      def vjp_fwd(self, nzs_in, x):
        return self(x), x

      def vjp_bwd(self, res, t, x_accum):
        if isinstance(x_accum, ad.GradAccum):
          x_accum.accum(t * 2.0 * res)
        return {self.tag: {'x': res, 'ct_in': t}}

    def square(x, tag='sq'):
      return Square(jax.typeof(x), tag)(x)

    with config.remat3(True):
      f = jax.checkpoint(lambda x: square(jnp.sin(x)))
      _, f_vjp = jax.vjp(f, 3.0)
      cts, logs = f_vjp.with_logs(1.0)
      self.assertAllClose(cts[0], 2 * jnp.sin(3.0) * jnp.cos(3.0))
      self.assertAllClose(logs, {'sq': {'x': jnp.sin(3.0), 'ct_in': 1.0}},
                          check_dtypes=False)
      (ct,) = f_vjp(1.0)  # plain call drops the logs without error
      self.assertAllClose(ct, 2 * jnp.sin(3.0) * jnp.cos(3.0))

      # under jit, and with with_logs itself traced
      _, logsj = jax.vjp(jax.jit(f), 3.0)[1].with_logs(1.0)
      self.assertAllClose(logsj['sq']['x'], jnp.sin(3.0))
      _, logst = jax.jit(lambda x, ct: jax.vjp(f, x)[1].with_logs(ct))(3.0, 1.0)
      self.assertAllClose(logst['sq']['x'], jnp.sin(3.0))

      # nested remat
      g = jax.checkpoint(lambda x: square(jax.checkpoint(
          lambda y: square(y, 'inner'))(x), 'outer'))
      cts, logs = jax.vjp(g, 2.0)[1].with_logs(1.0)
      self.assertAllClose(cts[0], 32.0, check_dtypes=False)
      self.assertEqual(set(logs), {'inner', 'outer'})
      self.assertAllClose(logs['inner']['x'], 2.0, check_dtypes=False)
      self.assertAllClose(logs['outer']['x'], 4.0, check_dtypes=False)

      # remat-of-scan stacks the body's logs across iterations
      def f_scan(xs):
        c, _ = jax.lax.scan(lambda c, x: (c + square(x), None), 0., xs)
        return c
      xs = jnp.arange(1., 4.)
      _, logss = jax.vjp(jax.checkpoint(f_scan), xs)[1].with_logs(1.0)
      self.assertArraysEqual(logss['sq']['x'], xs)

      # a checkpoint policy doesn't disturb the logs
      fp = jax.checkpoint(lambda x: square(jnp.sin(x)),
                          policy=jax.checkpoint_policies.nothing_saveable)
      _, logsp = jax.vjp(fp, 3.0)[1].with_logs(1.0)
      self.assertAllClose(logsp['sq']['x'], jnp.sin(3.0))

  @parameterized.parameters([False, True])
  def test_backward_pass_logging_remat_custom_vjp(self, remat3):
    @jax.custom_vjp
    def log_id(x):
      return x

    def log_id_fwd(x):
      return log_id(x), None

    def log_id_bwd(_, ct):
      return (ct,), {'canary': ct}

    log_id.defvjp_with_logs(log_id_fwd, log_id_bwd)

    with config.remat3(remat3):
      f = jax.checkpoint(lambda x: log_id(jnp.sin(x)) * 2.)
      _, f_vjp = jax.vjp(f, 1.0)
      cts, logs = f_vjp.with_logs(1.0)
      self.assertAllClose(cts[0], 2 * jnp.cos(1.0))
      self.assertAllClose(logs, {'canary': 2.0}, check_dtypes=False)
      (ct,) = f_vjp(1.0)  # plain call drops the logs without error
      self.assertAllClose(ct, 2 * jnp.cos(1.0))

      # under jit, and with with_logs itself traced
      _, logsj = jax.vjp(jax.jit(f), 1.0)[1].with_logs(1.0)
      self.assertAllClose(logsj, {'canary': 2.0}, check_dtypes=False)
      _, logst = jax.jit(lambda x, ct: jax.vjp(f, x)[1].with_logs(ct))(1.0, 1.0)
      self.assertAllClose(logst, {'canary': 2.0}, check_dtypes=False)

      # nested remat
      g = jax.checkpoint(lambda x: jax.checkpoint(
          lambda y: log_id(jnp.sin(y)))(x) * 2.)
      _, logsn = jax.vjp(g, 1.0)[1].with_logs(1.0)
      self.assertAllClose(logsn, {'canary': 2.0}, check_dtypes=False)

  def test_jvp_derived_from_lin(self):
    class RaiseToStaticPower(VJPHiPrimitive):
      def __init__(self, in_aval, *, power):
        self.in_avals = (in_aval,)
        self.out_aval = in_aval
        self.params = dict(power=power)
        super().__init__()

      def expand(self, x):
        return x ** self.power

      def lin(self, nzs_in, x):
        return self(x), x

      def linearized(self, x, t):
        return t * self.power * raise_to_static_power(x, self.power-1)

      jvp = jvp_from_lin
      vjp_fwd, vjp_bwd_retval = vjp_from_lin

      def batch_dim_rule(self, _axis_data, in_dims):
        return in_dims[0]

    def raise_to_static_power(x, power):
      return RaiseToStaticPower(jax.typeof(x), power=power)(x)

    def f(x):
      return raise_to_static_power(x, 3)

    self.assertEqual(jax.jvp(f, (2.0,), (1.0,)), (8.0, 12.0))
    self.assertEqual(jax.grad(f)(2.0), 12.0)
    self.assertEqual(jax.hessian(f)(2.0), 12.0)

  def test_jvp_from_lin_circular_error(self):
    class Sin(VJPHiPrimitive):
      def __init__(self, x_aval):
        self.in_avals = (x_aval,)
        self.out_aval = x_aval
        self.params = {}
        super().__init__()

      def expand(self, x):
        return jnp.sin(x)

      jvp = jvp_from_lin
      lin, linearized = linearize_from_jvp

    def sin(x):
      return Sin(jax.typeof(x))(x)

    with self.assertRaisesRegex(TypeError, 'jvp_from_lin'):
      jax.jvp(sin, (2.0,), (1.0,))

  def test_derived_rules_with_static_params(self):
    class ApplyAndScale(VJPHiPrimitive):
      def __init__(self, x_aval, *, f, scale):
        self.in_avals = (x_aval,)
        self.out_aval = x_aval
        self.params = dict(f=f, scale=scale)
        super().__init__()

      def expand(self, x):
        return self.scale * self.f(x)

      def jvp(self, primals, tangents):
        (x,), (t,) = primals, tangents
        return self(x), self.scale * jax.jvp(self.f, (x,), (t,))[1]

      lin, linearized = linearize_from_jvp
      vjp_fwd, vjp_bwd_retval = vjp_from_jvp

    def apply_and_scale(f, scale, x):
      return ApplyAndScale(jax.typeof(x), f=f, scale=scale)(x)

    f = lambda x: apply_and_scale(jnp.sin, 2.0, x)
    self.assertAllClose(f(2.0), 2 * jnp.sin(2.0))
    self.assertAllClose(jax.grad(f)(2.0), 2 * jnp.cos(2.0))
    self.assertAllClose(jax.jit(jax.grad(f))(2.0), 2 * jnp.cos(2.0))
    self.assertAllClose(jax.grad(jax.grad(f))(2.0), -2 * jnp.sin(2.0))
    self.assertAllClose(jax.linearize(f, 2.0)[1](1.0), 2 * jnp.cos(2.0))

  def test_rules_derived_from_jvp_error_messages(self):
    class Sin(VJPHiPrimitive):
      def __init__(self, x_aval):
        self.in_avals = (x_aval,)
        self.out_aval = x_aval
        self.params = {}
        super().__init__()

      def expand(self, x):
        return jnp.sin(x)

      def jvp(self, primals, tangents):
        (x,), (x_dot,) = primals, tangents
        return self(x), jnp.cos(x) * x_dot

    def sin(x):
      return Sin(jax.typeof(x))(x)

    with self.assertRaisesRegex(NotImplementedError, 'vjp_from_jvp'):
      jax.grad(sin)(2.0)
    with self.assertRaisesRegex(NotImplementedError, 'linearize_from_jvp'):
      jax.linearize(sin, 2.0)

  @jtu.with_explicit_mesh((2, 2), ('i', 'j'))
  def test_grad_remat_hitype(self, mesh):
    x = jnp.ones(4)
    y = jnp.ones(2)

    @jax.remat
    def f(x, y):
      tup = make_tup(x, y)
      x_ = get_tuple_element(tup, 0)
      y_ = get_tuple_element(tup, 1)
      return jnp.sum(x_ + jnp.concatenate((y_, y_)))

    f(x, y)
    jax.jit(jax.grad(f))(x, y)

  @jtu.with_explicit_mesh((2,), 'x')
  def test_shmap_grad_hitype(self, mesh):
    class Mul(VJPHiPrimitive):
      def __init__(self, aval):
        self.in_avals = (aval, aval)
        self.out_aval = aval
        self.params = {}
        super().__init__()

      def expand(self, x, y):
        return MulH(x.val * y.val)

      def vjp_fwd(self, nzs_in, x, y):
        return my_mul(x, y), (x, y)

      def vjp_bwd_retval(self, res, g):
        x, y = res
        return (my_mul(g, y), my_mul(g, x))

    @dataclass
    class MulH:
      val: Any

    @dataclass(frozen=True)
    class MulTy(HiType):
      ty: Ty

      def __repr__(self):
        return f"MulTy({self.ty})"

      def __hash__(self):
        return hash((self.ty,))

      def __eq__(self, other):
        if not isinstance(other, MulTy):
          return False
        return self.ty == other.ty

      def lo_ty(self):
        return [self.ty]

      def lower_val(self, hi_val: MulH):
        return [hi_val.val]

      def raise_val(self, lo_val):
        return MulH(lo_val)

      def to_tangent_aval(self) -> HiType:
        return MulTy(self.ty.to_tangent_aval())

      def vspace_zero(self):
        return MulHZero(self)()

      def to_ct_aval(self) -> HiType:
        return MulTy(self.ty.to_ct_aval())

      def shard(self, mesh, manual_axes, check_vma, spec):
        return MulTy(self.ty.shard(mesh, manual_axes, check_vma, spec.val))

      def unshard(self, mesh, check_vma, spec):
        return MulTy(self.ty.unshard(mesh, check_vma, spec.val))

    register_hitype(MulH, lambda m: MulTy(jax.typeof(m.val)))

    class MulHZero(VJPHiPrimitive):
      def __init__(self, mul_ty):
        self.in_avals = ()
        self.out_aval = mul_ty
        self.params = {}
        super().__init__()

      def expand(self):
        return MulH(ad.zeros_like_aval(self.out_aval.ty))

    @dataclass(frozen=True)
    class MulSpec(HiPspec):
      val: Any

      def to_lo(self):
        return [self.val]

      def to_tangent_spec(self):
        return MulSpec(self.val)

      def to_ct_spec(self):
        return MulSpec(self.val)

      def __repr__(self):
        return f"MulSpec({self.val})"

    def my_mul(x, y):
      return Mul(jax.typeof(x))(x, y)

    arr1 = jax.device_put(jnp.arange(8, dtype=jnp.float32), jax.P('x'))
    arr2 = jax.device_put(jnp.arange(8, dtype=jnp.float32), jax.P('x'))

    @jax.jit
    @jax.shard_map(in_specs=(MulSpec(jax.P('x')), MulSpec(jax.P('x'))),
                   out_specs=MulSpec(jax.P('x')))
    def f(x, y):
      return my_mul(x, y)

    _, f_vjp = jax.vjp(f, MulH(arr1), MulH(arr2))
    x = jax.device_put(jnp.ones((8,), dtype=jnp.float32), jax.P('x'))
    f_vjp(MulH(x))  # doesn't crash

  @parameterized.parameters([False, True])
  def test_ref_prim(self, jit):
    class Square(VJPHiPrimitive):
      def __init__(self, ref_aval):
        self.in_avals = (ref_aval,)
        self.out_aval = None
        self.params = {}
        self.effects = {state.WriteEffect(0)}
        super().__init__()

      def expand(self, ref):
        ref[...] = ref[...] ** 2

    x_ref = jax.new_ref(2.)

    def f(_, x_ref):
      Square(typeof(x_ref))(x_ref)

    if jit:
      f = jax.jit(f)

    f(0, x_ref)
    self.assertAllClose(x_ref[...], 4., check_dtypes=False)
    traced_jaxpr = jax.jit(f).trace(0, x_ref).jaxpr
    self.assertEqual(traced_jaxpr.effects,
                     {state.WriteEffect(traced_jaxpr.invars[1])})

  def test_lower_preserves_arg_names_for_shaped_arrays(self):
    x = jnp.array(1.0)
    lowered = jax.jit(square).lower(x)
    debug_info = lowered._lowering.compile_args['all_args_info'].debug_info
    self.assertEqual(debug_info.arg_names, ('x',))

  def test_lower_replicates_arg_names_for_hitypes(self):
    def f(x):
      return from_qarray(x)

    q = to_qarray(jnp.ones((2, 2), 'float32'))
    lowered = jax.jit(f).lower(q)
    debug_info = lowered._lowering.compile_args['all_args_info'].debug_info
    # QArrayTy.lo_ty() returns [int8[m,k], f32[m]], so 'x' is replicated twice
    self.assertEqual(debug_info.arg_names, ('x', 'x'))

  def test_nondiff_linearize(self):
    def f(x):
      return NonDiffPrim(jax.typeof(x))(x)
    _, f_lin = jax.linearize(f, jnp.ones((5,)))
    out_tangent = f_lin(jnp.ones((5,)))
    self.assertArraysEqual(out_tangent, jnp.zeros((5,)))


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

class HijaxTransformCoverageTest(jtu.JaxTestCase):
  # ------------
  # grad
  # ------------
  # with differentiable hijax arguments
  def test_hitypes_as_grad_args(self):
    box = immutbox_new((jnp.array(2.0), jnp.array(3.0)))

    def loss_fn(tup):
      x = immutbox_get(tup)[0]
      return x ** 2

    grads = jax.grad(loss_fn)(box)
    self.assertAllClose(immutbox_get(grads)[0], 4.0)

  # with non-differentiable hijax arguments
  def test_hitypes_as_nondiff_grad_args(self):
    box = immutbox_new((jnp.array(2.0), jnp.array(3.0)))
    x = jnp.array(3.0)

    def loss_fn(x, box):
      y = immutbox_get(box)[1]
      return x ** 2 + y

    grad = jax.grad(loss_fn)(x, box)
    self.assertAllClose(grad, 6.0, check_dtypes=False)

  # with hijax captured arguments
  def test_hitypes_as_captured_args(self):
    box = immutbox_new((jnp.array(2.0), jnp.array(3.0)))

    def loss_fn(x):
      y = immutbox_get(box)[1]
      return x ** 2 + y

    grad = jax.grad(loss_fn)(jnp.array(4.0))
    self.assertAllClose(grad, 8.0, check_dtypes=False)

  #------------
  # scan
  #------------
  # with hijax carry arguments
  def test_hitypes_as_scan_carry(self):
    box = immutbox_new((jnp.array(1.0), jnp.array(2.0)))

    def body(box, _):
      x, y = immutbox_get(box)
      return immutbox_new((x + 1.0, y + 2.0)), None

    box, _ = jax.lax.scan(body, box, None, length=5)
    x, y = immutbox_get(box)
    self.assertAllClose(x, 6.0, check_dtypes=False)
    self.assertAllClose(y, 12.0, check_dtypes=False)

  # with hijax captured arguments
  def test_hitypes_as_scan_captured(self):
    box = immutbox_new((jnp.array(3.0), jnp.array(4.0)))
    carry0 = jnp.array(1.0)
    xs = jnp.arange(5, dtype=jnp.float32)

    def body(carry, x):
      a, b = immutbox_get(box)
      carry = a * carry + b
      y = a * x + b
      return carry, immutbox_new(y)

    carry, ys_box = jax.lax.scan(body, carry0, xs)
    ys = immutbox_get(ys_box)
    self.assertAllClose(carry, 727.0, check_dtypes=False)
    self.assertAllClose(ys, 3.0 * xs + 4.0, check_dtypes=False)

  def test_grad_custom_vjp_optimize_remat_with_hijax(self):

    @jax.custom_vjp
    def f(x):
      return square(x)

    def f_fwd(x):
      y = square(x)
      return y, x  # (primal_out, residuals)

    def f_bwd(res, g):
      x = res
      return (g * 2.0 * x,)

    f.defvjp(f_fwd, f_bwd, optimize_remat=True)

    x = jnp.float32(3.0)
    result = jax.jit(jax.grad(f))(x)
    self.assertAllClose(result, jnp.float32(6.0))

  def test_custom_vjp_inlined_when_lower(self):

    @jax.custom_vjp
    def foo(x):
      return square(x)
    def foo_fwd(x):
      return foo(x), x
    def foo_bwd(res, g):
      return (g * 2.0 * res,)
    foo.defvjp(foo_fwd, foo_bwd)

    jaxpr = jax.jit(foo).trace(jnp.float32(0.0)).lojax.jaxpr

    has_custom_vjp = any(
        eqn.primitive is custom_vjp_call_p for eqn in jaxpr.eqns)
    self.assertFalse(has_custom_vjp,
        "custom_vjp_call_p should be inlined when lower=True")

  def test_custom_jvp_inlined_when_lower(self):

    @jax.custom_jvp
    def foo(x):
      return square(x)
    @foo.defjvp
    def foo_jvp(primals, tangents):
      x, = primals
      t, = tangents
      return square(x), t * 2.0 * x

    jaxpr = jax.jit(foo).trace(jnp.float32(0.0)).lojax.jaxpr

    has_custom_jvp = any(
        eqn.primitive is custom_jvp_call_p for eqn in jaxpr.eqns)
    self.assertFalse(has_custom_jvp,
        "custom_jvp_call_p should be inlined when lower=True")

  def test_custom_vjp_with_hiprimitive_is_high(self):
    @jax.custom_vjp
    def foo(x):
      return square(x)
    def foo_fwd(x):
      y = foo(x)
      return y, x
    def foo_bwd(res, g):
      return (g * 2.0 * res,)
    foo.defvjp(foo_fwd, foo_bwd)

    jaxpr = jax.make_jaxpr(foo)(jnp.float32(2.0))
    # The call_jaxpr should contain hi-primitives (square)
    self.assertTrue(jaxpr.is_high)

  def test_custom_vjp_with_hiprimitive_lowered(self):

    @jax.custom_vjp
    def foo(x):
      return square(x)
    def foo_fwd(x):
      y = foo(x)
      return y, x
    def foo_bwd(res, g):
      return (g * 2.0 * res,)
    foo.defvjp(foo_fwd, foo_bwd)

    jaxpr = jax.jit(foo).trace(jnp.float32(0.0)).lojax.jaxpr

    # custom_vjp_call_p should be inlined (no custom_vjp_call eqns remain)
    has_custom_vjp = any(
        eqn.primitive is custom_vjp_call_p for eqn in jaxpr.eqns)
    self.assertFalse(has_custom_vjp,
        "custom_vjp_call_p with hi-primitives should be inlined when "
        "lower=True")
    # The hi-primitive (square) should also be lowered
    self.assertFalse(jaxpr.is_high,
        "Lowered jaxpr should not contain hi-primitives")

  def test_custom_jvp_with_hiprimitive_lowered(self):

    @jax.custom_jvp
    def foo(x):
      return square(x)
    @foo.defjvp
    def foo_jvp(primals, tangents):
      x, = primals
      t, = tangents
      return square(x), t * 2.0 * x

    jaxpr = jax.jit(foo).trace(jnp.float32(0.0)).lojax.jaxpr

    has_custom_jvp = any(
        eqn.primitive is custom_jvp_call_p for eqn in jaxpr.eqns)
    self.assertFalse(has_custom_jvp,
        "custom_jvp_call_p with hi-primitives should be inlined when "
        "lower=True")
    self.assertFalse(jaxpr.is_high,
        "Lowered jaxpr should not contain hi-primitives")

  def test_dce_sink_basic(self):
    x = jnp.array(1.0)
    def f(x):
      y = x + 1.0
      jax.lax.dce_sink(y)
      return x
    self.assertEqual(f(x), 1.0)
    self.assertEqual(jax.jit(f)(x), 1.0)

  def test_dce_sink_autodiff(self):
    def f(x):
      y = x * 2.0
      jax.lax.dce_sink(y)
      return y * 3.0
    self.assertEqual(jax.jvp(f, (2.0,), (1.0,)), (12.0, 6.0))
    self.assertEqual(jax.grad(f)(2.0), 6.0)
    _, f_lin = jax.linearize(f, 2.0)
    self.assertEqual(f_lin(1.0), 6.0)

  def test_dce_sink_vmap(self):
    def f(x):
      jax.lax.dce_sink(x, prevent_mlir_dce=True)
      return x * 2.0
    out = jax.vmap(f)(jnp.arange(4.0))
    self.assertArraysAllClose(out, jnp.arange(4.0) * 2.0)

  @jtu.with_explicit_mesh((2,), ('x',))
  def test_dce_sink_under_explicit_mesh(self, mesh):
    x = jax.device_put(jnp.arange(10, dtype=jnp.float32), jax.P('x'))
    def f(x):
      y = x + 1.0
      jax.lax.dce_sink(y, prevent_mlir_dce=True)
      return x
    hlo = jax.jit(f).lower(x).compile().as_text()
    self.assertIn("dce_sink", hlo)
    self.assertIn("custom_call", hlo)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
