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

import dataclasses
import itertools

from absl.testing import absltest
import numpy as np

import jax
from jax import numpy as jnp

from jax._src import config
from jax._src import core
from jax._src import hijax
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.repro import repro_test_util as rtu

from jax._src import test_util as jtu


config.parse_flags_with_absl()
jtu.request_cpu_devices(8)

## QARRAY

# hivalues do not need to be hashable, but hitypes do.
@dataclasses.dataclass(frozen=True)  # not NamedTuple, which is a pytree
class QArray:
  qvalue: jax.Array  # dtype = int8
  scale: jax.Array  # rank 0, dtype = float32


# not necessary to be a dataclass, but good test because it bypasses the
# HiType constructor!! But has to be hashable.
@dataclasses.dataclass(frozen=True)
class QArrayTy(hijax.HiType):
  # Use as abstract value for hi values. In this example, the out_avals for
  # Q and in_avals for DQ.
  shape: tuple[int, int]

  def _qarray_aval(self):
    return core.ShapedArray(self.shape, jnp.dtype('int8'))

  def lo_ty(self):
    return [self._qarray_aval(),
            core.ShapedArray((), jnp.dtype('float32'))]

  def lower_val(self, q: QArray):
    return [q.qvalue, q.scale]

  def raise_val(self, qvalue: jax.Array, scale: jax.Array):
    return QArray(qvalue, scale)

  def to_tangent_aval(self):
    return core.ShapedArray(self.shape, jnp.dtype('float32'))

  def dec_rank(self, size: int, axis_spec: QArraySpec):
    a = self._qarray_aval().dec_rank(size, axis_spec.axis)
    return QArrayTy(a.shape)

  def inc_rank(self, size: int, axis_spec: QArraySpec):
    a = self._qarray_aval().inc_rank(size, axis_spec.axis)
    return QArrayTy(a.shape)


hijax.register_hitype(QArray, lambda q: QArrayTy(q.qvalue.shape))


@dataclasses.dataclass(frozen=True)
class QArraySpec(hijax.MappingSpec):
  axis: int


class Q(hijax.VJPHiPrimitive):
  def __init__(self, unquantized_aval: core.ShapedArray):
    if unquantized_aval.dtype != jnp.dtype('float32'): raise TypeError
    quantized_aval = QArrayTy(unquantized_aval.shape)
    self.in_avals = (unquantized_aval,)
    self.out_aval = quantized_aval
    self.params = {}
    super().__init__()

  def expand(self, x: jax.Array) -> QArray:
    scale = jnp.max(jnp.abs(x)) / 127
    qvalue = jnp.round(x / scale).astype(jnp.int8)
    return QArray(qvalue, scale)

  def vjp_fwd(self, nzs_in: tuple[bool, ...], x: jax.Array) -> tuple[QArray, None]:
    return self(x), None

  def vjp_bwd_retval(self, _: None, g: jax.Array) -> tuple[jax.Array,]:
    return g,

  def jvp(self,
          primals: tuple[jax.Array, ...],
          tangents: tuple[jax.Array]) -> tuple[QArray, QArray]:
    x, = primals
    x_tan, = tangents
    return self(x), self(x_tan)

  def batch(self, _axis_data: batching.AxisData,
            args, in_dims: tuple[int, ...]):
    (x,), (d,) = args, in_dims
    return make_qarray(x), QArraySpec(axis=d)


def make_qarray(x: jax.Array) -> QArray:
  return Q(jax.typeof(x))(x)


class DQ(hijax.VJPHiPrimitive):
  def __init__(self, quantized_aval: QArrayTy):
    unquantized_aval = core.ShapedArray(quantized_aval.shape, jnp.dtype('float32'))
    self.in_avals = (quantized_aval,)
    self.out_aval = unquantized_aval
    self.params = {}
    super().__init__()

  def expand(self, qx: QArray) -> jax.Array:
    return qx.qvalue.astype(jnp.float32) * qx.scale

  def vjp_fwd(self, nzs_in: tuple[bool, ...], qx: QArray) -> tuple[jax.Array, None]:
    return self(qx), None

  def vjp_bwd_retval(self, _: None, g: jax.Array) -> tuple[jax.Array,]:
    return g,

  def jvp(self, primals: tuple[QArray, ...],
          tangents: tuple[QArray, ...]
          ) -> tuple[jax.Array, jax.Array]:
    x, = primals
    x_tan, = tangents
    x_dq = self(x)
    x_tan_dq = self(x_tan)
    return x_dq, x_tan_dq

  def batch(self, _axis_data: batching.AxisData,
            args, in_dims: tuple[QArraySpec, ...]):
    (x,), (d,) = args, in_dims
    return make_dqarray(x), d.axis


def make_dqarray(qx: QArray) -> jax.Array:
  return DQ(jax.typeof(qx))(qx)


## TUPLES
@dataclasses.dataclass
class HiTup:  # a HiValue whose type is TupTy
  elts: tuple
  def __repr__(self):
    return 'HiTup{' + ','.join(map(repr, self.elts)) + '}'

Ty = core.AbstractValue

@dataclasses.dataclass(frozen=True)
class TupTy(hijax.HiType):
  tys: tuple[Ty, ...]

  def __repr__(self):
    return 'TupTy{' + ','.join(a.str_short() for a in self.tys) + '}'

  def __hash__(self):
    return hash(self.tys)

  def __eq__(self, other):
    return isinstance(other, type(self)) and self.tys == other.tys

  def lo_ty(self):
    return list(self.tys)

  def lower_val(self, hi_val: HiTup):
    return [lo for ty, elt in zip(self.tys, hi_val.elts)
            for lo in ty.lower_val(elt)]

  def raise_val(self, *elts_flat):
    elts_iter = iter(elts_flat)
    return HiTup(tuple(ty.raise_val(*itertools.islice(elts_iter, len(ty.lo_ty())))
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

  def vspace_add(self, x_tup: "TupTy", y_tup: "TupTy") -> jax.Array:
    n = len(self.tys)
    x_elts = [get_tuple_element(x_tup, i) for i in range(n)]
    y_elts = [get_tuple_element(y_tup, i) for i in range(n)]
    return make_tup(*(ty.vspace_add(x, y)
                      for ty, x, y in zip(self.tys, x_elts, y_elts)))

hijax.register_hitype(HiTup, lambda t: TupTy(tuple(map(jax.typeof, t.elts))))

@dataclasses.dataclass(frozen=True)
class TupSpec(hijax.MappingSpec):
  val: tuple

@dataclasses.dataclass(frozen=True)
class TupP(hijax.HiPspec):
  val: tuple

  def to_lo(self) -> tuple[jax.PartitionSpec, ...]:
    return self.val

class MakeTup(hijax.VJPHiPrimitive):
  from jax._src.interpreters import ad  # type: ignore

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

  def transpose(self, ct, *maybe_accums):
    cts = [get_tuple_element(ct, i)
            for i in range(len(self.out_aval.tys))]
    for ct_, accum in zip(cts, maybe_accums):
      if isinstance(accum, ad.GradAccum):
        accum.accum(ct_)

  def vjp_fwd(self, nz_in: tuple[bool, ...], *args: jax.Array) -> tuple["MakeTup", None]:
    return make_tup(*args), None

  def vjp_bwd_retval(self, _res, g: HiTup) -> tuple[jax.Array, ...]:
    return g.elts

  def batch(self, _axis_data, args, in_dims):
    return make_tup(*args), TupSpec(in_dims)

class GetTupElt(hijax.VJPHiPrimitive):
  def __init__(self, in_aval: "TupTy", idx: int):
    self.in_avals = in_aval,
    self.out_aval = in_aval.tys[idx]
    self.params = dict(idx=idx)
    super().__init__()

  def expand(self, tup: HiTup):
    return tup.elts[self.idx]

  def jvp(self, primals, tangents):
    (tup,), (tup_dot,) = primals, tangents
    return (get_tuple_element(tup, self.idx),
            get_tuple_element(tup_dot, self.idx))

  def transpose(self, g, tup_accum):
    tup_ty, = self.in_avals
    elts = map(ad.zeros_like_aval, tup_ty.tys)
    elts[self.idx] = g
    tup_accum.accum(make_tup(*elts))

  def vjp_fwd(self, nz_in: tuple[bool, ], tup):
    return get_tuple_element(tup, self.idx), None

  def vjp_bwd_retval(self, _res, g):
    tup_ty, = self.in_avals
    elts = map(ad.zeros_like_aval, tup_ty.tys)
    elts[self.idx] = g
    return make_tup(*elts),

  def batch(self, _axis_data, args, in_dims):
    (x,), (d,) = args, in_dims
    return get_tuple_element(x, self.idx), d.val[self.idx]

def make_tup(*elts: jax.Array) -> HiTup:
  return MakeTup(map(jax.typeof, elts))(*elts)

def get_tuple_element(tup: HiTup, idx: int) -> jax.Array:
  return GetTupElt(jax.typeof(tup), idx)(tup)


@jtu.with_config(jax_traceback_filtering="off",
                 jax_enable_checks=True)
class ReproHiTest(rtu.ReproTestBase):

  def test_qarray_basic(self):  # adapted from hijax_test.py
    @jax.jit
    def f(x: jax.Array) -> jax.Array:
      xq: QArray = make_qarray(x)
      xd: jax.Array = make_dqarray(xq)
      return jnp.sum(xd)

    x = np.arange(4., dtype=np.float32)
    self.collect_and_check(f, x)

  def test_qarray_boundary(self):
    """QArray crosses the jit boundary as input and output."""
    @jax.jit
    def f(qx: QArray) -> QArray:
      xd: jax.Array = make_dqarray(qx)
      return make_qarray(42. * xd)

    x = np.arange(4., dtype=np.float32)
    qx = make_qarray(x)
    qy = f(qx)
    y = make_dqarray(qy)
    expected = make_dqarray(make_qarray(42. * x))
    self.assertAllClose(y, expected)
    self.collect_and_check(lambda qx: make_dqarray(f(qx)), qx)

  def test_qarray_grad(self):  # adapted from hijax_test.py
    #self.skipTest("TODO: hijax")
    @jax.jit
    def h(x: QArray):
      @jax.jit
      def g(x: QArray):
        @jax.jit
        def f(x: QArray) -> QArray:
          return make_qarray(42. * make_dqarray(x))
        return jnp.sum(make_dqarray(x))
      return jax.grad(g)(x)

    x = np.arange(3., dtype=np.float32)
    def m(x: jax.Array) -> jax.Array:
      xq = make_qarray(x)
      return h(xq)

    m(x)
    self.collect_and_check(m, x)

  def test_qarray_jvp(self):  # adapted from hijax_test.py
    #self.skipTest("TODO: hijax")
    @jax.jit
    def f(x: QArray) -> QArray:
      xd = make_dqarray(x)
      return make_qarray(42. * xd)

    x = np.arange(3., dtype=np.float32)
    def g(x: jax.Array):
      xq = make_qarray(x)
      tan = make_qarray(jnp.full_like(x, 0.1))
      v = jax.jvp(f, (xq,), (tan,))
      return v[1]

    _ = g(x)
    # del y
    # g = jax.grad(f)(x)
    # del g
    # t = jax.jit(g).trace(x)
    # print("Hi Jaxpr: ", t.jaxpr)
    # print("Lo Jaxpr:", t.lojax.jaxpr)
    self.collect_and_check(g, x)

  def test_qarray_vmap_basic(self):  # adapted from hijax_test.py
    #@jax.jit
    def f(x: QArray) -> QArray:
      x_dq = make_dqarray(x)
      return make_qarray(42. * x_dq)

    x = np.arange(24, dtype=np.float32).reshape((2, 3, 4))
    def g(x: jax.Array):
      yqs = jax.vmap(f, axis_size=3,
                     in_axes=QArraySpec(1),
                     out_axes=QArraySpec(1))(make_qarray(x))
      return make_dqarray(yqs)

    #t = jax.jit(g).trace(x)
    #print("Hi Jaxpr: ", t.jaxpr)
    # print("Lo Jaxpr:", t.lojax.jaxpr)
    self.collect_and_check(g, x)

  def test_qarray_vmap_nested(self):  # adapted from hijax_test.py
    #@jax.jit
    def f(x: QArray) -> QArray:
      x_dq = make_dqarray(x)
      return make_qarray(42. * x_dq)

    x = np.arange(120, dtype=np.float32).reshape((2, 3, 4, 5))
    def g(x: jax.Array):
      f_map_1 = jax.vmap(f, in_axes=QArraySpec(2), axis_size=5,
                         out_axes=QArraySpec(2))
      f_map_2 = jax.vmap(f_map_1, in_axes=QArraySpec(1), axis_size=3,
                         out_axes=QArraySpec(1))
      yqs = f_map_2(make_qarray(x))
      return make_dqarray(yqs)

    _ = g(x)
    t = jax.jit(g).trace(x)
    print("Hi Jaxpr: ", t.jaxpr)
    #print("Lo Jaxpr:", t.lojax.jaxpr)
    self.collect_and_check(g, x)

  @jtu.parameterized_filterable(
    kwargs=[dict(with_jit=with_jit)
                for with_jit in [False, True]
    ])
  def test_tuple_basic(self, with_jit: bool):
    rtu.maybe_skip_known_failure("TODO: hijax")
    def f():
      tup = make_tup(1, 2)
      return get_tuple_element(tup, 1)

    if with_jit:
      f = jax.jit(f)

    self.collect_and_check(f)

  def test_tuple_vmap(self):
    tup = make_tup(jnp.arange(3.), jnp.arange(3.))
    def f(tup):
      return jax.vmap(lambda x: x, in_axes=TupSpec((0, 0)),
                      out_axes=TupSpec((0, 0)), axis_size=3)(tup)
    f(tup)

  def test_tuple_vmap_infer(self):
    tup = make_tup(jnp.arange(3.), jnp.arange(3.))
    jax.vmap(lambda _: make_tup(jnp.ones(3), jnp.ones(3)),
             in_axes=TupSpec((0, 0)),
             out_axes=batching.infer, axis_size=3)(tup)

  # def test_tuple_vmap_match(self):
  #   tup = make_tup(jnp.arange(3.), jnp.arange(3.))
  #   jax.vmap(lambda _: make_tup(jnp.ones(3), jnp.ones(3)),
  #            in_axes=TupSpec((0, 0)), out_axes=TupSpec((0, 0)), axis_size=3)(tup)

  def test_tuple_vmap_primitive(self):
    tup = make_tup(jnp.arange(3.), 5.)
    def f(tup):
      a, b = get_tuple_element(tup, 0), get_tuple_element(tup, 1)
      return make_tup(b, a)
    jax.vmap(f, in_axes=TupSpec((0, None)),
             out_axes=TupSpec((None, 0)), axis_size=3)(tup)

  @jtu.parameterized_filterable(
    kwargs=[dict(with_jit=with_jit)
                for with_jit in [False, True]
    ])
  def test_tuple_scan(self, with_jit):
    rtu.maybe_skip_known_failure("TODO: hijax")
    tup = make_tup(jnp.arange(3.), jnp.arange(3. * 4).reshape(3, 4))
    def body(_, x):
      self.assertEqual(jax.typeof(x), TupTy((jax.typeof(jnp.zeros(())), jax.typeof(jnp.arange(4.)))))
      a = get_tuple_element(x, 0)
      b = get_tuple_element(x, 1)
      return (), make_tup(a + 1, b * 2)
    def f():
      return jax.lax.scan(body, (), tup, length=3)
    if with_jit:
      f = jax.jit(f)
    (), tup2 = f()
    a = get_tuple_element(tup2, 0)
    b = get_tuple_element(tup2, 1)
    self.assertAllClose(a, jnp.arange(3.) + 1)
    self.assertAllClose(b, jnp.arange(3. * 4).reshape(3, 4) * 2)

    self.collect_and_check(f)

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

  @jtu.parameterized_filterable(
    kwargs=[dict(with_jit=with_jit)
                for with_jit in [False, True]
    ])
  def test_tuple_vjp(self, with_jit: bool):
    rtu.maybe_skip_known_failure("TODO: hijax")
    def f(t):
      e0 = get_tuple_element(t, 0)
      e1 = get_tuple_element(t, 1)
      return make_tup(5. * e0, e0 + 3. * e1)
    if with_jit:
      f = jax.jit(f)

    def g():
      out, f_vjp = jax.vjp(f, make_tup(2., 3.))
      ct, = f_vjp(make_tup(.1, .2))
      return ct

    ct = g()
    self.assertEqual(.7, get_tuple_element(ct, 0))
    self.assertEqual(.6, get_tuple_element(ct, 1))

    self.collect_and_check(g)

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

  @jtu.parameterized_filterable(
    kwargs=[dict(jit=jit)
                for jit in [False, True]
    ])
  def test_tuple_ref_to_tuple(self, jit):
    def f():
      tup = make_tup(1, 2)
      ref = jax.new_ref(tup)
      tup_ = ref[...]
      return get_tuple_element(tup_, 1)

    if jit:
      f = jax.jit(f)

    self.assertEqual(f(), 2)

  @jtu.parameterized_filterable(
    kwargs=[dict(jit=jit)
                for jit in [False, True]
    ])
  def test_tuple_run_state(self, jit):
    from jax.experimental import pallas as pl
    def f():
      @pl.run_state
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


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
