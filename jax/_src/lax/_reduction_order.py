# Copyright 2026 The JAX Authors.
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

from collections.abc import Sequence
from typing import Any

import numpy as np

from jax._src import core, dtypes, pjit, reduction_order as ro, util
from jax._src.lax import lax
from jax._src.partition_spec import PartitionSpec as P
from jax._src.typing import Array, DTypeLike


### Hijax lowering & batching entry points

def reduce_op(prim: core.Primitive, x: Any, axes: Sequence[int], out_s: Any,
              order: ro.Order) -> Any:
  if len(axes) != 1:
    raise ValueError(
        f"{prim.name} with composite reduction_order requires 1 axis,"
        f" got {axes}.")
  out_aval = prim.abstract_eval(
      core.typeof(x), axes=tuple(axes), out_sharding=out_s)[0]
  fn = lax.reduce_max if prim is lax.reduce_max_p else lax.reduce_sum
  res = _eval_reduce(fn, prim.name, x, axes[0], out_s, order)
  return _match_out_aval(res, out_aval)


def dot_general(lhs: Any, rhs: Any, dnums: Any, precision: Any, pref_dt: Any,
                out_s: Any, order: ro.Order) -> Any:
  (lc, rc), (lb, rb) = dnums
  if len(lc) != 1 or len(rc) != 1:
    raise ValueError(
        "dot with composite reduction_order requires 1 contracting axis.")
  out_aval = lax.dot_general_p.abstract_eval(
      core.typeof(lhs), core.typeof(rhs), dimension_numbers=dnums,
      precision=precision, preferred_element_type=pref_dt,
      out_sharding=out_s)[0]
  res = _eval_dot(
      lhs, rhs, lc[0], rc[0], lb, rb, precision, pref_dt, out_s, order)
  return _match_out_aval(res, out_aval)


def shift_order_for_vmap(
    order: ro.Order, out_bdim: int, old_out_ndim: int, shift_inner: bool = True
) -> ro.Order:
  rec = lambda o, b=out_bdim, n=old_out_ndim: shift_order_for_vmap(
      o, b, n, shift_inner)
  match order:
    case ro.Ring(d, ax) if ax is not None:
      oax = util.canonicalize_axis(ax, old_out_ndim)
      return ro.Ring(d, oax + int(out_bdim <= oax))
    case ro.Cast(dt, body):
      return ro.Cast(dt, rec(body))
    case ro.Pad(mult, body, pad):
      return ro.Pad(mult, rec(body), pad)
    case ro.Tile(t, inner, outer, sp, pad):
      in_ord = rec(inner, out_bdim + int(shift_inner), old_out_ndim + 1)
      return ro.Tile(t, in_ord, rec(outer), sp, pad)
    case _:
      return order


### Reduction-order interpreter

def _eval_reduce(
    reduce_fn, op_name: str, x: Array, ax: int, out_s: Any, order: ro.OrderLike
) -> Array:
  match order:
    case ro.Cast(dt, body):
      res = _eval_reduce(reduce_fn, op_name, x, ax, out_s, body)
      return lax.convert_element_type(res, dt)
    case ro.Pad(_, body):
      x = _pad_axis(x, ax, order, op_name)
      return _eval_reduce(reduce_fn, op_name, x, ax, out_s, body)
    case ro.Tile(tile, inner, outer, split):
      x = _tile(_pad_axis(x, ax, order, op_name), ax, tile)
      if split is ro.Split.block:
        partials = _eval_reduce(reduce_fn, op_name, x, ax + 1, None, inner)
      else:
        inner_s = _insert_none_spec(out_s, ax)
        partials = _eval_reduce(reduce_fn, op_name, x, ax, inner_s, inner)
      return _eval_reduce(reduce_fn, op_name, partials, ax, out_s, outer)
    case ro.Ring():
      if op_name == "reduce_max":
        raise ValueError(
            "Ring reduction order is not supported for reduce_max.")
      return _eval_ring(x, ax, out_s, order)
    case ro.ReductionOrder.unreduced:
      _check_unreduced_step(core.typeof(x), ax, out_s, op_name)
      return reduce_fn(x, (ax,), out_sharding=out_s)
    case _:
      return reduce_fn(x, (ax,), out_sharding=out_s, reduction_order=order)


def _eval_dot(
    a: Array, b: Array, ca: int, cb: int, lb: tuple[int, ...],
    rb: tuple[int, ...], precision: Any, pref_dt: Any, out_s: Any,
    order: ro.OrderLike,
) -> Array:
  match order:
    case ro.Cast(dt, body):
      res = _eval_dot(a, b, ca, cb, lb, rb, precision, pref_dt, out_s, body)
      return lax.convert_element_type(res, dt)
    case ro.Pad(_, body):
      a, b = _pad_dot_operands(a, b, ca, cb, order)
      return _eval_dot(a, b, ca, cb, lb, rb, precision, pref_dt, out_s, body)
    case ro.Tile(bk, inner, outer, split):
      a, b = _pad_dot_operands(a, b, ca, cb, order)
      strided = split is ro.Split.strided
      ps, eff_out_s = _dot_partials_sharding(
          core.typeof(a), core.typeof(b), ((ca, cb), (lb, rb)), out_s,
          strided)
      a, b = _tile(a, ca, bk), _tile(b, cb, bk)
      slb = tuple(d + int(d > ca) for d in lb)
      srb = tuple(d + int(d > cb) for d in rb)
      ca_in, cb_in = (ca, cb) if strided else (ca + 1, cb + 1)
      lb_in = (ca + 1, *slb) if strided else (ca, *slb)
      rb_in = (cb + 1, *srb) if strided else (cb, *srb)
      partials = _eval_dot(
          a, b, ca_in, cb_in, lb_in, rb_in, precision, pref_dt, ps, inner)
      return _eval_reduce(
          lax.reduce_sum, "reduce_sum", partials, 0, eff_out_s, outer)
    case ro.Ring():
      tiled = ro.Tile(1, ro.ReductionOrder.linear, order, ro.Split.block, None)
      return _eval_dot(a, b, ca, cb, lb, rb, precision, pref_dt, out_s, tiled)
    case ro.ReductionOrder.unreduced:
      _check_unreduced_step(core.typeof(a), ca, out_s, "dot_general")
      _check_unreduced_step(core.typeof(b), cb, out_s, "dot_general")
      return lax.dot_general(
          a, b, (((ca,), (cb,)), (lb, rb)), precision=precision,
          preferred_element_type=pref_dt, out_sharding=out_s)
    case _:
      return lax.dot_general(
          a, b, (((ca,), (cb,)), (lb, rb)), precision=precision,
          preferred_element_type=pref_dt, out_sharding=out_s,
          reduction_order=order)


def _eval_ring(arr: Array, ax: int, out_s: Any, ring_order: ro.Ring) -> Array:
  from jax._src import shard_map  # pyrefly: ignore[missing-import]
  from jax._src.lax import control_flow, parallel as lax_parallel, slicing
  aval = core.typeof(arr)
  m_axis, oax = _check_ring_args(aval, ax, out_s, ring_order)
  direction, s, out_ndim = ring_order.direction, arr.shape[ax], arr.ndim - 1
  d_len = arr.shape[oax + int(oax >= ax)]
  if d_len % (2 * s if direction is ro.RingDir.bidir else s):
    raise ValueError(f"Ring split_axis size {d_len} not divisible over {s}.")
  chunk = d_len // s
  to_w, from_w = _wire_ops(arr.dtype)
  mesh, s32 = getattr(arr, "sharding", aval.sharding).mesh, np.int32

  def _run_dir(z: Array, r: Array, sign: int) -> Array:
    perm = [(i, (i + sign) % s) for i in range(s)]
    sl = lambda idx: slicing.dynamic_index_in_dim(
        z, idx, axis=oax, keepdims=False)
    idx_at = lambda h: lax.rem(
        r - s32(sign) * (lax.convert_element_type(h, s32) + s32(1))
        + s32(2 * s),
        s32(s),
    )
    body = lambda h, acc: lax.optimization_barrier(lax.add(
        from_w(lax_parallel.ppermute(to_w(acc), m_axis, perm)),
        lax.optimization_barrier(sl(idx_at(h))),
    ))
    return control_flow.fori_loop(s32(1), s32(s), body, sl(idx_at(s32(0))))

  def _scatter_ring(local_arr: Array) -> Array:
    v = lax.squeeze(local_arr, (ax,))
    vc = lax.reshape(v, (*v.shape[:oax], s, chunk, *v.shape[oax + 1:]))
    r = lax.convert_element_type(lax_parallel.axis_index(m_axis), s32)
    if direction in (ro.RingDir.asc, ro.RingDir.desc):
      return _run_dir(vc, r, 1 if direction is ro.RingDir.asc else -1)
    top = slicing.slice_in_dim(vc, 0, chunk // 2, axis=oax + 1)
    bot = slicing.slice_in_dim(vc, chunk // 2, chunk, axis=oax + 1)
    return lax.concatenate([_run_dir(top, r, 1), _run_dir(bot, r, -1)], oax)

  out_specs = P(*([None] * oax + [m_axis] + [None] * (out_ndim - 1 - oax)))
  out = shard_map.shard_map(
      _scatter_ring, mesh=mesh, in_specs=shard_map.Infer,
      out_specs=out_specs, axis_names=frozenset({m_axis}),
  )(arr)
  if out_s is None:
    parts = lax.tuple_delete(aval.sharding.spec.partitions, (ax,))
    out_s = aval.sharding.update(spec=P(*parts))
  if core.typeof(out).sharding != out_s:
    return pjit.reshard(out, type(out_s)(mesh, out_s.spec))
  return out


def _tile(x: Array, ax: int, tile: int) -> Array:
  return lax.reshape(
      x, (*x.shape[:ax], x.shape[ax] // tile, tile, *x.shape[ax + 1:]))


def _match_out_aval(res: Array, out_aval: core.ShapedArray) -> Array:
  av = core.typeof(res)
  if av.dtype == out_aval.dtype and av.weak_type == out_aval.weak_type:
    return res
  return lax._convert_element_type(res, out_aval.dtype, out_aval.weak_type)


def _pad_axis(
    arr: Array, ax: int, order: ro.Pad | ro.Tile, op_name: str,
    pad_val: Any = None,
) -> Array:
  div = order.tile if isinstance(order, ro.Tile) else order.multiple
  if div <= 0:
    raise ValueError(f"Expected positive tile/multiple, got {div}.")
  if rem := arr.shape[ax] % div:
    if order.pad is None:
      raise ValueError(f"Axis size {arr.shape[ax]} not divisible by {div}")
    if pad_val is None:
      pad_val = (
          lax._get_max_identity(arr.dtype)
          if op_name == "reduce_max"
          else _sum_pad_val(arr.dtype, order.pad)
      )
    pads = [(0, (div - rem) if i == ax else 0, 0) for i in range(arr.ndim)]
    arr = lax.pad(arr, pad_val, pads)
  return arr


def _pad_dot_operands(
    a: Array, b: Array, ca: int, cb: int, order: ro.Pad | ro.Tile
) -> tuple[Array, Array]:
  is_nonfloat_absent = (
      order.pad is ro.PadVal.absent
      and not dtypes.issubdtype(a.dtype, np.floating)
  )
  b_pv = None if is_nonfloat_absent else lax._get_sum_identity(b.dtype)
  return (
      _pad_axis(a, ca, order, "dot_general"),
      _pad_axis(b, cb, order, "dot_general", pad_val=b_pv),
  )


def _insert_none_spec(out_s: Any, ax: int) -> Any:
  if out_s is None:
    return None
  p = out_s.spec.partitions
  p_pad = p + (None,) * max(0, ax - len(p))
  return out_s.update(
      spec=out_s.spec.update(partitions=(*p_pad[:ax], None, *p_pad[ax:]))
  )


def _sum_pad_val(dtype: DTypeLike, fill: Any) -> np.ndarray:
  if fill is ro.PadVal.absent:
    if dtypes.issubdtype(dtype, np.floating):
      return np.array(-0.0, dtype=dtype)
    if dtypes.issubdtype(dtype, np.complexfloating):
      return np.array(complex(-0.0, -0.0), dtype=dtype)
  return lax._get_sum_identity(dtype)


def _wire_ops(dt: np.dtype):
  if dtypes.issubdtype(dt, np.floating):
    u_dt = lax._UINT_BY_BITS.get(dtypes.itemsize_bits(dt))
    if u_dt is not None:
      return (
          lambda z: lax.bitcast_convert_type(z, u_dt),
          lambda w: lax.bitcast_convert_type(lax.optimization_barrier(w), dt),
      )
  return (lambda z: z), lax.optimization_barrier


### Sharding checks

def _check_unreduced_step(
    aval: core.ShapedArray, ax: int, out_s: Any, op_name: str
) -> None:
  spec = aval.sharding.spec.partitions[ax]
  if (
      core.getu(aval, None)[0]
      or spec is None
      or out_s is None
      or frozenset(lax.flatten_spec_no_none((spec,))) != out_s.spec.unreduced
      or aval.sharding.shard_shape(aval.shape)[ax] != 1
  ):
    raise core.ShardingTypeError(
        f"{op_name} with reduction_order='unreduced' requires 1 element per"
        " Explicit shard on reduced axis matching out_sharding.unreduced;"
        f" got {aval.str_short()} -> {out_s}.")


def _check_ring_args(
    aval: core.ShapedArray, ax: int, out_s: Any, ring_order: ro.Ring
) -> tuple[Any, int]:
  if core.getu(aval, None)[0] or (out_s is not None and out_s.spec.unreduced):
    raise core.ShardingTypeError(
        f"Ring does not support unreduced; got {aval.str_short()}.")
  spec_ax = aval.sharding.spec.partitions[ax]
  if spec_ax is None or (isinstance(spec_ax, tuple) and len(spec_ax) != 1):
    raise core.ShardingTypeError(
        f"Ring requires a single Explicit mesh axis on axis {ax},"
        f" got {spec_ax!r}.")
  m_axis = spec_ax[0] if isinstance(spec_ax, tuple) else spec_ax
  if (k_local := aval.sharding.shard_shape(aval.shape)[ax]) != 1:
    raise core.ShardingTypeError(
        f"Ring requires 1 partial per shard, got {k_local}.")
  out_ndim, oax = aval.ndim - 1, ring_order.split_axis
  out_parts = out_s.spec.partitions[:out_ndim] if out_s is not None else ()
  matches = [
      i for i, p in enumerate(out_parts)
      if p is not None
      and (m_axis in p if isinstance(p, tuple) else p == m_axis)
  ]
  if oax is None and len(matches) == 1:
    oax = matches[0]
  if oax is None:
    raise ValueError(
        "Ring requires split_axis (or out_sharding sharding an output axis on"
        " the reduced mesh axis).")
  oax = util.canonicalize_axis(oax, out_ndim)
  if matches and matches != [oax]:
    raise core.ShardingTypeError(
        f"Ring split_axis={oax} conflicts with out_sharding={out_s} on mesh"
        f" axis {m_axis!r}.")
  return m_axis, oax


def _dot_partials_sharding(a_aval, b_aval, dnums, out_s: Any, strided: bool):
  (ca, cb), (lb, rb) = dnums
  sa = a_aval.sharding.spec.partitions[ca]
  sb = b_aval.sharding.spec.partitions[cb]
  if sa != sb and (
      (sa is not None and sb is not None)
      or (out_s is not None and out_s.spec.unreduced)
  ):
    raise core.ShardingTypeError(
        f"dot_general requires matching contracting sharding; got {sa!r} and"
        f" {sb!r}.")
  if out_s is None:
    if strided or (sa is None) == (sb is None):
      return None, None
    out_s = lax._dot_general_sharding_rule(
        a_aval, b_aval, dimension_numbers=(((ca,), (cb,)), (lb, rb)),
        precision=None, preferred_element_type=None, out_sharding=None)
  out_ndim = a_aval.ndim + b_aval.ndim - len(lb) - 2
  parts = out_s.spec.partitions
  p_pad = parts + (None,) * (out_ndim - len(parts))
  if strided:
    part_s = out_s.update(spec=out_s.spec.update(partitions=(None, *p_pad)))
    return part_s, out_s
  blk_spec = sa or sb
  blk_axes = set(lax.flatten_spec_no_none((blk_spec,)))
  spec = P(blk_spec, *(_strip_axes(q, blk_axes) for q in p_pad))
  return out_s.update(spec=spec), out_s


def _strip_axes(q: Any, axes: set[Any]) -> Any:
  if q is None or not axes:
    return q
  tup = q if isinstance(q, tuple) else (q,)
  qs = tuple(x for x in tup if x not in axes)
  return None if not qs else (qs[0] if len(qs) == 1 else qs)


### ragged_dot_general (kept separate from dot_general; no unreduced support)

def ragged_dot_general(
    lhs: Any, rhs: Any, gs: Any, *, ragged_dot_dimension_numbers: Any,
    out_sharding: Any, reduction_order: ro.Order, **kw: Any) -> Any:
  rd = ragged_dot_dimension_numbers
  if _has_unreduced(reduction_order):
    raise NotImplementedError(
        "ragged_dot_general does not support reduction_order='unreduced'.")
  (lc, rc), (lb, rb) = rd.dot_dimension_numbers
  if len(lc) != 1 or len(rc) != 1:
    raise ValueError(
        "ragged_dot with composite reduction_order requires 1 contracting"
        " axis.")
  out_aval = lax.ragged_dot_general_p.abstract_eval(
      core.typeof(lhs), core.typeof(rhs), core.typeof(gs),
      ragged_dot_dimension_numbers=rd, out_sharding=out_sharding, **kw)[0]
  res = _eval_ragged_dot(
      lhs, rhs, gs, lc[0], rc[0], tuple(lb), tuple(rb),
      tuple(rd.lhs_ragged_dimensions), tuple(rd.rhs_group_dimensions),
      out_sharding, reduction_order, kw)
  return _match_out_aval(res, out_aval)


def _eval_ragged_dot(
    a: Array, b: Array, gs: Array, ca: int, cb: int, lb: tuple[int, ...],
    rb: tuple[int, ...], lr: tuple[int, ...], rg: tuple[int, ...],
    out_s: Any, order: ro.OrderLike, kw: dict[str, Any],
) -> Array:
  rec = lambda a, b, o: _eval_ragged_dot(
      a, b, gs, ca, cb, lb, rb, lr, rg, out_s, o, kw)
  match order:
    case ro.Cast(dt, body):
      return lax.convert_element_type(rec(a, b, body), dt)
    case _ if ca in lr and not ro.is_leaf_order(order):
      raise ValueError("ragged_dot with composite reduction_order cannot"
                       " reduce the ragged contracting axis.")
    case ro.Pad(_, body):
      return rec(*_pad_dot_operands(a, b, ca, cb, order), body)
    case ro.Tile(bk, inner, outer, split):
      a, b = _pad_dot_operands(a, b, ca, cb, order)
      strided = split is ro.Split.strided
      ps = _ragged_partials_sharding(
          core.typeof(a), core.typeof(b), ca, cb, out_s, strided)
      a, b = _tile(a, ca, bk), _tile(b, cb, bk)
      shift = lambda ds, c: tuple(d + int(d > c) for d in ds)
      ca_in, cb_in = (ca, cb) if strided else (ca + 1, cb + 1)
      lb_in = (ca + int(strided), *shift(lb, ca))
      rb_in = (cb + int(strided), *shift(rb, cb))
      if gs.ndim > 1:
        gs = lax.broadcast(gs, (a.shape[lb_in[0]],))
      partials = _eval_ragged_dot(
          a, b, gs, ca_in, cb_in, lb_in, rb_in, shift(lr, ca), shift(rg, cb),
          ps, inner, kw)
      return _eval_reduce(
          lax.reduce_sum, "reduce_sum", partials, 0, out_s, outer)
    case ro.Ring():
      return rec(a, b, ro.Tile(1, ro.ReductionOrder.linear, order,
                               ro.Split.block, None))
    case _:
      dnums = lax.RaggedDotDimensionNumbers(
          (((ca,), (cb,)), (lb, rb)), lr, rg)
      return lax.ragged_dot_general(
          a, b, gs, dnums, out_sharding=out_s, reduction_order=order, **kw)


def _has_unreduced(order: ro.OrderLike) -> bool:
  match order:
    case ro.Tile(_, inner, outer):
      return _has_unreduced(inner) or _has_unreduced(outer)
    case ro.Cast(_, body) | ro.Pad(_, body):
      return _has_unreduced(body)
    case _:
      return order is ro.ReductionOrder.unreduced


def _ragged_partials_sharding(
    a_aval, b_aval, ca: int, cb: int, out_s: Any, strided: bool) -> Any:
  sa = a_aval.sharding.spec.partitions[ca]
  sb = b_aval.sharding.spec.partitions[cb]
  if sa != sb and sa is not None and sb is not None:
    raise core.ShardingTypeError(
        f"ragged_dot requires matching contracting sharding; got {sa!r} and"
        f" {sb!r}.")
  if out_s is None:
    return None
  parts = out_s.spec.partitions
  if strided:
    return out_s.update(spec=out_s.spec.update(partitions=(None, *parts)))
  blk_spec = sa or sb
  blk_axes = set(lax.flatten_spec_no_none((blk_spec,)))
  return out_s.update(
      spec=P(blk_spec, *(_strip_axes(q, blk_axes) for q in parts)))
