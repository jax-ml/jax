# Copyright 2018 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import namedtuple

import itertools as it

import numpy as onp

from six.moves import reduce

from .. import core
from ..core import Trace, Tracer, new_master, pack, AbstractTuple, JaxTuple
from ..abstract_arrays import ShapedArray, make_shaped_array, array_types
from ..ad_util import add_jaxvals_p, zeros_like_p, zeros_like_jaxval
from ..linear_util import transformation, transformation_with_aux, wrap_init
from ..tree_util import register_pytree_node
from ..util import unzip2, partial, safe_map

map = safe_map


def batch(fun, in_vals, in_dims, out_dim_dst):
  sizes = reduce(set.union, map(dimsize, in_dims, in_vals))
  if not sizes:
    return fun.call_wrapped(*in_vals), None  # no mapped dimensions
  elif len(sizes) == 1:
    sz = sizes.pop()
    return batch_transform(fun, sz, in_dims, out_dim_dst).call_wrapped(in_vals)
  else:
    raise TypeError("got inconsistent map dimension sizes: {}".format(sizes))


@transformation
def batch_transform(size, in_dims, out_dim_dst, vals):
  with new_master(BatchTrace) as master:
    trace = BatchTrace(master, core.cur_sublevel())
    in_tracers = map(partial(BatchTracer, trace), vals, in_dims)
    out_tracer = yield in_tracers
    out_tracer = trace.full_raise(out_tracer)
    out_val, out_dim = out_tracer.val, out_tracer.batch_dim
    del master
  yield moveaxis(size, out_dim_dst, out_dim, out_val)


@transformation_with_aux
def batch_subtrace(master, dims, *vals):
  trace = BatchTrace(master, core.cur_sublevel())
  ans = yield map(partial(BatchTracer, trace), vals, dims)
  out_tracer = trace.full_raise(ans)
  out_val, out_dim = out_tracer.val, out_tracer.batch_dim
  yield out_val, out_dim


### tracer


class BatchTracer(Tracer):
  __slots__ = ['val', 'batch_dim']

  def __init__(self, trace, val, batch_dim):
    self.trace = trace
    self.val = val
    self.batch_dim = batch_dim

  @property
  def aval(self):
    batched_aval = get_aval(self.val)
    return remove_batch_dim_from_aval(self.batch_dim, batched_aval)

  def unpack(self):
    t = type(self.batch_dim)
    if t is tuple:
      batch_dims = self.batch_dim
    elif t is int:
      batch_dims = [self.batch_dim] * len(self.val)
    elif t is type(None):
      return tuple(self.val)
    else:
      raise TypeError(t)
    return map(partial(BatchTracer, self.trace), self.val, batch_dims)

  def full_lower(self):
    if self.batch_dim is None:
      return core.full_lower(self.val)
    else:
      return self

class BatchTrace(Trace):
  def pure(self, val):
    return BatchTracer(self, val, None)

  def lift(self, val):
    return BatchTracer(self, val, None)

  def sublift(self, val):
    return BatchTracer(self, val.val, val.batch_dim)

  def process_primitive(self, primitive, tracers, params):
    vals_in, dims_in = unzip2((t.val, t.batch_dim) for t in tracers)
    if all(bdim is None for bdim in dims_in):
      return primitive.bind(*vals_in, **params)
    else:
      # TODO(mattjj,phawkins): if no rule implemented, could vmap-via-map here
      batched_primitive = get_primitive_batcher(primitive)
      val_out, dim_out = batched_primitive(vals_in, dims_in, **params)
      return BatchTracer(self, val_out, dim_out)

  def process_call(self, call_primitive, f, tracers, params):
    vals, dims = unzip2((t.val, t.batch_dim) for t in tracers)
    if all(bdim is None for bdim in dims):
      return call_primitive.bind(f, *vals, **params)
    else:
      f, dim_out = batch_subtrace(f, self.master, dims)
      val_out = call_primitive.bind(f, *vals, **params)
      return BatchTracer(self, val_out, dim_out())

  def post_process_call(self, _, out_tracer):
    val, dim = out_tracer.val, out_tracer.batch_dim
    master = self.master
    def todo(x):
      trace = BatchTrace(master, core.cur_sublevel())
      return BatchTracer(trace, val, dim)

    return val, todo

  def pack(self, tracers):
    vals = pack([t.val for t in tracers])
    batch_dim = tuple(t.batch_dim for t in tracers)
    return BatchTracer(self, vals, batch_dim)


### abstract values


def get_aval(x):
  if isinstance(x, Tracer):
    return raise_to_shaped(x.aval)
  else:
    return shaped_aval(x)

def shaped_aval(x):
  try:
    return pytype_aval_mappings[type(x)](x)
  except KeyError:
    raise TypeError("{} is not a valid type for batching".format(type(x)))

def raise_to_shaped(aval):
  if type(aval) is AbstractTuple:
    return AbstractTuple(map(raise_to_shaped, aval))
  elif isinstance(aval, ShapedArray):
    return ShapedArray(aval.shape, aval.dtype)
  else:
    raise TypeError(type(aval))

def remove_batch_dim_from_aval(bdim, aval):
  t = type(aval)
  if t is AbstractTuple:
    if type(bdim) is tuple:
      return AbstractTuple(map(remove_batch_dim_from_aval, bdim, aval))
    else:
      return AbstractTuple(map(partial(remove_batch_dim_from_aval, bdim), aval))
  elif t is ShapedArray:
    if bdim is None:
      return ShapedArray(aval.shape, aval.dtype)
    else:
      assert 0 <= bdim < aval.ndim
      unbatched_shape = tuple(onp.delete(aval.shape, bdim))
      return ShapedArray(unbatched_shape, aval.dtype)
  else:
    raise TypeError(t)

def add_batch_dim_to_aval(bdim, size, aval):
  t = type(aval)
  if t is AbstractTuple:
    if type(bdim) is tuple:
      return AbstractTuple(map(add_batch_dim_to_aval, bdim, size, aval))
    else:
      return AbstractTuple(map(partial(add_batch_dim_to_aval, bdim, size), aval))
  elif t is ShapedArray:
    if bdim is None:
      return ShapedArray(aval.shape, aval.dtype)
    else:
      assert 0 <= bdim <= aval.ndim
      batched_shape = tuple(onp.insert(aval.shape, bdim, size))
      return ShapedArray(batched_shape, aval.dtype)
  else:
    raise TypeError(t)

pytype_aval_mappings = {}

def shaped_jaxtuple(xs):
  return AbstractTuple(map(shaped_aval, xs))

pytype_aval_mappings[JaxTuple] = shaped_jaxtuple

for t in array_types:
  pytype_aval_mappings[t] = make_shaped_array


### primitives


primitive_batchers = {}

def get_primitive_batcher(p):
  try:
    return primitive_batchers[p]
  except KeyError:
    raise NotImplementedError(
        "Batching rule for '{}' not implemented".format(p))

def defvectorized(prim):
  primitive_batchers[prim] = partial(vectorized_batcher, prim)

def vectorized_batcher(prim, batched_args, batch_dims, **params):
  assert all(batch_dims[0] == bd for bd in batch_dims[1:])
  return prim.bind(*batched_args, **params), batch_dims[0]

def defbroadcasting(prim):
  primitive_batchers[prim] = partial(broadcast_batcher, prim)

def broadcast_batcher(prim, batched_args, batch_dims, **params):
  args = map(bdim_at_front, batched_args, batch_dims)
  ndim = max(map(onp.ndim, args))  # special case to handle scalar broadcasting
  args = map(partial(handle_scalar_broadcasting, ndim), args, batch_dims)
  return prim.bind(*args, **params), 0

def defreducer(prim):
  primitive_batchers[prim] = partial(reducer_batcher, prim)

def reducer_batcher(prim, batched_args, batch_dims, axes, **params):
  operand, = batched_args
  bdim, = batch_dims
  axes = tuple(onp.where(onp.less(axes, bdim), axes, onp.add(axes, 1)))
  bdim_out = list(onp.delete(onp.arange(operand.ndim), axes)).index(bdim)
  if 'input_shape' in params:
    params = dict(params, input_shape=operand.shape)
  return prim.bind(operand, axes=axes, **params), bdim_out

# set up primitive batches for ad_util primitives

def add_batched(batched_args, batch_dims):
  bdx, bdy = batch_dims
  xs, ys = batched_args
  if bdx == bdy:
    return add_jaxvals_p.bind(xs, ys), bdx
  else:
    sz = (dimsize(bdx, xs) | dimsize(bdy, ys)).pop()
    move_bdim = partial(bdim_at_front, broadcast_size=sz, force_broadcast=True)
    xs, ys = map(move_bdim, batched_args, batch_dims)
    return add_jaxvals_p.bind(xs, ys), 0
primitive_batchers[add_jaxvals_p] = add_batched

def zeros_like_batched(batched_args, batch_dims):
  val, = batched_args
  bdim, = batch_dims
  return zeros_like_jaxval(val), bdim
primitive_batchers[zeros_like_p] = zeros_like_batched


### util

# These utilities depend on primitives for things like broadcasting, reshaping,
# and transposition on arrays. To avoid a circular import from depending on
# lax.py, these functions use method dispatch on their arguments, which could be
# DeviceArrays, numpy.ndarrays, or traced versions of those. This strategy
# almost works, except for broadcast, for which raw numpy.ndarrays don't have a
# method. To handle that case, the `broadcast` function uses a try/except.


def bdim_at_front(x, bdim, broadcast_size=1, force_broadcast=False):
  return moveaxis(broadcast_size, 0, bdim, x, force_broadcast=force_broadcast)

def move_dim_to_front(x, dim):
  assert dim is not None
  return moveaxis(None, 0, dim, x)

def dimsize(dim, x):
  aval = get_aval(x)
  if type(aval) is AbstractTuple:
    if type(dim) is tuple:
      return reduce(set.union, map(dimsize, dim, x))
    elif type(dim) is int:
      return reduce(set.union, map(partial(dimsize, dim), x))
    elif dim is None:
      return set()
    else:
      raise TypeError(type(dim))
  else:
    if type(dim) is int:
      return {x.shape[dim]}
    elif dim is None:
      return set()
    else:
      raise TypeError(type(dim))

def moveaxis(sz, dst, src, x, force_broadcast=True):
  aval = get_aval(x)
  if type(aval) is AbstractTuple:
    if type(src) is tuple and type(dst) is tuple:
      return pack(map(partial(moveaxis, sz), dst, src, x))
    elif type(src) is tuple:
      return pack(map(partial(moveaxis, sz, dst), src, x))
    elif type(dst) is tuple:
      srcs = (src,) * len(dst)
      return pack(map(partial(moveaxis, sz), dst, srcs, x))
    else:
      return pack(map(partial(moveaxis, sz, dst, src), x))
  elif isinstance(aval, ShapedArray):
    dst_ = (dst % aval.ndim) if dst is not None and aval.ndim else dst
    if src == dst_:
      return x
    else:
      if src is None:
        x = broadcast(x, sz, force_broadcast=force_broadcast)
        src = 0
        dst_ = dst % (aval.ndim + 1)
      if src == dst_:
        return x
      else:
        perm = [i for i in range(onp.ndim(x)) if i != src]
        perm.insert(dst_, src)
        return x.transpose(perm)
  else:
    raise TypeError(type(aval))

def broadcast(x, sz, force_broadcast=False):
  aval = get_aval(x)
  if type(aval) is AbstractTuple:
    return pack(map(partial(broadcast, sz=sz), x))
  elif isinstance(aval, ShapedArray):
    # for scalars, maybe don't actually broadcast
    if not onp.ndim(x) and not force_broadcast:
      return x

    # see comment at the top of this section
    if isinstance(x, onp.ndarray) or onp.isscalar(x):
      return onp.broadcast_to(x, (sz,) + onp.shape(x))
    else:
      return x.broadcast((sz,))  # should be a JAX arraylike
  else:
    raise TypeError(type(x))

def handle_scalar_broadcasting(nd, x, bdim):
  assert isinstance(get_aval(x), ShapedArray)
  if bdim is None or nd == onp.ndim(x):
    return x
  else:
    return x.reshape(x.shape + (1,) * (nd - x.ndim))
