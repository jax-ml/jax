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
from collections.abc import Sequence

from jax._src import core
from jax._src import dispatch
from jax._src.interpreters import batching
from jax._src.interpreters import pxla
from jax._src.interpreters import xla
from jax._src.state import discharge
from jax._src.state import indexing
from jax._src.state import primitives
from jax._src.state import types
from jax._src.lax import lax
import jax
import jax.numpy as jnp
import numpy as np

# Indexing #####################################################################

def _ndindexer_to_index(ndidx):
  """Converts an indexing.NDIndexer into a regular Python index."""
  index = []
  for idx in ndidx.indices:
    if isinstance(idx, indexing.Slice):
      index.append(slice(idx.start, idx.size, idx.stride))
    else:
      index.append(idx)
  return tuple(index)

# Accumulator ##################################################################

_monoids = {
  'add': (lax.add, lax._get_sum_identity),
  'mul': (lax.mul, lax._get_prod_identity),
  'bitwise_or': (lax.bitwise_or, lax._get_bitwise_or_identity),
  'bitwise_and': (lax.bitwise_and, lax._get_bitwise_and_identity),
  'bitwise_xor': (lax.bitwise_xor, lax._get_bitwise_or_identity),
  'max': (lax.max, lax._get_max_identity),
  'min': (lax.min, lax._get_min_identity),
}

class Accumulator:
  """An ArrayRef that you can only accumulate into.

  An Accumulator is like an ArrayRef, but you cannot arbitrarily read from it
  or write to it. Instead, you can only accumulate into it. Intuitively,
  accumulating into an Accumulator is like executing a `+=` operation. More
  formally, the operation can be any elementwise monoidal operation.
  """
  _aval: AbstractAccumulator
  _buf: jax.Array
  def __init__(self, aval, buf):
    self._aval = aval
    self._buf = buf
  aval = property(lambda self: self._aval)
  shape = property(lambda self: self._aval.shape)
  dtype = property(lambda self: self._aval.dtype)
  sharding = property(lambda self: self._buf.sharding)
  @property
  def at(self): return self._aval.at.fget(self)
  def accumulate(self, val, idx): return self._aval._accumulate(self, val, idx)
  def __getitem__(self, idx): return self._aval._getitem(self, idx)
  def __repr__(self) -> str: return 'Accumulator' + repr(self._buf)
  def __len__(self) -> int: return self._aval._len(self)

core.pytype_aval_mappings[Accumulator] = lambda x: x._aval

accumulator_p = core.Primitive('accumulator')
accumulator_p.is_effectful = lambda params: True  # type: ignore
accumulator_p.ref_primitive = True

def accumulator(init_val, f):
  """Returns an accumulator.

  f must be one of "add", "mul", "bitwise_or", "bitwise_and", "bitwise_xor",
  "max", or "min". It determines the element-wise operation used to accumulate.
  """
  if f not in _monoids:
    want = ', '.join(repr(k) for k in _monoids.keys())
    raise ValueError(f'Unrecognized accumulation function {repr(f)}; expected one of {want}')
  return accumulator_p.bind(init_val, f=f)

@accumulator_p.def_effectful_abstract_eval
def accumulator_abstract_eval(init_aval, f):
  effects = {core.internal_mutable_array_effect}
  return AbstractAccumulator(init_aval, f), effects

@accumulator_p.def_impl
def _accumulator_impl(init_val, f):
  aval = AbstractAccumulator(core.get_aval(init_val), f)
  return Accumulator(aval, lax._array_copy(init_val))

batching.defvectorized(accumulator_p)
xla.canonicalize_dtype_handlers[Accumulator] = lambda x: x

def _shard_accumulator(xs, shardings, layouts, copy_semantics):
  return pxla.shard_args(shardings, layouts, copy_semantics, [x._buf for x in xs])
pxla.shard_arg_handlers[Accumulator] = _shard_accumulator

# AbstractAccumulator ##########################################################

class AbstractAccumulator(types.AbstractRef):
  """The aval of an Accumulator."""
  __slots__ = ["inner_aval", "f", "memory_space", "foo"]

  def __init__(self, inner_aval: core.AbstractValue, f):
    self.inner_aval = inner_aval
    self.f = f
    self.memory_space = None

  def _accumulate(self, tracer, val, idx):
    ndidx = indexing.NDIndexer.from_indices_shape(idx, tracer.shape)
    flat, tree = jax.tree.flatten(ndidx)
    return accumulate_p.bind(tracer, val, *flat, tree=tree, f=self.f)

  def _getitem(self, tracer, idx) -> jax.Array:
    ndidx = indexing.NDIndexer.from_indices_shape(idx, tracer.shape)
    flat, tree = jax.tree.flatten(ndidx)
    return accumulator_get_p.bind(tracer, *flat, tree=tree)

  @core.aval_property
  def at(self):
    return _At(self)

  def _f(self):
    return _monoids[self.f][0]

  def _identity(self):
    return _monoids[self.f][1](self.dtype)

  def __repr__(self) -> str:
    return f'Accumulator[{self.inner_aval.str_short()}]'

core.pytype_aval_mappings[AbstractAccumulator] = lambda x: x

class _At:
  def __init__(self, acc):
    self.acc = acc

  def __getitem__(self, idx):
    return _AtAccumulator(self.acc, idx)

class _AtAccumulator:
  def __init__(self, acc, idx):
    self.acc = acc
    self.idx = idx

  def accumulate(self, val):
    return self.acc.accumulate(val, self.idx)

# accumulator_get ##############################################################

accumulator_get_p = core.Primitive('accumulator_get')
accumulator_get_p.ref_primitive = True
dispatch.simple_impl(accumulator_get_p)

@accumulator_get_p.def_effectful_abstract_eval
def accumulator_get_abstract_eval(acc, *flat, tree):
  ndidx = jax.tree.unflatten(tree, flat)
  new_shape = ndidx.transform_shape(acc.inner_aval.shape)
  return acc.inner_aval.update(shape=new_shape), {types.ReadEffect(0)}

@discharge.register_discharge_rule(accumulator_get_p)
def _accumulator_get_discharge_rule(in_avals: Sequence[core.AbstractValue],
                                    out_avals: Sequence[core.AbstractValue],
                                    x, *flat, tree):
  del in_avals, out_avals
  idx = _ndindexer_to_index(jax.tree.unflatten(tree, flat))
  return (None,) * (1 + len(flat)), x[idx]

def _accumulator_get_vmap(batched_args, batched_dims, *, tree):
  raise ValueError("You can't read accumulators inside a vmap")

batching.primitive_batchers[accumulator_get_p] = _accumulator_get_vmap

# accumulate ###################################################################

accumulate_p = core.Primitive("accumulate")
accumulate_p.multiple_results = True

dispatch.simple_impl(accumulate_p)

@accumulate_p.def_effectful_abstract_eval
def accumulate_abstract_eval(acc, val, *flat, tree, f):
  return [], {types.AccumEffect(0), core.internal_mutable_array_effect}

@discharge.register_discharge_rule(accumulate_p)
def _accumulate_discharge_rule(in_avals: Sequence[core.AbstractValue],
                               out_avals: Sequence[core.AbstractValue],
                               x, val, *flat, tree, f):
  aval = in_avals[0]
  assert isinstance(aval, AbstractAccumulator)
  idx = _ndindexer_to_index(jax.tree.unflatten(tree, flat))
  result = jnp.vectorize(aval._f())(x[idx], val)
  return (x.at[idx].set(result),) + (None,) * (1 + len(flat)), []

def _accumulate_vmap(batched_args, batched_dims, *, trace, tree, f):
  # Extract arguments.
  ref, val, *flat_idx = batched_args
  ref_dim, val_dim, *flat_idx_dims = batched_dims
  idx = jax.tree.unflatten(tree, flat_idx)
  idx_dims = jax.tree.unflatten(tree, flat_idx_dims)

  # Extract aval.
  aval = core.get_aval(ref)
  assert isinstance(aval, AbstractAccumulator)

  # Check what's batched.
  ref_is_batched = ref_dim is not batching.not_mapped
  val_is_batched = val_dim is not batching.not_mapped
  idx_is_batched = any(i_dim is not batching.not_mapped
                       for i_dim in flat_idx_dims)

  # If ref is not batched, pretend it is. We'll accumulate into a mapped
  # accumulator and reduce the mapped accumulator at the end of vmap.
  if not ref_is_batched:
    if ref not in trace.accumulator_states:
      init_val = jnp.full((trace.axis_data.size,) + ref.shape, aval._identity(), dtype=aval.dtype)
      trace.accumulator_states[ref] = accumulator(init_val, f=aval.f)
    ref = trace.accumulator_states[ref]
    ref_dim = 0
    ref_is_batched = True

  # Update the index.
  new_idx = primitives._batch_indexer(
      idx, idx_dims, trace.axis_data.size, ref.shape, ref_dim, idx_is_batched)

  # Find the batching dimension. This depends on whether the non-slice indexers
  # are contigious or not.
  is_int_indexing, _, _ = indexing.unpack_ndindexer(new_idx)
  int_indexers_contiguous = (
      any(is_int_indexing) and
      bool(np.all(np.diff(np.where(is_int_indexing)[0]) == 1))
  )
  if not int_indexers_contiguous:
    batched_dim_in_result = 0
  else:
    batched_dim_in_result = is_int_indexing.index(True)

  # Batch or reshape val.
  if val_is_batched:
    val = batching.moveaxis(
        val, val_dim, batched_dim_in_result)
  else:
    val = batching.broadcast(
        val, trace.axis_data.size, batched_dim_in_result)

  # Perform the index.
  flat, tree = jax.tree.flatten(new_idx)
  accumulate_p.bind(ref, val, *flat, tree=tree, f=aval.f)

  # TODO: mwhittaker - Allow returning mutable arrays?
  return [], []

batching.primitive_batchers[accumulate_p] = _accumulate_vmap

def _map_accumulator(size, axis, aval):
  mapped = core.mapped_aval(size, axis, aval.inner_aval)
  return AbstractAccumulator(mapped, aval.f)

def _unmap_accumulator(size, axis, explicit_mesh_axis, aval):
  unmapped = core.unmapped_aval(size, axis, aval.inner_aval, explicit_mesh_axis)
  return AbstractAccumulator(unmapped, aval.f)

core.aval_mapping_handlers[AbstractAccumulator] = (_map_accumulator, _unmap_accumulator)

core.Tracer.accumulate = lambda self, val, idx: self.aval._accumulate(self, val, idx)
