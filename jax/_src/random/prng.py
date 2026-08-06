# Copyright 2021 The JAX Authors.
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

from collections.abc import Callable, Iterator, Sequence
from functools import partial, reduce
import math
from typing import Any, NamedTuple

import numpy as np

from jax._src import api
from jax._src import config as config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import ffi
from jax._src import numpy as jnp
from jax._src import pretty_printer as pp
from jax._src.sharding import Sharding
from jax._src.named_sharding import NamedSharding
from jax._src.mesh import AbstractMesh, get_concrete_mesh
from jax._src import source_info_util
from jax._src import tree_util
from jax._src import typing

from jax._src.dtypes import float0
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.lax import lax

from jax._src.lib import gpu_prng
from jax._src.lib import xla_client as xc
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.numpy.array_methods import (
    _array_operators, _set_array_base_attributes, _IndexUpdateHelper)
from jax._src.sharding_impls import (
    make_single_device_sharding, physical_sharding, logical_sharding)
from jax._src.typing import Array
from jax._src.util import safe_map, safe_zip

map, unsafe_map = safe_map, map
zip, unsafe_zip = safe_zip, zip

Device = xc.Device
Shard = Any  # TODO(jakevdp): fix circular imports and import Shard
Shape = tuple[int, ...]

UINT_DTYPES: dict[int, np.dtype] = {
    8: np.dtype('uint8'),
    16: np.dtype('uint16'),
    32: np.dtype('uint32'),
    64: np.dtype('uint64'),
}

if hasattr(gpu_prng, "registrations"):
  for platform, targets in gpu_prng.registrations().items():
    for name, value, api_version in targets:
      ffi.register_ffi_target(
          name, value, platform=platform, api_version=api_version
      )

# -- PRNG implementation interface

class PRNGImpl(NamedTuple):
  """Specifies PRNG key shape and operations.

  A PRNG implementation is determined by a key type ``K`` and a
  collection of functions that operate on such keys. The key type
  ``K`` is an array type with element type uint32 and shape specified
  by ``key_shape``. The type signature of each operations is::

    seed :: int[] -> K
    fold_in :: K -> int[] -> K
    split[shape] :: K -> K[*shape]
    random_bits[shape, bit_width] :: K -> uint<bit_width>[*shape]

  A PRNG implementation is adapted to an array-like object of keys
  ``K`` by the ``PRNGKeyArray`` class, which should be created via the
  ``random_seed`` function.
  """
  key_shape: Shape
  seed: Callable
  split: Callable
  random_bits: Callable
  fold_in: Callable
  name: str = '<unnamed>'
  tag: str = '?'

  def __hash__(self) -> int:
    return hash(self.tag)

  def __str__(self) -> str:
    return self.tag

  def pprint(self):
    ty = self.__class__.__name__
    return (pp.text(f"{ty} [{self.tag}] {{{self.name}}}:") +
            pp.nest(2, pp.group(pp.brk() + pp.join(pp.brk(), [
              pp.text(f"{k} = {v}") for k, v in self._asdict().items()
            ]))))


prngs: dict[str, PRNGImpl] = {}

def register_prng(impl: PRNGImpl):
  if impl.name in prngs:
    raise ValueError(f'PRNG with name {impl.name} already registered: {impl}')
  prngs[impl.name] = impl


# -- PRNG key arrays

def _check_prng_key_data(impl, key_data: typing.Array):
  ndim = len(impl.key_shape)
  if not all(hasattr(key_data, attr) for attr in ['ndim', 'shape', 'dtype']):
    raise TypeError("JAX encountered invalid PRNG key data: expected key_data "
                    f"to have ndim, shape, and dtype attributes. Got {key_data}")
  if key_data.ndim < 1:
    raise TypeError("JAX encountered invalid PRNG key data: expected "
                    f"key_data.ndim >= 1; got ndim={key_data.ndim}")
  if key_data.shape[-ndim:] != impl.key_shape:
    raise TypeError("JAX encountered invalid PRNG key data: expected key_data.shape to "
                    f"end with {impl.key_shape}; got shape={key_data.shape} for {impl=}")
  if key_data.dtype not in [np.uint32, float0]:
    raise TypeError("JAX encountered invalid PRNG key data: expected key_data.dtype = uint32; "
                    f"got dtype={key_data.dtype}")


class PRNGKeyArray(Array):
  """An array of PRNG keys backed by an RNG implementation.

  This class lifts the definition of a PRNG, provided in the form of a
  ``PRNGImpl``, into an array-like pytree class. Instances of this
  class behave like an array whose base elements are keys, hiding the
  fact that keys are typically arrays (of ``uint32`` dtype) themselves.

  PRNGKeyArrays are also restricted relative to JAX arrays in that
  they do not expose arithmetic operations. They instead expose
  wrapper methods around the PRNG implementation functions (``split``,
  ``random_bits``, ``fold_in``).
  """
  # TODO(jakevdp): potentially add tolist(), tobytes(),
  #    device_buffer, device_buffers, __cuda_interface__()

  _impl: PRNGImpl
  _base_array: Array
  _consumed: bool | np.ndarray  # Used in jax.experimental.key_reuse.
  _source_info: None | source_info_util.SourceInfo = None

  def __init__(self, impl, key_data: Any):
    assert not isinstance(key_data, core.Tracer)
    _check_prng_key_data(impl, key_data)
    self._impl = impl
    self._consumed = False  # TODO(jakevdp): default to True here?
    if isinstance(key_data, np.ndarray):
      aval = core.typeof(key_data)
      device = pxla.get_default_device()
      key_data = pxla.batched_device_put(
          aval, make_single_device_sharding(device),
          [np.asarray(key_data)], [device], committed=False)
    self._base_array = key_data

  def _replace_with(self, value: PRNGKeyArray):
    self._base_array._replace_with(value._base_array)  # pyrefly: ignore[missing-attribute]

  def block_until_ready(self):
    _ = self._base_array.block_until_ready()
    return self

  def copy_to_host_async(self):
    _ = self._base_array.copy_to_host_async()

  @property
  def aval(self):
    mt = self._base_array.aval.mat
    return keys_shaped_array(self._impl, self.shape, self.sharding, mt)

  @property
  def shape(self):
    return base_arr_shape_to_keys_shape(self._impl, self._base_array.shape)

  @property
  def size(self):
    return math.prod(self.shape)

  @property
  def ndim(self):
    return len(self.shape)

  @property
  def dtype(self):
    return KeyTy(self._impl)

  @property
  def nbytes(self):
    return self.itemsize * self.size

  @property
  def itemsize(self):
    return self.dtype.itemsize

  @property
  def _device(self) -> Device:
    assert hasattr(self._base_array, "_device")
    return self._base_array._device

  @property
  def _committed(self) -> bool:
    assert hasattr(self._base_array, "_committed")
    return self._base_array._committed

  @property
  def device(self) -> Device | Sharding:
    return self._base_array.device

  @property
  def is_fully_addressable(self) -> bool:
    return self._base_array.is_fully_addressable

  @property
  def is_fully_replicated(self) -> bool:
    return self._base_array.is_fully_replicated

  def devices(self) -> set[Device]:
    return self._base_array.devices()

  def delete(self) -> None:
    self._base_array.delete()

  def is_deleted(self) -> bool:
    return self._base_array.is_deleted()

  def on_device_size_in_bytes(self) -> int:
    return self._base_array.on_device_size_in_bytes()

  def unsafe_buffer_pointer(self) -> int:
    return self._base_array.unsafe_buffer_pointer()

  def addressable_data(self, index: int) -> PRNGKeyArray:
    return PRNGKeyArray(self._impl, self._base_array.addressable_data(index))

  @property
  def addressable_shards(self) -> list[Shard]:
    return [
        type(s)(
            device=s._device,
            sharding=s._sharding,
            global_shape=s._global_shape,
            data=PRNGKeyArray(self._impl, s._data),
        )
        for s in self._base_array.addressable_shards
    ]

  @property
  def global_shards(self) -> list[Shard]:
    return [
        type(s)(
            device=s._device,
            sharding=s._sharding,
            global_shape=s._global_shape,
            data=PRNGKeyArray(self._impl, s._data),
        )
        for s in self._base_array.global_shards
    ]

  @property
  def sharding(self):
    return logical_sharding(self.shape, self.dtype, self._base_array.sharding)

  @property
  def committed(self):
    return self._base_array.committed

  def _is_scalar(self):
    base_ndim = len(self._impl.key_shape)
    return self._base_array.ndim == base_ndim

  def __len__(self):
    if self._is_scalar():
      raise TypeError('len() of unsized object')
    return len(self._base_array)

  def __iter__(self) -> Iterator[PRNGKeyArray]:
    if self._is_scalar():
      raise TypeError('iteration over a 0-d key array')
    # TODO(frostig): we may want to avoid iteration by slicing because
    # a very common use of iteration is `k1, k2 = split(key)`, and
    # slicing/indexing may be trickier to track for linearity checking
    # purposes. Maybe we can:
    # * introduce an unpack primitive+traceable (also allow direct use)
    # * unpack upfront into shape[0] many keyarray slices
    # * return iter over these unpacked slices
    # Whatever we do, we'll want to do it by overriding
    # ShapedArray._iter when the element type is KeyTy...
    return (PRNGKeyArray(self._impl, k) for k in iter(self._base_array))

  def __bool__(self):
    raise TypeError("key array cannot be converted to boolean.")

  def __repr__(self):
    return (f'Array({self.shape}, dtype={self.dtype.name}) overlaying:\n'
            f'{self._base_array}')

  def pprint(self):
    pp_keys = pp.text('shape = ') + pp.text(str(self.shape))
    pp_impl = pp.text('impl = ') + self._impl.pprint()
    return str(pp.group(
      pp.text('PRNGKeyArray:') +
      pp.nest(2, pp.brk() + pp_keys + pp.brk() + pp_impl)))

  def copy(self):
    out = self.__class__(self._impl, self._base_array.copy())
    out._consumed = self._consumed  # TODO(jakevdp): is this correct?
    return out

  __hash__ = None
  __array_priority__ = 100

  def __array__(self, dtype: np.dtype | None = None, context: None = None,
                copy: bool | None = None) -> np.ndarray:
    del dtype, context, copy
    raise TypeError("JAX array with PRNGKey dtype cannot be converted to a NumPy array."
                    " Use jax.random.key_data(arr) if you wish to extract the underlying"
                    " integer array.")


  # Overwritten immediately below
  @property
  def at(self)                  -> _IndexUpdateHelper: assert False  # pyrefly: ignore[bad-override]
  @property
  def T(self)                   -> PRNGKeyArray: assert False
  def __getitem__(self, _, /)   -> PRNGKeyArray: assert False
  def flatten(self, *_, **__)   -> PRNGKeyArray: assert False
  def ravel(self, *_, **__)     -> PRNGKeyArray: assert False
  def reshape(self, *_, **__)   -> PRNGKeyArray: assert False
  def squeeze(self, *_, **__)   -> PRNGKeyArray: assert False
  def swapaxes(self, *_, **__)  -> PRNGKeyArray: assert False
  def take(self, *_, **__)      -> PRNGKeyArray: assert False
  def transpose(self, *_, **__) -> PRNGKeyArray: assert False

_set_array_base_attributes(PRNGKeyArray, include=[
    *(f"__{op}__" for op in _array_operators),
    'at', 'flatten', 'ravel', 'reshape',
    'squeeze', 'swapaxes', 'take', 'transpose', 'T'])


def prngkeyarray_flatten(x):
  return (x._base_array,), x._impl

def prngkeyarray_unflatten(impl, children):
  base_array, = children
  return PRNGKeyArray(impl, base_array)

tree_util.dispatch_registry.register_node(
    PRNGKeyArray, prngkeyarray_flatten, prngkeyarray_unflatten)


# TODO(frostig): remove, rerouting callers directly to random_seed
def seed_with_impl(impl: PRNGImpl, seed: int | typing.ArrayLike) -> PRNGKeyArray:
  return random_seed(seed, impl=impl)


def keys_shaped_array(impl, shape, sharding, mat):
  aval = core.ShapedArray(shape, KeyTy(impl))
  return core.update_aval_with_sharding(aval, sharding, mat=mat)

def base_arr_shape_to_keys_shape(impl, base_arr_shape):
  base_ndim = len(impl.key_shape)
  return base_arr_shape[:-base_ndim]


class KeyTyRules:
  allow_conversion: bool = False

  @staticmethod
  def full(shape, fill_value, dtype):
    physical_shape = (*shape, *dtype._impl.key_shape)
    if hasattr(fill_value, 'dtype') and dtypes.issubdtype(fill_value.dtype, dtypes.prng_key):
      key_data = jnp.broadcast_to(random_unwrap(fill_value), physical_shape)
    else:
      key_data = lax.full(physical_shape, fill_value, dtype=np.dtype('uint32'))
    # TODO(frostig,mattjj,vanderplas,lenamartens): consider this consumed from
    # the outset.
    return random_wrap(key_data, impl=dtype._impl)

  @staticmethod
  def physical_element_aval(dtype) -> core.ShapedArray:
    return core.ShapedArray(dtype._impl.key_shape, np.dtype('uint32'))

  @staticmethod
  def physical_const(val) -> Array:
    return val._base_array

  @staticmethod
  def result_handler(sticky_device, aval):
    def handler(_, buf):
      buf.aval = core.ShapedArray(buf.shape, buf.dtype)
      return PRNGKeyArray(aval.dtype._impl, buf)
    return handler

  @staticmethod
  def global_sharded_result_handler(aval, out_sharding, committed):
    phys_aval = core.physical_aval(aval)
    phys_handler_maker = pxla.global_result_handlers[core.ShapedArray]

    phys_sharding = physical_sharding(aval, out_sharding)
    phys_handler = phys_handler_maker(phys_aval, phys_sharding, committed)
    def handler(bufs):
      return PRNGKeyArray(aval.dtype._impl, bufs)
    return phys_handler.wrap(handler)

  @staticmethod
  def make_sharded_array(aval, sharding, arrays, committed):
    phys_aval = core.physical_aval(aval)
    phys_handler_maker = pxla.global_result_handlers[core.ShapedArray]
    phys_arrays = [random_unwrap(arr) for arr in arrays]

    phys_sharding = physical_sharding(aval, sharding)
    phys_handler = phys_handler_maker(phys_aval, phys_sharding, committed)
    phys_result = phys_handler(phys_arrays)
    return PRNGKeyArray(aval.dtype._impl, phys_result)

  @staticmethod
  def device_put_sharded(vals, aval, sharding, devices):
    physical_buffers = tree_util.tree_map(random_unwrap, vals)
    physical_result = api.device_put_sharded(physical_buffers, list(devices))
    return random_wrap(physical_result, impl=aval.dtype._impl)

  @staticmethod
  def device_put_replicated(val, aval, sharding, devices):
    physical_aval = core.physical_aval(aval)
    physical_buf = random_unwrap(val)
    phys_sharding = physical_sharding(aval, sharding)
    physical_result = pxla.batched_device_put(
        physical_aval, phys_sharding, [physical_buf] * len(devices), devices)  # pyrefly: ignore[bad-argument-type]
    return random_wrap(physical_result, impl=aval.dtype._impl)

  @staticmethod
  def tangent_dtype(_):
    return dtypes.float0

  # TODO(mattjj,frostig): even though the key dtype shouldn't appear in
  # tangents, our ad.replace_float0s in custom_jvp/vjp means passing in zeros
  # like the primal to user rules
  @staticmethod
  def zero(_):
    return np.zeros((), dtypes.float0)


class KeyTy(dtypes.ExtendedDType):
  _impl: PRNGImpl  # TODO(mattjj,frostig): protocol really
  _rules = KeyTyRules
  type = dtypes.prng_key

  def __init__(self, impl):
    self._impl = impl

  @property
  def name(self) -> str:
    return f'key<{self._impl.tag}>'

  @property
  def itemsize(self) -> int:
    return math.prod(self._impl.key_shape) * np.dtype('uint32').itemsize

  def __repr__(self) -> str:
    return self.name

  def __eq__(self, other):
    return type(other) is KeyTy and self._impl == other._impl

  def __hash__(self) -> int:
    return hash((self.__class__, self._impl))


core.pytype_aval_mappings[PRNGKeyArray] = lambda x: x.aval
dtypes.register_canonicalize_value_handler(PRNGKeyArray, None)


def key_array_shard_arg_handler(xs: Sequence[PRNGKeyArray], shardings, layouts,
                                copy_semantics):
  arrs = [x._base_array for x in xs]
  phys_shardings = [physical_sharding(x.aval, sharding)
                    for x, sharding in zip(xs, shardings)]
  # TODO(yashkatariya): `layouts` should be converted to physical layouts.
  return pxla.shard_args(phys_shardings, layouts, copy_semantics, arrs)


pxla.shard_arg_handlers[PRNGKeyArray] = key_array_shard_arg_handler


def key_array_constant_handler(val, aval):
  arr = val._base_array
  return mlir.get_constant_handler(type(arr))(arr, aval)
mlir.register_constant_handler(PRNGKeyArray, key_array_constant_handler)


# -- primitives

def iterated_vmap_unary(n, f):
  for _ in range(n):
    f = api.vmap(f)
  return f

# TODO(frostig): Revise the following two functions? These basically
# undo the singleton dimensions added by `batching.defbroadcasting`.
# It works, but introduces some possibly-redundant squeezes. Can we
# borrow from other broadcasting primitives instead?

def squeeze_vmap(f, left):
  def squeeze_vmap_f(x, y):
    if left:
      x = jnp.squeeze(x, axis=0)
      axes = (None, 0)
    else:
      y = jnp.squeeze(y, axis=0)
      axes = (0, None)
    return api.vmap(f, in_axes=axes, out_axes=0)(x, y)
  return squeeze_vmap_f

def iterated_vmap_binary_bcast(shape1, shape2, f):
  ndim1, ndim2 = len(shape1), len(shape2)
  if ndim1 == ndim2 == 0:
    return f
  if 0 in [ndim1, ndim2]:
    if ndim1 == 0:
      return lambda x, y: iterated_vmap_unary(ndim2, lambda y: f(x, y))(y)
    else:
      return lambda x, y: iterated_vmap_unary(ndim1, lambda x: f(x, y))(x)
  assert len(shape1) == len(shape2)
  for sz1, sz2 in reversed(zip(shape1, shape2)):
    if sz1 == sz2:
      f = api.vmap(f, out_axes=0)
    else:
      assert sz1 == 1 or sz2 == 1, (sz1, sz2)
      f = squeeze_vmap(f, sz1 == 1)
  return f


def random_seed(seeds: int | typing.ArrayLike, impl: PRNGImpl) -> PRNGKeyArray:
  # Avoid overflow error in X32 mode by first converting ints to int64.
  # This breaks JIT invariance for large ints, but supports the common
  # use-case of instantiating with Python hashes in X32 mode.
  if isinstance(seeds, int):
    seeds_arr = jnp.asarray(np.int64(seeds))
  else:
    seeds_arr = jnp.asarray(seeds)
  if config.random_seed_offset.value:
    seeds_arr += config.random_seed_offset.value
  return random_seed_p.bind(seeds_arr, impl=impl)

random_seed_p = core.Primitive('random_seed')
ad.defjvp_zero(random_seed_p)
batching.defvectorized(random_seed_p)

@random_seed_p.def_abstract_eval
def random_seed_abstract_eval(seeds_aval, *, impl):
  return keys_shaped_array(impl, seeds_aval.shape, seeds_aval.sharding,
                           seeds_aval.mat)

@random_seed_p.def_impl
def random_seed_impl(seeds, *, impl):
  base_arr = random_seed_impl_base(seeds, impl=impl)
  return PRNGKeyArray(impl, base_arr)

def random_seed_impl_base(seeds, *, impl):
  seed = iterated_vmap_unary(np.ndim(seeds), impl.seed)
  return seed(seeds)

def random_seed_lowering(ctx, seeds, *, impl):
  aval, = ctx.avals_in
  seed = iterated_vmap_unary(aval.ndim, impl.seed)
  seed_lowering = mlir.lower_fun(seed, multiple_results=False)
  return mlir.delegate_lowering(
      ctx, seed_lowering, seeds,
      avals_out=map(core.physical_aval, ctx.avals_out))

mlir.register_lowering(random_seed_p, random_seed_lowering)


def random_split(keys, shape: Shape):
  return random_split_p.bind(keys, shape=shape)

random_split_p = core.Primitive('random_split')
ad.defjvp_zero(random_split_p)
batching.defvectorized(random_split_p)

@random_split_p.def_abstract_eval
def random_split_abstract_eval(keys_aval, *, shape):
  # TODO(yashkatariya): random_split should take sharding as an arg too so we
  # don't choose None here?
  if keys_aval.sharding.mesh.empty:
    out_sharding = core.get_cur_mesh_sharding()
  else:
    new_spec = (*keys_aval.sharding.spec, *[None] * len(shape))
    out_sharding = keys_aval.sharding.update(spec=new_spec)
  return keys_shaped_array(keys_aval.dtype._impl, (*keys_aval.shape, *shape),
                           out_sharding, keys_aval.mat)

@random_split_p.def_impl
def random_split_impl(keys, *, shape):
  base_arr = random_split_impl_base(
      keys._impl, keys._base_array, keys.ndim, shape=shape)
  return PRNGKeyArray(keys._impl, base_arr)

def random_split_impl_base(impl, base_arr, keys_ndim, *, shape):
  split = iterated_vmap_unary(keys_ndim, lambda k: impl.split(k, shape))
  return split(base_arr)

def random_split_lowering(ctx, keys, *, shape):
  aval, = ctx.avals_in
  impl = aval.dtype._impl
  split = iterated_vmap_unary(aval.ndim, lambda k: impl.split(k, shape))
  split_lowering = mlir.lower_fun(split, multiple_results=False)
  return mlir.delegate_lowering(
      ctx, split_lowering, keys,
      avals_in=[core.physical_aval(aval)],
      avals_out=map(core.physical_aval, ctx.avals_out))

mlir.register_lowering(random_split_p, random_split_lowering)


def random_fold_in(keys, msgs):
  msgs = jnp.asarray(msgs)
  keys, msgs = core.auto_insert_reshard(keys, msgs)
  return random_fold_in_p.bind(keys, msgs)

random_fold_in_p = core.Primitive('random_fold_in')
ad.defjvp_zero(random_fold_in_p)
batching.defbroadcasting(random_fold_in_p)

@random_fold_in_p.def_abstract_eval
def random_fold_in_abstract_eval(keys_aval, msgs_aval):
  shape = lax.broadcasting_shape_rule(
      'random_fold_in', keys_aval, msgs_aval)
  sharding = lax.broadcasting_sharding_rule(
      'random_fold_in', keys_aval, msgs_aval)
  vma = core.standard_vma_rule('random_fold_in', keys_aval, msgs_aval)
  out_mat = core.ManualAxisType(varying=vma)
  return core.ShapedArray(shape, keys_aval.dtype, sharding=sharding,
                          manual_axis_type=out_mat)

@random_fold_in_p.def_impl
def random_fold_in_impl(keys, msgs):
  base_arr = random_fold_in_impl_base(
      keys._impl, keys._base_array, msgs, keys.shape)
  return PRNGKeyArray(keys._impl, base_arr)

def random_fold_in_impl_base(impl, base_arr, msgs, keys_shape):
  fold_in = iterated_vmap_binary_bcast(
      keys_shape, np.shape(msgs), impl.fold_in)
  return fold_in(base_arr, msgs)

def random_fold_in_lowering(ctx, keys, msgs):
  keys_aval, msgs_aval = ctx.avals_in
  impl = keys_aval.dtype._impl
  fold_in = iterated_vmap_binary_bcast(
      keys_aval.shape, msgs_aval.shape, impl.fold_in)
  fold_in_lowering = mlir.lower_fun(fold_in, multiple_results=False)
  return mlir.delegate_lowering(
      ctx, fold_in_lowering, keys, msgs,
      avals_in=[core.physical_aval(keys_aval), msgs_aval],
      avals_out=map(core.physical_aval, ctx.avals_out))

mlir.register_lowering(random_fold_in_p, random_fold_in_lowering)


def random_bits(keys, *, bit_width, shape, out_sharding=None):
  return random_bits_p.bind(keys, bit_width=bit_width, shape=shape, out_sharding=out_sharding)

random_bits_p = core.Primitive('random_bits')
ad.defjvp_zero(random_bits_p)
batching.defvectorized(random_bits_p)

@random_bits_p.def_abstract_eval
def random_bits_abstract_eval(keys_aval, *, bit_width, shape, out_sharding=None):
  out_shape = (*keys_aval.shape, *shape)
  out_dtype = dtypes.dtype(f'uint{bit_width}')
  if out_sharding is not None:
    out_sharding.check_compatible_aval(out_shape)
  elif keys_aval.sharding.mesh.empty:
    out_sharding = core.get_cur_mesh_sharding()
  else:
    new_spec = (*keys_aval.sharding.spec, *[None] * len(shape))
    out_sharding = keys_aval.sharding.update(spec=new_spec)
  return core.ShapedArray(out_shape, out_dtype, sharding=out_sharding,
                          manual_axis_type=keys_aval.mat)

@random_bits_p.def_impl
def random_bits_impl(keys, *, bit_width, shape, out_sharding=None):
  res = random_bits_impl_base(keys._impl, keys._base_array, keys.ndim,
                              bit_width=bit_width, shape=shape)
  if out_sharding is not None:
    if isinstance(out_sharding, NamedSharding) and isinstance(out_sharding.mesh, AbstractMesh):
      concrete_mesh = get_concrete_mesh()
      if not concrete_mesh.empty:
        out_sharding = NamedSharding(concrete_mesh, out_sharding.spec)
    res = api.device_put(res, out_sharding)
  return res

def random_bits_impl_base(impl, base_arr, keys_ndim, *, bit_width, shape):
  bits = iterated_vmap_unary(
      keys_ndim, lambda k: impl.random_bits(k, bit_width, shape))
  return bits(base_arr)

def random_bits_lowering(ctx, keys, *, bit_width, shape, out_sharding=None):
  del out_sharding
  aval, = ctx.avals_in
  impl = aval.dtype._impl
  bits = iterated_vmap_unary(
      aval.ndim, lambda k: impl.random_bits(k, bit_width, shape))
  bits_lowering = mlir.lower_fun(bits, multiple_results=False)
  ctx_new = ctx.replace(avals_in=[core.physical_aval(aval)])
  out = bits_lowering(ctx_new, keys)
  ctx.set_tokens_out(ctx_new.tokens_out)
  return out

mlir.register_lowering(random_bits_p, random_bits_lowering)


# The following wrap/unwrap primitives are at least a stopgap for
# backwards compatibility, namely when `config.jax_enable_custom_prng`
# is False. We need to convert key arrays to and from underlying
# uint32 base array, and we may need to do so under a jit. For
# example, we want to support:
#
#   keys = jax.jit(random.split)(key)
#
# where `key` and `keys` are both acceptably old-style uint32 arrays
# so long as enable_custom_prng is False. The way we handle this is
# that `random.split` adapts the input/output by converting to/from
# key arrays across its call to `random_split`. So we rely on these
# wrap/unwrap casting primitives to allow that conversion under jit.
#
# We may want to keep both around for testing and debugging escape
# hatches. We can rename them `unsafe` for emphasis, and/or issue a
# warning on entry to the traceable.
#
# TODO(frostig): Consider removal once we always enable_custom_prng.

def random_wrap(base_arr, *, impl):
  _check_prng_key_data(impl, base_arr)
  return random_wrap_p.bind(base_arr, impl=impl)

random_wrap_p = core.Primitive('random_wrap')
ad.defjvp_zero(random_wrap_p)

@random_wrap_p.def_abstract_eval
def random_wrap_abstract_eval(base_arr_aval, *, impl):
  shape = base_arr_shape_to_keys_shape(impl, base_arr_aval.shape)
  sharding = logical_sharding(shape, KeyTy(impl), base_arr_aval.sharding)
  return keys_shaped_array(impl, shape, sharding, base_arr_aval.mat)

@random_wrap_p.def_impl
def random_wrap_impl(base_arr, *, impl):
  return PRNGKeyArray(impl, base_arr)

def random_wrap_lowering(ctx, base_arr, *, impl):
  return [base_arr]

def random_wrap_batch_rule(batched_args, batch_dims, *, impl):
  x, = batched_args
  d, = batch_dims
  x = batching.bdim_at_front(x, d, 1)
  return random_wrap(x, impl=impl), 0

mlir.register_lowering(random_wrap_p, random_wrap_lowering)
batching.primitive_batchers[random_wrap_p] = random_wrap_batch_rule


def random_unwrap(keys):
  if not dtypes.issubdtype(keys.dtype, dtypes.prng_key):
    raise TypeError(f'random_unwrap takes key array operand, got {keys.dtype=}')
  return random_unwrap_p.bind(keys)

random_unwrap_p = core.Primitive('random_unwrap')
ad.defjvp_zero(random_unwrap_p)
batching.defvectorized(random_unwrap_p)

@random_unwrap_p.def_abstract_eval
def random_unwrap_abstract_eval(keys_aval):
  return core.physical_aval(keys_aval)

@random_unwrap_p.def_impl
def random_unwrap_impl(keys):
  return keys._base_array

def random_unwrap_lowering(ctx, keys):
  return [keys]

mlir.register_lowering(random_unwrap_p, random_unwrap_lowering)


def iota_2x32_shape(shape):
  """Reshaped ``uint64`` iota, as two parallel ``uint32`` arrays.

  Setting aside representation, this function essentially computes the
  equivalent of::

    jax.lax.iota(dtype=np.uint64, size=math.prod(shape)).reshape(shape)

  However:

  * It returns two parallel ``uint32`` arrays instead of one
    ``uint64`` array. This renders it invariant under either setting of
    the system-wide ``jax_enable_x64`` configuration flag.

  * It lowers in a way such that the compiler's automatic SPMD
    partitioner recognizes its partitionability.

  For example::

    >>> import numpy as np
    >>> from jax import lax
    >>> from jax._src.random import prng

    >>> prng.iota_2x32_shape((3, 4))
    [Array([[0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0]], dtype=uint32),
     Array([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]], dtype=uint32)]

    >>> def reshaped_iota(shape):
    ...   return lax.iota(size=math.prod(shape), dtype=np.uint32).reshape(shape)
    ...
    >>> reshaped_iota((3, 4))
    Array([[ 0,  1,  2,  3],
           [ 4,  5,  6,  7],
           [ 8,  9, 10, 11]], dtype=uint32)

  Args:
    shape: the output shape

  Returns:
    A pair of ``uint32`` arrays ``(counts_hi, counts_lo)``, both of
    shape ``shape``, representing the higher-order and lower-order 32
    bits of the 64 bit unsigned iota.
  """
  if len(shape) == 0:
    return (jnp.zeros((), np.dtype('uint32')),) * 2
  return iota_2x32_shape_p.bind(shape=shape)

iota_2x32_shape_p = core.Primitive('iota_2x32_shape')
iota_2x32_shape_p.multiple_results = True
iota_2x32_shape_p.def_impl(partial(dispatch.apply_primitive, iota_2x32_shape_p))

@iota_2x32_shape_p.def_abstract_eval
def iota_2x32_shape_abstract_eval(*, shape):
  return (core.ShapedArray(shape, np.dtype('uint32')),) * 2

def bcast_iotas_to_reshaped_iota(
    add: Callable[[ir.Value, ir.Value], ir.Value],
    mul: Callable[[core.DimSize, ir.Value], ir.Value],
    shape: core.Shape,
    iotas: Sequence[ir.Value]) -> ir.Value:
  strides: core.Shape = (*(np.cumprod(shape[1:][::-1])[::-1]), 1)
  return reduce(add, [mul(s, i) for i, s in zip(iotas, strides)])

def iota_2x32_shape_lowering(ctx, *, shape):
  aval_out, _ = ctx.avals_out
  aval_u64 = core.ShapedArray(shape, np.dtype('uint64'))

  def _add(x: ir.Value, y: ir.Value) -> ir.Value:
    return mlir.hlo.add(x, y)

  def _mul(x: core.DimSize, y: ir.Value) -> ir.Value:
    if core.is_constant_dim(x):
      x_const = mlir.ir_constant(np.array(x, np.dtype('uint64')))
    else:
      x_shape, = mlir.eval_dynamic_shape(ctx, (x,))
      x_const = hlo.convert(
          ir.RankedTensorType.get(
              [],
              mlir.dtype_to_ir_type(np.dtype('uint64'))), x_shape)  # pyrefly: ignore[bad-argument-type]
    x_bcast = mlir.broadcast_in_dim(ctx, x_const, aval_u64,
                                    broadcast_dimensions=[])
    return mlir.hlo.multiply(x_bcast, y)

  assert len(shape) > 0

  iotas = [mlir.iota(ctx, aval_u64, dimension=dimension)
           for dimension in range(len(shape))]
  counts = bcast_iotas_to_reshaped_iota(_add, _mul, shape, iotas)
  shift = mlir.ir_constant(np.array(32, np.dtype('uint64')))
  shift = mlir.broadcast_in_dim(ctx, shift, aval_u64,
                                broadcast_dimensions=[])
  counts_shifted = mlir.hlo.shift_right_logical(counts, shift)
  result_type = mlir.aval_to_ir_type(ctx.module_context, aval_out)
  counts_lo = mlir.hlo.convert(result_type, counts)
  counts_hi = mlir.hlo.convert(result_type, counts_shifted)
  return counts_hi, counts_lo
mlir.register_lowering(iota_2x32_shape_p, iota_2x32_shape_lowering)
