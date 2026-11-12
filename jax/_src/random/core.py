# Copyright 2018 The JAX Authors.
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

from collections.abc import Hashable, Sequence
from functools import partial
import math
from operator import index
import typing
import warnings

import numpy as np

from jax._src import config
from jax._src import core
from jax._src import dispatch
from jax._src import dtypes
from jax._src import mesh as mesh_lib
from jax._src import numpy as jnp
from jax._src import shard_map
from jax._src.random import prng
from jax._src.random import threefry2x32
from jax._src.mesh import get_abstract_mesh, use_abstract_mesh
from jax._src.sharding_impls import NamedSharding, PartitionSpec as P
from jax._src.api import Inline, jit, jvp, vmap
from jax._src.interpreters import ad
from jax._src.interpreters import batching
from jax._src.interpreters import mlir
from jax._src.lax import control_flow as lax_control_flow
from jax._src.lax.erf import erf_inv_core, ndtr
from jax._src.lax import lax
from jax._src.lax import special as lax_special
from jax._src.named_sharding import remove_size_one_mesh_axis_from_spec
from jax._src.numpy import einsum as jnp_einsum
from jax._src.numpy import linalg as jnp_linalg
from jax._src.numpy.util import (
    _arraylike, check_arraylike, promote_dtypes_inexact)
from jax._src.pjit import reshard
from jax._src.sharding_impls import canonicalize_sharding
from jax._src.typing import Array, ArrayLike, DType, DTypeLike
from jax._src.util import canonicalize_axis

# Internal type for PRNG keys.
Key = typing.NewType("Key", Array)

RealArray = ArrayLike
IntegerArray = ArrayLike
# TODO: Import or define these to match
# https://github.com/numpy/numpy/blob/main/numpy/typing/_dtype_like.py.
DTypeLikeInt = DTypeLike
DTypeLikeUInt = DTypeLike
DTypeLikeFloat = DTypeLike
Shape = Sequence[int]

PRNGImpl = prng.PRNGImpl

UINT_DTYPES = prng.UINT_DTYPES


### utilities

def _check_broadcast_shapes(name: str, shape: tuple | Shape | None, *args: ArrayLike):
  arg_shapes = [np.shape(a) for a in args]
  if shape is None:
    if arg_shapes:
      shape = lax.broadcast_shapes(*arg_shapes)
    else:
      shape = ()
  else:
    shape = core.canonicalize_shape(shape)
    _check_shape(name, shape, *arg_shapes)
  return shape


def _check_all_safe_to_cast(name: str, dtype: DTypeLike, *args):
  for arg in args:
    if not dtypes.safe_to_cast(arg, dtype):
      raise dtypes.TypePromotionError(f"In arguments to {name}, cannot safely cast argument of type {jnp.asarray(arg).dtype} to {dtype}")


def _isnan(x: ArrayLike) -> Array:
  return lax.ne(x, x)


def _check_prng_key(name: str, key: Key | ArrayLike, *,
                    allow_batched: bool = False) -> tuple[Key, bool]:
  if isinstance(key, Array) and dtypes.issubdtype(key.dtype, dtypes.prng_key):
    wrapped_key = key
    wrapped = False
  elif _arraylike(key):
    # Call random_wrap here to surface errors for invalid keys.
    wrapped_key = prng.random_wrap(key, impl=default_prng_impl())
    wrapped = True
    if config.legacy_prng_key.value == config.LegacyPrngKeyState.ERROR:
      raise ValueError(
        'Legacy uint32 key array passed as key to jax.random function. '
        'Please create keys using jax.random.key(). If use of a raw key array '
        'was intended, set jax_legacy_prng_key="allow".')
    elif config.legacy_prng_key.value == config.LegacyPrngKeyState.WARN:
      warnings.warn(
        'Legacy uint32 key array passed as key to jax.random function. '
        'Please create keys using jax.random.key(). If use of a raw key array '
        'was intended, set jax_legacy_prng_key="allow".', stacklevel=2)
    elif config.enable_custom_prng.value:
      # TODO(jakevdp): possibly remove this warning condition.
      warnings.warn(
          'Raw arrays as random keys to jax.random functions are deprecated. '
          'Assuming valid threefry2x32 key for now.',
          FutureWarning)
  else:
    raise TypeError(f'unexpected PRNG key type {type(key)}')

  if (not allow_batched) and wrapped_key.ndim:
    raise ValueError(f"{name} accepts a single key, but was given a key array of"
                     f" shape {np.shape(key)} != (). Use jax.vmap for batching.")

  return Key(wrapped_key), wrapped


def _return_prng_keys(was_wrapped: bool, key: Key) -> Key:
  # TODO(frostig): remove once we always enable_custom_prng
  assert dtypes.issubdtype(key.dtype, dtypes.prng_key)
  if config.enable_custom_prng.value:
    return key
  else:
    return prng.random_unwrap(key) if was_wrapped else key


def _random_bits(
    key: Key, bit_width: int, shape: Shape,
    out_sharding: NamedSharding | P | None = None,
) -> Array:
  assert dtypes.issubdtype(key.dtype, dtypes.prng_key)
  return prng.random_bits(
      key, bit_width=bit_width, shape=shape, out_sharding=out_sharding)[0]


# TODO(frostig,vanderplas): remove from public API altogether, or at
# least change to return after asserting presence in `prng.prngs`
def default_prng_impl():
  """Get the default PRNG implementation.

  The default implementation is determined by ``config.jax_default_prng_impl``,
  which specifies it by name.
  """
  impl_name = config.default_prng_impl.value
  assert impl_name in prng.prngs, impl_name
  return prng.prngs[impl_name]


### key operations

# Wrapper around prng.PRNGImpl meant to hide its attributes from the
# public API.
# TODO(frostig,vanderplas): consider hiding all the attributes of
# PRNGImpl and directly returning it.
class PRNGSpec:
  """Specifies a PRNG key implementation."""

  __slots__ = ['_impl']
  _impl: PRNGImpl

  def __init__(self, impl):
    self._impl = impl

  def __repr__(self) -> str:
    return f"PRNGSpec({self._impl.name!r})"

  def __str__(self)  -> str:
    return str(self._impl)

  def __hash__(self) -> int:
    return hash(self._impl)

  def __eq__(self, other) -> bool:
    return isinstance(other, PRNGSpec) and self._impl == other._impl


# TODO(frostig,vanderplas): remove PRNGImpl from this union when it's
# no longer in the public API because `default_prng_impl` is gone
PRNGSpecDesc = str | PRNGSpec | PRNGImpl | Hashable
KeyDTypeLike = str | prng.KeyTy


def resolve_prng_impl(
    impl_spec: PRNGSpecDesc | KeyDTypeLike | None,
) -> PRNGImpl:
  if impl_spec is None:
    return default_prng_impl()
  if type(impl_spec) is PRNGImpl:
    # TODO(frostig,vanderplas): remove this case once we remove
    # default_prng_impl (and thus PRNGImpl) from the public API and
    # PRNGImpl from jex. We won't need to handle these then, and we
    # can remove them from the input type annotation above as well.
    return impl_spec
  if type(impl_spec) is PRNGSpec:
    return impl_spec._impl
  if isinstance(impl_spec, prng.KeyTy):
    return impl_spec._impl
  if type(impl_spec) is str:
    if impl_spec in prng.prngs:
      return prng.prngs[impl_spec]

    keys_fmt = ', '.join(f'"{s}"' for s in prng.prngs.keys())
    raise ValueError(f'unrecognized PRNG implementation "{impl_spec}". '
                     f'Did you mean one of: {keys_fmt}?')

  t = type(impl_spec)
  raise TypeError(f"unrecognized type {t} for specifying PRNG implementation.")


def key_dtype(impl_spec: PRNGSpecDesc | None = None) -> prng.KeyTy:
  """Get the dtype corresponding to a PRNG implementation."""
  return prng.KeyTy(resolve_prng_impl(impl_spec))


def _key(ctor_name: str, seed: int | ArrayLike,
         impl_spec: PRNGSpecDesc | None) -> Key:
  impl = resolve_prng_impl(impl_spec)
  if hasattr(seed, 'dtype') and dtypes.issubdtype(seed.dtype, dtypes.prng_key):
    raise TypeError(
        f"{ctor_name} accepts a scalar seed, but was given a PRNG key.")
  if np.ndim(seed):
    raise TypeError(
        f"{ctor_name} accepts a scalar seed, but was given an array of "
        f"shape {np.shape(seed)} != (). Use jax.vmap for batching")
  return Key(prng.random_seed(seed, impl=impl))

def key(seed: int | ArrayLike, *,
        impl: PRNGSpecDesc | None = None,
        dtype: KeyDTypeLike | None = None) -> Array:  # TODO(jakevdp): make this Key
  """Create a pseudo-random number generator (PRNG) key given an integer seed.

  The result is a scalar array containing a key, whose dtype indicates
  the default PRNG implementation, as determined by the optional
  ``dtype`` or ``impl`` argument or, otherwise, by the ``jax_default_prng_impl``
  config flag at the time when this function is called.

  Args:
    seed: a 64- or 32-bit integer used as the value of the key.
    impl: optional string specifying the PRNG implementation (e.g.
      ``'threefry2x32'``). Deprecated in favor of ``dtype``.
    dtype: optional dtype or string name specifying the PRNG implementation
      (e.g. ``jax.random.key_dtype('threefry2x32')`` or ``'threefry2x32'``).

  Returns:
    A scalar PRNG key array, consumable by random functions as well as ``split``
    and ``fold_in``.
  """
  if dtype is not None:
    if impl is not None:
      raise ValueError(
          "Cannot specify both `impl` and `dtype` arguments to jax.random.key")
    impl = dtype
  return _key('key', seed, impl)

def PRNGKey(seed: int | ArrayLike, *,
            impl: PRNGSpecDesc | None = None) -> Array:
  """Create a legacy PRNG key given an integer seed.

  This function produces old-style legacy PRNG keys, which are arrays
  of dtype ``uint32``. For more, see the note in the `PRNG keys
  <https://docs.jax.dev/en/latest/jax.random.html#prng-keys>`_
  section. When possible, :func:`jax.random.key` is recommended for
  use instead.

  The resulting key does not carry a PRNG implementation. The returned
  key matches the implementation given by the optional ``impl``
  argument or, otherwise, determined by the ``jax_default_prng_impl``
  config flag. Callers must ensure that same implementation is set as
  the default when passing this key as an argument to other functions
  (such as ``jax.random.split`` and ``jax.random.normal``).

  Args:
    seed: a 64- or 32-bit integer used as the value of the key.
    impl: optional string specifying the PRNG implementation (e.g.
      ``'threefry2x32'``)

  Returns:
    A PRNG key, consumable by random functions as well as ``split``
    and ``fold_in``.
  """
  return _return_prng_keys(True, _key('PRNGKey', seed, impl))


def fold_in(
    key: Key | ArrayLike,  # TODO(jakevdp): make this Key
    data: IntegerArray,
) -> Array:  # TODO(jakevdp): make this Key
  """Folds in data to a PRNG key to form a new PRNG key.

  Args:
    key: a PRNG key (from ``key``, ``split``, ``fold_in``).
    data: a 32-bit integer representing data to be folded into the key.

  Returns:
    A new PRNG key that is a deterministic function of the inputs and is
    statistically safe for producing a stream of new pseudo-random values.
  """
  key, wrapped = _check_prng_key("fold_in", key)
  if np.ndim(data):
    raise TypeError("fold_in accepts a scalar, but was given an array of"
                    f"shape {np.shape(data)} != (). Use jax.vmap for batching.")
  key_out = prng.random_fold_in(key, jnp.asarray(data, dtype='uint32'))
  return _return_prng_keys(wrapped, key_out)


def _split(key: Key, num: int | tuple[int, ...] = 2) -> Key:
  # Alternative to split() to use within random samplers.
  # TODO(frostig): remove and use split(); we no longer need to wait
  # to always enable_custom_prng
  assert dtypes.issubdtype(key.dtype, dtypes.prng_key)
  if key.ndim:
    raise TypeError("split accepts a single key, but was given a key array of "
                    f"shape {key.shape} != (). Use jax.vmap for batching.")
  shape = tuple(num) if isinstance(num, Sequence) else (num,)
  return prng.random_split(key, shape=shape)

def split(
    key: Key | ArrayLike,  # TODO(jakevdp): make this Key
    num: int | tuple[int, ...] = 2,
) -> Array:  # TODO(jakevdp): make this Key
  """Splits a PRNG key into `num` new keys by adding a leading axis.

  Args:
    key: a PRNG key (from ``key``, ``split``, ``fold_in``).
    num: optional, a positive integer (or tuple of integers) indicating
      the number (or shape) of keys to produce. Defaults to 2.

  Returns:
    An array-like object of `num` new PRNG keys.
  """
  typed_key, wrapped = _check_prng_key("split", key)
  return _return_prng_keys(wrapped, _split(typed_key, num))


def _key_impl(keys: Key) -> PRNGImpl:
  assert dtypes.issubdtype(keys.dtype, dtypes.prng_key)
  keys_dtype = typing.cast(prng.KeyTy, keys.dtype)
  return keys_dtype._impl

def _key_spec(keys: Key) -> str | PRNGSpec:
  impl = _key_impl(keys)
  return impl.name if impl.name in prng.prngs else PRNGSpec(impl)

def key_impl(keys: Key | ArrayLike) -> str | PRNGSpec:  # TODO(jakevdp): make this Key
  typed_keys, _ = _check_prng_key("key_impl", keys, allow_batched=True)
  return _key_spec(typed_keys)


def _key_data(keys: Key) -> Array:
  assert dtypes.issubdtype(keys.dtype, dtypes.prng_key)
  return prng.random_unwrap(keys)

def key_data(keys: Key | ArrayLike) -> Array:  # TODO(jakevdp): make this Key
  """Recover the bits of key data underlying a PRNG key array."""
  keys, _ = _check_prng_key("key_data", keys, allow_batched=True)
  return _key_data(keys)


def wrap_key_data(key_bits_array: Array, *,
                  impl: PRNGSpecDesc | None = None,
                  dtype: KeyDTypeLike | None = None) -> Array:  # TODO(jakevdp): make this Key
  """Wrap an array of key data bits into a PRNG key array.

  Args:
    key_bits_array: a ``uint32`` array with trailing shape corresponding to
      the key shape of the PRNG implementation specified by ``impl``.
    impl: optional, specifies a PRNG implementation, as in ``random.key``.
    dtype: optional dtype or string name specifying the PRNG implementation
      (e.g. ``jax.random.key_dtype('threefry2x32')`` or ``'threefry2x32'``).

  Returns:
    A PRNG key array, whose dtype is a subdtype of ``jax.dtypes.prng_key``
      corresponding to ``impl``, and whose shape equals the leading shape
      of ``key_bits_array.shape`` up to the key bit dimensions.

  Examples:
    Construct a key, and extract its data and dtype:

    >>> import jax
    >>> key = jax.random.key(42)
    >>> data = jax.random.key_data(key)
    >>> dtype = key.dtype

    Reconstruct an equivalent key with :func:`wrap_key_data`:

    >>> new_key = jax.random.wrap_key_data(data, dtype=dtype)
    >>> key == new_key
    Array(True, dtype=bool)
  """
  if dtype is not None:
    if impl is not None:
      raise ValueError(
          "Cannot specify both `impl` and `dtype` arguments to"
          " jax.random.wrap_key_data")
    impl = dtype
  impl_obj = resolve_prng_impl(impl)
  return prng.random_wrap(key_bits_array, impl=impl_obj)


### random samplers


def _check_shape(name: str, shape: Shape, *param_shapes) -> None:
  if param_shapes:
    shape_ = lax.broadcast_shapes(shape, *param_shapes)  # pyrefly: ignore[no-matching-overload]
    if shape != shape_:
      msg = ("{} parameter shapes must be broadcast-compatible with shape "
             "argument, and the result of broadcasting the shapes must equal "
             "the shape argument, but got result {} for shape argument {}.")
      raise ValueError(msg.format(name, shape_, shape))


def _prepend_none_sharding(
    out_sharding: NamedSharding | None,
) -> NamedSharding | None:
  if out_sharding is None:
    return None
  spec = out_sharding.spec
  return out_sharding.update(
      spec=spec.update(partitions=(None, *spec.partitions)))


def canonicalize_sharding_for_samplers(
    out_sharding: NamedSharding | P | None, name: str, shape: Shape
) -> NamedSharding | None:
  s = canonicalize_sharding(out_sharding, name)
  if s is None:
    return None
  if s.spec.unreduced or s.spec.reduced:
    raise NotImplementedError(
        f"unreduced/reduced out_sharding for {name} is not implemented.")
  if not s.mesh._any_axis_explicit:
    return None
  s.spec._check_compatible_wrt_shape(shape)
  spec = s.spec._normalized_spec_for_aval(len(shape))
  if config.remove_size_one_mesh_axis_from_type.value:
    spec = remove_size_one_mesh_axis_from_spec(spec, s.mesh)
  return s.update(spec=spec)


def _as_axis_tuple(spec_elem) -> tuple[typing.Any, ...]:
  if spec_elem is None:
    return ()
  return spec_elem if isinstance(spec_elem, tuple) else (spec_elem,)


def _is_sub_spec(ps, os) -> bool:
  ps_t, os_t = _as_axis_tuple(ps), _as_axis_tuple(os)
  return ps_t == os_t[:len(ps_t)]


def _align_params(shape: Shape, out_sharding: NamedSharding | None, *params):
  if out_sharding is None:
    return params
  out_s = core.get_sharding(out_sharding, shape)
  out_parts = out_s.spec.partitions
  aligned = []
  for p in params:
    aval = core.typeof(p)
    s = aval.sharding
    if s.mesh.empty or not aval.ndim or aval.ndim > len(shape):
      aligned.append(p)
      continue
    out_tail = out_parts[len(shape) - aval.ndim:]
    cur_parts = s.spec.partitions + (None,) * (
        aval.ndim - len(s.spec.partitions))
    new_parts = tuple(
        os if (ps is not None and ps != os and not core.definitely_equal(d, 1))
        else ps
        for d, ps, os in zip(aval.shape, cur_parts, out_tail)
    )
    if new_parts != cur_parts:
      p = reshard(p, out_s.update(spec=P(*new_parts)))
    aligned.append(p)
  return tuple(aligned)


def _check_param_sharding(
    name: str, shape: Shape, out_sharding: NamedSharding | None, *params
) -> None:
  if out_sharding is None:
    return
  out_s = core.get_sharding(out_sharding, shape)
  out_parts = out_s.spec.partitions
  for p in params:
    aval = core.typeof(p)
    s = aval.sharding
    if s.mesh.empty or aval.ndim > len(shape):
      continue
    if (s.mesh != out_s.mesh and
        (s.mesh._any_axis_explicit or out_s.mesh._any_axis_explicit)):
      raise core.ShardingTypeError(
          f"{name} parameter mesh {s.mesh} must match out_sharding mesh"
          f" {out_s.mesh}.")
    out_tail = out_parts[len(shape) - aval.ndim:]
    if any(not _is_sub_spec(ps, os)
           for ps, os in zip(s.spec.partitions, out_tail)):
      raise core.ShardingTypeError(
          f"{name} parameter sharding {s.spec} is incompatible with"
          f" out_sharding {out_s.spec}.")


def _sample_sharding(
    name: str, shape: Shape, out_sharding: NamedSharding | None, *params
) -> NamedSharding | None:
  if out_sharding is not None:
    _check_param_sharding(name, shape, out_sharding, *params)
    return out_sharding
  if core.trace_state_clean() and mesh_lib.get_concrete_mesh().empty:
    return None
  avals = []
  for p in params:
    aval = p if isinstance(p, core.ShapedArray) else core.typeof(p)
    s = aval.sharding
    if not s.mesh._any_axis_explicit or aval.ndim > len(shape):
      continue
    s = core.get_sharding(s, aval.shape)
    if all(x is None for x in s.spec.partitions):
      continue
    pad = len(shape) - aval.ndim
    spec = P(*((None,) * pad + s.spec.partitions))
    avals.append(core.ShapedArray(
        (1,) * pad + aval.shape, aval.dtype, sharding=s.update(spec=spec)))
  if not avals:
    return None
  if len(avals) > 1:
    specs = [a.sharding.spec.partitions + (None,) * (
        len(shape) - len(a.sharding.spec.partitions)) for a in avals]
    refined_specs = []
    for dim_specs in zip(*specs):
      non_none = [s for s in dim_specs if s is not None]
      finest = max(
          non_none, key=lambda s: len(_as_axis_tuple(s)), default=None)
      if not non_none:
        refined_specs.append(None)
      elif finest is not None and all(_is_sub_spec(s, finest) for s in non_none):
        refined_specs.append(finest)
      else:
        refined_specs = []
        break
    if refined_specs:
      avals = [
          a.update(sharding=a.sharding.update(spec=P(*(
              rs if ps is not None else None
              for ps, rs in zip(sp, refined_specs)))))
          for a, sp in zip(avals, specs)
      ]
  try:
    return lax.broadcasting_sharding_rule(name, *avals)
  except lax.lax_utils.DuplicateSpecError as e:
    raise core.ShardingTypeError(str(e)) from e


def _make_shard_slicer(mesh, in_aval, target_parts):
  in_parts = in_aval.sharding.spec.partitions
  in_parts = in_parts + (None,) * (in_aval.ndim - len(in_parts))
  target_parts = target_parts + (None,) * (in_aval.ndim - len(target_parts))
  rem_shape, rem_in, rem_target = [], [], []
  for d, in_p, p in zip(in_aval.shape, in_parts, target_parts):
    if core.definitely_equal(d, 1) or in_p == p:
      rem_shape.append(d)
      rem_in.append(in_p)
      rem_target.append(in_p)
    elif in_p is not None and _is_sub_spec(in_p, p):
      in_t, out_t = _as_axis_tuple(in_p), _as_axis_tuple(p)
      rem = out_t[len(in_t):]
      rem_shape.append(d // math.prod(mesh.shape[a] for a in in_t))
      rem_in.append(None)
      rem_target.append(rem[0] if len(rem) == 1 else rem)
    else:
      rem_shape.append(d)
      rem_in.append(in_p)
      rem_target.append(p)
  in_aval = in_aval.update(
      shape=tuple(rem_shape), sharding=NamedSharding(mesh, P(*rem_in)))
  target_aval = in_aval.update(sharding=NamedSharding(mesh, P(*rem_target)))
  with use_abstract_mesh(mesh):
    rule = lax._naryop_collective_rule([in_aval], [target_aval])
  if rule.in_collective is None:
    return lambda x: x
  return lambda x: rule.in_collective(x)[0]


def _check_unsharded_trailing_axis(
    name: str, out_shape: Shape, out_sharding: NamedSharding | None, param,
    *, param_has_trailing: bool,
):
  out_s = (core.get_sharding(out_sharding, out_shape)
           if out_sharding is not None else
           _sample_sharding(name, out_shape, None, param))
  p_s = core.typeof(param).sharding
  p_parts = p_s.spec.partitions + (None,) * (
      np.ndim(param) - len(p_s.spec.partitions))
  has_trail = param_has_trailing or np.ndim(param) == len(out_shape)
  if ((out_s is not None and out_s.spec.partitions[-1] is not None) or
      (not p_s.mesh.empty and has_trail and p_parts and
       p_parts[-1] is not None)):
    raise core.ShardingTypeError(
        f"{name} cannot shard the trailing dimension.")
  _check_param_sharding(name, out_shape, out_s, param)
  batch_s = (None if out_s is None else out_s.update(
      spec=out_s.spec.update(partitions=out_s.spec.partitions[:-1])))
  return out_s, batch_s


@typing.overload
def bits(key: Key | ArrayLike,
         shape: Shape = (),
         dtype: DTypeLikeUInt | None = None,
         *,
         num_streams: None = None,
         out_sharding: NamedSharding | P | None = None) -> Array:
  ...

@typing.overload
def bits(key: Key | ArrayLike,
         shape: Shape = (),
         dtype: DTypeLikeUInt | None = None,
         *,
         num_streams: int,
         out_sharding: NamedSharding | P | None = None) -> tuple[Array, ...]:
  ...

@typing.overload
def bits(key: Key | ArrayLike,
         shape: Shape = (),
         dtype: DTypeLikeUInt | None = None,
         *,
         num_streams: int | None = None,
         out_sharding: NamedSharding | P | None = None) -> Array | tuple[Array, ...]:
  ...

def bits(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
         shape: Shape = (),
         dtype: DTypeLikeUInt | None = None,
         *,
         num_streams: int | None = None,
         out_sharding: NamedSharding | P | None = None) -> Array | tuple[Array, ...]:
  """Sample uniform bits in the form of unsigned integers.

  Args:
    key: a PRNG key used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ``()``.
    dtype: optional, an unsigned integer dtype for the returned values (default
      ``uint64`` if ``jax_enable_x64`` is true, otherwise ``uint32``).
    num_streams: optional, the number of independent random streams to return.
      If ``None`` (default), returns a single array. If an integer (currently
      supported values are ``1``, ``2``, or ``4``), returns a tuple of
      ``num_streams`` independent random arrays of the given shape and dtype.
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified shape and dtype, or a tuple of
    ``num_streams`` such arrays if ``num_streams`` is specified.
  """
  key, _ = _check_prng_key("bits", key)
  if dtype is None:
    dtype = dtypes.default_uint_dtype()
  else:
    dtype = dtypes.check_and_canonicalize_user_dtype(dtype)
  if not dtypes.issubdtype(dtype, np.unsignedinteger):
    raise ValueError("dtype argument to `bits` must be an unsigned int dtype, "
                     f"got {dtype}")
  if isinstance(num_streams, bool) or num_streams not in (None, 1, 2, 4):
    raise ValueError(f"num_streams must be None, 1, 2, or 4, got {num_streams}")
  shape = core.canonicalize_shape(shape)
  out_sharding = canonicalize_sharding(out_sharding, "bits")
  bit_width = dtype.itemsize * 8
  num_s = 1 if num_streams is None else num_streams
  out = prng.random_bits(key, bit_width, shape, num_s, out_sharding)
  return out[0] if num_streams is None else out


def uniform(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
            shape: Shape = (),
            dtype: DTypeLikeFloat | None = None,
            minval: RealArray = 0.,
            maxval: RealArray = 1.,
            *,
            out_sharding: NamedSharding | P | None = None) -> Array:
  """Sample uniform random values in [minval, maxval) with given shape/dtype.

  Args:
    key: a PRNG key used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    minval: optional, a minimum (inclusive) value broadcast-compatible with shape for the range (default 0).
    maxval: optional, a maximum (exclusive) value broadcast-compatible with shape for the range (default 1).
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified shape and dtype.
  """
  key, _ = _check_prng_key("uniform", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  shape = core.canonicalize_shape(shape)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `uniform` must be a float dtype, "
                     f"got {dtype}")
  out_sharding = canonicalize_sharding(out_sharding, "uniform")
  return _uniform(key, minval, maxval, shape, dtype, out_sharding)


@jit(static_argnums=(3, 4, 5))
def _uniform(
    key: Key, minval, maxval, shape, dtype, out_sharding=None
) -> Array:
  _check_shape("uniform", shape)
  if not dtypes.issubdtype(dtype, np.floating):
    raise TypeError("uniform only accepts floating point dtypes.")

  minval = lax.convert_element_type(minval, dtype)
  maxval = lax.convert_element_type(maxval, dtype)
  sample_s = _sample_sharding("uniform", shape, out_sharding, minval, maxval)
  minval, maxval = _align_params(shape, sample_s, minval, maxval)
  minval = lax.broadcast_to_rank(minval, len(shape))
  maxval = lax.broadcast_to_rank(maxval, len(shape))

  finfo = dtypes.finfo(dtype)
  nbits, nmant = finfo.bits, finfo.nmant

  if nbits not in (8, 16, 32, 64):
    raise TypeError(
        f"uniform only accepts 8-, 16-, 32-, or 64-bit dtypesgot {dtype}."
    )

  rng_bits = nbits
  if nmant < 8:
    rng_bits = 8
  bits = _random_bits(key, rng_bits, shape, out_sharding=sample_s)

  if config.legacy_random_uniform.value:
    uint_dtype = UINT_DTYPES[nbits]
    if rng_bits != nbits:
      bits = lax.convert_element_type(bits, uint_dtype)

    # The strategy here is to randomize only the mantissa bits with an exponent
    # of 1 (after applying the bias), then shift and scale to the desired range.
    # The bit-level transformation we use relies on Numpy and XLA having
    # bit-for-bit equivalent float representations, which might not be true on
    # all platforms.
    float_bits = lax.shift_right_logical(
        bits, jnp.array(rng_bits - nmant, uint_dtype)
    )
    float_bits = lax.bitwise_or(
        float_bits,
        # The double cast is because the TPU backend does not implement `view`
        # on float64 values => do the `view` in NumPy first, but then ensure
        # that we have a JAX array that won't be canonicalized further.
        jnp.asarray(
            np.array(1.0, dtype).view(float_bits.dtype), dtype=float_bits.dtype
        ),
    )
    floats = lax.bitcast_convert_type(float_bits, dtype) - jnp.array(1.0, dtype)
  else:
    # Convert (nmant + 1)-bit random integers to floats and scale by
    # 2**-(nmant + 1) to sample uniformly on [0, 1) with spacing 2**-(nmant + 1).
    n_float_bits = nmant + 1
    float_bits = lax.shift_right_logical(
        bits, jnp.array(rng_bits - n_float_bits, bits.dtype)
    )
    floats = (
        lax.convert_element_type(float_bits, dtype)
        * jnp.array(2.0 ** -n_float_bits, dtype)
    )
  return lax.max(
      minval,
      lax.reshape(floats * (maxval - minval) + minval, shape))


def _convert_and_clip_integer(val: Array, dtype: DType) -> Array:
  """
  Convert integer-typed val to specified integer dtype, clipping to dtype
  range rather than wrapping.

  Args:
    val: value to be converted
    dtype: dtype of output

  Returns:
    equivalent of val in new dtype

  Examples
  --------
  Normal integer type conversion will wrap:

  >>> val = jnp.uint32(0xFFFFFFFF)
  >>> val.astype('int32')
  Array(-1, dtype=int32)

  This function clips to the values representable in the new type:

  >>> _convert_and_clip_integer(val, 'int32')
  Array(2147483647, dtype=int32)
  """
  assert isinstance(val, Array)
  if not (dtypes.issubdtype(dtype, np.integer) and dtypes.issubdtype(val.dtype, np.integer)):
    raise TypeError("_convert_and_clip_integer only accepts integer dtypes.")

  min_val = lax._const(val, max(dtypes.iinfo(dtype).min,
                                dtypes.iinfo(val.dtype).min))
  max_val = lax._const(val, min(dtypes.iinfo(dtype).max,
                                 dtypes.iinfo(val.dtype).max))
  return jnp.clip(val, min_val, max_val).astype(dtype)


def randint(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
            shape: Shape,
            minval: IntegerArray,
            maxval: IntegerArray,
            dtype: DTypeLikeInt | None = None,
            *,
            out_sharding: NamedSharding | P | None = None) -> Array:
  """Sample uniform random values in [minval, maxval) with given shape/dtype.

  Args:
    key: a PRNG key used as the random key.
    shape: a tuple of nonnegative integers representing the shape.
    minval: int or array of ints broadcast-compatible with ``shape``, a minimum
      (inclusive) value for the range.
    maxval: int or array of ints broadcast-compatible with ``shape``, a maximum
      (exclusive) value for the range.
    dtype: optional, an int dtype for the returned values (default int64 if
      jax_enable_x64 is true, otherwise int32).
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified shape and dtype.

  .. note::

     :func:`randint` uses a modulus-based computation that is known to produce
     slightly biased values in some cases. The magnitude of the bias scales as
     ``(maxval - minval) * ((2 ** nbits ) % (maxval - minval)) / 2 ** nbits``:
     in words, the bias goes to zero when ``(maxval - minval)`` is a power of 2,
     and otherwise the bias will be small whenever ``(maxval - minval)`` is
     small compared to the range of the sampled type.

     To reduce this bias, 8-bit and 16-bit values will always be sampled at 32-bit and
     then cast to the requested type. If you find yourself sampling values for which
     this bias may be problematic, a possible alternative is to sample via uniform::

       def randint_via_uniform(key, shape, minval, maxval, dtype):
         u = jax.random.uniform(key, shape, minval=minval - 0.5, maxval=maxval - 0.5)
         return u.round().astype(dtype)

     But keep in mind this method has its own biases due to floating point rounding
     errors, and in particular there may be some integers in the range
     ``[minval, maxval)`` that are impossible to produce with this approach.
  """
  key, _ = _check_prng_key("randint", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      int if dtype is None else dtype)
  shape = core.canonicalize_shape(shape)
  if not dtypes.issubdtype(dtype, np.integer):
    raise TypeError(f"randint only accepts integer dtypes, got {dtype}")

  out_sharding = canonicalize_sharding(out_sharding, "randint")
  info = dtypes.iinfo(dtype)
  dtype_for_sampling = dtype
  if info.bits < 32:
    # Sample in 32 bits to avoid biased results.
    dtype_for_sampling = np.dtype('int32')
    minval = jnp.asarray(minval).astype('int32').clip(info.min, info.max)
    maxval = jnp.asarray(maxval).astype('int32').clip(info.min, info.max + 1)

  return _randint(
      key, minval, maxval, shape, dtype_for_sampling, out_sharding
  ).astype(dtype)


@jit(static_argnums=(3, 4, 5))
def _randint(
    key: Key, minval, maxval, shape, dtype, out_sharding=None
) -> Array:
  # We have three imperfect options for generating random integers in an arbitrary
  # user-specified range:
  #
  # 1. Rejection sampling. This produces unbiased results, but involves a dynamic
  #    number of iterations, so it's not suitable for computation on accelerators.
  # 2. Generate floating point values between minval and maxval, and cast to int.
  #    This introduces bias for large ranges due to floating point rounding error:
  #    many integers within a given range would never be sampled.
  # 3. Generate numbers in a range that is a power of 2, and use an integer modulus
  #    to shift them into the desired range. This produces a biased distribution
  #    when the desired range is not a power of 2, which scales as
  #    O[bias] ~= (desired_range ** 2) / full_range.
  #
  # Given these three imperfect options, we opt for a modified version of (3), where we
  # sample 2 * nbits bits per value, because it is efficient and works well in most cases
  # of interest. To help users avoid inadvertently producing biased results, we always
  # generate samples in at least 32 bits.
  _check_shape("randint", shape, np.shape(minval), np.shape(maxval))
  if not dtypes.issubdtype(dtype, np.integer):
    raise TypeError(f"randint only accepts integer dtypes, got {dtype}")

  check_arraylike("randint", minval, maxval)
  minval = jnp.asarray(minval)
  maxval = jnp.asarray(maxval)
  sample_s = _sample_sharding("randint", shape, out_sharding, minval, maxval)
  minval, maxval = _align_params(shape, sample_s, minval, maxval)
  if not dtypes.issubdtype(minval.dtype, np.integer):
    minval = minval.astype(int)
  if not dtypes.issubdtype(maxval.dtype, np.integer):
    maxval = maxval.astype(int)

  # Flag where maxval is greater than the maximum value of dtype
  # in order to handle cases like randint(key, shape, 0, 256, 'uint8')
  maxval_out_of_range = lax.gt(
    maxval, _convert_and_clip_integer(jnp.array(dtypes.iinfo(dtype).max, dtype), maxval.dtype))

  minval = _convert_and_clip_integer(minval, dtype)
  maxval = _convert_and_clip_integer(maxval, dtype)
  minval = lax.broadcast_to_rank(minval, len(shape))
  maxval = lax.broadcast_to_rank(maxval, len(shape))
  nbits = dtypes.iinfo(dtype).bits

  if nbits not in (8, 16, 32, 64):
    raise TypeError(f"randint only accepts 8-, 16-, 32-, or 64-bit dtypes, got {dtype}")

  # This algorithm is biased whenever (maxval - minval) is not a power of 2.
  # We generate double the number of random bits required by the dtype so as to
  # reduce that bias.
  k1, k2 = _split(key)
  rbits = lambda key: _random_bits(key, nbits, shape, out_sharding=sample_s)
  higher_bits, lower_bits = rbits(k1), rbits(k2)

  unsigned_dtype = UINT_DTYPES[nbits]
  span = lax.convert_element_type(maxval - minval, unsigned_dtype)

  # Ensure that span=1 when maxval <= minval, so minval is always returned;
  # https://github.com/jax-ml/jax/issues/222
  span = lax.select(maxval <= minval, lax.full_like(span, 1), span)

  # When maxval is out of range, the span has to be one larger.
  # If span is already the maximum representable value, this will wrap to zero,
  # causing remainders below to have no effect, which is the correct semantics.
  span = lax.select(
    maxval_out_of_range & (maxval > minval),
    lax.add(span, lax._const(span, 1)),
    span)

  # To compute a remainder operation on an integer that might have twice as many
  # bits as we can represent in the native unsigned dtype, we compute a
  # multiplier equal to 2**nbits % span. To avoid overflow, we use the identity:
  #  (a * b) % N = [(a % N) * (b % N)] % N
  multiplier = lax.rem(lax._const(span, 2 ** (nbits // 2)), span)
  multiplier = lax.rem(lax.mul(multiplier, multiplier), span)

  random_offset = lax.add(lax.mul(lax.rem(higher_bits, span), multiplier),
                          lax.rem(lower_bits, span))
  random_offset = lax.rem(random_offset, span)
  return lax.add(minval, lax.convert_element_type(random_offset, dtype))


def permutation(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
                x: int | ArrayLike,
                axis: int = 0,
                independent: bool = False,
                *,
                out_sharding: NamedSharding | P | None = None) -> Array:
  """Returns a randomly permuted array or range.

  Args:
    key: a PRNG key used as the random key.
    x: int or array. If x is an integer, randomly shuffle np.arange(x).
      If x is an array, randomly shuffle its elements.
    axis: int, optional. The axis which x is shuffled along. Default is 0.
    independent: bool, optional. If set to True, each individual vector along
      the given axis is shuffled independently. Default is False.
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A shuffled version of x or array range
  """
  key, _ = _check_prng_key("permutation", key)
  check_arraylike("permutation", x)
  axis = canonicalize_axis(axis, np.ndim(x) or 1)
  if not np.ndim(x):
    if not np.issubdtype(lax.dtype(x), np.integer):
      raise TypeError("x must be an integer or at least 1-dimensional")
    r = core.concrete_or_error(
        int, x, "argument x of jax.random.permutation()")
    out_s = canonicalize_sharding_for_samplers(out_sharding, "permutation", (r,))
    out = _shuffle(key, jnp.arange(r), axis)
    return reshard(out, out_s) if out_s is not None else out
  return _permutation(
      key, x, axis=axis, independent=independent, out_sharding=out_sharding)


def _permutation(key: Key, x, axis, independent, out_sharding=None):
  x = jnp.asarray(x)
  out_s = canonicalize_sharding_for_samplers(
      out_sharding, "permutation", x.shape)
  x_s = core.typeof(x).sharding
  if not x_s.mesh.empty and x_s.spec.partitions[axis] is not None:
    raise core.ShardingTypeError(
        f"permutation cannot shuffle along sharded axis {axis}; got"
        f" {x_s.spec}.")
  _check_param_sharding("permutation", x.shape, out_s, x)
  if independent or x.ndim == 1:
    if out_s is not None:
      sort_parts = (
          *out_s.spec.partitions[:axis], None,
          *out_s.spec.partitions[axis + 1:])
      sort_sharding = out_s.update(spec=P(*sort_parts))
      if core.typeof(x).sharding != sort_sharding:
        x = reshard(x, sort_sharding)
      out = _shuffle(key, x, axis, sort_sharding=sort_sharding)
      return (reshard(out, out_s)
              if out_s.spec.partitions[axis] is not None else out)
    sort_sharding = _sample_sharding("permutation", x.shape, None, x)
    return _shuffle(key, x, axis, sort_sharding=sort_sharding)
  ind = _shuffle(key, jnp.arange(x.shape[axis]), 0)
  if out_s is None:
    return jnp.take(x, ind, axis, unique_indices=True)
  x_parts = (
      *out_s.spec.partitions[:axis], None, *out_s.spec.partitions[axis + 1:])
  x_target_s = out_s.update(spec=P(*x_parts))
  if core.typeof(x).sharding != x_target_s:
    x = reshard(x, x_target_s)
  if out_s.spec.partitions[axis] is not None:
    ind = reshard(ind, out_s.update(spec=P(out_s.spec.partitions[axis])))
  idx = (*((slice(None),) * axis), ind)
  return x.at[idx].get(unique_indices=True, out_sharding=out_s)


@jit(static_argnums=(2, 3))
def _shuffle(key: Key, x, axis, sort_sharding=None) -> Array:
  # On parallel architectures, Fisher-Yates is more expensive than doing
  # multiple sorts. This algorithm is based on one developed and analyzed by
  # tjablin@. We sort according to randomly-generated 32bit keys, but those keys
  # may have collisions. If we repeat the process, using fresh 32bit keys for
  # each sort, then whenever all pairs of elements have been assigned distinct
  # keys at some iteration (or equivalently when the strings formed by
  # concatenating the successive keys for each element are all distinct) then we
  # are guaranteed to have a perfect sample (assuming that either the sort is
  # stable or that any bias is not value-dependent). Since checking uniqueness
  # at runtime may be expensive, we use a heuristic static stop criterion
  # developed by tjablin@. See tensorflow/compiler/tf2xla/random_ops.cc for more
  # info, and for the original implementation of this algorithm. See also
  # Section 2 of http://people.csail.mit.edu/costis/6896sp11/lec5s.pdf for
  # another analysis (where the keys are generated one bit at a time).
  exponent = 3  # see tjablin@'s analysis for explanation of this parameter
  uint32max = dtypes.iinfo(np.uint32).max
  if not core.is_constant_dim(x.size):
    raise NotImplementedError(
        "shape polymorphism for `permutation` or `shuffle`"
        f" for arrays of non-constant size: {x.size}")
  key, (x,) = random_insert_pvary("permutation", key, x)
  num_rounds = int(
      np.ceil(exponent * np.log(max(1, x.size)) / np.log(uint32max)))

  for _ in range(num_rounds):
    key, subkey = _split(key)
    sort_keys = _random_bits(subkey, 32, x.shape, out_sharding=sort_sharding)
    _, x = lax.sort_key_val(sort_keys, x, axis)

  return x


def choice(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
           a: int | ArrayLike,
           shape: Shape = (),
           replace: bool = True,
           p: RealArray | None = None,
           axis: int = 0,
           mode: str | None = None) -> Array:
  """Generates a random sample from a given array.

  .. warning::
    If ``p`` has fewer non-zero elements than the requested number of samples,
    as specified in ``shape``, and ``replace=False``, the output of this
    function is ill-defined. Please make sure to use appropriate inputs.

  Args:
    key: a PRNG key used as the random key.
    a : array or int. If an ndarray, a random sample is generated from
      its elements. If an int, the random sample is generated as if a were
      arange(a).
    shape : tuple of ints, optional. Output shape.  If the given shape is,
      e.g., ``(m, n)``, then ``m * n`` samples are drawn.  Default is (),
      in which case a single value is returned.
    replace : boolean.  Whether the sample is with or without replacement.
      Default is True.
    p : 1-D array-like, The probabilities associated with each entry in a.
      If not given the sample assumes a uniform distribution over all
      entries in a.
    axis: int, optional. The axis along which the selection is performed.
      The default, 0, selects by row.
    mode: optional, "high" or "low" for how many bits to use in the gumbel sampler
      when `p is None` and `replace = False`. The default is determined by the
      ``use_high_dynamic_range_gumbel`` config, which defaults to "low". With mode="low",
      in float32 sampling will be biased for choices with probability less than about
      1E-7; with mode="high" this limit is pushed down to about 1E-14. mode="high"
      approximately doubles the cost of sampling.

  Returns:
    An array of shape `shape` containing samples from `a`.
  """
  key, _ = _check_prng_key("choice", key)
  if not isinstance(shape, Sequence):
    raise TypeError("shape argument of jax.random.choice must be a sequence, "
                    f"got {shape}")
  check_arraylike("choice", a)
  arr = jnp.asarray(a)
  if arr.ndim == 0:
    n_inputs = core.concrete_or_error(int, a, "The error occurred in jax.random.choice()")
  else:
    axis = canonicalize_axis(axis, arr.ndim)
    n_inputs = arr.shape[axis]
  n_draws = math.prod(shape)
  out_shape = (tuple(shape) if arr.ndim == 0 else
               arr.shape[:axis] + tuple(shape) + arr.shape[axis + 1:])
  if n_draws == 0:
    return jnp.zeros(out_shape, dtype=arr.dtype)
  if n_inputs <= 0:
    raise ValueError("a must be greater than 0 unless no samples are taken")
  if not replace and n_draws > n_inputs:
    raise ValueError(
        f"Cannot take a larger sample (size {n_draws}) than "
        f"population (size {n_inputs}) when 'replace=False'")

  if p is None:
    if replace:
      ind = randint(key, shape, 0, n_inputs)
      result = ind if arr.ndim == 0 else jnp.take(arr, ind, axis)
    else:
      slices = (slice(None),) * axis + (slice(n_draws),)
      result = permutation(key, n_inputs if arr.ndim == 0 else arr, axis)[slices]
  else:
    check_arraylike("choice", p)
    p_arr, = promote_dtypes_inexact(p)
    if p_arr.shape != (n_inputs,):
      raise ValueError(
          "p must be None or a 1D vector with the same size as a.shape[axis]. "
          f"p has shape {p_arr.shape} and a.shape[axis] is {n_inputs}.")
    if replace:
      p_cuml = jnp.cumsum(p_arr)
      r = p_cuml[-1] * (1 - uniform(key, shape, dtype=p_cuml.dtype))
      ind = jnp.searchsorted(p_cuml, r).astype(int)
    else:
      # Gumbel top-k trick: https://timvieira.github.io/blog/post/2019/09/16/algorithms-for-sampling-without-replacement/
      g = gumbel(key, (n_inputs,), dtype=p_arr.dtype, mode=mode) + jnp.log(p_arr)
      ind = lax.top_k(g, k=n_draws)[1].astype(int)
    result = ind if arr.ndim == 0 else jnp.take(arr, ind, axis)

  return result.reshape(out_shape)


def normal(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
           shape: Shape = (),
           dtype: DTypeLikeFloat | None = None,
           *,
           mode: str | None = None,
           out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample standard normal random values with given shape and float dtype.

  The values are returned according to the probability density function:

  .. math::
     f(x) = \frac{1}{\sqrt{2\pi}}e^{-x^2/2}

  on the domain :math:`-\infty < x < \infty`

  Args:
    key: a PRNG key used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    mode: optional, "low", "high", or "highest" for how many bits of entropy to
      convert into floating-point samples. Defaults to ``jax_random_normal_mode``
      config ("low").
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified shape and dtype.
  """
  key, _ = _check_prng_key("normal", key)
  shape = core.canonicalize_shape(shape)
  if mode is None:
    mode = config.random_normal_mode.value
  if mode not in ("low", "high", "highest"):
    raise ValueError(f"Must provide valid mode for normal, got: {mode}")
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.inexact):
    raise ValueError(f"dtype argument to `normal` must be a float or complex dtype, "
                     f"got {dtype}")
  out_sharding = canonicalize_sharding(out_sharding, "normal")
  return _normal(key, shape, dtype, mode=mode, out_sharding=out_sharding)


def _signed_midpoint_and_complement(
    bits: Array | tuple[Array, Array], compute_dtype: DType, mode: str
) -> tuple[Array, Array, Array]:
  """Convert random integer bits into `(x, |x|, 1 - |x|)` for `x` uniform on `(-1, 1)`.

  To sample from a normal distribution, we map a uniform random value `x` in
  `(-1, 1)` through `sqrt(2) * erfinv(x)`. If `x` is converted to float first,
  values close to `+/-1` (which map into the tails of the normal distribution)
  round to `1.0` or lose precision when subtracting `1 - |x|` in floating point.

  Instead, this helper computes both `|x|` (used near `0`) and `q = 1 - |x|`
  (used near `|x| = 1`) directly from the integer bits before converting to
  float.
  """
  w = dtypes.finfo(compute_dtype).bits
  uint_dtype = UINT_DTYPES[w]
  if mode == "high":
    assert not isinstance(bits, tuple)
    bits_u = lax.convert_element_type(bits, uint_dtype)
    input_bits = dtypes.iinfo(bits.dtype).bits
    if input_bits < w:
      bits_u = lax.shift_left(
          bits_u, np.array(w - input_bits, dtype=uint_dtype)
      )
    sign_bit = lax.bitwise_and(
        bits_u, np.array(1 << (w - 1), dtype=uint_dtype)
    )
    # Shift left by 1 to drop `sign_bit`, leaving `2 * k` in bits `1 .. w-1`
    # (where `k` is the `(w - 1)`-bit integer in `0 .. 2^(w-1) - 1`).
    two_k = lax.shift_left(bits_u, np.array(1, dtype=uint_dtype))

    # Map `k` to `(k + 0.5) / 2^(w-1) = (2 * k + 1) * 2^-w`:
    # - at `k = 0`, `+x` and `-x` land at `+2^-w` and `-2^-w` instead of
    #   mapping both `+0` and `-0` to `0.0`;
    # - at `k = 2^(w-1) - 1`, `1 - |x| = 2^-w > 0`, avoiding `erfinv(1) = inf`.
    abs_x_int = lax.bitwise_or(two_k, np.array(1, dtype=uint_dtype))

    # Since `|x| = (2 * k + 1) * 2^-w`, `1 - |x| = ((2^w - 1) - 2 * k) * 2^-w`,
    # whose numerator `(2^w - 1) - two_k` is the bitwise complement `~two_k`.
    q_int = lax.bitwise_not(two_k)

    # Converting an unsigned integer to float rounds it to the nearest float,
    # and multiplying by `2^-w` shifts its exponent down by `w`, so `abs_x`
    # and `q` round to the nearest floats to `|x|` and `1 - |x|`.
    #
    # `erf_inv_core` computes `1 - x^2` as `q * (1 + |x|)`:
    # - Near `|x| = 1`, `abs_x` rounds up to `1.0`, giving `1 + |x| == 2.0`,
    #   while `q` stays positive (`q_int >= 1`), avoiding `erfinv(1) = inf`.
    # - Near `x = 0`, `q` rounds up to `1.0`, giving `q * (1 + |x|) == 1.0`,
    #   while `erf_inv_core` scales its result by `x` so small values stay
    #   distinct.
    scale = np.array(2.0 ** -w, compute_dtype)
    abs_x = lax.convert_element_type(abs_x_int, compute_dtype) * scale
    q = lax.convert_element_type(q_int, compute_dtype) * scale
  elif mode == "highest":
    assert isinstance(bits, tuple) and len(bits) == 2
    hi, lo = bits
    sign_bit = lax.bitwise_and(hi, np.array(1 << (w - 1), dtype=uint_dtype))
    # Same construction across two `w`-bit words `(hi, lo)`: shift left by 1
    # into `(two_k_hi, two_k_lo)` to drop `sign_bit`.
    two_k_hi = lax.bitwise_or(
        lax.shift_left(hi, np.array(1, dtype=uint_dtype)),
        lax.shift_right_logical(lo, np.array(w - 1, dtype=uint_dtype)),
    )
    two_k_lo = lax.shift_left(lo, np.array(1, dtype=uint_dtype))
    abs_x_hi = two_k_hi
    abs_x_lo = lax.bitwise_or(two_k_lo, np.array(1, dtype=uint_dtype))
    q_hi = lax.bitwise_not(two_k_hi)
    q_lo = lax.bitwise_not(two_k_lo)
    # Scale the high and low words separately by `2^-w` and `2^(-2*w)` before
    # adding in floating point so tail values down to `2^(-2*w)` do not round to
    # zero when `q_hi == 0`.
    scale_hi = np.array(2.0 ** -w, compute_dtype)
    scale_lo = np.array(2.0 ** (-2 * w), compute_dtype)
    abs_x = (
        lax.convert_element_type(abs_x_hi, compute_dtype) * scale_hi
        + lax.convert_element_type(abs_x_lo, compute_dtype) * scale_lo
    )
    q = (
        lax.convert_element_type(q_hi, compute_dtype) * scale_hi
        + lax.convert_element_type(q_lo, compute_dtype) * scale_lo
    )
  else:
    raise ValueError(f"Unsupported mode: {mode}")
  # Attach `sign_bit` to `abs_x` to form signed `x`.
  x = lax.bitcast_convert_type(
      lax.bitwise_or(lax.bitcast_convert_type(abs_x, uint_dtype), sign_bit),
      compute_dtype,
  )
  return x, abs_x, q


def _bits_to_normal(
    bits: Array | tuple[Array, Array], dtype: DType, mode: str
) -> Array:
  """Convert random unsigned integer bits into standard normal samples.

  If `x` is uniform on `(-1, 1)`, `sqrt(2) * erfinv(x)` is standard normal.
  Public inverse functions (`lax.erf_inv` and `jax.scipy.special.ndtri`) take a
  single floating-point input, so values near `x = +/-1` (in the tails of the
  normal distribution) lose precision or round to `1.0`.

  In `"high"` and `"highest"` modes, `_signed_midpoint_and_complement` produces
  both `x` and `q = 1 - |x|` in integer arithmetic before casting to float, and
  we pass both directly to `erf_inv_core(..., has_far_tail=True, is_ndtri=True)`,
  which uses `x` near zero and `q` in the tails to evaluate `sqrt(2) * erfinv(x)`
  without losing tail precision.
  """
  if mode not in ("high", "highest"):
    raise ValueError(f"Unsupported mode for _bits_to_normal: {mode}")
  compute_dtype = (
      dtype if dtype in (np.float32, np.float64) else np.dtype(np.float32)
  )
  x, _, q = _signed_midpoint_and_complement(bits, compute_dtype, mode)
  z = erf_inv_core(x, q, has_far_tail=True, is_ndtri=True)
  return lax.convert_element_type(z, dtype)


@jit(static_argnums=(1, 2, 3, 4))
def _normal(
    key: Key, shape, dtype, mode: str = "low", out_sharding=None
) -> Array:
  if dtypes.issubdtype(dtype, np.complexfloating):
    sqrt2 = np.array(np.sqrt(2), dtype)

    key_re, key_im = _split(key)
    real_dtype = np.array(0, dtype).real.dtype
    _re = _normal_real(
        key_re, shape, real_dtype, mode=mode, out_sharding=out_sharding
    ).astype(dtype)
    _im = _normal_real(
        key_im, shape, real_dtype, mode=mode, out_sharding=out_sharding
    ).astype(dtype)
    return (_re + 1j * _im) / sqrt2
  else:
    return _normal_real(
        key, shape, dtype, mode=mode, out_sharding=out_sharding)


@jit(static_argnums=(1, 2, 3, 4))
def _normal_real(
    key: Key, shape, dtype, mode: str = "low", out_sharding=None
) -> Array:
  _check_shape("normal", shape)
  if mode == "low":
    lo = np.nextafter(np.array(-1.0, dtype), np.array(0.0, dtype), dtype=dtype)
    hi = np.array(1.0, dtype)
    u = uniform(key, shape, dtype, lo, hi, out_sharding=out_sharding)
    return lax.mul(np.array(np.sqrt(2), dtype), lax_special.erf_inv(u))
  compute_dtype = (
      dtype if dtype in (np.float32, np.float64) else np.dtype(np.float32)
  )
  word_bits = dtypes.finfo(compute_dtype).bits
  if mode == "highest":
    hi_bits, lo_bits = prng.random_bits(
        key, bit_width=word_bits, shape=shape, num_streams=2,
        out_sharding=out_sharding)
    bits: Array | tuple[Array, Array] = (hi_bits, lo_bits)
  else:
    bits = _random_bits(key, word_bits, shape, out_sharding=out_sharding)
  return lax.reshape(_bits_to_normal(bits, dtype, mode=mode), shape)


def multivariate_normal(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
                        mean: RealArray,
                        cov: RealArray,
                        shape: Shape | None = None,
                        dtype: DTypeLikeFloat | None = None,
                        method: str = 'cholesky',
                        *,
                        out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample multivariate normal random values with given mean and covariance.

  The values are returned according to the probability density function:

  .. math::
     f(x;\mu, \Sigma) = (2\pi)^{-k/2} \det(\Sigma)^{-1}e^{-\frac{1}{2}(x - \mu)^T \Sigma^{-1} (x - \mu)}

  where :math:`k` is the dimension, :math:`\mu` is the mean (given by ``mean``) and
  :math:`\Sigma` is the covariance matrix (given by ``cov``).

  Args:
    key: a PRNG key used as the random key.
    mean: a mean vector of shape ``(..., n)``.
    cov: a positive definite covariance matrix of shape ``(..., n, n)``. The
      batch shape ``...`` must be broadcast-compatible with that of ``mean``.
    shape: optional, a tuple of nonnegative integers specifying the result
      batch shape; that is, the prefix of the result shape excluding the last
      axis. Must be broadcast-compatible with ``mean.shape[:-1]`` and
      ``cov.shape[:-2]``. The default (None) produces a result batch shape by
      broadcasting together the batch shapes of ``mean`` and ``cov``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    method: optional, a method to compute the factor of ``cov``.
      Must be one of ``'svd'``, ``'eigh'``, and ``'cholesky'``. Default
      ``'cholesky'``. For singular or ill-conditioned covariance matrices, use
      ``'svd'`` or ``'eigh'``, or add a small diagonal jitter to ``cov``; see
      the note below.
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified dtype and shape given by
    ``shape + mean.shape[-1:]`` if ``shape`` is not None, or else
    ``broadcast_shapes(mean.shape[:-1], cov.shape[:-2]) + mean.shape[-1:]``.

  .. note::

     :func:`multivariate_normal` works by factoring :math:`\Sigma = F F^T` and
     transforming standard normal draws :math:`z \sim \mathcal{N}(0, I)` via
     :math:`\mu + F z`. The default ``method='cholesky'`` is the fastest
     decomposition, but can fail and return ``NaN`` values if ``cov`` is
     singular or ill-conditioned enough that floating-point rounding makes some
     eigenvalues non-positive (for example, large smooth Gaussian process kernel
     matrices, especially in ``float32``).

     Whenever ``cov`` is already available in factored form
     :math:`\Sigma = F F^T`—where :math:`F` has shape ``(n, k)`` (such as a
     Cholesky factor, a low-rank factor with ``k < n``, or scaled features)—it
     is better to skip :func:`multivariate_normal` and transform standard normal
     draws directly: draw ``z = jax.random.normal(key, (*batch_shape, k))`` and
     compute ``mean + z @ F.T`` (or ``mean + F @ z`` for a single sample). This
     costs :math:`O(n k)` per sample without an :math:`O(n^3)` matrix
     factorization, avoids squaring the condition number by forming
     ``F @ F.T``, and works even when ``k < n`` makes :math:`F F^T` singular.

     When only a dense ``cov`` matrix is available and it may be singular or
     ill-conditioned, either add a small diagonal jitter scaled to ``cov`` (e.g.
     ``cov + eps * jnp.eye(n)``) or pass ``method='eigh'`` or ``method='svd'``
     (note that eigendecomposition and SVD are substantially slower than
     Cholesky, particularly on TPU).

  Examples:
    Draw a single sample from a 2D multivariate normal distribution:

    >>> import jax
    >>> import jax.numpy as jnp
    >>> key = jax.random.key(0)
    >>> mean = jnp.array([1.0, 2.0])
    >>> cov = jnp.array([[2.0, 1.0],
    ...                  [1.0, 2.0]])
    >>> jax.random.multivariate_normal(key, mean, cov)
    Array([3.2947626, 5.6278143], dtype=float32)

    Draw a batch of 4 samples by passing ``shape``:

    >>> jax.random.multivariate_normal(key, mean, cov, shape=(4,))
    Array([[3.2947626 , 5.6278143 ],
           [0.386805  , 1.5971162 ],
           [1.2490301 , 0.93395376],
           [0.29954183, 2.2552586 ]], dtype=float32)

    If the covariance is already known in factored form ``cov = F @ F.T`` (for
    example, a ``3 x 2`` low-rank factor ``F`` whose ``3 x 3`` covariance
    ``F @ F.T`` has rank 2 and is therefore singular), sample directly from
    ``F`` rather than forming ``cov``:

    >>> F = jnp.array([[1.0, 0.0],
    ...                [1.0, 1.0],
    ...                [0.0, 1.0]])
    >>> mean = jnp.array([1.0, 2.0, 3.0])
    >>> z = jax.random.normal(key, (F.shape[-1],))
    >>> mean + F @ z  # single sample from N(mean, F @ F.T)
    Array([2.622642 , 5.647907 , 5.0252647], dtype=float32)
    >>> z_batch = jax.random.normal(key, (10_000, F.shape[-1]))
    >>> samples = mean + z_batch @ F.T  # batch of samples from N(mean, F @ F.T)
    >>> jnp.allclose(jnp.cov(samples, rowvar=False), F @ F.T, atol=0.1)
    Array(True, dtype=bool)
  """
  key, _ = _check_prng_key("multivariate_normal", key)
  mean, cov = promote_dtypes_inexact(mean, cov)
  if method not in {'svd', 'eigh', 'cholesky'}:
    raise ValueError("method must be one of {'svd', 'eigh', 'cholesky'}")
  if dtype is None:
    dtype = mean.dtype
  else:
    dtype = dtypes.check_and_canonicalize_user_dtype(dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(
        "dtype argument to `multivariate_normal` must be a float dtype,"
        f" got {dtype}")
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  out_sharding = canonicalize_sharding(out_sharding, "multivariate_normal")
  return _multivariate_normal(
      key, mean, cov, shape=shape, dtype=dtype, method=method,
      out_sharding=out_sharding)


@jit(static_argnums=(3, 4, 5, 6))
def _multivariate_normal(
    key: Key, mean, cov, shape, dtype, method, out_sharding=None
) -> Array:
  if not np.ndim(mean) >= 1:
    msg = "multivariate_normal requires mean.ndim >= 1, got mean.ndim == {}"
    raise ValueError(msg.format(np.ndim(mean)))
  if not np.ndim(cov) >= 2:
    msg = "multivariate_normal requires cov.ndim >= 2, got cov.ndim == {}"
    raise ValueError(msg.format(np.ndim(cov)))
  n = mean.shape[-1]
  if np.shape(cov)[-2:] != (n, n):
    msg = ("multivariate_normal requires cov.shape == (..., n, n) for n={n}, "
           "but got cov.shape == {shape}.")
    raise ValueError(msg.format(n=n, shape=np.shape(cov)))

  if shape is None:
    shape = lax.broadcast_shapes(mean.shape[:-1], cov.shape[:-2])
  else:
    _check_shape("normal", shape, mean.shape[:-1], cov.shape[:-2])

  out_shape = (*shape, n)
  out_s = canonicalize_sharding_for_samplers(
      out_sharding, "multivariate_normal", out_shape)
  mean_s = core.typeof(mean).sharding
  cov_s = core.typeof(cov).sharding
  if not mean_s.mesh.empty and mean_s.spec.partitions[-1] is not None:
    raise core.ShardingTypeError(
        "multivariate_normal mean trailing axis must be unsharded; got"
        f" {mean_s.spec}.")
  if (not cov_s.mesh.empty and
      any(p is not None for p in cov_s.spec.partitions[-2:])):
    raise core.ShardingTypeError(
        "multivariate_normal cov matrix axes must be unsharded; got"
        f" {cov_s.spec}.")

  if out_s is not None:
    batch_parts = out_s.spec.partitions[:-1]
    sample_sharding = out_s.update(spec=P(*batch_parts, None))
    cov_batch_s = out_s.update(spec=P(*batch_parts, None, None))
    _check_param_sharding(
        "multivariate_normal", out_shape, sample_sharding, mean)
    _check_param_sharding(
        "multivariate_normal", (*shape, n, n), cov_batch_s, cov)
  else:
    cov_lead = core.ShapedArray(
        cov.shape[:-1], cov.dtype,
        sharding=cov_s.update(spec=P(*cov_s.spec.partitions[:-1])))
    sample_sharding = _sample_sharding(
        "multivariate_normal", out_shape, None, mean, cov_lead)

  key, (mean, cov) = random_insert_pvary(
      "multivariate_normal", key, mean, cov)
  if sample_sharding is not None and cov.ndim > 2:
    n_batch = cov.ndim - 2
    cov_tail = sample_sharding.spec.partitions[len(shape) - n_batch:-1]
    shape_tail = shape[len(shape) - n_batch:]
    cov_cur_s = core.typeof(cov).sharding
    cov_cur_parts = (
        (None,) * n_batch
        if cov_cur_s.mesh.empty else cov_cur_s.spec.partitions[:-2])
    cov_parts = tuple(
        p if core.definitely_equal(d, sd) else cp
        for p, d, sd, cp in zip(
            cov_tail, cov.shape[:-2], shape_tail, cov_cur_parts))
    cov_sharding = sample_sharding.update(spec=P(*cov_parts, None, None))
    if cov_cur_s != cov_sharding:
      cov = reshard(cov, cov_sharding)

  if method == 'svd':
    (u, s, _) = jnp_linalg.svd(cov)
    factor = u * jnp.sqrt(s[..., None, :])
  elif method == 'eigh':
    (w, v) = jnp_linalg.eigh(cov)
    factor = v * jnp.sqrt(jnp.maximum(w[..., None, :], 0))
  else:  # 'cholesky'
    factor = jnp_linalg.cholesky(cov)
  normal_samples = normal(
      key, out_shape, dtype, out_sharding=sample_sharding)
  einsum_sharding = out_s if out_s is not None else sample_sharding
  mean, = _align_params(out_shape, einsum_sharding, mean)
  with config.numpy_rank_promotion('allow'):
    return mean + jnp_einsum.einsum(
        '...ij,...j->...i', factor, normal_samples,
        out_sharding=einsum_sharding)


def truncated_normal(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
                     lower: RealArray,
                     upper: RealArray,
                     shape: Shape | None = None,
                     dtype: DTypeLikeFloat | None = None,
                     *,
                     mode: str = "low",
                     out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample truncated standard normal random values with given shape and dtype.

  The values are returned according to the probability density function:

  .. math::
     f(x) \propto e^{-x^2/2}

  on the domain :math:`\rm{lower} < x < \rm{upper}`.

  Args:
    key: a PRNG key used as the random key.
    lower: a float or array of floats representing the lower bound for
      truncation. Must be broadcast-compatible with ``upper``.
    upper: a float or array of floats representing the  upper bound for
      truncation. Must be broadcast-compatible with ``lower``.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``lower`` and ``upper``. The
      default (None) produces a result shape by broadcasting ``lower`` and
      ``upper``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    mode: optional, "low" or "high" for the sampling precision mode (default
      "low"). Set to "high" for accurate sampling in the tails and at narrow
      truncation intervals.
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified dtype and shape given by ``shape`` if
    ``shape`` is not None, or else by broadcasting ``lower`` and ``upper``.
    Returns values in the open interval ``(lower, upper)``.
  """
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  if mode not in ("low", "high"):
    raise ValueError(
        f"Must provide valid mode for truncated_normal, got: {mode}"
    )
  key, _ = _check_prng_key("truncated_normal", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `truncated_normal` must be a float "
                     f"dtype, got {dtype}")
  out_sharding = canonicalize_sharding(out_sharding, "truncated_normal")
  return _truncated_normal(
      key, lower, upper, shape, dtype, mode, out_sharding)


def _ndtr_and_erf_shifted(
    x: Array, dx: Array
) -> tuple[Array, Array, Array]:
  """Evaluates (2*ndtr(x+dx), 2*ndtr(-(x+dx)), erf((x+dx)/sqrt(2))) for tiny |dx|."""
  x_fin = jnp.where(jnp.isfinite(x), x, 0.0)
  xm = x + dx
  rem = dx - ((x_fin + dx) - x_fin)
  sqrt_2_over_pi = np.array(math.sqrt(2.0 / math.pi), dtype=x.dtype)
  inv_sqrt2 = np.array(1.0 / math.sqrt(2.0), dtype=x.dtype)
  corr = rem * sqrt_2_over_pi * jnp.exp(-0.5 * jnp.square(xm))
  return (
      2.0 * ndtr(xm) + corr,
      2.0 * ndtr(-xm) - corr,
      lax_special.erf(xm * inv_sqrt2) + corr,
  )


@jit(static_argnums=(3, 4, 5, 6))
def _truncated_normal(
    key: Key, lower, upper, shape, dtype, mode: str = "low", out_sharding=None
) -> Array:
  if shape is None:
    shape = lax.broadcast_shapes(np.shape(lower), np.shape(upper))
  else:
    _check_shape("truncated_normal", shape, np.shape(lower), np.shape(upper))
  if not dtypes.issubdtype(dtype, np.floating):
    raise TypeError("truncated_normal only accepts floating point dtypes.")
  lower = lax.convert_element_type(lower, dtype)
  upper = lax.convert_element_type(upper, dtype)
  sample_s = _sample_sharding(
      "truncated_normal", shape, out_sharding, lower, upper)
  lower, upper = _align_params(shape, sample_s, lower, upper)

  if mode == "low":
    sqrt2 = np.array(np.sqrt(2), dtype)
    a = lax_special.erf(lower / sqrt2)
    b = lax_special.erf(upper / sqrt2)
    u = uniform(key, shape, dtype, minval=a, maxval=b, out_sharding=sample_s)
    out = sqrt2 * lax_special.erf_inv(u)
    # Clamp the value to the open interval (lower, upper) to make sure that
    # rounding (or if we chose `a` for `u`) doesn't push us outside of the range.
    return jnp.clip(
        out,
        lax.nextafter(lax.stop_gradient(lower), np.array(np.inf, dtype=dtype)),
        lax.nextafter(lax.stop_gradient(upper), np.array(-np.inf, dtype=dtype)))

  lower = lax.stop_gradient(lower)
  upper = lax.stop_gradient(upper)
  if np.ndim(lower) or np.ndim(upper):
    lower = lax.broadcast_to_rank(lower, len(shape))
    upper = lax.broadcast_to_rank(upper, len(shape))
    lower, upper = (
        lower + jnp.zeros_like(upper, dtype=dtype),
        upper + jnp.zeros_like(lower, dtype=dtype),
    )
    if sample_s is not None:
      s = core.get_sharding(sample_s, shape)
      cur_parts = core.typeof(lower).sharding.spec.partitions
      cur_parts = cur_parts + (None,) * (len(shape) - len(cur_parts))
      bound_parts = tuple(
          sp if core.definitely_equal(d, sd) else cp
          for d, sd, sp, cp in zip(
              lower.shape, shape, s.spec.partitions, cur_parts)
      )
      if any(p is not None for p in bound_parts):
        bound_s = s.update(spec=P(*bound_parts))
        if core.typeof(lower).sharding != bound_s:
          lower = reshard(lower, bound_s)
          upper = reshard(upper, bound_s)
  min_out = lax.nextafter(lower, np.array(np.inf, dtype=dtype))
  max_out = lax.nextafter(upper, np.array(-np.inf, dtype=dtype))

  # Compute in at least float32 so 16-bit dtypes get full 32-bit quantile
  # resolution and avoid underflow/rounding in the tails.
  compute_dtype = (
      np.dtype("float32") if dtypes.finfo(dtype).bits < 32 else dtype
  )
  lo = lax.convert_element_type(lower, compute_dtype)
  hi = lax.convert_element_type(upper, compute_dtype)

  # Shift integration bounds inward by half a destination-dtype ULP so we
  # truncate at the rounding-interval boundaries of [min_out, max_out] rather
  # than folding [lower, lower + 0.5*ulp) onto min_out via jnp.clip.
  lo_next = lax.convert_element_type(min_out, compute_dtype)
  hi_prev = lax.convert_element_type(max_out, compute_dtype)
  valid = lo_next <= hi_prev
  d_lo = jnp.where(valid & jnp.isfinite(lo), 0.5 * (lo_next - lo), 0.0)
  d_hi = jnp.where(valid & jnp.isfinite(hi), -0.5 * (hi - hi_prev), 0.0)

  # Reflect intervals with positive midpoint so tail bounds lie in (-inf, 0],
  # where ndtr(x) -> 0 retains full floating-point exponent precision.
  flip = lo > -hi
  a, da = jnp.where(flip, -hi, lo), jnp.where(flip, -d_hi, d_lo)
  b, db = jnp.where(flip, -lo, hi), jnp.where(flip, -d_lo, d_hi)

  finfo = dtypes.finfo(compute_dtype)
  qa_l, qa_r, ea = _ndtr_and_erf_shifted(a, da)
  qb_l, qb_r, eb = _ndtr_and_erf_shifted(b, db)
  delta_q = jnp.where(b <= -1.0, qb_l - qa_l, eb - ea)

  bits = lax.reshape(
      _random_bits(key, finfo.bits, shape, out_sharding=sample_s), shape)
  s, _, q_s = _signed_midpoint_and_complement(bits, compute_dtype, mode="high")
  neg = s < 0
  step = (0.5 * q_s) * delta_q

  q_l = jnp.clip(jnp.where(neg, qa_l + step, qb_l - step), finfo.tiny, 2.0)
  q_r = jnp.clip(jnp.where(neg, qa_r - step, qb_r + step), finfo.tiny, 2.0)
  q_erf = jnp.minimum(q_l, q_r)

  x_end = jnp.where(neg, ea + step, eb - step)
  x_mid = jnp.where(q_s < 0.5, x_end, 0.5 * (ea + eb) + 0.5 * s * delta_q)
  x_erf = jnp.where(
      q_l < 0.5, q_l - 1.0, jnp.where(q_r < 0.5, 1.0 - q_r, x_mid)
  )

  z = erf_inv_core(x_erf, q_erf, has_far_tail=True, is_ndtri=True)
  if np.ndim(flip):
    flip = flip | jnp.zeros_like(z, dtype=bool)
  out = lax.convert_element_type(jnp.where(flip, -z, z), dtype)
  # Clamp the value to the open interval (lower, upper) to make sure that
  # rounding doesn't push us outside of the range.
  return jnp.clip(out, min_out, max_out)


def bernoulli(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
              p: RealArray = 0.5,
              shape: Shape | None = None,
              mode: str = 'low',
              *,
              out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample Bernoulli random values with given shape and mean.

  The values are distributed according to the probability mass function:

  .. math::
     f(k; p) = p^k(1 - p)^{1 - k}

  where :math:`k \in \{0, 1\}` and :math:`0 \le p \le 1`.

  Args:
    key: a PRNG key used as the random key.
    p: optional, a float or array of floats for the mean of the random
      variables. Must be broadcast-compatible with ``shape``. Default 0.5.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Must be broadcast-compatible with ``p.shape``. The default (None)
      produces a result shape equal to ``p.shape``.
    mode: optional, "high" or "low" for how many bits to use when sampling.
      default='low'. Set to "high" for correct sampling at small values of
      `p`. When sampling in float32, bernoulli samples with mode='low' produce
      incorrect results for p < ~1E-7. mode="high" approximately doubles the
      cost of sampling.
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with boolean dtype and shape given by ``shape`` if ``shape``
    is not None, or else ``p.shape``.
  """
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  if mode not in ['high', 'low']:
    raise ValueError(f"got {mode=}, expected 'high' or 'low'")
  key, _ = _check_prng_key("bernoulli", key)
  dtype = lax.dtype(p)
  if not dtypes.issubdtype(dtype, np.floating):
    msg = "bernoulli probability `p` must have a floating dtype, got {}."
    raise TypeError(msg.format(dtype))
  out_sharding = canonicalize_sharding(out_sharding, "bernoulli")
  p = lax.convert_element_type(p, dtype)
  return _bernoulli(key, p, shape=shape, mode=mode, out_sharding=out_sharding)


@jit(static_argnames=['shape', 'mode', 'out_sharding'])
def _bernoulli(
    key: Key, p: Array, shape: Shape | None, mode: str, out_sharding=None
) -> Array:
  if shape is None:
    # TODO: Use the named part of `p` as well
    shape = np.shape(p)
  else:
    _check_shape("bernoulli", shape, np.shape(p))
  dtype = lax.dtype(p)
  sample_s = _sample_sharding("bernoulli", shape, out_sharding, p)
  p, = _align_params(shape, sample_s, p)

  if mode == 'high':
    u1, u2 = lax.unstack(uniform(
        key, (2, *shape), dtype,
        out_sharding=_prepend_none_sharding(sample_s)))
    # resolution of uniform samples is 2 ** -n_bits
    n_bits = dtypes.finfo(dtype).nmant + (
        0 if config.legacy_random_uniform.value else 1
    )
    u2 *= 2 ** -n_bits
    return u2 < p - u1
  else:
    return uniform(key, shape, dtype, out_sharding=sample_s) < p


def beta(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
         a: RealArray,
         b: RealArray,
         shape: Shape | None = None,
         dtype: DTypeLikeFloat | None = None,
         *,
         method: str = "exact",
         out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample Beta random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(x;a,b) \propto x^{a - 1}(1 - x)^{b - 1}

  on the domain :math:`0 \le x \le 1`.

  Args:
    key: a PRNG key used as the random key.
    a: a float or array of floats broadcast-compatible with ``shape``
      representing the first parameter "alpha".
    b: a float or array of floats broadcast-compatible with ``shape``
      representing the second parameter "beta".
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``a`` and ``b``. The default
      (None) produces a result shape by broadcasting ``a`` and ``b``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    method: optional, the sampling algorithm to use, either ``'exact'`` (the
      default) or ``'approximate'``. The ``'exact'`` method is a rejection
      sampler. The ``'approximate'`` method is loop-free and faster but carries
      a small bias. The gradient w.r.t. ``a`` and ``b`` differs between the two
      methods because of the ambiguity in defining a gradient for random variates.
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified dtype and shape given by ``shape`` if
    ``shape`` is not None, or else by broadcasting ``a`` and ``b``.
  """
  key, _ = _check_prng_key("beta", key)
  if method not in {'exact', 'approximate'}:
    raise ValueError("method argument to `beta` must be one of "
                     f"{{'exact', 'approximate'}}, got {method!r}")
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `beta` must be a float "
                     f"dtype, got {dtype}")
  _check_all_safe_to_cast("beta", dtype, a, b)
  shape = _check_broadcast_shapes("beta", shape, a, b)
  out_sharding = canonicalize_sharding(out_sharding, "beta")
  return _beta(key, a, b, method, shape, dtype, out_sharding)


@jit(static_argnums=(3, 4, 5, 6))
def _beta(
    key: Key, a, b, method, shape, dtype, out_sharding=None
) -> Array:
  if shape is None:
    shape = lax.broadcast_shapes(np.shape(a), np.shape(b))
  else:
    _check_shape("beta", shape, np.shape(a), np.shape(b))

  key, (a, b) = random_insert_pvary("jax.random.beta", key, a, b)

  a = lax.convert_element_type(a, dtype)
  b = lax.convert_element_type(b, dtype)
  sample_s = _sample_sharding("beta", shape, out_sharding, a, b)
  a, b = _align_params(shape, sample_s, a, b)
  key_a, key_b = _split(key)
  log_gamma_a = loggamma(
      key_a, a, shape, dtype, method=method, out_sharding=sample_s)
  log_gamma_b = loggamma(
      key_b, b, shape, dtype, method=method, out_sharding=sample_s)
  # Compute gamma_a / (gamma_a + gamma_b) without losing precision.
  log_max = lax.max(log_gamma_a, log_gamma_b)
  gamma_a_scaled = jnp.exp(log_gamma_a - log_max)
  gamma_b_scaled = jnp.exp(log_gamma_b - log_max)
  return gamma_a_scaled / (gamma_a_scaled + gamma_b_scaled)


def cauchy(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
           shape: Shape = (),
           dtype: DTypeLikeFloat | None = None,
           *,
           out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample Cauchy random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(x) \propto \frac{1}{x^2 + 1}

  on the domain :math:`-\infty < x < \infty`

  Args:
    key: a PRNG key used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified shape and dtype.
  """
  key, _ = _check_prng_key("cauchy", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `cauchy` must be a float "
                     f"dtype, got {dtype}")
  shape = core.canonicalize_shape(shape)
  out_sharding = canonicalize_sharding(out_sharding, "cauchy")
  return _cauchy(key, shape, dtype, out_sharding)


@jit(static_argnums=(1, 2, 3))
def _cauchy(key: Key, shape, dtype, out_sharding=None) -> Array:
  _check_shape("cauchy", shape)
  u = uniform(key, shape, dtype, minval=dtypes.finfo(dtype).eps, maxval=1.,
              out_sharding=out_sharding)
  pi = lax._const(u, np.pi)
  return lax.tan(lax.mul(pi, lax.sub(u, lax._const(u, 0.5))))


def dirichlet(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
              alpha: RealArray,
              shape: Shape | None = None,
              dtype: DTypeLikeFloat | None = None,
              *,
              out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample Dirichlet random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(\{x_i\}; \{\alpha_i\}) \propto \prod_{i=1}^k x_i^{\alpha_i - 1}

  Where :math:`k` is the dimension, and :math:`\{x_i\}` satisfies

  .. math::
     \sum_{i=1}^k x_i = 1

  and :math:`0 \le x_i \le 1` for all :math:`x_i`.

  Args:
    key: a PRNG key used as the random key.
    alpha: an array of shape ``(..., n)`` used as the concentration
      parameter of the random variables.
    shape: optional, a tuple of nonnegative integers specifying the result
      batch shape; that is, the prefix of the result shape excluding the last
      element of value ``n``. Must be broadcast-compatible with
      ``alpha.shape[:-1]``. The default (None) produces a result shape equal to
      ``alpha.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified dtype and shape given by
    ``shape + (alpha.shape[-1],)`` if ``shape`` is not None, or else
    ``alpha.shape``.
  """
  key, _ = _check_prng_key("dirichlet", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `dirichlet` must be a float "
                     f"dtype, got {dtype}")
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  out_sharding = canonicalize_sharding(out_sharding, "dirichlet")
  return _dirichlet(key, alpha, shape, dtype, out_sharding)


@jit(static_argnums=(2, 3, 4))
def _dirichlet(key: Key, alpha, shape, dtype, out_sharding=None) -> Array:
  from jax._src.nn.functions import softmax  # pyrefly: ignore[missing-import]

  if not np.ndim(alpha) >= 1:
    msg = "dirichlet requires alpha.ndim >= 1, got alpha.ndim == {}"
    raise ValueError(msg.format(np.ndim(alpha)))

  if shape is None:
    shape = np.shape(alpha)[:-1]
  else:
    _check_shape("dirichlet", shape, np.shape(alpha)[:-1])

  alpha = lax.convert_element_type(alpha, dtype)
  out_shape = shape + np.shape(alpha)[-1:]
  out_s, _ = _check_unsharded_trailing_axis(
      "dirichlet", out_shape, out_sharding, alpha, param_has_trailing=True)
  # Compute gamma in log space, otherwise small alpha can lead to poor behavior.
  log_gamma_samples = loggamma(
      key, alpha, out_shape, dtype, out_sharding=out_s)
  return softmax(log_gamma_samples, -1)


def exponential(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
                shape: Shape = (),
                dtype: DTypeLikeFloat | None = None,
                *,
                out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample Exponential random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(x) = e^{-x}

  on the domain :math:`0 \le x < \infty`.

  Args:
    key: a PRNG key used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified shape and dtype.
  """
  key, _ = _check_prng_key("exponential", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `exponential` must be a float "
                     f"dtype, got {dtype}")
  shape = core.canonicalize_shape(shape)
  out_sharding = canonicalize_sharding(out_sharding, "exponential")
  return _exponential(key, shape, dtype, out_sharding)


@jit(static_argnums=(1, 2, 3))
def _exponential(key: Key, shape, dtype, out_sharding=None) -> Array:
  _check_shape("exponential", shape)
  u = uniform(key, shape, dtype, out_sharding=out_sharding)
  # taking 1 - u to move the domain of log to (0, 1] instead of [0, 1)
  return lax.neg(lax.log1p(lax.neg(u)))


def _gamma_one(key: Key, alpha, log_space) -> Array:
  # Ref: A simple method for generating gamma variables, George Marsaglia and Wai Wan Tsang
  # The algorithm can also be founded in:
  # https://en.wikipedia.org/wiki/Gamma_distribution#Generating_gamma-distributed_random_variables
  zero = lax._const(alpha, 0)
  one = lax._const(alpha, 1)
  two = lax._const(alpha, 2)
  minus_one = lax._const(alpha, -1)
  one_over_two = lax._const(alpha, 0.5)
  one_over_three = lax._const(alpha, 1. / 3.)
  squeeze_const = lax._const(alpha, 0.0331)
  dtype = lax.dtype(alpha)

  zero = core.pvary(zero, tuple(core.typeof(alpha).mat.varying))
  one = core.pvary(one, tuple(core.typeof(alpha).mat.varying))
  minus_one = core.pvary(minus_one, tuple(core.typeof(alpha).mat.varying))
  two = core.pvary(two, tuple(core.typeof(alpha).mat.varying))

  # for alpha < 1, we boost alpha to alpha + 1 and get a sample according to
  #   Gamma(alpha) ~ Gamma(alpha+1) * Uniform()^(1 / alpha)
  # When alpha is very small, this boost can be problematic because it may result
  # in floating point underflow; for this reason we compute it in log space if
  # specified by the `log_space` argument:
  #   log[Gamma(alpha)] ~ log[Gamma(alpha + 1)] + log[Uniform()] / alpha
  # Note that log[Uniform()] ~ -Exponential(), but to avoid problems at x=0
  # exponential is computed in terms of log[1 - Uniform()]; we must account for this
  # so that log-space and non-log-space samples match.
  boost_mask = lax.ge(alpha, one)
  alpha_orig = alpha
  alpha = lax.select(boost_mask, alpha, lax.add(alpha, one))

  d = lax.sub(alpha, one_over_three)
  c = lax.div(one_over_three, lax.sqrt(d))

  def _cond_fn(kXVU):
    _, X, V, U = kXVU
    # TODO: use lax.cond when its batching rule is supported
    # The reason is to avoid evaluating second condition which involves log+log
    # if the first condition is satisfied
    cond = lax.bitwise_and(
        lax.ge(U, lax.sub(one, lax.mul(squeeze_const, lax.mul(X, X)))),
        lax.ge(lax.log(U), lax.add(lax.mul(X, one_over_two),
                                   lax.mul(d, lax.add(lax.sub(one, V),
                                                      lax.log(V))))))
    return cond

  def _body_fn(kXVU):
    def _next_kxv(kxv):
      key = kxv[0]
      key, subkey = _split(key)
      x = normal(subkey, (), dtype=dtype)
      v = lax.add(one, lax.mul(x, c))
      return key, x, v

    key = kXVU[0]
    key, x_key, U_key = _split(key, 3)
    _, x, v = lax_control_flow.while_loop(lambda kxv: lax.le(kxv[2], zero),
                                          _next_kxv, (x_key, zero, minus_one))
    X = lax.mul(x, x)
    V = lax.mul(lax.mul(v, v), v)
    U = uniform(U_key, (), dtype=dtype)
    return key, X, V, U

  # initial state is chosen such that _cond_fn will return True
  key, subkey = _split(key)
  _, _, V, _ = lax_control_flow.while_loop(
      _cond_fn, _body_fn, (key, zero, one, two))
  if log_space:
    log_samples = lax.neg(exponential(subkey, (), dtype=dtype))
    log_boost = lax.select(boost_mask | (log_samples == 0), zero,
                           lax.mul(log_samples, lax.div(one, alpha_orig)))
    return lax.add(lax.add(lax.log(d), lax.log(V)), log_boost)
  else:
    samples = 1 - uniform(subkey, (), dtype=dtype)
    boost = lax.select(boost_mask, one,
                       lax.pow(samples, lax.div(one, alpha_orig)))
    return lax.mul(lax.mul(d, V), boost)


def _gamma_grad_one_shard(a, x, *, use_map):
  out_shape = lax.broadcast_shapes(np.shape(a), np.shape(x))
  if 0 in out_shape:
    return jnp.empty(out_shape, dtype=x.dtype)
  if use_map and not get_abstract_mesh().auto_axes:
    alphas = jnp.reshape(jnp.broadcast_to(a, out_shape), -1)
    xs = jnp.reshape(jnp.broadcast_to(x, out_shape), -1)
    return lax_control_flow.map(
        lambda args: lax_special.random_gamma_grad(*args),
        (alphas, xs)).reshape(out_shape)
  return lax_special.random_gamma_grad(a, x)


def _gamma_grad_lowering(ctx, a, x, *, use_map=False):
  out_aval, = ctx.avals_out
  out_s = out_aval.sharding
  manual_axes = frozenset(out_s.mesh.explicit_axes)
  out_parts = out_s.spec.partitions
  one_shard = partial(_gamma_grad_one_shard, use_map=use_map)
  def _lower_fn(a_val, x_val):
    if not manual_axes or all(p is None for p in out_parts):
      return one_shard(a_val, x_val)
    slice_fns = [
        _make_shard_slicer(
            out_s.mesh, av,
            out_parts[len(out_parts) - av.ndim:] if av.ndim else ())
        for av in ctx.avals_in
    ]
    def _per_shard(local_a, local_x):
      return one_shard(slice_fns[0](local_a), slice_fns[1](local_x))
    return shard_map.shard_map(
        _per_shard, mesh=out_s.mesh, out_specs=out_s.spec,
        axis_names=manual_axes, check_vma=False)(a_val, x_val)
  lowering = mlir.lower_fun(_lower_fn, multiple_results=False)
  avals_in = [v.update(manual_axis_type=core.empty_mat) for v in ctx.avals_in]
  avals_out = [out_aval.update(manual_axis_type=core.empty_mat)]
  return mlir.delegate_lowering(
      ctx, lowering, a, x, avals_in=avals_in, avals_out=avals_out)


_gamma_grad_p = lax.standard_naryop([lax._float, lax._float],
                                    'random_gamma_grad')
mlir.register_lowering(
    _gamma_grad_p, _gamma_grad_lowering, shard_map_passthrough=True)
mlir.register_lowering(
    _gamma_grad_p, partial(_gamma_grad_lowering, use_map=True),
    platform='cpu', shard_map_passthrough=True)
ad.primitive_jvps[_gamma_grad_p] = lambda primals, tangents: jvp(
    lax_special.random_gamma_grad, primals,
    tuple(map(ad.instantiate_zeros, tangents)))


def _gamma_grad(sample, a, *, log_space, **_):
  if np.shape(a) != np.shape(sample):
    a = jnp.broadcast_to(a, np.shape(sample))
  a = lax.convert_element_type(a, sample.dtype)
  if log_space:
    # d[log(sample)] = d[sample] / sample
    s = lax.exp(sample)
    s = lax.select(lax.eq(s, lax._const(s, 0)),
                   lax.full_like(s, dtypes.finfo(s.dtype).tiny), s)
    return _gamma_grad_p.bind(a, s) / s
  return _gamma_grad_p.bind(a, sample)


def _gamma_from_keys(keys, a, *, impl, log_space, use_vmap):
  a_shape = np.shape(a)
  alphas = a.reshape(-1)
  if use_vmap and impl is threefry2x32.threefry_prng_impl:
    samples = vmap(partial(_gamma_one, log_space=log_space))(keys, alphas)
  else:
    samples = lax_control_flow.map(
        lambda args: _gamma_one(*args, log_space=log_space), (keys, alphas))
  return jnp.reshape(samples, a_shape)


def _gamma_impl_local(key: Key, a, *, log_space, use_vmap=False):
  a_shape = np.shape(a)
  if 0 in a_shape:
    return jnp.empty_like(a)
  split_count = math.prod(a_shape[key.ndim:])
  keys = vmap(_split, in_axes=(0, None))(key.flatten(), split_count).flatten()
  return _gamma_from_keys(
      keys, a, impl=_key_impl(key), log_space=log_space, use_vmap=use_vmap)


def _gamma_batching_rule(
    axis_data, batched_args, batch_dims, *, log_space, out_sharding):
  k, a = batched_args
  bk, ba = batch_dims
  if bk is None and ba is None:
    return random_gamma_p.bind(
        k, a, log_space=log_space, out_sharding=out_sharding), None
  mesh_axis = axis_data.explicit_mesh_axis
  k = batching.bdim_at_front(k, bk, axis_data.size, mesh_axis=mesh_axis)
  a = batching.bdim_at_front(a, ba, axis_data.size, mesh_axis=mesh_axis)
  if out_sharding is not None:
    out_sharding = batching.get_sharding_for_vmap(
        axis_data, canonicalize_sharding(out_sharding, 'random_gamma'), 0)
  return random_gamma_p.bind(
      k, a, log_space=log_space, out_sharding=out_sharding), 0


random_gamma_p = core.Primitive('random_gamma')
dispatch.simple_impl(random_gamma_p)


def _random_gamma_sharding_rule(key_aval, a_aval, *, out_sharding,
                                name='random_gamma'):
  k_s, a_s = key_aval.sharding, a_aval.sharding
  meshes = [s.mesh for s in (k_s, a_s, out_sharding)
            if s is not None and not s.mesh.empty]
  if len(set(meshes)) > 1:
    raise core.ShardingTypeError(
        f'Mismatched meshes in {name}: {meshes}.')
  if out_sharding is not None:
    out_s = core.get_sharding(out_sharding, a_aval.shape)
    out_parts = out_s.spec.partitions
    if (any(not _is_sub_spec(kp, op)
            for kp, op in zip(k_s.spec.partitions, out_parts)) or
        any(not _is_sub_spec(ap, op)
            for ap, op in zip(a_s.spec.partitions, out_parts))):
      raise core.ShardingTypeError(
          f'Incompatible shardings {k_s.spec}, {a_s.spec} vs'
          f' {out_s.spec}.')
    return out_s
  mesh = meshes[0] if meshes else get_abstract_mesh()
  k_parts = k_s.spec.partitions + (None,) * (
      a_aval.ndim - len(k_s.spec.partitions))
  a_parts = a_s.spec.partitions + (None,) * (
      a_aval.ndim - len(a_s.spec.partitions))
  out_parts = []
  for kp, ap in zip(k_parts, a_parts):
    if _is_sub_spec(kp, ap):
      out_parts.append(ap)
    elif _is_sub_spec(ap, kp):
      out_parts.append(kp)
    else:
      raise core.ShardingTypeError(
          f'Incompatible key {k_s.spec} and param {a_s.spec} shardings.')
  try:
    return core.get_sharding(NamedSharding(mesh, P(*out_parts)), a_aval.shape)
  except lax.lax_utils.DuplicateSpecError as e:
    raise core.ShardingTypeError(str(e)) from e


def _random_gamma_abstract_eval(key_aval, a_aval, *, log_space, out_sharding):
  del log_space
  if a_aval.shape[:key_aval.ndim] != key_aval.shape:
    raise ValueError(
        f'random_gamma leading shape of a {a_aval.shape} must match key shape'
        f' {key_aval.shape}.')
  out_sharding = canonicalize_sharding(out_sharding, 'random_gamma')
  sharding = prng.resolve_random_sharding(
      random_gamma_p, _random_gamma_sharding_rule,
      key_aval, a_aval, out_sharding=out_sharding)
  vma = core.standard_vma_rule('random_gamma', key_aval, a_aval)
  return a_aval.update(
      sharding=sharding, manual_axis_type=core.ManualAxisType(varying=vma))


random_gamma_p.def_abstract_eval(_random_gamma_abstract_eval)

ad.defjvp2(
    random_gamma_p, None,
    lambda tangent, ans, key, a, **kwds: tangent * _gamma_grad(ans, a, **kwds))


def _random_gamma_lowering(ctx, key, a, *, log_space, out_sharding):
  key_aval, a_aval = ctx.avals_in
  out_aval, = ctx.avals_out
  impl = key_aval.dtype._impl
  out_s = out_aval.sharding
  manual_axes = frozenset(out_s.mesh.explicit_axes)
  out_parts = out_s.spec.partitions
  k_ndim = key_aval.ndim
  trail_shape = a_aval.shape[k_ndim:]
  trail_parts = out_parts[k_ndim:]

  def _lower_fn(key_base, a_val):
    if not manual_axes or all(p is None for p in out_parts):
      return _gamma_impl_local(
          prng.random_wrap(key_base, impl=impl), a_val,
          log_space=log_space, use_vmap=True)
    slice_a = _make_shard_slicer(out_s.mesh, a_aval, out_parts)
    if all(p is None for p in trail_parts):
      slice_k = _make_shard_slicer(
          out_s.mesh, core.typeof(key_base), out_parts[:k_ndim])
      def _per_shard_unsplit(local_k, local_a):
        return _gamma_impl_local(
            prng.random_wrap(slice_k(local_k), impl=impl), slice_a(local_a),
            log_space=log_space, use_vmap=True)
      return shard_map.shard_map(
          _per_shard_unsplit, mesh=out_s.mesh, out_specs=out_s.spec,
          axis_names=manual_axes, check_vma=False)(key_base, a_val)

    k_lead = key_aval.sharding.spec.partitions
    k_lead = k_lead + (None,) * (k_ndim - len(k_lead))
    subkey_s = NamedSharding(out_s.mesh, P(*k_lead, *trail_parts))
    subkeys = prng.random_unwrap(
        prng.random_split(
            prng.random_wrap(key_base, impl=impl),
            shape=trail_shape, out_sharding=subkey_s))
    slice_sub = _make_shard_slicer(
        out_s.mesh, core.typeof(subkeys), out_parts)
    def _per_shard(local_subkeys, local_a):
      if 0 in local_a.shape:
        return jnp.empty_like(local_a)
      local_subkeys = slice_sub(local_subkeys)
      local_a = slice_a(local_a)
      keys = prng.random_wrap(
          local_subkeys.reshape(-1, *impl.key_shape), impl=impl)
      return _gamma_from_keys(
          keys, local_a, impl=impl, log_space=log_space, use_vmap=True)
    return shard_map.shard_map(
        _per_shard, mesh=out_s.mesh, out_specs=out_s.spec,
        axis_names=manual_axes, check_vma=False)(subkeys, a_val)

  lowering = mlir.lower_fun(_lower_fn, multiple_results=False)
  phys_key = core.physical_aval(key_aval).update(
      manual_axis_type=core.empty_mat)
  a_in = a_aval.update(manual_axis_type=core.empty_mat)
  out_in = out_aval.update(manual_axis_type=core.empty_mat)
  return mlir.delegate_lowering(
      ctx, lowering, key, a, avals_in=[phys_key, a_in], avals_out=[out_in])


mlir.register_lowering(
    random_gamma_p, _random_gamma_lowering, shard_map_passthrough=True)
batching.fancy_primitive_batchers[random_gamma_p] = _gamma_batching_rule

def gamma(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
          a: RealArray,
          shape: Shape | None = None,
          dtype: DTypeLikeFloat | None = None,
          *,
          method: str = 'exact',
          out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample Gamma random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(x;a) \propto x^{a - 1} e^{-x}

  on the domain :math:`0 \le x < \infty`, with :math:`a > 0`.

  This is the standard gamma density, with a unit scale/rate parameter.
  Dividing the sample output by the rate is equivalent to sampling from
  *gamma(a, rate)*, and multiplying the sample output by the scale is equivalent
  to sampling from *gamma(a, scale)*.

  Args:
    key: a PRNG key used as the random key.
    a: a float or array of floats broadcast-compatible with ``shape``
      representing the parameter of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``a``. The default (None)
      produces a result shape equal to ``a.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    method: optional, the sampling algorithm to use, either ``'exact'`` (the
      default) or ``'approximate'``. The ``'exact'`` method is a rejection
      sampler. The ``'approximate'`` method is loop-free and faster but carries
      a small bias. The gradient w.r.t. `a` differs between the two methods
      because of the ambiguity in defining a gradient for random variates.
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``a.shape``.

  See Also:
    loggamma : sample gamma values in log-space, which can provide improved
      accuracy for small values of ``a``.
  """
  key, _ = _check_prng_key("gamma", key)
  if method not in {'exact', 'approximate'}:
    raise ValueError("method argument to `gamma` must be one of "
                     f"{{'exact', 'approximate'}}, got {method!r}")
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `gamma` must be a float "
                     f"dtype, got {dtype}")
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  out_sharding = canonicalize_sharding(out_sharding, "gamma")
  if method == 'approximate':
    return _gamma_approx(
        key, a, shape=shape, dtype=dtype, out_sharding=out_sharding)
  return _gamma(key, a, shape=shape, dtype=dtype, out_sharding=out_sharding)


def loggamma(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
             a: RealArray,
             shape: Shape | None = None,
             dtype: DTypeLikeFloat | None = None,
             *,
             method: str = 'exact',
             out_sharding: NamedSharding | P | None = None) -> Array:
  """Sample log-gamma random values with given shape and float dtype.

  This function is implemented such that the following will hold for a
  dtype-appropriate tolerance::

    np.testing.assert_allclose(jnp.exp(loggamma(*args)), gamma(*args), rtol=rtol)

  The benefit of log-gamma is that for samples very close to zero (which occur frequently
  when `a << 1`) sampling in log space provides better precision.

  Args:
    key: a PRNG key used as the random key.
    a: a float or array of floats broadcast-compatible with ``shape``
      representing the parameter of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``a``. The default (None)
      produces a result shape equal to ``a.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    method: optional, the sampling algorithm to use, either ``'exact'`` (the
      default) or ``'approximate'``. The ``'exact'`` method is a rejection
      sampler. The ``'approximate'`` method is loop-free and faster but carries
      a small bias. The gradient w.r.t. `a` differs between the two methods
      because of the ambiguity in defining a gradient for random variates.
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``a.shape``.

  See Also:
    gamma : standard gamma sampler.
  """
  key, _ = _check_prng_key("loggamma", key)
  if method not in {'exact', 'approximate'}:
    raise ValueError("method argument to `loggamma` must be one of "
                     f"{{'exact', 'approximate'}}, got {method!r}")
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `gamma` must be a float "
                     f"dtype, got {dtype}")
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  out_sharding = canonicalize_sharding(out_sharding, "loggamma")
  if method == 'approximate':
    return _gamma_approx(
        key, a, shape=shape, dtype=dtype, log_space=True,
        out_sharding=out_sharding)
  return _gamma(
      key, a, shape=shape, dtype=dtype, log_space=True,
      out_sharding=out_sharding)


@jit(static_argnames=('shape', 'dtype', 'log_space', 'out_sharding'))
def _gamma(
    key: Key, a, shape, dtype, log_space=False, out_sharding=None
) -> Array:
  if shape is None:
    shape = np.shape(a)
  else:
    _check_shape("gamma", shape, np.shape(a))

  a = lax.convert_element_type(a, dtype)
  _check_param_sharding("gamma", shape, out_sharding, a)
  if np.shape(a) != shape:
    a = jnp.broadcast_to(a, shape)
  key, (a,) = random_insert_pvary('gamma', key, a)
  return random_gamma_p.bind(
      key, a, log_space=log_space, out_sharding=out_sharding)


# Coefficients of the chi-square quantile series used by `_gamma_approx` (see
# there for how they are applied). Each inner tuple k (k=0..6) holds the
# coefficients of x^k nu^(-k/2), written as a quadratic in 1 / nu: entry [k][j]
# multiplies nu^(-j) for j=0..2. The leading constant is set to exactly 1 rather
# than the published 1.0000886; see `_gamma_approx` for why.
_CHI2_QUANTILE_COEF = (
    (1.0, -0.2237368, -0.01513904),
    (0.4713941, 0.02607083, -0.008986007),
    (0.0001348028, 0.01128186, 0.02277679),
    (-0.008553069, -0.01153761, -0.01323293),
    (0.00312558, 0.005169654, -0.006950356),
    (-0.0008426812, 0.00253001, 0.001060438),
    (0.00009780499, -0.001450117, 0.001565326),
)

# Number of Stuart boosting steps used by `_gamma_approx`. Four steps keep the
# base draw at shape a + 4 >= 4, where the chi-square quantile series is
# accurate, even as a -> 0.
_LOGGAMMA_NUM_BOOSTS = 4


def _loggamma_chisquare(key: Key, a, out_sharding=None) -> Array:
  r"""Sample ``log(Gamma(a, 1))`` via the chi-square quantile series.

  Cubes a 6th-degree polynomial in a standard normal applied to the chi-square
  distribution with ``nu = 2 * a`` degrees of freedom, using
  ``Gamma(a, 1) = chi2(2a) / 2``. The polynomial is evaluated by Horner's method
  both in the normal variate and in ``1 / nu``. The series is only accurate for
  ``a`` away from zero, so callers boost ``a`` to keep it there; see
  :func:`_gamma_approx` for the references and overall algorithm.
  """
  nu = 2 * a
  x = _normal(key, np.shape(a), a.dtype, out_sharding=out_sharding)
  u = 1 / nu
  # x^k nu^(-k/2) = t^k, so the series is a degree-6 Horner polynomial in t.
  t = x * lax.rsqrt(nu)

  row_coeffs = [lax.polynomial(u, row) for row in _CHI2_QUANTILE_COEF]
  poly = lax.polynomial(t, row_coeffs)

  # chi2 = nu poly^3 and Gamma(a, 1) = chi2 / 2 = a poly^3, so log gains log(a).
  # `poly` should never actually reach values <= 0, but clip just in case
  # someone changes some other param
  poly = jnp.maximum(poly, dtypes.finfo(a.dtype).tiny)
  return jnp.log(a) + 3 * jnp.log(poly)


@jit(static_argnames=('shape', 'dtype', 'log_space', 'out_sharding'))
def _gamma_approx(
    key: Key, a, shape, dtype, log_space=False, out_sharding=None
) -> Array:
  r"""Loop-free approximate sampler for ``Gamma(a, 1)``.

  Backs the ``method='approximate'`` path of :func:`gamma` and :func:`loggamma`,
  returning ``log(Gamma(a, 1))`` when ``log_space`` is true and ``Gamma(a, 1)``
  otherwise. It avoids the rejection loops of the exact sampler, trading a small
  bias for a fully vectorized computation that is much faster, especially on
  accelerators.

  The sampler has two ingredients. The first is a base draw from a chi-square
  quantile series: since :math:`\mathrm{Gamma}(a, 1) = \chi^2_{2a} / 2`, a draw
  is obtained by cubing a 6th-degree polynomial in a standard normal evaluated at
  :math:`\nu = 2a` degrees of freedom (eq. 18.37 of Johnson, Kotz & Balakrishnan,
  originally Goldstein's Algorithm 451). This series is accurate to about 0.05%
  relative error on the quantiles for shape :math:`\geq 4` (Zar, 1978, table 2)
  but degrades near zero. The second is Stuart's boosting identity (Stuart,
  1962), which for independent uniforms :math:`U_k` on :math:`(0, 1)` gives

  .. math::
     \log G_a = \log G_{a + n} + \sum_{k=0}^{n-1} \frac{\log U_k}{a + k}.

  We take the base draw :math:`G_{a + n}` at the boosted shape :math:`a + n`,
  where the series is accurate, and let the sum carry the small-``a`` behavior,
  which it does exactly. :math:`\log U_k` is drawn as
  :math:`-\mathrm{Exponential}(1)` to avoid evaluating ``log`` at zero.

  With ``n = _LOGGAMMA_NUM_BOOSTS = 4`` the base shape stays :math:`\geq 4` even
  as :math:`a \to 0`. The cubed polynomial can become non-positive for an
  extreme normal draw, which would imply a negative chi-square value; we clip it
  to the smallest positive float, though it should never actually reach that
  value.

  One coefficient in ``_CHI2_QUANTILE_COEF`` departs from the published series:
  the leading constant is set to exactly 1 rather than the published 1.0000886.
  As :math:`\nu \to \infty` the cube root :math:`(\chi^2 / \nu)^{1/3} \to 1`, so
  this term must be 1; the published value is a fit artifact that scales every
  draw by :math:`1.0000886^3`, adding a constant :math:`3 \log(1.0000886)
  \approx 2.7 \times 10^{-4}` bias to ``log(Gamma)``. That bias is negligible
  against the :math:`O(1 / \sqrt{a})` spread at small ``a``, but at large ``a``
  the spread shrinks below it and it would otherwise dominate the sampling error.

  The computation runs in at least float32 and narrows to ``dtype`` only at the
  end: its cost is the rng and transcendentals, which run in float32 regardless,
  so a narrower compute dtype gives no speedup and only loses accuracy in the
  draws. Samples below the smallest representable ``dtype`` value underflow to
  :math:`-\infty` in log space, or to ``0`` otherwise.

  References:
    Stuart, A. (1962). Gamma-distributed products of independent random
    variables. Biometrika, 49(3-4), 564-565.

    Johnson, N. L., Kotz, S., & Balakrishnan, N. (1994). Continuous Univariate
    Distributions, vol. 1 (2nd ed.), eq. 18.37. Wiley.

    Goldstein, R. B. (1973). Algorithm 451: Chi-square quantiles. Communications
    of the ACM, 16(8), 483-485.

    Zar, J. H. (1978). Approximations for the percentage points of the
    chi-squared distribution. Journal of the Royal Statistical Society Series C,
    27(3), 280-290.
  """
  # compute at least in float32; don't use dtypes.promote_types, fails in strict mode
  compute_dtype = np.dtype('float32') if dtypes.finfo(dtype).bits < 32 else dtype
  shape = _check_broadcast_shapes("gamma", shape, a)
  a = lax.convert_element_type(a, compute_dtype)
  sample_s = _sample_sharding("gamma", shape, out_sharding, a)
  a, = _align_params(shape, sample_s, a)
  if np.shape(a) != shape:
    a = jnp.broadcast_to(a, shape)
  key, (a,) = random_insert_pvary('gamma', key, a)

  key_base, key_boost = _split(key)
  n = _LOGGAMMA_NUM_BOOSTS

  # Base draw at the boosted shape a + n, where the series is accurate.
  log_base = _loggamma_chisquare(key_base, a + n, out_sharding=sample_s)

  # Boosting correction sum_k log(U_k) / (a + k), with log(U_k) = -Exp(1).
  log_u = -_exponential(
      key_boost, (n, *shape), compute_dtype,
      out_sharding=_prepend_none_sharding(sample_s))
  log_sample = log_base + sum(log_u[i] / (a + i) for i in range(n))

  out = log_sample if log_space else lax.exp(log_sample)
  return lax.convert_element_type(out, dtype)


def _threefry_uniform_f32(subkey: Key, counts1: Array, counts2: Array) -> Array:
  raw_k = prng.random_unwrap(subkey)
  b0, b1 = threefry2x32.threefry2x32_p.bind(
      raw_k[0], raw_k[1], counts1, counts2)
  bits = lax.bitwise_xor(b0, b1)
  if config.legacy_random_uniform.value:
    float_bits = lax.bitwise_or(
        lax.shift_right_logical(bits, np.uint32(32 - 23)),
        np.uint32(np.float32(1.0).view(np.uint32)))
    return lax.bitcast_convert_type(float_bits, np.float32) - np.float32(1.0)
  float_bits = lax.shift_right_logical(bits, np.uint32(32 - 24))
  return lax.convert_element_type(float_bits, np.float32) * np.float32(2.0**-24)


@jit(static_argnums=(2, 3, 4))
def _poisson_knuth(
    key: Key, lam, shape, dtype, max_iters, counts=None
) -> Array:
  # Knuth's algorithm for generating Poisson random variates.
  # Reference:
  # https://en.wikipedia.org/wiki/Poisson_distribution#Generating_Poisson-distributed_random_variables
  draw_u = ((lambda sk: uniform(sk, shape, np.float32))
            if counts is None
            else (lambda sk: _threefry_uniform_f32(sk, *counts)))

  def body_fn(carry):
    i, k, rng, log_prod = carry
    rng, subkey = _split(rng)
    k = lax.select(log_prod > -lam, k + 1, k)
    u = draw_u(subkey)
    return i + 1, k, rng, log_prod + jnp.log(u)

  def cond_fn(carry):
    i, log_prod = carry[0], carry[3]
    return (log_prod > -lam).any() & (i < max_iters)

  k_init = lax.full_like(lam, 0, dtype, shape)
  log_rate_init = lax.full_like(lam, 0, np.float32, shape)
  k = lax_control_flow.while_loop(
      cond_fn, body_fn, (0, k_init, key, log_rate_init))[1]
  return (k - 1).astype(dtype)


@jit(static_argnums=(2, 3, 4))
def _poisson_rejection(
    key: Key, lam, shape, dtype, max_iters, counts=None
) -> Array:
  # Transformed rejection due to Hormann.
  # Reference:
  # http://citeseer.ist.psu.edu/viewdoc/citations;jsessionid=1BEB35946CC807879F55D42512E5490C?doi=10.1.1.48.3054.
  log_lam = lax.log(lam)
  b = 0.931 + 2.53 * lax.sqrt(lam)
  a = -0.059 + 0.02483 * b
  inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
  v_r = 0.9277 - 3.6224 / (b - 2)
  draw_u = ((lambda sk: uniform(sk, shape, lam.dtype))
            if counts is None
            else (lambda sk: _threefry_uniform_f32(sk, *counts)))

  def body_fn(carry):
    i, k_out, accepted, key = carry
    key, subkey_0, subkey_1 = _split(key, 3)

    u = draw_u(subkey_0) - 0.5
    v = draw_u(subkey_1)
    u_shifted = 0.5 - abs(u)

    k = lax.floor((2 * a / u_shifted + b) * u + lam + 0.43)
    s = lax.log(v * inv_alpha / (a / (u_shifted * u_shifted) + b))
    t = -lam + k * log_lam - lax_special.lgamma(k + 1)

    accept1 = (u_shifted >= 0.07) & (v <= v_r)
    reject = (k < 0) | ((u_shifted < 0.013) & (v > u_shifted))
    accept2 = s <= t
    accept = accept1 | (~reject & accept2)

    k_out = lax.select(accept & ~accepted, k, k_out)
    accepted |= accept

    return i + 1, k_out, accepted, key

  def cond_fn(carry):
    i, k_out, accepted, key = carry
    return (~accepted).any() & (i < max_iters)

  k_init = lax.full_like(lam, -1, lam.dtype, shape)
  accepted = lax.full_like(lam, False, np.dtype('bool'), shape)
  k = lax_control_flow.while_loop(
      cond_fn, body_fn, (0, k_init, accepted, key))[1]
  return k.astype(dtype)


def _poisson_one_key(key: Key, lam, shape, dtype, counts=None) -> Array:
  # The implementation matches TensorFlow and NumPy:
  # https://github.com/tensorflow/tensorflow/blob/v2.2.0-rc3/tensorflow/core/kernels/random_poisson_op.cc
  # https://github.com/numpy/numpy/blob/v1.18.3/numpy/random/src/distributions/distributions.c#L574
  # For lambda < 10, we use the Knuth algorithm; otherwise, we use transformed
  # rejection sampling.
  use_knuth = ~jnp.isfinite(lam) | (lam < 10)
  lam_knuth = lax.select(lam < 10, lam, lax.full_like(lam, 0.0))
  # The acceptance probability for rejection sampling maxes out at 89% as
  # λ -> ∞, so pick some arbitrary large value.
  lam_rejection = lax.select(use_knuth, lax.full_like(lam, 1e5), lam)
  max_iters = dtype.type(dtypes.iinfo(dtype).max)  # insanely conservative
  result = lax.select(
      use_knuth,
      _poisson_knuth(key, lam_knuth, shape, dtype, max_iters, counts),
      _poisson_rejection(key, lam_rejection, shape, dtype, max_iters, counts),
  )
  return lax.select(lam == 0, jnp.zeros_like(result), result)


def _poisson_impl_local(key: Key, lam, *, dtype, counts=None) -> Array:
  trail_shape = np.shape(lam)[key.ndim:]
  if 0 in np.shape(lam):
    return lax.full(np.shape(lam), 0, dtype)
  fn = lambda k, l: _poisson_one_key(k, l, trail_shape, dtype, counts)
  return prng.iterated_vmap_unary(key.ndim, fn)(key, lam)


def _poisson_batching_rule(
    axis_data, batched_args, batch_dims, *, dtype, out_sharding):
  k, lam = batched_args
  bk, bl = batch_dims
  if bk is None and bl is None:
    return _random_poisson_p.bind(
        k, lam, dtype=dtype, out_sharding=out_sharding), None
  mesh_axis = axis_data.explicit_mesh_axis
  k = batching.bdim_at_front(k, bk, axis_data.size, mesh_axis=mesh_axis)
  lam = batching.bdim_at_front(lam, bl, axis_data.size, mesh_axis=mesh_axis)
  if out_sharding is not None:
    out_sharding = batching.get_sharding_for_vmap(
        axis_data, canonicalize_sharding(out_sharding, 'poisson'), 0)
  return _random_poisson_p.bind(
      k, lam, dtype=dtype, out_sharding=out_sharding), 0


_random_poisson_p = core.Primitive('random_poisson')
dispatch.simple_impl(_random_poisson_p)
ad.defjvp_zero(_random_poisson_p)


def _random_poisson_abstract_eval(key_aval, lam_aval, *, dtype, out_sharding):
  if lam_aval.shape[:key_aval.ndim] != key_aval.shape:
    raise ValueError(
        f'random_poisson leading shape of lam {lam_aval.shape} must match key'
        f' shape {key_aval.shape}.')
  if not dtypes.issubdtype(dtype, np.integer):
    raise ValueError(f'random_poisson requires an integer dtype, got {dtype}.')
  out_sharding = canonicalize_sharding(out_sharding, 'poisson')
  sharding = prng.resolve_random_sharding(
      _random_poisson_p,
      partial(_random_gamma_sharding_rule, name='random_poisson'),
      key_aval, lam_aval, out_sharding=out_sharding)
  vma = core.standard_vma_rule('random_poisson', key_aval, lam_aval)
  return core.ShapedArray(
      lam_aval.shape, dtype, sharding=sharding,
      manual_axis_type=core.ManualAxisType(varying=vma))


_random_poisson_p.def_abstract_eval(_random_poisson_abstract_eval)


def _random_poisson_lowering(ctx, key, lam, *, dtype, out_sharding):
  key_aval, lam_aval = ctx.avals_in
  out_aval, = ctx.avals_out
  impl = key_aval.dtype._impl
  out_s = out_aval.sharding
  manual_axes = frozenset(out_s.mesh.explicit_axes)
  out_parts = out_s.spec.partitions
  k_ndim = key_aval.ndim
  trail_shape = lam_aval.shape[k_ndim:]
  trail_parts = out_parts[k_ndim:]
  trail_s = NamedSharding(out_s.mesh, P(*trail_parts))

  def _lower_fn(key_base, lam_val):
    if 0 in lam_aval.shape:
      return lax.full(out_aval.shape, 0, dtype, sharding=out_s)
    if not manual_axes or all(p is None for p in out_parts):
      return _poisson_impl_local(
          prng.random_wrap(key_base, impl=impl), lam_val, dtype=dtype)
    slice_k = _make_shard_slicer(
        out_s.mesh, core.typeof(key_base), out_parts[:k_ndim])
    slice_lam = _make_shard_slicer(out_s.mesh, lam_aval, out_parts)
    if trail_s.is_fully_replicated:
      def _per_shard_unsplit(local_k, local_lam):
        return _poisson_impl_local(
            prng.random_wrap(slice_k(local_k), impl=impl),
            slice_lam(local_lam), dtype=dtype)
      return shard_map.shard_map(
          _per_shard_unsplit, mesh=out_s.mesh, out_specs=out_s.spec,
          axis_names=manual_axes, check_vma=False)(key_base, lam_val)

    if not config.threefry_partitionable.value:
      raise NotImplementedError(
          'out_sharding requires jax_threefry_partitionable=True.')
    counts1, counts2 = prng.iota_2x32_shape(trail_shape, out_sharding=trail_s)
    def _per_shard(local_k, local_lam, local_c1, local_c2):
      return _poisson_impl_local(
          prng.random_wrap(slice_k(local_k), impl=impl),
          slice_lam(local_lam), dtype=dtype,
          counts=(local_c1, local_c2))
    return shard_map.shard_map(
        _per_shard, mesh=out_s.mesh, out_specs=out_s.spec,
        axis_names=manual_axes, check_vma=False)(
            key_base, lam_val, counts1, counts2)

  lowering = mlir.lower_fun(_lower_fn, multiple_results=False)
  phys_key = core.physical_aval(key_aval).update(
      manual_axis_type=core.empty_mat)
  lam_in = lam_aval.update(manual_axis_type=core.empty_mat)
  out_in = out_aval.update(manual_axis_type=core.empty_mat)
  return mlir.delegate_lowering(
      ctx, lowering, key, lam, avals_in=[phys_key, lam_in], avals_out=[out_in])


mlir.register_lowering(
    _random_poisson_p, _random_poisson_lowering, shard_map_passthrough=True)
batching.fancy_primitive_batchers[_random_poisson_p] = _poisson_batching_rule


@jit(static_argnums=(2, 3, 4))
def _poisson(key: Key, lam, shape, dtype, out_sharding=None) -> Array:
  _check_param_sharding("poisson", shape, out_sharding, lam)
  lam = lax.convert_element_type(lam, np.float32)
  if np.shape(lam) != shape:
    lam = jnp.broadcast_to(lam, shape)
  key, (lam,) = random_insert_pvary('poisson', key, lam)
  return _random_poisson_p.bind(key, lam, dtype=dtype, out_sharding=out_sharding)


# Below this mean, `_poisson_approx` samples by inverting the truncated cdf
# directly; above, it inverts the Peizer-Pratt normal approximation of the cdf,
# whose max cdf error at the split is ~6e-5 (chi-square-detectable only beyond
# ~3e7 samples).
_POISSON_SPLIT = 7.0

# Number of pmf terms for the truncated cdf; the truncated tail mass
# Pr[X > 19 | lam = 7] ~ 4e-5 keeps the truncated-branch total variation below
# the Peizer-Pratt error at the split (~6e-5).
_POISSON_NUM_TERMS = 20

# Peizer-Pratt continuity correction constant (0 for simplicity, 0.02 for more
# accuracy).
_PEIZER_PRATT_EPS = 0.02


@jit(static_argnums=(2, 3, 4))
def _poisson_approx(key: Key, lam, shape, dtype, out_sharding=None) -> Array:
  r"""Loop-free approximate sampler for ``Poisson(lam)``.

  Backs the ``method='approximate'`` path of :func:`poisson`. It avoids the
  rejection loops of the exact sampler, trading a small bias for a fully
  vectorized computation that is much faster.

  The total variation distance from the exact distribution stays below 1e-4,
  worst around ``lam == _POISSON_SPLIT``.

  A standard normal draw is mapped monotonically to a count by inverting an
  approximate cdf, in :func:`_poisson_from_normal`. Below `_POISSON_SPLIT` the
  pmf is truncated to `_POISSON_NUM_TERMS` terms and the cdf inverted directly;
  above, the sampler inverts the very accurate normal approximation of the cdf
  by Peizer and Pratt (1968), as presented in Johnson, Kemp & Kotz (2005), page
  168.

  The computation runs in float32, or float64 if ``lam`` is a 64-bit float
  array or ``dtype`` is a 64-bit type: above ``lam ~ 2**24`` the samples land
  on the float32 integer lattice, so wider means need the wider compute dtype
  to resolve them, and a 64-bit ``dtype`` is the only way to represent such
  counts in the output.

  References:
    Peizer, D. B., & Pratt, J. W. (1968). A Normal Approximation for Binomial,
    F, Beta, and other Common, Related Tail Probabilities, I. Journal of the
    American Statistical Association, 63(324), 1416-1456.

    Johnson, N. L., Kemp, A. W., & Kotz, S. (2005). Univariate discrete
    distributions (3rd ed), page 168. Wiley.
  """
  # compute at least in float32; don't use dtypes.promote_types, fails in
  # strict mode
  lam_dtype = dtypes.dtype(lam)
  lam_is_f64 = (dtypes.issubdtype(lam_dtype, np.floating)
                and dtypes.finfo(lam_dtype).bits > 32)
  if lam_is_f64 or np.dtype(dtype).itemsize > 4:
    compute_dtype = np.dtype('float64')
  else:
    compute_dtype = np.dtype('float32')

  sample_sharding = _sample_sharding("poisson", shape, out_sharding, lam)
  lam = lax.convert_element_type(lam, compute_dtype)
  if np.ndim(lam):
    lam = lax.expand_dims(lam, tuple(range(len(shape) - np.ndim(lam))))
    if (sample_sharding is not None and
        any(p is not None for p in sample_sharding.spec.partitions) and
        core.typeof(lam).sharding != sample_sharding):
      lam = reshard(jnp.broadcast_to(lam, shape), sample_sharding)
  key, (lam,) = random_insert_pvary('poisson', key, lam)
  z = _normal(key, shape, compute_dtype, out_sharding=sample_sharding)
  return lax.convert_element_type(_poisson_from_normal(z, lam), dtype)


def _poisson_from_normal(z, lam):
  """Map a standard normal variate to an approximate Poisson(lam) variate."""
  valid = (lam > 0) & jnp.isfinite(lam)
  lam_safe = jnp.where(valid, lam, 1.0)
  small = _poisson_cdf_inversion(ndtr(z), lam_safe)
  large = _poisson_peizer_pratt(z, lam_safe)
  out = jnp.where(lam_safe < _POISSON_SPLIT, small, large)
  return jnp.where(lam == 0, 0.0, jnp.where(valid, out, -1.0))


def _poisson_cdf_inversion(u, lam):
  """Invert the cdf truncated to the first `_POISSON_NUM_TERMS` values."""
  # unrolled pmf recurrence instead of cumsum(exp(logpmf(arange))): keeps every
  # op elementwise at the broadcast shape, so xla fuses the inversion into one
  # kernel instead of materializing shape x _POISSON_NUM_TERMS intermediates
  pmf = lax.exp(-lam)
  cdf = pmf
  # if the final cdf >= 1 due to rounding, the last possible value is never
  # selected, no overflow
  count = lax.convert_element_type(cdf < u, u.dtype)
  for k in range(1, _POISSON_NUM_TERMS - 1):
    pmf *= lam / k
    cdf += pmf
    count += lax.convert_element_type(cdf < u, u.dtype)
  return count


def _poisson_peizer_pratt(z, lam):
  """Invert the Peizer-Pratt cdf approximation Pr[X <= x] ~ ndtr(z(x)).

  The sample is the smallest integer x with z(x) >= z, i.e., ceil of the real
  root of z(x) = z. The exact z(x) is evaluated at 2 integers around an
  analytical guess to select the right value, correcting the guess as long as
  it falls within +/-1 of the root.
  """
  # the guess error stays within (-1 + 0.6, 1 + 0.6) of the root over
  # lam >= _POISSON_SPLIT, |z| <= 8.4 (the float64 range of normal draws), so
  # shifting by 0.6 centers it in the +/-1 window around ceil
  x = _peizer_pratt_root(z, lam) + 0.6

  # select the smallest integer c with z(c) >= z among {c-1, c, c+1},
  # c = ceil(x); z at negative integers is -inf since Pr[X <= x] = 0 there
  c = jnp.ceil(x)

  def zint(c):
    return jnp.where(c < 0, -np.inf, _peizer_pratt_z(jnp.maximum(c, 0.0), lam))

  def root_ge(c):
    return lax.convert_element_type(zint(c) >= z, x.dtype)

  out = c + 1 - root_ge(c) - root_ge(c - 1)
  return jnp.maximum(out, 0.0)


def _peizer_pratt_root(z, lam):
  """Cornish-Fisher approximation of the root of `_peizer_pratt_z(x, .) = z`."""
  s = jnp.sqrt(lam)
  return (
      lam + s * z - 2 / 3 + jnp.square(z) / 6 + z * (1 - jnp.square(z)) / (72 * s))


def _peizer_pratt_z(x, lam):
  """Peizer-Pratt z(x) such that Pr[X <= x] ~ ndtr(z(x)), for x > -1/2."""
  y = (x + 0.5) / lam
  t = _peizer_pratt_t(y)
  return (x - lam + 2 / 3 + _PEIZER_PRATT_EPS / (x + 1)) * jnp.sqrt((1 + t) / lam)


def _peizer_pratt_t(y):
  """T(y) = (1 - y^2 + 2 y log y) / (1 - y)^2, stable around y = 1."""
  d = y - 1
  small = jnp.abs(d) < 0.25

  # series: T(1 + d) = sum_{k>=1} 2 (-1)^k d^k / ((k+1) (k+2)); 9 terms are the
  # fewest that keep the truncation error at the cutoff (~2e-8) below float32
  # resolution; in float64 the Peizer-Pratt error dominates anyway
  coeffs = [2 * (-1) ** k / ((k + 1) * (k + 2)) for k in range(1, 10)]
  series = d * lax.polynomial(d, coeffs, unroll=len(coeffs))

  y_safe = jnp.where(small, 2.0, y)
  d_safe = jnp.where(small, 1.0, d)
  direct = (1 - jnp.square(y_safe) + 2 * y_safe * jnp.log(y_safe)) / jnp.square(
      d_safe)

  return jnp.where(small, series, direct)


def poisson(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
            lam: RealArray,
            shape: Shape | None = None,
            dtype: DTypeLikeInt | None = None,
            *,
            method: str = 'exact',
            out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample Poisson random values with given shape and integer dtype.

  The values are distributed according to the probability mass function:

  .. math::
     f(k; \lambda) = \frac{\lambda^k e^{-\lambda}}{k!}

  Where `k` is a non-negative integer and :math:`\lambda > 0`.

  Args:
    key: a PRNG key used as the random key.
    lam: rate parameter (mean of the distribution), must be >= 0. Must be broadcast-compatible with ``shape``
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default (None) produces a result shape equal to ``lam.shape``.
    dtype: optional, a integer dtype for the returned values (default int64 if
      jax_enable_x64 is true, otherwise int32).
    method: optional, the sampling algorithm to use, either ``'exact'`` (the
      default) or ``'approximate'``. The ``'exact'`` method uses rejection
      sampling and supports only the threefry2x32 RNG. The ``'approximate'``
      method is loop-free and faster but approximate: the total variation
      distance from the exact distribution is below 1e-4.
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape is not None, or else by ``lam.shape``.
  """
  key, _ = _check_prng_key("poisson", key)
  if method not in {'exact', 'approximate'}:
    raise ValueError("method argument to `poisson` must be one of "
                     f"{{'exact', 'approximate'}}, got {method!r}")
  dtype = dtypes.check_and_canonicalize_user_dtype(
      int if dtype is None else dtype)
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  else:
    shape = np.shape(lam)
  if lax.broadcast_shapes(np.shape(lam), shape) != shape:
    raise ValueError("lam shape must be broadcastable to shape argument; "
                     f"got lam.shape {np.shape(lam)}, shape {shape}")
  out_sharding = canonicalize_sharding_for_samplers(
      out_sharding, "poisson", shape)
  if method == 'approximate':
    return _poisson_approx(key, lam, shape, dtype, out_sharding)
  # TODO(frostig): generalize underlying poisson implementation and
  # remove this check
  keys_dtype = typing.cast(prng.KeyTy, key.dtype)
  key_impl = keys_dtype._impl
  if key_impl is not threefry2x32.threefry_prng_impl:
    raise NotImplementedError(
        "`poisson` with method='exact' is only implemented for the "
        f'threefry2x32 RNG, not {key_impl}')
  return _poisson(key, lam, shape, dtype, out_sharding)


def gumbel(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
           shape: Shape = (),
           dtype: DTypeLikeFloat | None = None,
           mode: str | None = None,
           *,
           out_sharding: NamedSharding | P | None = None) -> Array:
  """Sample Gumbel random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(x) = e^{-(x + e^{-x})}

  Args:
    key: a PRNG key used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    mode: optional, "high" or "low" for how many bits to use when sampling.
      The default is determined by the ``use_high_dynamic_range_gumbel`` config,
      which defaults to "low". When drawing float32 samples, with mode="low" the
      uniform resolution is such that the largest possible gumbel logit is ~16;
      with mode="high" this is increased to ~32, at approximately double the
      computational cost.
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified shape and dtype.
  """
  key, _ = _check_prng_key("gumbel", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `gumbel` must be a float "
                     f"dtype, got {dtype}")
  shape = core.canonicalize_shape(shape)
  if mode is None:
    mode = "high" if config.use_high_dynamic_range_gumbel.value else "low"
  if mode not in ("highest", "high", "low"):
    raise ValueError("Must provide valid mode for gumbel got: %s" % mode)
  out_sharding = canonicalize_sharding(out_sharding, "gumbel")
  return _gumbel(
      key, shape=shape, dtype=dtype, mode=mode, out_sharding=out_sharding)

def _safe_int_to_float(
    bits: Array | tuple[Array, Array], dtype: DTypeLike
) -> Array:
  """Converts bits: (hi, lo) tuple or u32[2,...] into f32[...] in (0,1)."""
  hi, lo = bits
  if hi.dtype != np.uint32 or lo.dtype != np.uint32 or dtype != np.float32:
    raise RuntimeError("_safe_int_to_float only works for u32 -> f32")
  finfo = dtypes.finfo(dtype)
  hiclz = lax.clz(hi)
  loclz = lax.clz(lo)

  mantissa = lax.bitwise_or(
      lax.shift_left(hi, hiclz),
      jnp.where(
          hiclz == 32,
          lax.shift_left(lo, loclz),
          lax.shift_right_logical(lo, finfo.bits - hiclz)))
  mantissa = lax.shift_right_logical(
      mantissa, np.array(finfo.bits - finfo.nmant - 1, dtype=np.uint32))
  mantissa = lax.bitwise_and(
      mantissa, np.array((1 << finfo.nmant) - 1, dtype=np.uint32))
  exp = lax.shift_left(
      (-finfo.minexp - jnp.where(hiclz == 32, 32 + loclz, hiclz)),
      np.array(finfo.nmant, dtype=np.uint32))
  exp = lax.bitwise_and(exp, np.array(
      (1 << (finfo.bits - 2)) - (1 << (finfo.nmant - 1)), dtype=np.uint32))
  return lax.bitwise_or(exp, mantissa).view(dtype)


@jit(static_argnums=(1, 2, 3, 4))
def _gumbel(key: Key, shape, dtype, mode, out_sharding=None) -> Array:
  _check_shape("gumbel", shape)
  info = dtypes.finfo(dtype)
  if dtype == np.float32 and mode == "highest":
    bits = _random_bits(
        key, info.bits, shape=(2,) + shape,
        out_sharding=_prepend_none_sharding(out_sharding))
    hi, lo = bits[0], bits[1]
    neg_hi = lax.bitwise_not(hi)
    neg_lo = lax.bitwise_not(lo)
    # 1 - bits in u64 fixed point.
    neg_hi = neg_hi + (
        neg_lo == np.uint32((1 << info.bits) - 1)
    ).astype(np.uint32)
    neg_lo = neg_lo + np.uint32(1)
    flip_mask = hi < np.uint32(1 << (info.bits - 1))
    sel_hi = jnp.where(flip_mask, hi, neg_hi)
    sel_lo = jnp.where(flip_mask, lo, neg_lo)
    x = _safe_int_to_float((sel_hi, sel_lo), dtype=np.float32)
    # use log1p for (0,0.5) and log for [0.5, 1).
    return jnp.where(flip_mask,
        -jnp.log(-jnp.log1p(-x)), -jnp.log(-jnp.log(x)))
  elif mode == "high" or mode == "highest":
    high, low = lax.unstack(_uniform(
        key, minval=0., maxval=1., shape=(2,) + shape, dtype=dtype,
        out_sharding=_prepend_none_sharding(out_sharding)))
    # TODO(parkers): The condition is to protect against rounding up but
    # we should be able to add safely with the right addition operation.
    n_bits = info.nmant + (0 if config.legacy_random_uniform.value else 1)
    x = jnp.where(high >= 0.5, high, high + 2 ** -n_bits * low + max(
        2 ** -(2 * n_bits + 1), info.tiny))
    return -jnp.log(-jnp.log1p(-x))
  else:
    return -jnp.log(-jnp.log(
        _uniform(key, minval=info.tiny, maxval=1., shape=shape, dtype=dtype,
                 out_sharding=out_sharding)))


def categorical(
  key: Key | ArrayLike,  # TODO(jakevdp): make this Key
  logits: RealArray,
  axis: int = -1,
  shape: Shape | None = None,
  replace: bool = True,
  mode: str | None = None,
  *,
  out_sharding: NamedSharding | P | None = None
) -> Array:
  """Sample random values from categorical distributions.

  Sampling with replacement uses the Gumbel max trick. Sampling without replacement uses
  the Gumbel top-k trick. See [1] for reference.

  Args:
    key: a PRNG key used as the random key.
    logits: Unnormalized log probabilities of the categorical distribution(s) to sample from,
      so that `softmax(logits, axis)` gives the corresponding probabilities.
    axis: Axis along which logits belong to the same categorical distribution.
    shape: Optional, a tuple of nonnegative integers representing the result shape.
      Must be broadcast-compatible with ``np.delete(logits.shape, axis)``.
      The default (None) produces a result shape equal to ``np.delete(logits.shape, axis)``.
    replace: If True (default), perform sampling with replacement. If False, perform
      sampling without replacement.
    mode: optional, "high" or "low" for how many bits to use in the gumbel sampler.
      The default is determined by the ``use_high_dynamic_range_gumbel`` config,
      which defaults to "low". With mode="low", in float32 sampling will be biased
      for events with probability less than about 1E-7; with mode="high" this limit
      is pushed down to about 1E-14. mode="high" approximately doubles the cost of
      sampling.
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with int dtype and shape given by ``shape`` if ``shape``
    is not None, or else ``np.delete(logits.shape, axis)``.

  References:
    .. [1] Wouter Kool, Herke van Hoof, Max Welling. "Stochastic Beams and Where to Find
      Them: The Gumbel-Top-k Trick for Sampling Sequences Without Replacement".
      Proceedings of the 36th International Conference on Machine Learning, PMLR
      97:3499-3508, 2019. https://proceedings.mlr.press/v97/kool19a.html.
  """
  key, _ = _check_prng_key("categorical", key)
  check_arraylike("categorical", logits)
  logits_arr = jnp.asarray(logits)
  batch_shape = tuple(np.delete(logits_arr.shape, axis))
  if shape is None:
    shape = batch_shape
  else:
    shape = core.canonicalize_shape(shape)
    _check_shape("categorical", shape, batch_shape)
  out_sharding = canonicalize_sharding_for_samplers(
      out_sharding, "categorical", shape)
  return _categorical(
      key, logits_arr, shape=shape, batch_shape=batch_shape, axis=axis,
      replace=replace, mode=mode, out_sharding=out_sharding)


def _categorical(
    key: Key, logits_arr, shape, batch_shape, axis, replace, mode,
    out_sharding=None,
) -> Array:
  prefix_ndim = len(shape) - len(batch_shape)
  axis = canonicalize_axis(axis, logits_arr.ndim)
  out_s = out_sharding
  logits_s = core.typeof(logits_arr).sharding
  if not logits_s.mesh.empty and logits_s.spec.partitions[axis] is not None:
    raise core.ShardingTypeError(
        f"categorical cannot sample along sharded axis {axis}; got"
        f" {logits_s.spec}.")
  cat_axis = prefix_ndim + axis
  g_out_shape = (
      *shape[:cat_axis], logits_arr.shape[axis], *shape[cat_axis:])
  out_g_s = (
      None if out_s is None else out_s.update(
          spec=P(*out_s.spec.partitions[:cat_axis], None,
                 *out_s.spec.partitions[cat_axis:])))
  _check_param_sharding("categorical", g_out_shape, out_g_s, logits_arr)
  key, (logits_arr,) = random_insert_pvary("categorical", key, logits_arr)

  if replace:
    gumbel_sharding = _sample_sharding(
        "categorical", g_out_shape, out_g_s, logits_arr)
    logits_arr, = _align_params(g_out_shape, gumbel_sharding, logits_arr)
    g = gumbel(key, g_out_shape, logits_arr.dtype, mode=mode,
               out_sharding=gumbel_sharding)
    return jnp.argmax(
        g + jnp.broadcast_to(logits_arr, g_out_shape), axis=cat_axis)

  k = math.prod(shape[:prefix_ndim])
  if k > logits_arr.shape[axis]:
    raise ValueError(
        f"Number of samples without replacement ({k}) cannot exceed number"
        f" of categories ({logits_arr.shape[axis]}).")
  logits_parts = (
      (None,) * logits_arr.ndim
      if logits_s.mesh.empty else logits_s.spec.partitions)
  logits_target_s = (
      None if out_g_s is None else core.get_sharding(
          out_g_s.update(
              spec=P(*(p if core.definitely_equal(d, gd) else lp
                       for p, d, gd, lp in zip(
                           out_g_s.spec.partitions[prefix_ndim:],
                           logits_arr.shape,
                           g_out_shape[prefix_ndim:],
                           logits_parts)))),
          logits_arr.shape))
  gumbel_sharding = _sample_sharding(
      "categorical", logits_arr.shape, logits_target_s, logits_arr)
  logits_arr, = _align_params(
      logits_arr.shape, gumbel_sharding, logits_arr)
  logits_arr = logits_arr + gumbel(
      key, logits_arr.shape, logits_arr.dtype, mode=mode,
      out_sharding=gumbel_sharding)
  _, indices = lax.top_k(jnp.moveaxis(logits_arr, axis, -1), k)
  return lax.reshape(jnp.moveaxis(indices, -1, 0), shape, out_sharding=out_s)


def laplace(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
            shape: Shape = (),
            dtype: DTypeLikeFloat | None = None,
            *,
            out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample Laplace random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
    f(x) = \frac{1}{2}e^{-|x|}

  Args:
    key: a PRNG key used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified shape and dtype.
  """
  key, _ = _check_prng_key("laplace", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `laplace` must be a float "
                     f"dtype, got {dtype}")
  shape = core.canonicalize_shape(shape)
  out_sharding = canonicalize_sharding(out_sharding, "laplace")
  return _laplace(key, shape, dtype, out_sharding)


@jit(static_argnums=(1, 2, 3))
def _laplace(key: Key, shape, dtype, out_sharding=None) -> Array:
  _check_shape("laplace", shape)
  u = uniform(key, shape, dtype, minval=-1. + dtypes.finfo(dtype).epsneg,
              maxval=1., out_sharding=out_sharding)
  return lax.mul(lax.sign(u), lax.log1p(lax.neg(lax.abs(u))))


def logistic(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
             shape: Shape = (),
             dtype: DTypeLikeFloat | None = None,
             *,
             out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample logistic random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(x) = \frac{e^{-x}}{(1 + e^{-x})^2}

  Args:
    key: a PRNG key used as the random key.
    shape: optional, a tuple of nonnegative integers representing the result
      shape. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified shape and dtype.
  """
  key, _ = _check_prng_key("logistic", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `logistic` must be a float "
                     f"dtype, got {dtype}")
  shape = core.canonicalize_shape(shape)
  out_sharding = canonicalize_sharding(out_sharding, "logistic")
  return _logistic(key, shape, dtype, out_sharding)


@jit(static_argnums=(1, 2, 3))
def _logistic(key: Key, shape, dtype, out_sharding=None):
  _check_shape("logistic", shape)
  x = uniform(key, shape, dtype, minval=dtypes.finfo(dtype).tiny, maxval=1.,
              out_sharding=out_sharding)
  return lax.sub(lax.log(x), lax.log1p(lax.neg(x)))


def pareto(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
           b: RealArray,
           shape: Shape | None = None,
           dtype: DTypeLikeFloat | None = None,
           *,
           out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample Pareto random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(x; b) = b / x^{b + 1}

  on the domain :math:`1 \le x < \infty` with :math:`b > 0`

  Args:
    key: a PRNG key used as the random key.
    b: a float or array of floats broadcast-compatible with ``shape``
      representing the parameter of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``b``. The default (None)
      produces a result shape equal to ``b.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``b.shape``.
  """
  key, _ = _check_prng_key("pareto", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `pareto` must be a float "
                     f"dtype, got {dtype}")
  shape = _check_broadcast_shapes("pareto", shape, b)
  _check_all_safe_to_cast("pareto", dtype, b)
  out_sharding = canonicalize_sharding(out_sharding, "pareto")
  return _pareto(key, b, shape, dtype, out_sharding)


@jit(static_argnums=(2, 3, 4))
def _pareto(key: Key, b, shape, dtype, out_sharding=None) -> Array:
  b = lax.convert_element_type(b, dtype)
  sample_s = _sample_sharding("pareto", shape, out_sharding, b)
  b, = _align_params(shape, sample_s, b)
  e = exponential(key, shape, dtype, out_sharding=sample_s)
  return lax.exp(e / b)


def t(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
      df: RealArray,
      shape: Shape | None = None,
      dtype: DTypeLikeFloat | None = None,
      *,
      out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample Student's t random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(t; \nu) \propto \left(1 + \frac{t^2}{\nu}\right)^{-(\nu + 1)/2}

  Where :math:`\nu > 0` is the degrees of freedom, given by the parameter ``df``.

  Args:
    key: a PRNG key used as the random key.
    df: a float or array of floats broadcast-compatible with ``shape``
      representing the degrees of freedom parameter of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``df``. The default (None)
      produces a result shape equal to ``df.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``df.shape``.
  """
  key, _ = _check_prng_key("t", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `t` must be a float "
                     f"dtype, got {dtype}")
  shape = _check_broadcast_shapes("t", shape, df)
  _check_all_safe_to_cast("t", dtype, df)
  out_sharding = canonicalize_sharding(out_sharding, "t")
  return _t(key, df, shape, dtype, out_sharding)


@jit(static_argnums=(2, 3, 4))
def _t(key: Key, df, shape, dtype, out_sharding=None) -> Array:
  if shape is None:
    shape = np.shape(df)
  else:
    _check_shape("t", shape, np.shape(df))

  df = lax.convert_element_type(df, dtype)
  sample_s = _sample_sharding("t", shape, out_sharding, df)
  df, = _align_params(shape, sample_s, df)
  key_n, key_g = _split(key)
  n = normal(key_n, shape, dtype, out_sharding=sample_s)
  two = lax._const(n, 2)
  half_df = lax.div(df, two)
  g = gamma(key_g, half_df, shape, dtype, out_sharding=sample_s)
  return n * jnp.sqrt(half_df / g)


def chisquare(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
              df: RealArray,
              shape: Shape | None = None,
              dtype: DTypeLikeFloat | None = None,
              *,
              method: str = "exact",
              out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample Chisquare random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(x; \nu) \propto x^{\nu/2 - 1}e^{-x/2}

  on the domain :math:`0 < x < \infty`, where :math:`\nu > 0` represents the
  degrees of freedom, given by the parameter ``df``.

  Args:
    key: a PRNG key used as the random key.
    df: a float or array of floats broadcast-compatible with ``shape``
      representing the parameter of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``df``. The default (None)
      produces a result shape equal to ``df.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    method: optional, the sampling algorithm to use, either ``'exact'`` (the
      default) or ``'approximate'``. The ``'exact'`` method is a rejection
      sampler. The ``'approximate'`` method is loop-free and faster but carries
      a small bias. The gradient w.r.t. ``df`` differs between the two
      methods because of the ambiguity in defining a gradient for random variates.
    out_sharding: optional, Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``df.shape``.
  """
  key, _ = _check_prng_key("chisquare", key)
  if method not in {"exact", "approximate"}:
    raise ValueError("method argument to `chisquare` must be one of "
                     f"{{'exact', 'approximate'}}, got {method!r}")
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError("dtype argument to `chisquare` must be a float "
                     f"dtype, got {dtype}")
  shape = _check_broadcast_shapes("chisquare", shape, df)
  _check_all_safe_to_cast("chisquare", dtype, df)
  out_sharding = canonicalize_sharding(out_sharding, "chisquare")
  return _chisquare(key, df, method, shape, dtype, out_sharding)


@jit(static_argnums=(2, 3, 4, 5))
def _chisquare(key: Key, df, method, shape, dtype, out_sharding=None) -> Array:
  df = lax.convert_element_type(df, dtype)
  sample_s = _sample_sharding("chisquare", shape, out_sharding, df)
  df, = _align_params(shape, sample_s, df)
  two = lax._const(df, 2)
  log_g = loggamma(
      key, a=lax.div(df, two), shape=shape, dtype=dtype, method=method,
      out_sharding=sample_s)
  return lax.mul(jnp.exp(log_g), two)


def f(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
      dfnum: RealArray,
      dfden: RealArray,
      shape: Shape | None = None,
      dtype: DTypeLikeFloat | None = None,
      *,
      out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample F-distribution random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
     f(x; \nu_1, \nu_2) \propto x^{\nu_1/2 - 1}\left(1 + \frac{\nu_1}{\nu_2}x\right)^{
      -(\nu_1 + \nu_2) / 2}

  on the domain :math:`0 < x < \infty`. Here :math:`\nu_1` is the degrees of
  freedom of the numerator (``dfnum``), and :math:`\nu_2` is the degrees of
  freedom of the denominator (``dfden``).

  Args:
    key: a PRNG key used as the random key.
    dfnum: a float or array of floats broadcast-compatible with ``shape``
      representing the numerator's ``df`` of the distribution.
    dfden: a float or array of floats broadcast-compatible with ``shape``
      representing the denominator's ``df`` of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``dfnum`` and ``dfden``.
      The default (None) produces a result shape equal to ``dfnum.shape``,
      and ``dfden.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    out_sharding: optional, specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``df.shape``.
  """
  key, _ = _check_prng_key("f", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError("dtype argument to `f` must be a float "
                     f"dtype, got {dtype}")
  shape = _check_broadcast_shapes("f", shape, dfnum, dfden)
  _check_all_safe_to_cast("f", dtype, dfnum, dfden)
  out_sharding = canonicalize_sharding(out_sharding, "f")
  return _f(key, dfnum, dfden, shape, dtype, out_sharding)


@jit(static_argnums=(3, 4, 5))
def _f(key: Key, dfnum, dfden, shape, dtype, out_sharding=None) -> Array:
  dfden = lax.convert_element_type(dfden, dtype)
  dfnum = lax.convert_element_type(dfnum, dtype)
  sample_s = _sample_sharding("f", shape, out_sharding, dfnum, dfden)
  dfnum, dfden = _align_params(shape, sample_s, dfnum, dfden)
  key_dfd, key_dfn = _split(key)
  chi2_dfn = chisquare(key_dfn, dfnum, shape, dtype, out_sharding=sample_s)
  chi2_dfd = chisquare(key_dfd, dfden, shape, dtype, out_sharding=sample_s)
  return lax.div(chi2_dfn / dfnum, chi2_dfd / dfden)


def rademacher(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
               shape: Shape = (),
               dtype: DTypeLikeInt | None = None,
               *,
               out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample from a Rademacher distribution.

  The values are distributed according to the probability mass function:

  .. math::
     f(k) = \frac{1}{2}(\delta(k - 1) + \delta(k + 1))

  on the domain :math:`k \in \{-1, 1\}`, where :math:`\delta(x)` is the dirac delta function.

  Args:
    key: a PRNG key.
    shape: The shape of the returned samples. Default ().
    dtype: The type used for samples.
    out_sharding: optional, specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A jnp.array of samples, of shape `shape`. Each element in the output has
    a 50% change of being 1 or -1.

  """
  key, _ = _check_prng_key("rademacher", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      int if dtype is None else dtype)
  shape = core.canonicalize_shape(shape)
  out_sharding = canonicalize_sharding(out_sharding, "rademacher")
  return _rademacher(key, shape, dtype, out_sharding)


@jit(static_argnums=(1, 2, 3))
def _rademacher(key: Key, shape, dtype, out_sharding=None) -> Array:
  bernoulli_samples = bernoulli(key=key, p=0.5, shape=shape,
                                out_sharding=out_sharding).astype(dtype)
  return (2 * bernoulli_samples - 1).astype(dtype)


def maxwell(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
            shape: Shape = (),
            dtype: DTypeLikeFloat | None = None,
            *,
            out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample from a one sided Maxwell distribution.

  The values are distributed according to the probability density function:

  .. math::
     f(x) \propto x^2 e^{-x^2 / 2}

  on the domain :math:`0 \le x < \infty`.

  Args:
    key: a PRNG key.
    shape: optional, the batch dimensions of the result. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    out_sharding: optional, specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A jnp.array of samples, of shape `shape`.

  """
  # Generate samples using:
  # sqrt(X^2 + Y^2 + Z^2), X,Y,Z ~N(0,1)
  key, _ = _check_prng_key("maxwell", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `maxwell` must be a float "
                     f"dtype, got {dtype}")
  shape = core.canonicalize_shape(shape)
  out_sharding = canonicalize_sharding(out_sharding, "maxwell")
  return _maxwell(key, shape, dtype, out_sharding)


@jit(static_argnums=(1, 2, 3))
def _maxwell(key: Key, shape, dtype, out_sharding=None) -> Array:
  if out_sharding is not None:
    out_s = core.get_sharding(out_sharding, shape)
    out_sharding = out_s.update(
        spec=out_s.spec.update(partitions=(*out_s.spec.partitions, None)))
  norm_rvs = normal(
      key=key, shape=shape + (3,), dtype=dtype, out_sharding=out_sharding)
  return jnp_linalg.norm(norm_rvs, axis=-1)


def double_sided_maxwell(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
                         loc: RealArray,
                         scale: RealArray,
                         shape: Shape = (),
                         dtype: DTypeLikeFloat | None = None) -> Array:
  r"""Sample from a double sided Maxwell distribution.

  The values are distributed according to the probability density function:

  .. math::
     f(x;\mu,\sigma) \propto z^2 e^{-z^2 / 2}

  where :math:`z = (x - \mu) / \sigma`, with the center :math:`\mu` specified by
  ``loc`` and the scale :math:`\sigma` specified by ``scale``.

  Args:
    key: a PRNG key.
    loc: The location parameter of the distribution.
    scale: The scale parameter of the distribution.
    shape: The shape added to the parameters loc and scale broadcastable shape.
    dtype: The type used for samples.

  Returns:
    A jnp.array of samples.

  """
  key, _ = _check_prng_key("double_sided_maxwell", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `double_sided_maxwell` must be a float"
                     f" dtype, got {dtype}")
  shape = core.canonicalize_shape(shape)
  return _double_sided_maxwell(key, loc, scale, shape, dtype)


@jit(static_argnums=(3, 4))
def _double_sided_maxwell(key: Key, loc, scale, shape, dtype) -> Array:
  params_shapes = lax.broadcast_shapes(np.shape(loc), np.shape(scale))
  if not shape:
    shape = params_shapes

  shape = shape + params_shapes
  maxwell_key, rademacher_key = _split(key)
  maxwell_rvs = maxwell(maxwell_key, shape=shape, dtype=dtype)
  # Generate random signs for the symmetric variates.
  random_sign = rademacher(rademacher_key, shape=shape, dtype=dtype)
  assert random_sign.shape == maxwell_rvs.shape

  return random_sign * maxwell_rvs * scale + loc


def weibull_min(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
                scale: RealArray,
                concentration: RealArray,
                shape: Shape = (),
                dtype: DTypeLikeFloat | None = None) -> Array:
  r"""Sample from a Weibull distribution.

  The values are distributed according to the probability density function:

  .. math::
     f(x;\sigma,c) \propto x^{c - 1} \exp(-(x / \sigma)^c)

  on the domain :math:`0 < x < \infty`, where :math:`c > 0` is the concentration
  parameter, and :math:`\sigma > 0` is the scale parameter.

  Args:
    key: a PRNG key.
    scale: The scale parameter of the distribution.
    concentration: The concentration parameter of the distribution.
    shape: The shape added to the parameters loc and scale broadcastable shape.
    dtype: The type used for samples.

  Returns:
    A jnp.array of samples.

  """
  key, _ = _check_prng_key("weibull_min", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(f"dtype argument to `weibull_min` must be a float "
                     f"dtype, got {dtype}")
  shape = core.canonicalize_shape(shape)
  return _weibull_min(key, scale, concentration, shape, dtype)


@jit(static_argnums=(3, 4))
def _weibull_min(key: Key, scale, concentration, shape, dtype) -> Array:
  random_uniform = uniform(
      key=key, shape=shape, minval=0, maxval=1, dtype=dtype)

  # Inverse weibull CDF.
  return jnp.power(-jnp.log1p(-random_uniform), 1.0/concentration) * scale


def orthogonal(
  key: Key | ArrayLike,  # TODO(jakevdp): make this Key
  n: int,
  shape: Shape = (),
  dtype: DTypeLikeFloat | None = None,
  m: int | None = None,
  *,
  out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample uniformly from the orthogonal group O(n).

  If the dtype is complex, sample uniformly from the unitary group U(n).

  For unequal rows and columns, this samples a semi-orthogonal matrix instead.
  That is, if :math:`A` is the resulting matrix and :math:`A^*` is its conjugate
  transpose, then:

  - If :math:`n \leq m`, the rows are mutually orthonormal: :math:`A A^* = I_n`.
  - If :math:`m \leq n`, the columns are mutually orthonormal: :math:`A^* A = I_m`.

  Args:
    key: a PRNG key used as the random key.
    n: an integer indicating the number of rows.
    shape: optional, the batch dimensions of the result. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    m: an integer indicating the number of columns. Defaults to `n`.
    out_sharding: optional, specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array of shape `(*shape, n, m)` and specified dtype.

  References:
    .. [1] Mezzadri, Francesco. (2007). "How to generate random matrices from
           the classical compact groups". Notices of the American Mathematical
           Society, 54(5), 592-604. https://arxiv.org/abs/math-ph/0609050.
  """
  if m is None:
    _m = n
  else:
    _m = m
  shape = core.canonicalize_shape(shape)
  key, _ = _check_prng_key("orthogonal", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  _check_shape("orthogonal", shape)
  n = core.concrete_or_error(
      index, n, "The error occurred in jax.random.orthogonal()")
  _m = core.concrete_or_error(
      index, _m, "The error occurred in jax.random.orthogonal()")
  out_sharding = canonicalize_sharding_for_samplers(
      out_sharding, "orthogonal", (*shape, n, _m))
  return _orthogonal(key, n, _m, shape, dtype, out_sharding)


@jit(static_argnums=(1, 2, 3, 4, 5))
def _orthogonal(key: Key, n, _m, shape, dtype, out_sharding=None):
  out_s = out_sharding
  z_sharding = (
      None if out_s is None else
      out_s.update(spec=P(*out_s.spec.partitions[:len(shape)], None, None)))
  z = normal(
      key, (*shape, max(n, _m), min(n, _m)), dtype, out_sharding=z_sharding)
  q, r = jnp_linalg.qr(z)
  d = jnp_linalg.diagonal(r)
  x = q * jnp.expand_dims(jnp.sign(d), -2)
  out = x.mT if n < _m else x
  if out_s is not None and core.typeof(out).sharding != out_s:
    out = reshard(out, out_s)
  return out

def generalized_normal(
  key: Key | ArrayLike,  # TODO(jakevdp): make this Key
  p: RealArray,
  shape: Shape = (),
  dtype: DTypeLikeFloat | None = None,
  *,
  out_sharding: NamedSharding | P | None = None,
) -> Array:
  r"""Sample from the generalized normal distribution.

  The values are returned according to the probability density function:

  .. math::
     f(x;p) \propto e^{-|x|^p}

  on the domain :math:`-\infty < x < \infty`, where :math:`p > 0` is the
  shape parameter.

  Args:
    key: a PRNG key used as the random key.
    p: a float or array of floats broadcast-compatible with ``shape``
      representing the shape parameter.
    shape: optional, the batch dimensions of the result. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    out_sharding: optional, specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified shape and dtype.
  """
  shape = core.canonicalize_shape(shape)
  key, _ = _check_prng_key("generalized_normal", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  _check_shape("generalized_normal", shape)
  out_sharding = canonicalize_sharding(out_sharding, "generalized_normal")
  return _generalized_normal(key, p, shape, dtype, out_sharding)


@jit(static_argnums=(2, 3, 4))
def _generalized_normal(key: Key, p, shape, dtype, out_sharding=None):
  sample_s = _sample_sharding("generalized_normal", shape, out_sharding, p)
  p, = _align_params(shape, sample_s, p)
  keys = split(key)
  g = gamma(keys[0], 1 / p, shape, dtype, out_sharding=sample_s)
  r = rademacher(keys[1], shape, dtype, out_sharding=sample_s)
  return r * g ** (1 / p)

def ball(
  key: Key | ArrayLike,  # TODO(jakevdp): make this Key
  d: int,
  p: RealArray = 2,
  shape: Shape = (),
  dtype: DTypeLikeFloat | None = None,
  *,
  out_sharding: NamedSharding | P | None = None,
):
  """Sample uniformly from the unit Lp ball.

  Reference: https://arxiv.org/abs/math/0503650.

  Args:
    key: a PRNG key used as the random key.
    d: a nonnegative int representing the dimensionality of the ball.
    p: a float representing the p parameter of the Lp norm.
    shape: optional, the batch dimensions of the result. Default ().
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    out_sharding: optional, specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array of shape `(*shape, d)` and specified dtype.
  """
  check_arraylike("ball", p)
  shape = core.canonicalize_shape(shape)
  key, _ = _check_prng_key("ball", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  _check_shape("ball", shape)
  d = core.concrete_or_error(index, d, "The error occurred in jax.random.ball()")
  out_sharding = canonicalize_sharding(out_sharding, "ball")
  return _ball(key, p, d, shape, dtype, out_sharding)


@jit(static_argnums=(2, 3, 4, 5))
def _ball(key: Key, p, d, shape, dtype, out_sharding=None):
  out_shape = (*shape, d)
  out_s, batch_s = _check_unsharded_trailing_axis(
      "ball", out_shape, out_sharding, p, param_has_trailing=False)
  p, = _align_params(out_shape, out_s, p)
  k1, k2 = split(key)
  g = generalized_normal(k1, p, out_shape, dtype, out_sharding=out_s)
  e = exponential(k2, shape, dtype, out_sharding=batch_s)
  return g / (((jnp.abs(g) ** p).sum(-1) + e) ** (1 / p))[..., None]


def rayleigh(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
             scale: RealArray,
             shape: Shape | None = None,
             dtype: DTypeLikeFloat | None = None,
             *,
             out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample Rayleigh random values with given shape and float dtype.

  The values are returned according to the probability density function:

  .. math::
     f(x;\sigma) \propto xe^{-x^2/(2\sigma^2)}

  on the domain :math:`-\infty < x < \infty`, and where :math:`\sigma > 0` is the scale
  parameter of the distribution.

  Args:
    key: a PRNG key used as the random key.
    scale: a float or array of floats broadcast-compatible with ``shape``
      representing the parameter of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``scale``. The default (None)
      produces a result shape equal to ``scale.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    out_sharding: Optional. Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``scale.shape``.
  """
  key, _ = _check_prng_key("rayleigh", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError("dtype argument to `rayleigh` must be a float "
                     f"dtype, got {dtype}")
  shape = _check_broadcast_shapes("rayleigh", shape, scale)
  _check_all_safe_to_cast("rayleigh", dtype, scale)
  out_sharding = canonicalize_sharding(out_sharding, "rayleigh")
  return _rayleigh(key, scale, shape, dtype, out_sharding)


@jit(static_argnums=(2, 3, 4))
def _rayleigh(key: Key, scale, shape, dtype, out_sharding=None) -> Array:
  scale = lax.convert_element_type(scale, dtype)
  sample_s = _sample_sharding("rayleigh", shape, out_sharding, scale)
  scale, = _align_params(shape, sample_s, scale)
  u = uniform(key, shape, dtype, out_sharding=sample_s)
  scale = jnp.broadcast_to(scale, shape)
  log_u = lax.log(u)
  n_two = lax._const(scale, -2)
  sqrt_u = lax.sqrt(lax.mul(log_u, n_two))
  ray = lax.mul(scale, sqrt_u)
  return ray

def wald(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
         mean: RealArray,
         shape: Shape | None = None,
         dtype: DTypeLikeFloat | None = None,
         *,
         out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample Wald random values with given shape and float dtype.

  The values are returned according to the probability density function:

  .. math::
     f(x;\mu) = \frac{1}{\sqrt{2\pi x^3}} \exp\left(-\frac{(x - \mu)^2}{2\mu^2 x}\right)

  on the domain :math:`-\infty < x < \infty`, and where :math:`\mu > 0` is the location
  parameter of the distribution.


  Args:
    key: a PRNG key used as the random key.
    mean: a float or array of floats broadcast-compatible with ``shape``
      representing the mean parameter of the distribution.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``mean``. The default
      (None) produces a result shape equal to ``np.shape(mean)``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    out_sharding: optional, specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``mean.shape``.
  """
  key, _ = _check_prng_key("wald", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError("dtype argument to `wald` must be a float "
                     f"dtype, got {dtype}")
  shape = _check_broadcast_shapes("wald", shape, mean)
  _check_all_safe_to_cast("wald", dtype, mean)
  out_sharding = canonicalize_sharding(out_sharding, "wald")
  return _wald(key, mean, shape, dtype, out_sharding)


@jit(static_argnums=(2, 3, 4))
def _wald(key: Key, mean, shape, dtype, out_sharding=None) -> Array:
  k1, k2 = _split(key, 2)
  mean = lax.convert_element_type(mean, dtype)
  sample_s = _sample_sharding("wald", shape, out_sharding, mean)
  mean, = _align_params(shape, sample_s, mean)
  mean = jnp.broadcast_to(mean, shape)
  v = normal(k1, shape, dtype, out_sharding=sample_s)
  z = uniform(k2, shape, dtype, out_sharding=sample_s)
  y = lax.integer_pow(v, 2)
  y_sq = lax.integer_pow(y, 2)
  mean_sq = lax.integer_pow(mean, 2)
  sqrt_term = lax.sqrt(4 * mean * y + mean_sq * y_sq)
  x = mean + mean_sq * y / 2 - mean / 2 * sqrt_term
  w = lax.select(lax.le(z,  mean / (mean + x)), x, mean_sq / x)
  return w

def geometric(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
              p: RealArray,
              shape: Shape | None = None,
              dtype: DTypeLikeInt | None = None,
              *,
              out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample Geometric random values with given shape and float dtype.

  The values are returned according to the probability mass function:

  .. math::
      f(k;p) = p(1-p)^{k-1}

  on the domain :math:`0 < p < 1`.

  Args:
    key: a PRNG key used as the random key.
    p: a float or array of floats broadcast-compatible with ``shape``
      representing the probability of success of an individual trial.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``p``. The default
      (None) produces a result shape equal to ``np.shape(p)``.
    dtype: optional, a int dtype for the returned values (default int64 if
      jax_enable_x64 is true, otherwise int32).
    out_sharding: optional, Specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``p.shape``.
  """
  key, _ = _check_prng_key("geometric", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      int if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.integer):
    raise ValueError("dtype argument to `geometric` must be an int "
                     f"dtype, got {dtype}")
  shape = _check_broadcast_shapes("geometric", shape, p)
  out_sharding = canonicalize_sharding(out_sharding, "geometric")
  return _geometric(key, p, shape, dtype, out_sharding)

@jit(static_argnums=(2, 3, 4))
def _geometric(key: Key, p, shape, dtype, out_sharding=None) -> Array:
  check_arraylike("geometric", p)
  p, = promote_dtypes_inexact(p)
  sample_s = _sample_sharding("geometric", shape, out_sharding, p)
  p, = _align_params(shape, sample_s, p)
  u = uniform(key, shape, p.dtype, out_sharding=sample_s)
  # TODO(jakevdp): switch to log_u = lax.log1p(u - 1)
  # For now we map u=0 to u=1 to avoid inf in log_u without otherwise
  # changing samples produced for a given key.
  u = jnp.where(u == 0, 1, u)
  log_u = lax.log(u)
  log_one_minus_p = lax.log1p(-p)
  log_one_minus_p = jnp.broadcast_to(log_one_minus_p, shape)
  g = lax.floor(lax.div(log_u, log_one_minus_p)) + 1
  return g.astype(dtype)


def triangular(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
               left: RealArray,
               mode: RealArray,
               right: RealArray,
               shape: Shape | None = None,
               dtype: DTypeLikeFloat | None = None,
               *,
               out_sharding: NamedSharding | P | None = None) -> Array:
  r"""Sample Triangular random values with given shape and float dtype.

  The values are returned according to the probability density function:

  .. math::
      f(x; a, b, c) = \frac{2}{c-a} \left\{ \begin{array}{ll} \frac{x-a}{b-a} & a \leq x \leq b \\ \frac{c-x}{c-b} & b \leq x \leq c \end{array} \right.

  on the domain :math:`a \leq x \leq c`.

  Args:
    key: a PRNG key used as the random key.
    left: a float or array of floats broadcast-compatible with ``shape``
      representing the lower limit parameter of the distribution.
    mode: a float or array of floats broadcast-compatible with ``shape``
      representing the peak value parameter of the distribution, value must
      fulfill the condition ``left <= mode <= right``.
    right: a float or array of floats broadcast-compatible with ``shape``
      representing the upper limit parameter of the distribution, must be
      larger than ``left``.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``left``,``mode`` and ``right``.
      The default (None) produces a result shape equal to ``left.shape``, ``mode.shape``
      and ``right.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    out_sharding: optional, specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified dtype and with shape given by ``shape`` if
    ``shape`` is not None, or else by ``left.shape``, ``mode.shape`` and ``right.shape``.
  """
  key, _ = _check_prng_key("triangular", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError("dtype argument to `triangular` must be a float "
                     f"dtype, got {dtype}")
  shape = _check_broadcast_shapes("triangular", shape, left, mode, right)
  _check_all_safe_to_cast("triangular", dtype, left, mode, right)
  out_sharding = canonicalize_sharding(out_sharding, "triangular")
  return _triangular(key, left, mode, right, shape, dtype, out_sharding)


@jit(static_argnums=(4, 5, 6), inline=Inline.JAX_EARLY)
def _triangular(
    key: Key, left, mode, right, shape, dtype, out_sharding=None
) -> Array:
  # https://en.wikipedia.org/wiki/Triangular_distribution#Generating_triangular-distributed_random_variates
  left = lax.convert_element_type(left, dtype)
  mode = lax.convert_element_type(mode, dtype)
  right = lax.convert_element_type(right, dtype)
  sample_s = _sample_sharding(
      "triangular", shape, out_sharding, left, mode, right)
  left, mode, right = _align_params(shape, sample_s, left, mode, right)
  left = jnp.broadcast_to(left, shape)
  right = jnp.broadcast_to(right, shape)
  mode = jnp.broadcast_to(mode, shape)
  fc = (mode - left) / (right - left)
  u = uniform(key, shape, dtype, out_sharding=sample_s)
  out1 = left + lax.sqrt(u * (right - left) * (mode - left))
  out2 = right - lax.sqrt((1 - u) * (right - left) * (right - mode))
  tri = lax.select(u < fc, out1, out2)
  return tri


def lognormal(key: Key | ArrayLike,  # TODO(jakevdp): make this Key
              sigma: RealArray = np.float32(1),
              shape: Shape | None = None,
              dtype: DTypeLikeFloat | None = None,
              *,
              out_sharding: NamedSharding | P | None = None) -> Array:
  r""" Sample lognormal random values with given shape and float dtype.

  The values are distributed according to the probability density function:

  .. math::
      f(x) = \frac{1}{x\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(\log x)^2}{2\sigma^2}\right)

  on the domain :math:`x > 0`.

  Args:
    key: a PRNG key used as the random key.
    sigma: a float or array of floats broadcast-compatible with ``shape`` representing
      the standard deviation of the underlying normal distribution. Default 1.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. The default (None) produces a result shape equal to ``()``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    out_sharding: optional, specifies how the output array should be sharded
      across devices in multi-device computation. Can be a
      :class:`~jax.sharding.NamedSharding`, a :class:`~jax.sharding.PartitionSpec`
      (``P``), or ``None`` (default). When specified, the output will be sharded
      according to the given sharding specification. Primarily used in explicit
      sharding mode.
      See the `explicit sharding tutorial <https://docs.jax.dev/en/latest/201/sharding.html#explicit-sharding-mode-makes-sharding-queryable-at-trace-time>`_
      for more details.

  Returns:
    A random array with the specified dtype and with shape given by ``shape``.
  """
  key, _ = _check_prng_key("lognormal", key)
  dtype = dtypes.check_and_canonicalize_user_dtype(float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.inexact):
    raise ValueError(f"dtype argument to `lognormal` must be a float or complex dtype, "
                    f"got {dtype}")
  shape = _check_broadcast_shapes("lognormal", shape, sigma)
  _check_all_safe_to_cast("lognormal", dtype, sigma)
  out_sharding = canonicalize_sharding(out_sharding, "lognormal")
  return _lognormal(key, sigma, shape, dtype, out_sharding)


@jit(static_argnums=(2, 3, 4), inline=Inline.JAX_EARLY)
def _lognormal(key: Key, sigma, shape, dtype, out_sharding=None) -> Array:
  sigma = lax.convert_element_type(sigma, dtype)
  sample_s = _sample_sharding("lognormal", shape, out_sharding, sigma)
  sigma, = _align_params(shape, sample_s, sigma)
  scaled_norm = normal(key, shape, dtype, out_sharding=sample_s) * sigma
  return lax.exp(scaled_norm)


def _stirling_approx_tail(k):
  stirling_tail_vals = jnp.array(
      [
          0.0810614667953272,
          0.0413406959554092,
          0.0276779256849983,
          0.02079067210376509,
          0.0166446911898211,
          0.0138761288230707,
          0.0118967099458917,
          0.0104112652619720,
          0.00925546218271273,
          0.00833056343336287,
      ],
      dtype=k.dtype,
  )
  use_tail_values = k <= 9
  k = lax.clamp(lax._const(k, 0.0), k, lax._const(k, 9.0))
  kp1sq = (k + 1) * (k + 1)
  approx = (1.0 / 12 - (1.0 / 360 - 1.0 / 1260 / kp1sq) / kp1sq) / (k + 1)
  k = jnp.floor(k)
  return lax.select(
      use_tail_values, stirling_tail_vals[jnp.asarray(k, dtype='int32')], approx)


@jit(static_argnums=(3, 4, 5), inline=Inline.JAX_EARLY)
def _binomial_inversion(key: Key, count, prob, shape, dtype, max_iters):
  if config.enable_checks.value:
    assert dtypes.issubdtype(prob.dtype, np.floating)

  log1minusprob = jnp.log1p(-prob)

  def body_fn(carry):
    i, num_geom, geom_sum, key = carry
    subkey, key = split(key)
    num_geom_out = lax.select(geom_sum <= count, num_geom + 1, num_geom)
    u = uniform(subkey, shape, prob.dtype)
    geom = jnp.ceil(jnp.log(u) / log1minusprob)
    geom_sum = geom_sum + geom
    return i + 1, num_geom_out, geom_sum, key

  def cond_fn(carry):
    i, geom_sum = carry[0], carry[2]
    return (geom_sum <= count).any() & (i < max_iters)

  num_geom_init = lax.full_like(prob, 0, prob.dtype, shape)
  geom_sum_init = lax.full_like(prob, 0, prob.dtype, shape)
  carry = (0, num_geom_init, geom_sum_init, key)
  k = lax_control_flow.while_loop(cond_fn, body_fn, carry)[1]
  return (k - 1).astype(dtype)


@jit(static_argnums=(3, 4, 5), inline=Inline.JAX_EARLY)
def _btrs(key: Key, count, prob, shape, dtype, max_iters):
  # transforman-rejection algorithm
  # https://www.tandfonline.com/doi/abs/10.1080/00949659308811496
  stddev = jnp.sqrt(count * prob * (1 - prob))
  b = 1.15 + 2.53 * stddev
  a = -0.0873 + 0.0248 * b + 0.01 * prob
  c = count * prob + 0.5
  v_r = 0.92 - 4.2 / b
  r = prob / (1 - prob)
  alpha = (2.83 + 5.1 / b) * stddev
  m = jnp.floor((count + 1) * prob)

  def body_fn(carry):
    i, k_out, accepted, key = carry
    key, subkey_0, subkey_1 = split(key, 3)
    u = uniform(subkey_0, shape, prob.dtype)
    v = uniform(subkey_1, shape, prob.dtype)
    u = u - 0.5
    us = 0.5 - jnp.abs(u)
    accept1 = (us >= 0.07) & (v <= v_r)
    k = jnp.floor((2 * a / us + b) * u + c)
    reject = (k < 0) | (k > count)
    v = jnp.log(v * alpha / (a / (us * us) + b))
    ub = (
      (m + 0.5) * jnp.log((m + 1) / (r * (count - m + 1)))
      + (count + 1) * jnp.log((count - m + 1) / (count - k + 1))
      + (k + 0.5) * jnp.log(r * (count - k + 1) / (k + 1))
      + _stirling_approx_tail(m)
      + _stirling_approx_tail(count - m)
      - _stirling_approx_tail(k)
      - _stirling_approx_tail(count - k)
    )
    accept2 = v <= ub
    accept = accept1 | (~reject & accept2)
    k_out = lax.select(accept, k, k_out)
    accepted |= accept
    return i + 1, k_out, accepted, key

  def cond_fn(carry):
    i, accepted = carry[0], carry[2]
    return (~accepted).any() & (i < max_iters)

  k_init = lax.full_like(prob, -1, prob.dtype, shape)
  carry = (0, k_init, jnp.full(shape, False, bool), key)
  return lax_control_flow.while_loop(cond_fn, body_fn, carry)[1].astype(dtype)


@jit(static_argnums=(3, 4), inline=Inline.JAX_EARLY)
def _binomial(key: Key, count, prob, shape, dtype) -> Array:
  # The implementation matches TensorFlow and TensorFlow Probability:
  # https://github.com/tensorflow/tensorflow/blob/v2.2.0-rc3/tensorflow/core/kernels/random_binomial_op.cc
  # and tensorflow_probability.substrates.jax.distributions.Binomial
  # For n * p < 10, we use the binomial inverse algorithm; otherwise btrs.
  if shape is None:
    shape = jnp.broadcast_shapes(np.shape(count), np.shape(prob))
  else:
    _check_shape("binomial", shape, np.shape(count), np.shape(prob))
  (prob,) = promote_dtypes_inexact(prob)
  count = lax.convert_element_type(count, prob.dtype)
  count = jnp.broadcast_to(count, shape)
  prob = jnp.broadcast_to(prob, shape)
  p_lt_half = prob < 0.5
  q = lax.select(p_lt_half, prob, 1.0 - prob)
  count_nan_or_neg = _isnan(count) | (count < 0.0)
  count_inf = jnp.isinf(count)
  q_is_nan = _isnan(q)
  q_l_0 = q < 0.0
  q = lax.select(q_is_nan | q_l_0, lax.full_like(q, 0.01), q)
  use_inversion = count_nan_or_neg | (count * q <= 10.0)

  # consistent with np.random.binomial behavior for float count input
  count = jnp.floor(count)

  count_inv = lax.select(use_inversion, count, lax.full_like(count, 0.0))
  count_btrs = lax.select(use_inversion, lax.full_like(count, 1e4), count)
  q_btrs = lax.select(use_inversion, lax.full_like(q, 0.5), q)
  max_iters = dtype.type(dtypes.finfo(dtype).max)
  samples = lax.select(
    use_inversion,
    _binomial_inversion(key, count_inv, q, shape, dtype, max_iters),
    _btrs(key, count_btrs, q_btrs, shape, dtype, max_iters),
  )
  # ensure nan q always leads to nan output and nan or neg count leads to nan
  # as discussed in https://github.com/jax-ml/jax/pull/16134#pullrequestreview-1446642709
  invalid = (q_l_0 | q_is_nan | count_nan_or_neg)
  samples = lax.select(
    invalid,
    jnp.full_like(samples, np.nan, dtype),
    samples,
  )

  # +inf count leads to inf
  samples = lax.select(
    count_inf & (~invalid),
    jnp.full_like(samples, np.inf, dtype),
    samples,
  )

  samples = lax.select(
    p_lt_half | count_nan_or_neg | q_is_nan | count_inf,
    samples,
    count.astype(dtype) - samples,
  )
  return samples


def binomial(
    key: Key | Array,  # TODO(jakevdp): make this Key
    n: RealArray,
    p: RealArray,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
) -> Array:
  r"""Sample Binomial random values with given shape and float dtype.

  The values are returned according to the probability mass function:

  .. math::
      f(k;n,p) = \binom{n}{k}p^k(1-p)^{n-k}

  on the domain :math:`0 < p < 1`, and where :math:`n` is a nonnegative integer
  representing the number of trials and :math:`p` is a float representing the
  probability of success of an individual trial.

  Args:
    key: a PRNG key used as the random key.
    n: a float or array of floats broadcast-compatible with ``shape``
      representing the number of trials.
    p: a float or array of floats broadcast-compatible with ``shape``
      representing the probability of success of an individual trial.
    shape: optional, a tuple of nonnegative integers specifying the result
      shape. Must be broadcast-compatible with ``n`` and ``p``.
      The default (None) produces a result shape equal to ``np.broadcast(n, p).shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).

  Returns:
    A random array with the specified dtype and with shape given by
    ``np.broadcast(n, p).shape``.
  """
  key, _ = _check_prng_key("binomial", key)
  check_arraylike("binomial", n, p)
  dtype = dtypes.check_and_canonicalize_user_dtype(
      float if dtype is None else dtype)
  if not dtypes.issubdtype(dtype, np.floating):
    raise ValueError(
        f"dtype argument to `binomial` must be a float dtype, got {dtype}"
      )
  if shape is not None:
    shape = core.canonicalize_shape(shape)
  return _binomial(key, n, p, shape, dtype)


# Functions related to key reuse checking
random_clone_p = core.Primitive("random_clone")
dispatch.simple_impl(random_clone_p)
random_clone_p.def_abstract_eval(lambda x: x)
batching.defvectorized(random_clone_p)
mlir.register_lowering(
    random_clone_p, lambda _, k: [k], shard_map_passthrough=True
)


def multinomial(
    key: Key | Array,  # TODO(jakevdp): make this Key
    n: RealArray,
    p: RealArray,
    *,
    shape: Shape | None = None,
    dtype: DTypeLikeFloat | None = None,
    unroll: int | bool = 1,
):
  r"""Sample from a multinomial distribution.

  The probability mass function is

  .. math::
      f(x;n,p) = \frac{n!}{x_1! \ldots x_k!} p_1^{x_1} \ldots p_k^{x_k}

  Args:
    key: PRNG key.
    n: number of trials. Should have shape broadcastable to ``p.shape[:-1]``.
    p: probability of each outcome, with outcomes along the last axis.
    shape: optional, a tuple of nonnegative integers specifying the result batch
      shape, that is, the prefix of the result shape excluding the last axis.
      Must be broadcast-compatible with ``p.shape[:-1]``. The default (None)
      produces a result shape equal to ``p.shape``.
    dtype: optional, a float dtype for the returned values (default float64 if
      jax_enable_x64 is true, otherwise float32).
    unroll: optional, unroll parameter passed to :func:`jax.lax.scan` inside the
      implementation of this function.

  Returns:
    An array of counts for each outcome with the specified dtype and with shape
      ``p.shape`` if ``shape`` is None, otherwise ``shape + (p.shape[-1],)``.
  """

  key, _ = _check_prng_key("multinomial", key)
  check_arraylike("multinomial", n, p)
  n, p = promote_dtypes_inexact(n, p)

  if shape is None:
    shape = p.shape[:-1]
  else:
    shape = core.canonicalize_shape(shape)
    _check_shape("multinomial", shape, np.shape(n), p.shape[:-1])
  n = jnp.broadcast_to(n, shape)
  p = jnp.broadcast_to(p, (*shape, p.shape[-1]))

  def f(remainder, ratio_key):
    ratio, key = ratio_key
    count = binomial(key, remainder, ratio.clip(0, 1), dtype=remainder.dtype)
    return remainder - count, count

  p = jnp.moveaxis(p, -1, 0)

  remaining_probs = lax_control_flow.cumsum(p, 0, reverse=True)
  ratios = p / jnp.where(remaining_probs == 0, 1, remaining_probs)

  keys = split(key, ratios.shape[0])
  remainder, counts = lax_control_flow.scan(f, n, (ratios, keys), unroll=unroll)
  # final remainder should be zero

  return jnp.moveaxis(counts, 0, -1).astype(dtype)


def clone(
    key: Key | Array,  # TODO(jakevdp): make this Key
) -> Array:  # TODO(jakevdp): make this Key
  """Clone a key for reuse

  Outside the context of key reuse checking (see :mod:`jax.experimental.key_reuse`)
  this function operates as an identity.

  Examples:

    >>> import jax
    >>> key = jax.random.key(0)
    >>> data = jax.random.uniform(key)
    >>> cloned_key = jax.random.clone(key)
    >>> same_data = jax.random.uniform(cloned_key)
    >>> assert data == same_data
  """
  return random_clone_p.bind(key)


def random_insert_pvary(name, key: Key, *args):
  if not config._check_vma.value or not config.auto_pcast.value:
    return key, args
  if not args:
    return key, args
  key_vma = core.typeof(key).mat.varying
  out = []
  for a in args:
    arg_vma = (aval.mat.varying
               if isinstance(aval := core.typeof(a), core.ShapedArray)
               else frozenset())
    # If key is less varying than the args, then it's an error and user should
    # pvary at their level because it has key-reuse implications. They can
    # shard the keys passed to shard_map correctly so as to avoid key-reuse
    # getting correctly varying keys. But JAX shouldn't auto-pvary the key.
    if key_vma - arg_vma:
      a = core.pvary(a, tuple(k for k in key_vma if k not in arg_vma))
    if core.typeof(key).mat != core.typeof(a).mat:
      raise TypeError(
          f"{name} requires all arguments to have matching type. Got key type:"
          f" {core.typeof(key)} vs arg type: {core.typeof(a)}. Use"
          " jax.lax.pcast(..., to='varying') to make them match. If your key is"
          " less varying than arg, watch out for key-reuse problems.")
    out.append(a)
  return key, out
