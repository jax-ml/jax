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

"""Contains GPU-specific Pallas abstractions."""

from __future__ import annotations

import abc
import collections
from collections.abc import Callable, Iterable, Sequence
import dataclasses
import enum
import itertools as it
import math
from typing import Any, ClassVar, Literal, Union

import jax
from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import effects
from jax._src import pretty_printer as pp
from jax._src import tree_util
from jax._src.lib.mlir.dialects import arith as arith_dialect
from jax._src.pallas import core as pallas_core
from jax._src.pallas import helpers as pallas_helpers
from jax._src.pallas import primitives as pallas_primitives
import jax._src.pallas.utils as pallas_utils
from jax._src.state import discharge as state_discharge
from jax._src.state import indexing
from jax._src.state import types as state_types
import jax.experimental.mosaic.gpu as mgpu
from jax.experimental.mosaic.gpu import utils as mgpu_utils
import jax.numpy as jnp
from jaxlib.mlir import ir


_Ref = pallas_core.AbstractMemoryRef | state_types.TransformedRef
AbstractMemoryRef = pallas_core.AbstractMemoryRef

DimensionSemantics = Literal["parallel", "sequential"]

# We align all our SMEM allocations to 1024 bytes. TMA and WGMMA are very
# sensitive to alignment and while this is quite conservative, it gets the job
# done. We should make this more refined in the future.
SMEM_ALIGNMENT = 1024


def is_trivial_index(idx, shape) -> bool:
  """Checks if the index selects the entire shape."""

  # Slices that select the entire dimension.
  def _slices(d):
    slices = [slice(b, e, s) for b, e, s in it.product([0, None], [d, None], [1, None])]
    return [indexing.Slice(0, d, 1), *slices]

  if isinstance(idx, tuple):
    return all(i in _slices(d) for d, i in zip(shape, idx))

  return idx is ... or (len(shape) == 1 and idx in _slices(shape[0]))


@dataclasses.dataclass(frozen=True, kw_only=True)
class CompilerParams(pallas_core.CompilerParams):
  """Mosaic GPU compiler parameters.

  Attributes:
    approx_math: If True, the compiler is allowed to use approximate
      implementations of some math operations, e.g. ``exp``. Defaults to False.
    dimension_semantics: A list of dimension semantics for each grid
      dimension of the kernel. Either "parallel" for dimensions that can
      execute in any order, or "sequential" for dimensions that must be
      executed sequentially.
    max_concurrent_steps: The maximum number of sequential stages that are
      active concurrently. Defaults to 1.
    delay_release: The number of steps to wait before reusing the input/output
      references. Defaults to 0, and must be strictly smaller than
      max_concurrent_steps. Generally, you'll want to set it to 1 if you don't
      await the WGMMA in the body.
    unsafe_no_auto_barriers: If True, Pallas will never automatically insert
      barrier instructions that ensure synchronous semantics of loads and stores.
      At the moment, the insertion is done conservatively and might regress
      performance. There are (at least) two conditions that must be satisfied
      for the use of this flag to be safe. First, no memory region is ever read
      *and* written to by the same thread (async copies are performed by
      background threads and do not count towards this rule). Secondly, no
      thread ever calls commit_smem(), reads from the committed SMEM and then
      issues an async copy overwriting that region (this is a very artificial
      and highly unlikely scenario).
    profile_space: The number of profiler events that can be collected in a
      single invocation. It is undefined behavior if a thread collects more
      events than this.
    profile_dir: The directory to which profiling traces will be written to.
  """
  BACKEND: ClassVar[pallas_core.Backend] = "mosaic_gpu"
  approx_math: bool = False
  dimension_semantics: Sequence[DimensionSemantics] | None = None
  max_concurrent_steps: int = 1
  delay_release: int = 0
  unsafe_no_auto_barriers: bool = False
  profile_space: int = 0
  profile_dir: str = ""
  lowering_semantics: mgpu.core.LoweringSemantics = mgpu.core.LoweringSemantics.Lane

  def __post_init__(self):
    if bool(self.profile_space) ^ bool(self.profile_dir):
      raise ValueError(
          "Either both profile_space and profile_dir must be set, or neither."
      )


class MemorySpace(enum.Enum):
  #: Global memory.
  GMEM = "gmem"
  #: Shared memory.
  SMEM = "smem"
  #: Tensor memory.
  TMEM = "tmem"
  #: Registers.
  REGS = "regs"

  def __str__(self) -> str:
    return self.value

  def __call__(
      self,
      shape: tuple[int, ...],
      dtype: jnp.dtype,
      *,
      transforms: Sequence[MemoryRefTransform] = (),
      packed: bool | None = None,
      collective: bool | None = None
  ) -> pallas_core.MemoryRef:
    # A convenience function for constructing MemoryRef types.
    return GPUMemoryRef(shape, dtype, memory_space=self, transforms=transforms,
                        packed=packed, collective=collective)


class SemaphoreType(enum.Enum):
  REGULAR = "regular"
  BARRIER = "barrier"

  def __call__(self, shape: tuple[int, ...]):
    dtype: Any
    if self == SemaphoreType.BARRIER:
      dtype = pallas_core.BarrierSemaphore()
    else:
      dtype = pallas_core.Semaphore()
    return pallas_core.MemoryRef(shape, dtype, MemorySpace.GMEM)

  def get_array_aval(self) -> jax_core.ShapedArray:
    return self(()).get_array_aval()

  def get_ref_aval(self) -> _Ref:
    return self(()).get_ref_aval()


class PrimitiveSemantics(enum.Enum):
  """Thread semantics for a primitives at the Pallas user-level."""

  Warp = enum.auto()
  Warpgroup = enum.auto()


# Convenience constants for (lowering, primitive) thread semantics pairs.
LANExWG_SEMANTICS = (
    mgpu.LoweringSemantics.Lane, PrimitiveSemantics.Warpgroup)
LANExWARP_SEMANTICS = (
    mgpu.LoweringSemantics.Lane, PrimitiveSemantics.Warp)
WGxWG_SEMANTICS = (
    mgpu.LoweringSemantics.Warpgroup, PrimitiveSemantics.Warpgroup)


def kernel(
    body: Callable[..., None],
    out_shape: object,
    *,
    scratch_shapes: pallas_core.ScratchShapeTree = (),
    compiler_params: pallas_core.CompilerParams | None = None,
    **mesh_kwargs: object,
):
  if unwrap_out := not isinstance(out_shape, (tuple, list)):
    out_shape = (out_shape,)
  def wrapper(*operands):
    def stateful(operand_and_out_refs):
      operand_refs, out_refs = operand_and_out_refs
      mesh = Mesh(**mesh_kwargs)
      thread_name = mesh.thread_name if mesh.thread_name is not None else ()
      def cmap_body():
        pallas_primitives.run_scoped(
            lambda *scratch_refs: body(*operand_refs, *out_refs, *scratch_refs),
            *scratch_shapes,
            collective_axes=thread_name,
        )
      pallas_core.core_map(
          mesh, compiler_params=compiler_params
      )(cmap_body)
    _, outs = state_discharge.run_state(stateful)(
        (operands, pallas_helpers.empty_like(out_shape, backend="mosaic_gpu"))
    )
    return outs[0] if unwrap_out else outs
  return wrapper


def _is_known_divisible(value, divisor, fuel=10) -> bool:
  """Returns True if the value is statically known to be divisible by the divisor."""
  if divisor == 1:
    return True
  if fuel < 0:
    return False
  if not isinstance(value.owner, ir.Operation):
    return False
  def_op = value.owner.opview
  match def_op:
    case arith_dialect.IndexCastOp():
      return _is_known_divisible(value.owner.operands[0], divisor, fuel - 1)
    case arith_dialect.ConstantOp():
      return ir.IntegerAttr(def_op.value).value % divisor == 0
    case arith_dialect.MulIOp():
      return (_is_known_divisible(value.owner.operands[0], divisor, fuel // 2) or
              _is_known_divisible(value.owner.operands[1], divisor, (fuel + 1)// 2))
    case arith_dialect.SelectOp():
      return (_is_known_divisible(value.owner.operands[1], divisor, fuel // 2) and
              _is_known_divisible(value.owner.operands[2], divisor, (fuel + 1)// 2))

  return False


@dataclasses.dataclass(frozen=True)
class GPUMemoryRef(pallas_core.MemoryRef):
  transforms: Sequence[MemoryRefTransform] = ()

  # Whether to allow TMEM packing for sub 4-byte dtypes.
  packed: bool | None = dataclasses.field(default=None, kw_only=True)
  collective: bool | None = dataclasses.field(default=None, kw_only=True)

  def __post_init__(self):
    if self.memory_space != MemorySpace.TMEM:
      if self.packed is not None:
        raise ValueError("Packed option is only supported for TMEM.")
      if self.collective is not None:
        raise ValueError("Collective option is only supported for TMEM.")

  def get_ref_aval(self) -> _Ref:
    aval = jax_core.ShapedArray(self.shape, self.dtype)
    for t in self.transforms:
      aval = t(aval)
    if self.memory_space == MemorySpace.TMEM:
      ref = pallas_core.TransformedRef(
          AbstractTMEMRef(aval,
                          memory_space=self.memory_space,
                          packed=self.packed,
                          collective=self.collective), ()
      )
    else:
      ref = pallas_core.TransformedRef(
          AbstractMemoryRef(aval, memory_space=self.memory_space), ()
      )
    for t in reversed(self.transforms):
      ref = t.undo(ref)
    if not ref.transforms:
      return ref.ref
    return ref


def align_to(x: int, alignment: int):
  if rem := x % alignment:
    return x + alignment - rem
  return x


# A tree of `GPUMemoryRef`s.
_GPUMemoryRefTree = Any


def _ref_group_size(refs: _GPUMemoryRefTree) -> int:
  if isinstance(refs, GPUMemoryRef):
    refs = (refs,)
  size = 0
  for ref in jax.tree.leaves(refs):
    # Make sure that the start of each ref is aligned with `SMEM_ALIGNMENT`.
    size = align_to(size, SMEM_ALIGNMENT)
    if jnp.issubdtype(ref.dtype, jnp.integer):
      nbits = jnp.iinfo(ref.dtype).bits
    elif jnp.issubdtype(ref.dtype, jnp.floating):
      nbits = jnp.finfo(ref.dtype).bits
    else:
      raise NotImplementedError(f"Unsupported dtype: {ref.dtype}")
    ref_bits = math.prod(ref.shape) * nbits
    if ref_bits % 8:
      raise ValueError("Only byte-aligned shapes are supported.")
    size += ref_bits // 8
  return size


def flatten_ref_union(ref_union: AbstractRefUnion) -> tuple[_Ref, ...]:
  """Flattens a union of trees of references into a tuple of references.

  This is the moral equivalent of `jax.tree.leaves` for aliased references.
  """
  flat_refs = []
  union_bytes = 0
  for ref_group in ref_union.refs:
    byte_offset = 0
    for ref in jax.tree.leaves(ref_group):
      byte_offset = align_to(byte_offset, SMEM_ALIGNMENT)
      assert isinstance(ref, pallas_core.AbstractMemoryRef) or isinstance(
          ref, pallas_core.TransformedRef
      )
      if not isinstance(ref, pallas_core.TransformedRef):
        ref = pallas_core.TransformedRef(ref, transforms=())
      transform = ExtractAliasedRef.from_transformed_ref(ref, byte_offset)
      flat_refs.append(
          pallas_core.TransformedRef(
              ref_union, transforms=(transform, *ref.transforms)
          )
      )
      if jnp.issubdtype(ref.dtype, jnp.integer):
        nbits = jnp.iinfo(ref.dtype).bits
      elif jnp.issubdtype(ref.dtype, jnp.floating):
        nbits = jnp.finfo(ref.dtype).bits
      else:
        raise NotImplementedError(f"Unsupported dtype: {ref.dtype}")
      ref_bits = math.prod(ref.shape) * nbits
      if ref_bits % 8:
        raise ValueError("Only byte-aligned shapes are supported.")
      byte_offset += ref_bits // 8
    union_bytes = max(union_bytes, byte_offset)
  assert union_bytes == ref_union.shape[0]
  return tuple(flat_refs)


class AbstractRefUnion(pallas_core.AbstractMemoryRef):
  refs: Sequence[_GPUMemoryRefTree]

  def __init__(
      self,
      aval,
      refs: Sequence[_GPUMemoryRefTree],
      memory_space,
  ):
    self.refs = refs
    super().__init__(aval, memory_space=memory_space)

  def _iter(self, tracer):
    return iter(flatten_ref_union(tracer))

  def _getitem(self, tracer, index):
    return list(iter(tracer))[index]

  def _setitem(self, tracer, index, value):
    del tracer, index, value  # Unused.
    raise ValueError("Ref unions can't be assigned to.")


@dataclasses.dataclass(init=False, frozen=True)
class RefUnion(GPUMemoryRef):
  """A sequence of trees of refs that are allowed to reuse the same memory.

  One should not make assumptions as to how each ref will map to the underlying
  memory region, since arbitrary padding may be applied in between different
  refs.

  As such, ref unions are only safe to use when the groups of refs that we
  intend to alias have disjoint lifetimes (i.e. one should never attempt to read
  data using a different ref than the one that was used to write the data).
  """
  refs: Sequence[_GPUMemoryRefTree] = ()

  def __init__(self, *refs: _GPUMemoryRefTree):
    if any(ref.memory_space != SMEM for ref in jax.tree.leaves(refs)):
      raise NotImplementedError("Only SMEM refs can be aliased.")
    object.__setattr__(self, "refs", refs)
    num_bytes = max(map(_ref_group_size, self.refs))
    super().__init__(
        shape=(num_bytes,),
        dtype=jnp.int8,
        memory_space=SMEM,
        transforms=(),
    )

  def get_ref_aval(self) -> AbstractRefUnion:
    inner_aval = jax.core.ShapedArray(self.shape, self.dtype)
    refs_aval = jax.tree.map(lambda ref: ref.get_ref_aval(), self.refs)
    return AbstractRefUnion(inner_aval, refs_aval, memory_space=SMEM)


class MemoryRefTransform(pallas_core.MemoryRefTransform, abc.ABC):
  @abc.abstractmethod
  def to_gpu_transform(self) -> mgpu.MemRefTransform:
    pass

  def batch(self, leading_rank: int):
    """Returns a transform that accepts a ref with the extra `leading_rank` dims.

    The returned transform should leave the leading dimensions unchanged and
    only apply to the suffix of the shape.
    """
    raise NotImplementedError

  def __call__(self, aval: jax_core.ShapedArray) -> jax_core.ShapedArray:
    return aval.update(
        shape=self.to_gpu_transform().transform_shape(aval.shape)
    )

Index = Union[mgpu.DynamicSlice, slice, int, ir.Value]

@dataclasses.dataclass(frozen=True)
class TilingTransform(MemoryRefTransform):
  """Represents a tiling transformation for memory refs.

  A tiling of (X, Y) on an array of shape (M, N) will result in a transformed
  shape of (M // X, N // Y, X, Y). Ex. A (256, 256) block that is tiled with a
  tiling of (64, 32) will be tiled as (4, 8, 64, 32).
  """
  tiling: tuple[int, ...]

  def undo(self, ref: pallas_core.TransformedRef) -> pallas_core.TransformedRef:
    return dataclasses.replace(
        ref, transforms=(*ref.transforms, UntileRef(self.tiling))
    )

  def batch(self, leading_rank: int):
    return self

  def to_gpu_transform(self) -> mgpu.MemRefTransform:
    return mgpu.TileTransform(self.tiling)


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class UntileRef(state_types.Transform):
  tiling: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

  def transform_shape(self, shape):
    if shape is None:
      return None
    assert shape[-len(self.tiling) :] == self.tiling
    shape = shape[: -len(self.tiling)]  # Drop tiling
    return shape[: -len(self.tiling)] + tuple(
        block_dim * tiling_dim
        for block_dim, tiling_dim in zip(shape[-len(self.tiling) :], self.tiling)
    )

  def transform_dtype(self, dtype):
    return dtype

  def untransform_transpose(
      self, perm: tuple[int, ...]
  ) -> tuple[tuple[int, ...], state_types.Transform]:
    # The transpose in question is applied to the utiled ref so we
    # need to translate it by duplicating and offsetting the last part.
    off = len(perm)
    new_suffix = [i + off for i in perm[-len(self.tiling) :]]
    if set(new_suffix) != set(range(off, off + len(self.tiling))):
      raise ValueError(
          "Transpose cannot be moved before a tiling transform when it changes"
          f" the set of tiled dimensions. (permutation: {perm}, tiling:"
          f" {self.tiling})"
      )

    new_tiling = tuple(self.tiling[i - off] for i in new_suffix)
    return (*perm, *new_suffix), dataclasses.replace(self, tiling=new_tiling)

  def untransform_reshape(
      self, dtype: jnp.dtype, shape: tuple[int, ...]
  ) -> tuple[tuple[int, ...], state_types.Transform]:
    del dtype
    raise NotImplementedError("Reshapes don't commute with transposes.")

  def untransform_index(
      self, dtype: jnp.dtype | ir.Type, idxs: tuple[Index, ...]
  ) -> tuple[tuple[Index, ...], state_types.Transform]:
    del dtype
    untiled_idxs = idxs[: -len(self.tiling)]
    tiled_idxs = idxs[-len(self.tiling) :]
    idxs_after_tiling: list[Index] = []
    for idx, tile in zip(tiled_idxs, self.tiling):
      if isinstance(idx, slice):
        if idx.step is not None and idx.step != 1:
          raise NotImplementedError("Strided slices unsupported")
        if (idx.start is not None and idx.start % tile) or (idx.stop is not None and idx.stop % tile):
          raise ValueError("Non-empty slices must be tile aligned")
        idxs_after_tiling.append(slice(idx.start // tile, idx.stop // tile))
      elif isinstance(idx, mgpu.DynamicSlice):
        if idx.length % tile:
          raise ValueError(
              f"Dynamic slice length ({idx.length}) is not divisible by the"
              f" tiling ({tile})"
          )
        if isinstance(idx.base, ir.Value):
          if not _is_known_divisible(idx.base, tile):
            raise ValueError(
                "Dynamic slice base index (which is a dynamic value) cannot be"
                f" statically proven to be divisible by the tiling ({tile})"
            )
          new_base = arith_dialect.divui(idx.base, mgpu.c(tile, idx.base.type))
        else:
          if idx.base % tile:
            raise ValueError(
                f"Dynamic slice base ({idx.base}) is not divisible by the"
                f" tiling ({tile})"
            )
          new_base = idx.base // tile
        idxs_after_tiling.append(mgpu.DynamicSlice(new_base, idx.length // tile))
      else:
        raise TypeError(f"Unsupported index type: {type(idx)}")
    return (*untiled_idxs, *idxs_after_tiling, *(slice(None) for _ in self.tiling)), self

  def undo_to_gpu_transform(self) -> mgpu.MemRefTransform:
    return mgpu.TileTransform(self.tiling)

  def pretty_print(self, context: jax_core.JaxprPpContext) -> pp.Doc:
    return pp.text(f"{{untile({list(self.tiling)})}}")


def _perm_inverse(permutation: tuple[int, ...]) -> tuple[int, ...]:
  inverse = [-1] * len(permutation)
  for i, p in enumerate(permutation):
    inverse[p] = i
  return tuple(inverse)


@dataclasses.dataclass(frozen=True)
class TransposeTransform(MemoryRefTransform):
  """Transpose a tiled memref."""
  permutation: tuple[int, ...]

  def __post_init__(self):
    if set(self.permutation) != set(range(len(self.permutation))):
      raise ValueError(f"Permutation {self.permutation} is not a permutation.")

  def batch(self, leading_rank: int):
    return TransposeTransform(
        (*range(leading_rank), *(d + leading_rank for d in self.permutation))
    )

  def undo(self, ref: pallas_core.TransformedRef) -> pallas_core.TransformedRef:
    return dataclasses.replace(
        ref,
        transforms=(
            *ref.transforms,
            TransposeRef(_perm_inverse(self.permutation)),
        ),
    )

  def to_gpu_transform(self) -> mgpu.MemRefTransform:
    return mgpu.TransposeTransform(self.permutation)


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class TransposeRef(state_types.Transform):
  permutation: tuple[int, ...] = dataclasses.field(metadata=dict(static=True))

  def transform_shape(self, shape):
    if shape is None:
      return None
    return tuple(shape[i] for i in self.permutation)

  def transform_dtype(self, dtype):
    return dtype

  def untransform_transpose(
      self, perm
  ) -> tuple[tuple[int, ...], state_types.Transform]:
    raise NotImplementedError(
        "Commuting of transpose over transpose is not supported."
    )

  def untransform_reshape(
      self, dtype: jnp.dtype | ir.Type, shape: tuple[int, ...]
  ) -> tuple[tuple[int, ...], state_types.Transform]:
    del shape, dtype
    raise NotImplementedError("Can't reshape a transposed memref.")

  def untransform_index(
      self, dtype: jnp.dtype | ir.Type, idxs: tuple[Index, ...]
  ) -> tuple[tuple[Index, ...], state_types.Transform]:
    del dtype
    removed_dims = [
        i for i, idx in enumerate(idxs) if not isinstance(idx, (slice, mgpu.ds))
    ]
    new_perm = tuple(
        p - sum(d < p for d in removed_dims)
        for p in self.permutation
        if p not in removed_dims
    )
    new_idxs = tuple(idxs[i] for i in _perm_inverse(self.permutation))
    return new_idxs, TransposeRef(new_perm)

  def undo_to_gpu_transform(self) -> mgpu.MemRefTransform:
    return mgpu.TransposeTransform(_perm_inverse(self.permutation))

  def pretty_print(self, context: jax_core.JaxprPpContext) -> pp.Doc:
    return pp.text(f"{{transpose({list(self.permutation)})}}")


@tree_util.register_pytree_node_class
@dataclasses.dataclass
class PeerMemRef(state_types.Transform):
  device_id: Any
  device_id_type: pallas_primitives.DeviceIdType

  def transform_shape(self, shape):
    return shape

  def transform_dtype(self, dtype):
    return dtype

  def untransform_index(
      self, idxs: tuple[Index, ...]
  ) -> tuple[tuple[Index, ...], state_types.Transform]:
    return idxs, self

  def tree_flatten(self):
    return (self.device_id,), (self.device_id_type,)

  @classmethod
  def tree_unflatten(cls, metadata, arrays):
    return cls(arrays[0], metadata[0])


def remote_ref(
    ref: _Ref,
    device_id: jax.typing.ArrayLike,
    device_id_type: pallas_primitives.DeviceIdType = pallas_primitives.DeviceIdType.MESH,
) -> pallas_core.TransformedRef:
  """Translate memref to a symmetric memref on a peer device."""
  if not isinstance(ref, pallas_core.TransformedRef):
    if not isinstance(jax_core.get_aval(ref), state_types.AbstractRef):
      raise TypeError("ref must be a reference")
    ref = pallas_core.TransformedRef(ref, transforms=())
  return pallas_core.TransformedRef(
      ref.ref, (*ref.transforms, PeerMemRef(device_id, device_id_type)),
  )


def transform_ref(
    ref: pallas_core.TransformedRef,
    transform: state_types.Transform
) -> pallas_core.TransformedRef:
  if not isinstance(ref, pallas_core.TransformedRef):
    if not isinstance(jax_core.get_aval(ref), state_types.AbstractRef):
      raise TypeError("ref must be a reference")
    ref = pallas_core.TransformedRef(ref, transforms=())
  return pallas_core.TransformedRef(
      ref.ref, (*ref.transforms, transform),
  )

def transpose_ref(
    ref: pallas_core.TransformedRef | Any,
    permutation: tuple[int, ...],
) -> pallas_core.TransformedRef:
  return transform_ref(ref, TransposeRef(permutation))

def untile_ref(ref, tiling: tuple[int, ...]) -> pallas_core.TransformedRef:
  return transform_ref(ref, UntileRef(tiling))

def unswizzle_ref(ref, swizzle: int) -> pallas_core.TransformedRef:
  return transform_ref(ref, UnswizzleRef(swizzle))


@tree_util.register_pytree_node_class
@dataclasses.dataclass(frozen=True)
class ExtractAliasedRef(state_types.Transform):
  """Bitcasts the underlying ref at the given offset to the given shape and dtype."""
  dtype: dtypes.DType
  shape: tuple[int, ...]
  offset: int

  @classmethod
  def from_transformed_ref(
      cls, ref: pallas_core.TransformedRef, byte_offset: int
  ):
    return cls(
        dtypes.dtype(ref.dtype), ref.ref.shape, byte_offset
    )

  def transform_shape(self, shape):
    if shape is None:
      return None
    return self.shape

  def transform_dtype(self, dtype):
    del dtype  # Unused.
    return self.dtype

  def tree_flatten(self):
    return (), (self.dtype, self.shape, self.offset)

  @classmethod
  def tree_unflatten(cls, metadata, arrays):
    assert not arrays
    return cls(*metadata)


@dataclasses.dataclass(frozen=True)
class SwizzleTransform(MemoryRefTransform):
  swizzle: int

  def __post_init__(self):
    if self.swizzle not in {32, 64, 128}:
      raise ValueError(
          f"Swizzle {self.swizzle} is not supported. Only 32, 64 and 128 are"
          " accepted."
      )

  def batch(self, leading_rank: int):
    return self

  def undo(self, ref: pallas_core.TransformedRef) -> pallas_core.TransformedRef:
    return dataclasses.replace(
        ref, transforms=(*ref.transforms, UnswizzleRef(self.swizzle))
    )

  def to_gpu_transform(self) -> mgpu.MemRefTransform:
    raise RuntimeError("SwizzleTransform does not have a GPU transform.")

  def undo_to_gpu_transform(self) -> mgpu.MemRefTransform:
    # There's no swizzle transform in mgpu right now. It's a separate arg.
    raise NotImplementedError

  def __call__(self, aval: jax_core.ShapedArray) -> jax_core.ShapedArray:
    swizzle_elems = (self.swizzle * 8) // pallas_utils.dtype_bitwidth(aval.dtype)
    if swizzle_elems != aval.shape[-1]:
      raise ValueError(
          f"Swizzle {self.swizzle} requires the trailing dimension to be of"
          f" size {swizzle_elems}, but got shape: {aval.shape}"
      )
    return aval


@tree_util.register_dataclass
@dataclasses.dataclass(frozen=True)
class UnswizzleRef(state_types.Transform):
  swizzle: int = dataclasses.field(metadata=dict(static=True))

  def swizzle_elems(self, dtype: jnp.dtype | ir.Type) -> int:
    if not isinstance(dtype, ir.Type):
      dtype = mgpu_utils.dtype_to_ir_type(dtype)
    return (self.swizzle * 8) // mgpu.bitwidth(dtype)

  def untransform_transpose(self, perm) -> tuple[tuple[int, ...], state_types.Transform]:
    if perm[-1] != len(perm) - 1:
      raise ValueError("Can't transpose the swizzled dimension.")

    return perm, self

  def untransform_reshape(
      self, dtype: jnp.dtype | ir.Type, shape: tuple[int, ...]
  ) -> tuple[tuple[int, ...], state_types.Transform]:
    if shape[-1] != self.swizzle_elems(dtype):
      raise ValueError(
          f"Reshape shape {shape} is not divisible by swizzle elements"
          f" {self.swizzle_elems(dtype)}"
      )
    return shape, self

  def untransform_index(
      self, dtype: jnp.dtype | ir.Type, idxs: tuple[Index, ...]
  ) -> tuple[tuple[Index, ...], state_types.Transform]:
    swizzle_elems = self.swizzle_elems(dtype)
    if not idxs:
      return idxs, self
    if not all(isinstance(idx, (slice, mgpu.ds)) for idx in idxs[-2:]):
      raise NotImplementedError(
          "Non-slice indices are not supported in 2 minormost dims"
      )
    last_idx = idxs[-1]
    if isinstance(last_idx, mgpu.DynamicSlice):
      if last_idx.base != 0 or last_idx.length != swizzle_elems:
        raise ValueError("Swizzled dims cannot be sliced")
    else:
      assert isinstance(last_idx, slice)
      if (
          (last_idx.step is not None and last_idx.step != 1)
          or (last_idx.start is not None and last_idx.start != 0)
          or (last_idx.stop is not None and last_idx.stop != swizzle_elems)
      ):
        raise ValueError("Swizzled dims cannot be sliced")
    return idxs, self

  def pretty_print(self, context: jax_core.JaxprPpContext) -> pp.Doc:
    return pp.text(f"{{unswizzle({self.swizzle})}}")


@dataclasses.dataclass
class BlockSpec(pallas_core.BlockSpec):
  transforms: Sequence[MemoryRefTransform] = ()

  def to_block_mapping(
      self,
      origin: pallas_core.OriginStr,
      array_aval: jax_core.ShapedArray,
      *,
      index_map_avals: Sequence[jax_core.AbstractValue],
      index_map_tree: tree_util.PyTreeDef,
      grid: pallas_core.GridMappingGrid,
      mapped_dims: tuple[int, ...],
  ) -> pallas_core.BlockMapping:
    bm = super().to_block_mapping(
        origin,
        array_aval,
        index_map_avals=index_map_avals,
        index_map_tree=index_map_tree,
        grid=grid,
        mapped_dims=mapped_dims,
    )
    block_inner_aval = bm.block_aval.inner_aval
    for t in self.transforms:
      block_inner_aval = t(block_inner_aval)
    return bm.replace(
        transformed_block_aval=bm.block_aval.update(
            inner_aval=block_inner_aval
        ),
        transforms=self.transforms,
    )


GMEM = MemorySpace.GMEM
SMEM = MemorySpace.SMEM
TMEM = MemorySpace.TMEM
REGS = MemorySpace.REGS


class barrier_dtype(dtypes.extended):
  pass


@dataclasses.dataclass(frozen=True)
class BarrierType(dtypes.ExtendedDType):
  type: ClassVar[Any] = barrier_dtype
  name: ClassVar[str] = "barrier"

  num_arrivals: int
  for_tensor_core: bool

  def __str__(self):
    return self.name


@dataclasses.dataclass(frozen=True)
class ClusterBarrierType(dtypes.ExtendedDType):
  type: ClassVar[Any] = barrier_dtype
  name: ClassVar[str] = "cluster_barrier"

  collective_axes: tuple[str | tuple[str, ...], ...]

  def __str__(self):
    return self.name


@dataclasses.dataclass(frozen=True, kw_only=True)
class Barrier:
  """Describes a barrier Ref.

  Attributes:
    num_arrivals: The number of arrivals that will be recorded by this barrier.
    num_barriers: The number of barriers that will be created. Individual
      barriers can be accessed by indexing into the barrier Ref.
    for_tensor_core: Whether this barrier is used for synchronizing with
      the tensor core. This should be set to True when waiting on Blackwell
      (TC Gen 5) asynchronous matmul instructions.
  """
  num_arrivals: int = 1
  num_barriers: int = 1
  for_tensor_core: bool = False

  def get_ref_aval(self) -> AbstractMemoryRef:
    aval = jax_core.ShapedArray(
        [self.num_barriers], BarrierType(self.num_arrivals,
                                         for_tensor_core=self.for_tensor_core)
    )
    return AbstractMemoryRef(aval, SMEM)

  def __post_init__(self):
    if self.num_arrivals < 1:
      raise ValueError(
          f"Num arrivals must be at least 1, but got {self.num_arrivals}"
      )

@dataclasses.dataclass(frozen=True, kw_only=True)
class ClusterBarrier:
  collective_axes: tuple[str | tuple[str, ...], ...]
  num_barriers: int = 1

  def get_ref_aval(self) -> AbstractMemoryRef:
    aval = jax_core.ShapedArray(
        [self.num_barriers], ClusterBarrierType(self.collective_axes)
    )
    return AbstractMemoryRef(aval, SMEM)


@dataclasses.dataclass(frozen=True)
class WGMMAAccumulatorRef:
  shape: tuple[int, int]
  dtype: jnp.dtype = jnp.float32
  _init: Any = state_types.uninitialized

  def get_ref_aval(self) -> AbstractMemoryRef:
    if self._init is not state_types.uninitialized:
      raise ValueError(
          "Preinitialized WGMMAAccumulatorRef only supported in pl.run_state."
      )
    return WGMMAAbstractAccumulatorRef(
        jax_core.ShapedArray(shape=self.shape, dtype=self.dtype), MemorySpace.REGS
    )

  @staticmethod
  def init(array):
    return WGMMAAccumulatorRef(array.shape, array.dtype, _init=array)


def _wgmma_ref_type_mapping(ref: WGMMAAccumulatorRef):
  aval = WGMMAAbstractAccumulatorRef(
      jax_core.ShapedArray(shape=ref.shape, dtype=ref.dtype), MemorySpace.REGS
  )
  return aval, ref._init
state_types._ref_type_aval_mappings[WGMMAAccumulatorRef] = _wgmma_ref_type_mapping


class WGMMAAbstractAccumulatorRef(AbstractMemoryRef):
  __slots__ = ["inner_aval", "memory_space"]

  def __repr__(self) -> str:
    return f'Accumulator{{{self.inner_aval.str_short()}}}'

  def update_weak_type(self, weak_type):
    return _as_accum(super().update_weak_type(weak_type))

  def update(self, inner_aval=None, memory_space=None):
    return _as_accum(super().update(inner_aval=None, memory_space=None))

  def _getitem(self, tracer, idx):
    from jax._src.pallas.mosaic_gpu.primitives import wgmma_accumulator_deref  # pytype: disable=import-error
    arr = wgmma_accumulator_deref(tracer)

    if not is_trivial_index(idx, tracer.shape):
      arr = arr[idx]

    return arr


def _as_accum(ref) -> WGMMAAbstractAccumulatorRef:
  return WGMMAAbstractAccumulatorRef(
      inner_aval=ref.inner_aval,
      memory_space=ref.memory_space,  # pytype: disable=attribute-error
  )

class AbstractTMEMRef(AbstractMemoryRef):
  __slots__ = ["inner_aval", "memory_space", "packed", "collective"]

  def __init__(self, inner_aval, memory_space, packed, collective):
    super().__init__(inner_aval, memory_space)
    self.packed = packed
    self.collective = collective

  def __repr__(self) -> str:
    return f'TMEM({self.inner_aval.str_short()},packed={self.packed})'


_WARPGROUP_AXIS_NAME = object()

@dataclasses.dataclass(frozen=True, kw_only=True)
class Mesh:
  grid: Sequence[int] = ()
  grid_names: Sequence[str] = ()
  cluster: Sequence[int] = ()
  cluster_names: Sequence[str] = ()
  # Those are NOT CUDA threads. On Hopper they correspond to warpgroups.
  num_threads: int | None = None
  thread_name: str | None = None

  def __post_init__(self):
    if len(self.cluster) > 3:
      raise ValueError(f"cluster= must be at most 3D, got {self}.")
    if len(self.grid_names) != len(self.grid):
      raise ValueError(
          f"grid_names must have the same length as grid, got {self}."
      )
    if len(self.cluster_names) != len(self.cluster):
      raise ValueError(
          f"cluster_names must have the same length as cluster, got {self}."
      )
    if (self.thread_name is None) != (self.num_threads is None):
      raise ValueError(
          "num_threads and thread_name must be either both set or both None,"
          f" got {self}"
      )
    if self.num_threads is not None and self.num_threads > 2048 // 128:
      raise ValueError(
          "Requested too many CUDA threads per block. Each Mosaic thread"
          " corresponds to 128 CUDA threads."
      )

  @property
  def backend(self) -> str:
    return "mosaic_gpu"

  @property
  def shape(self) -> collections.OrderedDict[object, int]:
    pairs: Iterable[tuple[object, int]]
    if self.num_threads is not None:
      pairs = zip(
          (*self.grid_names, *self.cluster_names, self.thread_name),
          (*self.grid, *self.cluster, self.num_threads),
      )
    else:
      pairs = zip(
          (*self.grid_names, *self.cluster_names),
          (*self.grid, *self.cluster),
      )
    return collections.OrderedDict(pairs)

  def discharges_effect(self, effect: jax_core.Effect):
    return effect is _wgmma_pipeline_effect or effect is _memory_effect

@dataclasses.dataclass(frozen=True, kw_only=True)
class WarpMesh:
  """Represents a mesh over individual warps within a warpgroup.

  When used in conjunction with `core_map`, the warp ID will be visible
  within the body of the wrapped scope by querying `lax.axis_index` with
  the specified axis name.
  """

  _NUM_WARPS_PER_WARPGROUP: ClassVar[int] = 4
  axis_name: str

  @property
  def shape(self):
    return collections.OrderedDict([
        (self.axis_name, self._NUM_WARPS_PER_WARPGROUP),
    ])

  def discharges_effect(self, effect: jax_core.Effect):
    del effect
    return False

def _gpu_mesh_discharge_rule(
    in_avals,
    out_avals,
    *args,
    mesh,
    jaxpr,
    compiler_params,
    interpret,
    debug,
    cost_estimate,
    name,
):
  if not isinstance(mesh, Mesh):
    raise TypeError(f"Mesh must be a `plgpu.Mesh`, got {type(mesh)}")
  if compiler_params and not isinstance(compiler_params, CompilerParams):
    raise TypeError(
        "Compiler params must be a `plgpu.CompilerParams`, got"
        f" {type(compiler_params)}"
    )
  if not compiler_params:
    compiler_params = CompilerParams()
  return pallas_core.default_mesh_discharge_rule(
      in_avals,
      out_avals,
      *args,
      jaxpr=jaxpr,
      mesh=mesh,
      compiler_params=compiler_params,
      debug=debug,
      interpret=interpret,
      cost_estimate=cost_estimate,
      name=name,
      memory_space=GMEM,
  )


pallas_core._core_map_mesh_rules[Mesh] = _gpu_mesh_discharge_rule


class MemoryEffect(jax_core.Effect):
  pass


effects.control_flow_allowed_effects.add_type(MemoryEffect)
_memory_effect = MemoryEffect()


class _WGMMAPipelineEffect(effects.Effect):
  pass


effects.control_flow_allowed_effects.add_type(_WGMMAPipelineEffect)
_wgmma_pipeline_effect = _WGMMAPipelineEffect()
