# Copyright 2023 The JAX Authors.
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

"""Module for pallas-core functionality."""
from __future__ import annotations

import collections
from collections.abc import Callable, Iterable, Iterator, Sequence
import contextlib
import copy
import dataclasses
import enum
import functools
import itertools
import threading
from typing import Any, ClassVar, Literal, Protocol, TypeAlias, Union, runtime_checkable
from collections.abc import Hashable

import jax
from jax._src import api_util
from jax._src import config
from jax._src import core as jax_core
from jax._src import dtypes
from jax._src import frozen_dict
from jax._src import linear_util as lu
from jax._src import state
from jax._src import tree_util
from jax._src import util
from jax._src.export._export import export
from jax._src.interpreters import mlir
from jax._src.interpreters import partial_eval as pe
from jax._src.state import discharge as state_discharge
from jax._src.state import indexing
from jax._src.state import types as state_types
from jax._src.state.types import TransformedRef
import jax.numpy as jnp


class DynamicGridDim:
  def __repr__(self):
    return "DynamicGridDim"
dynamic_grid_dim = DynamicGridDim()


partial = functools.partial
GridElement = int | jax_core.Array
GridName = Hashable
GridNames = tuple[Hashable, ...] | None
NamedGrid = tuple[tuple[GridName, int], ...]
TupleGrid = tuple[GridElement, ...]
Grid = Union[NamedGrid, TupleGrid]
StaticGrid = tuple[int, ...]
GridMappingGrid = tuple[int | DynamicGridDim, ...]
OriginStr = str  # The origin of a block spec, e.g. input[2]["field"]

# Datatype for semaphore values in interpret mode.
# For now, we choose a relatively uncommon datatype (i16) so it is more easily
# identifiable in kernels.
# TODO(justinfu): Handle semaphores with a custom extended dtype.
SEMAPHORE_INTERPRET_DTYPE = jnp.int16
SEMAPHORE_MAX_VALUE = jnp.iinfo(SEMAPHORE_INTERPRET_DTYPE).max

class AbstractSemaphoreTyRules:
  @staticmethod
  def pallas_interpret_element_aval(_) -> jax_core.ShapedArray:
    return jax_core.ShapedArray((), SEMAPHORE_INTERPRET_DTYPE)

  @staticmethod
  def physical_element_aval(_) -> jax_core.ShapedArray:
    return jax_core.ShapedArray((), jnp.int32)

# TODO(sharadmv): implement dtype rules for AbstractSemaphoreTy
class AbstractSemaphoreTy(dtypes.ExtendedDType):
  name: str
  _rules = AbstractSemaphoreTyRules

  def __repr__(self) -> str:
    return self.name

  def __eq__(self, other):
    return self.__class__ == other.__class__

  def __hash__(self) -> int:
    return hash(self.__class__)

class semaphore_dtype(dtypes.extended):
  """Common dtype for all kinds of semaphore dtypes.

  This is an abstract class that should never be instantiated, but rather
  exists for the sake of `jnp.issubdtype`.
  """

class semaphore(semaphore_dtype):
  """Regular semaphore dtype.

  Like its superclass, this class should never be instantiated.
  """

class Semaphore(AbstractSemaphoreTy):
  name = "semaphore"
  type = semaphore

class barrier_semaphore(semaphore_dtype):
  """Barrier semaphore dtype.

  Like its superclass, this class should never be instantiated.
  """

class BarrierSemaphore(AbstractSemaphoreTy):
  name = "barrier_semaphore"
  type = barrier_semaphore

Backend = Literal["mosaic_tpu", "triton", "mosaic_gpu"]

@runtime_checkable
class CompilerParams(Protocol):
  """Base class for compiler parameters."""
  BACKEND: ClassVar[Backend]

  # Subclasses must be dataclasses.
  __dataclass_fields__: ClassVar[dict[str, dataclasses.Field[Any]]]

@dataclasses.dataclass(frozen=True)
class Buffered:
  """Specifies how a block should be buffered for a pipeline.

  Attributes:
    buffer_count: The number of buffers to use for multiple buffering.
    use_lookahead: optional bool, indicates whether to use lookahead on the
      buffer. Enabling lookahead allows the pipeline to begin fetching the next
      changed block as soon as a slot is available, no matter how many
      iterations ahead that block is.
  """
  buffer_count: int
  use_lookahead: bool = False

split_list = util.split_list

map, unsafe_map = util.safe_map, map
zip, unsafe_zip = util.safe_zip, zip


class ShapedArrayWithMemorySpace(jax_core.ShapedArray):
  __slots__ = ["memory_space"]

  def __init__(self, shape, dtype, weak_type=False, sharding=None,
               vma=frozenset(), memory_space=None):
    super().__init__(shape, dtype, weak_type=weak_type, sharding=sharding,
                     vma=vma)
    self.memory_space = memory_space

  def __eq__(self, other):
    return super().__eq__(other) and self.memory_space == other.memory_space

  def __hash__(self):
    return hash((self.shape, self.dtype, self.weak_type, self.sharding,
                 self.vma, self.memory_space))

  def str_short(self, short_dtypes=False):
    dt_str = (dtypes.short_dtype_name(self.dtype) if short_dtypes else
              self.dtype.name)
    dt_str = dt_str.replace("void", "float0")
    shapestr = ",".join(map(str, self.shape))
    sharding_str = (f"{dt_str}[{shapestr}]({self.sharding})"
                    if self.sharding else "")
    memoryspace_str = ("" if self.memory_space is None
                       else f"<{self.memory_space}>")
    return f"{dt_str}{memoryspace_str}[{shapestr}]{sharding_str}"

  def update(
      self,
      shape=None,
      dtype=None,
      weak_type=None,
      sharding=None,
      vma=None,
      memory_space=None,
  ):
    if shape is None:
      shape = self.shape
    if dtype is None:
      dtype = self.dtype
    if weak_type is None:
      weak_type = self.weak_type
    if sharding is None:
      sharding = self.sharding
    if vma is None:
      vma = self.vma
    if memory_space is None:
      memory_space = self.memory_space
    return ShapedArrayWithMemorySpace(
        shape, dtype, weak_type, sharding=sharding, vma=vma,
        memory_space=memory_space
    )
mlir.ir_type_handlers[ShapedArrayWithMemorySpace] = mlir._array_ir_types


@dataclasses.dataclass(frozen=True)
class MemoryRef:
  """Like jax.ShapeDtypeStruct but with memory spaces."""
  shape: tuple[int, ...]
  dtype: jnp.dtype | dtypes.ExtendedDType
  # TODO(b/368122763): Unify memory space types across backends
  memory_space: Any

  def get_array_aval(self) -> jax_core.ShapedArray:
    dtype = self.dtype
    if not isinstance(dtype, (jnp.dtype, dtypes.ExtendedDType)):
      dtype = jnp.dtype(dtype)
    return ShapedArrayWithMemorySpace(
        self.shape, dtype, memory_space=self.memory_space
    )

  def get_ref_aval(self) -> TransformedRef | state.AbstractRef:
    # TODO(sharadmv): Clean this up. ShapedArrayWithMemorySpace fails when we
    # try to apply JAX ops to it.
    return state.AbstractRef(
        jax_core.ShapedArray(self.shape, self.dtype), self.memory_space)


class MemorySpace(enum.Enum):
  """Logical, device-agnostic memory spaces.

  Each memory space will be translated to a device-specific memory
  type during lowering.
  """
  ANY = "any"  # Unrestricted memory space (usually HBM)
  ERROR = "error"  # Memory space for checkify errors.
  INDEX = "index"  # Memory space for scalar prefetch arguments.
  KEY = "key"  # Memory space for PRNG keys.
  HOST = "host"  # Host memory space.

  def __str__(self) -> str:
    return self.value

  def __call__(self, shape: tuple[int, ...], dtype: jnp.dtype):
    # A convenience function for constructing MemoryRef types.
    return MemoryRef(shape, dtype, self)


@dataclasses.dataclass(frozen=True)
class PallasGridContext:
  grid: GridMappingGrid
  mapped_dims: tuple[int, ...]

  def size(self, axis: int) -> int | DynamicGridDim:
    valid_grid = tuple(self.grid)
    try:
      size = valid_grid[axis]
    except IndexError as e:
      raise ValueError(
          f"Axis {axis} is out of bounds for grid {self.grid}"
      ) from e
    return size


@dataclasses.dataclass
class PallasTracingEnv(threading.local):
  grid_context: PallasGridContext | None = None
  grid_env_stack: list[GridEnv] = dataclasses.field(default_factory=list)
  is_interpret_mode: bool = False
  dynamic_shapes: bool = False
  module_export_fn: Callable[[mlir.ir.Module], None] | None = None

_pallas_tracing_env = PallasTracingEnv()


def axis_frame() -> PallasGridContext:
  # This is like jax_core.axis_frame, except there should only ever be one
  # active PallasGridAxisName for a particular main_trace because we cannot
  # nest pallas_calls.
  env = _pallas_tracing_env
  assert env.grid_context is not None
  return env.grid_context


@dataclasses.dataclass(frozen=True)
class GridAxis:
  index: jax.Array
  size: int

# Stores the kernel execution position and the size along grid axes.
GridEnv = Sequence[GridAxis]

@contextlib.contextmanager
def grid_env(env: GridEnv) -> Iterator[None]:
  _pallas_tracing_env.grid_env_stack.append(env)
  try:
    yield
  finally:
   _pallas_tracing_env.grid_env_stack.pop()


def current_grid_env() -> GridEnv | None:
  if not _pallas_tracing_env.grid_env_stack:
    return None
  return _pallas_tracing_env.grid_env_stack[-1]


@dataclasses.dataclass(frozen=True)
class Element:
  """Use to index an array using an elementwise start index."""
  block_size: int
  padding: tuple[int, int] = (0, 0)

  def __str__(self):
    if self.padding == (0, 0):
      return f"Element({self.block_size})"
    return f"Element({self.block_size}, padding={self.padding})"

@dataclasses.dataclass(frozen=True)
class Squeezed:
  """Represents a one-sized block dimension that is squeezed out in the kernel."""

squeezed = Squeezed()

@dataclasses.dataclass(frozen=True)
class Blocked:
  """The default BlockShape type."""
  block_size: int

  def __str__(self):
    return f"Blocked({self.block_size})"

@dataclasses.dataclass(frozen=True)
class BoundedSlice:
  """Allows to specify a bounded slice of a dimension.

  Specifically, the index_map need to return a `pl.Slice/pl.ds` for this
  dimension. The start and size may be dynamic, as long as the size <=
  block_size.
  """
  block_size: int

  def __repr__(self):
    return f"BoundedSlice({self.block_size})"

BlockDim: TypeAlias = Element | Squeezed | Blocked | BoundedSlice


def default_index_map(ndim: int) -> Callable:
  return lambda *args: (0,) * ndim


def _canonicalize_block_dim(dim: BlockDim | int | None) -> BlockDim:
  match dim:
    case None:
      return squeezed
    case int():
      return Blocked(int(dim))
    case Squeezed() | Blocked() | Element() | BoundedSlice():
      return dim
    case _:
      # Handle case where the dim is a symbolic dimension so we assume it is
      # Blocked.
      if jax_core.is_symbolic_dim(dim):
        return Blocked(dim)
      try:
        return Blocked(int(dim))
      except Exception as e:
        raise ValueError(
            f"Unsupported block dimension type: {type(dim)}. Allowed types:"
            " `pl.Squeezed`, `pl.Blocked`, `pl.Element`, `int`, `None`."
        ) from e

def _canonicalize_block_shape(block_shape: Sequence[BlockDim | int | None]
                              ) -> tuple[BlockDim, ...]:
  return tuple(_canonicalize_block_dim(dim) for dim in block_shape)


def _get_block_dim_size(dim: BlockDim) -> int:
  match dim:
    case Squeezed():
      return 1
    case Blocked(block_size):
      return block_size
    case Element():
      return dim.block_size
    case BoundedSlice(block_size):
      return block_size
    case _:
      raise ValueError(f"Unsupported block shape type: {type(dim)}")


def _get_block_shape(block_shape: tuple[BlockDim, ...]) -> tuple[int, ...]:
  return tuple(_get_block_dim_size(dim) for dim in block_shape)

def _get_ref_block_shape(block_shape: tuple[BlockDim, ...]) -> tuple[int, ...]:
  # Special handling for squeezed here (don't include Squeezed dims in the Ref
  # shape).
  return tuple(
      _get_block_dim_size(dim)
      for dim in block_shape
      if not isinstance(dim, Squeezed)
  )


class _IndexMapFunc:
  """Helper class that checks for index_map equality."""

  def __init__(self, index_map):
    self.index_map = index_map
    functools.update_wrapper(self, self.index_map)

  def __eq__(self, other: object):
    if not isinstance(other, _IndexMapFunc):
      return NotImplemented
    return self.index_map == other.index_map

  def __call__(self, *args, **kwargs):
    out_indices = self.index_map(*args, **kwargs)
    if isinstance(out_indices, list):
      out_indices = tuple(out_indices)
    if not isinstance(out_indices, tuple):
      out_indices = (out_indices,)
    return out_indices


@dataclasses.dataclass
class BlockSpec:
  """Specifies how an array should be sliced for each invocation of a kernel.

  The `block_shape` is a sequence of `int | None`s, or `BlockDim` types (e.g.
  `pl.Element`, `pl.Squeezed`, `pl.Blocked`, `pl.BoundedSlice`). Each of these
  types specify the size of the block dimension. `None` is used to specify a
  dimension that is squeezed out of the kernel. The `BlockDim` types allow for
  more fine-grained control over the indexing of the dimension. The `index_map`
  needs to return a tuple of the same length as `block_shape`, which each entry
  depending on the type of `BlockDim`.

  See :ref:`pallas_blockspec` and the individual `BlockDim` type docstrings for
  more details.
  """
  # An internal canonicalized version is in BlockMapping.
  block_shape: Sequence[BlockDim | int | None] | None = None
  index_map: Callable[..., Any] | None = None
  memory_space: Any | None = dataclasses.field(kw_only=True, default=None)
  pipeline_mode: Buffered | None = None

  def __post_init__(self):
    if self.index_map is not None:
      self.index_map = _IndexMapFunc(self.index_map)

  def to_block_mapping(
      self,
      origin: OriginStr,
      array_aval: jax_core.ShapedArray,
      *,
      # Inputs for the index_map
      index_map_avals: Sequence[jax_core.AbstractValue],
      index_map_tree: tree_util.PyTreeDef,
      grid: GridMappingGrid,
      vmapped_dims: tuple[int, ...],
      debug: bool = False,
  ) -> BlockMapping:
    if self.index_map is None:
      index_map_func = default_index_map(len(array_aval.shape))
      index_map_dbg = api_util.debug_info("pallas_call index_map",
                                          default_index_map, (),{}
                                          )._replace(arg_names=("",) * len(index_map_avals))
      api_util.save_wrapped_fun_debug_info(index_map_func, index_map_dbg)
    else:
      index_map_func = self.index_map
    if self.block_shape is None:
      block_shape = _canonicalize_block_shape(array_aval.shape)
    else:
      block_shape = _canonicalize_block_shape(self.block_shape)
      if len(array_aval.shape) != len(block_shape):
        raise ValueError(
            f"Block shape for {origin} (= {block_shape}) "
            "must have the same number of dimensions as the "
            f"array shape {array_aval.shape}."
        )

    ref_block_shape = _get_ref_block_shape(block_shape)
    if isinstance(array_aval, jax_core.DShapedArray):
      # Get the "max" shape for the ragged array.
      block_array_aval = array_aval.update(shape=ref_block_shape)
      block_array_aval = jax_core.ShapedArray(
          block_array_aval.shape,
          block_array_aval.dtype,
          block_array_aval.weak_type,
      )
    elif isinstance(array_aval, ShapedArrayWithMemorySpace):
      block_array_aval = jax_core.ShapedArray(
          ref_block_shape, array_aval.dtype, array_aval.weak_type
      )
    else:
      block_array_aval = array_aval.update(shape=ref_block_shape)
    block_aval = state.AbstractRef(block_array_aval, self.memory_space)

    if (
        not jax_core.is_constant_shape(block_aval.shape)
        and not dynamic_shapes_export_enabled()
    ):
      raise ValueError(
          "shape polymorphism for Pallas does not support "
          "dynamically-shaped blocks. "
          f"Block spec for {origin} has block_shape: {block_aval.shape}"
      )

    fake_index_map_args, fake_index_map_kwargs = \
        index_map_tree.unflatten([False] * index_map_tree.num_leaves)
    debug_info = api_util.debug_info(
        "pallas_call index_map",
        index_map_func,
        fake_index_map_args,
        fake_index_map_kwargs,
    )
    flat_index_map_fun, index_map_out_tree_thunk = api_util.flatten_fun(
        lu.wrap_init(index_map_func, debug_info=debug_info), index_map_tree
    )
    with tracing_grid_env(grid, vmapped_dims):
      jaxpr, out_avals, consts = pe.trace_to_jaxpr_dynamic(
          flat_index_map_fun, index_map_avals
      )
    index_map_out_tree = index_map_out_tree_thunk()
    unflat_avals = tree_util.tree_unflatten(index_map_out_tree, out_avals)

    if len(unflat_avals) != len(block_shape):
      raise ValueError(
          f"Index map function {debug_info.func_src_info} for "
          f"{origin} must return "
          f"{len(block_shape)} values to match {block_shape=}. "
          f"Currently returning {len(unflat_avals)} values:"
      )
    # Verify types match
    for i, (idx_aval, bd) in enumerate(zip(unflat_avals, block_shape)):
      match bd:
        case BoundedSlice():
          if not isinstance(idx_aval, indexing.Slice):
            raise ValueError(
                "index_map returned a value of type"
                f" {type(idx_aval)} at position {i} with block dimension"
                f" {bd} when it should be pl.Slice"
            )
        case Blocked() | Element() | Squeezed() | int():
          if (
              not isinstance(idx_aval, jax_core.ShapedArray)
              and not idx_aval.shape
          ):
            raise ValueError(
                "index_map returned a value of type"
                f" {type(idx_aval)} at position {i} with block dimension"
                f" {bd} when it should be a scalar"
            )
    for i, ov in enumerate(out_avals):
      if ov.shape or ov.dtype not in [jnp.int32, jnp.int64]:
        raise ValueError(
            f"Index map function {debug_info.func_src_info} for "
            f"{origin} must return integer scalars. Output[{i}] has type "
            f"{ov}."
        )

    if consts:
      raise ValueError(
          f"Index map function {debug_info.func_src_info} for "
          f"{origin} must not capture constants: {consts}"
      )

    array_aval_shape = _max_shape_from_aval(array_aval)

    mapping = BlockMapping(
        block_shape=block_shape,
        transformed_block_aval=block_aval,  # There are no transforms by default
        index_map_jaxpr=jax_core.ClosedJaxpr(jaxpr, consts),
        index_map_out_tree=index_map_out_tree,
        array_shape_dtype=jax.ShapeDtypeStruct(
            array_aval_shape, array_aval.dtype
        ),
        origin=origin,
        pipeline_mode=self.pipeline_mode,
        debug=debug,
    )
    mapping.check_invariants()
    return mapping

  replace = dataclasses.replace


class NoBlockSpec:
  def __repr__(self):
    return "NoBlockSpec"
no_block_spec = NoBlockSpec()


# A PyTree of BlockSpec | NoBlockSpec.
# BlockSpecTree = Sequence[BlockSpec | NoBlockSpec, ...] | NoBlockSpec
BlockSpecTree = Any


class MemoryRefTransform(Protocol):
  """Transforms a memory reference on load or store."""

  def undo(self, ref: TransformedRef) -> TransformedRef:
    raise NotImplementedError("Abstract evaluation not implemented.")


@dataclasses.dataclass(frozen=True)
class BlockMapping:
  """An internal canonicalized version of BlockSpec.

  See the `check_invariants` method for precise specification.
  """
  # TODO(apaszke,sharadmv): Replace mapped dims in block_shape with a transform.
  # After all, it's just indexing out singleton dimensions.
  block_shape: tuple[BlockDim, ...]
  transformed_block_aval: state.AbstractRef
  index_map_jaxpr: jax_core.ClosedJaxpr
  index_map_out_tree: tree_util.PyTreeDef
  array_shape_dtype: jax.ShapeDtypeStruct  # The whole array
  origin: OriginStr
  transforms: Sequence[MemoryRefTransform] = ()
  pipeline_mode: Buffered | None = None
  debug: bool = False

  def check_invariants(self) -> None:
    if not config.enable_checks.value: return

    ref_block_shape = _get_ref_block_shape(self.block_shape)
    assert ref_block_shape == self.ref_aval.shape, (
        self.block_shape, self.ref_aval.shape)
    assert len(self.block_shape) == len(self.array_shape_dtype.shape), (
        self.block_shape, self.array_shape_dtype
    )

    assert not self.index_map_jaxpr.consts
    assert all(ov.shape == () and
               (ov.dtype == jnp.int32 or ov.dtype == jnp.int64)
               for ov in self.index_map_jaxpr.out_avals), (
               self.index_map_jaxpr.out_avals)

  def replace(self, **kwargs):
    new_self = dataclasses.replace(self, **kwargs)
    new_self.check_invariants()
    return new_self

  @property
  def block_aval(self) -> state.AbstractRef:
    # If you hit this, make sure you take transforms into account and use either
    # ref_aval or transformed_block_aval.
    assert not self.transforms, "Lowering failed to handle transforms"
    return self.transformed_block_aval

  @property
  def ref_aval(self) -> state.AbstractRef | TransformedRef:
    """Returns the abstract value of the Ref after transformations."""
    if not self.transforms:
      return self.transformed_block_aval
    ref = TransformedRef(self.transformed_block_aval, ())
    for transform in reversed(self.transforms):
      ref = transform.undo(ref)
    return ref

  def compute_start_indices_interpret(self, loop_idx, *args):
    discharged_jaxpr, discharged_consts = state_discharge.discharge_state(
        self.index_map_jaxpr.jaxpr, self.index_map_jaxpr.consts
    )
    jaxpr = jax_core.ClosedJaxpr(discharged_jaxpr, discharged_consts)
    block_indices_and_rest = jax_core.jaxpr_as_fun(jaxpr)(*loop_idx, *args)
    # Since we're passing in `Ref`s potentially, we need to split out their
    # updated values since we only care about the return values.
    block_indices, _ = split_list(block_indices_and_rest,
                                  [len(self.block_shape)])
    def _get_start_index(i, b):
      match b:
        case Squeezed() | Element():
          return i
        case Blocked(block_size):
          return block_size * i
        case _:
          raise ValueError(f"Unsupported block dim type: {type(b)}")
    return tuple(
        _get_start_index(i, b) for i, b in zip(block_indices, self.block_shape)
    )

  def has_trivial_window(self):
    """If block shape is same as the array shape and index_map returns 0s."""
    for b, s in zip(self.block_shape, self.array_shape_dtype.shape):
      if _get_block_dim_size(b) != s:
        return False
    for atom in self.index_map_jaxpr.jaxpr.outvars:
      if not (isinstance(atom, jax_core.Literal) and atom.val == 0):
        return False
    return True

  def __repr__(self):
    if self.debug:
      return (
          f"BlockMapping(block_shape={self.block_shape}, "
          f"transformed_block_aval={self.transformed_block_aval}, "
          f"index_map_jaxpr={self.index_map_jaxpr}, "
          f"index_map_out_tree={self.index_map_out_tree}, "
          f"array_shape_dtype={self.array_shape_dtype}, "
          f"origin={self.origin}, "
          f"transforms={self.transforms}, "
          f"pipeline_mode={self.pipeline_mode}, "
          f"debug={self.debug})"
      )
    return f"BlockMapping(block_shape={self.block_shape})"

  def __str__(self):
    return self.__repr__()


@contextlib.contextmanager
def tracing_grid_env(grid: GridMappingGrid, mapped_dims: tuple[int, ...]):
  if dynamic_shapes_export_enabled():
    assert all(i is dynamic_grid_dim or jax_core.is_dim(i) for i in grid)
  else:
    assert all(i is dynamic_grid_dim or isinstance(i, int) for i in grid)
  old_grid_context = _pallas_tracing_env.grid_context
  try:
    _pallas_tracing_env.grid_context = PallasGridContext(grid, mapped_dims)
    yield
  finally:
    _pallas_tracing_env.grid_context = old_grid_context


@contextlib.contextmanager
def pallas_export_experimental(dynamic_shapes: bool):
  old_dynamic_shapes = _pallas_tracing_env.dynamic_shapes
  try:
    _pallas_tracing_env.dynamic_shapes = dynamic_shapes
    yield
  finally:
    _pallas_tracing_env.dynamic_shapes = old_dynamic_shapes


def dynamic_shapes_export_enabled() -> bool:
  return _pallas_tracing_env.dynamic_shapes


@dataclasses.dataclass(frozen=True)
class GridMapping:
  """An internal canonicalized version of GridSpec.

  Encodes the calling conventions of the pallas_call primitive, the kernel,
  and the index maps.

  The pallas_call is invoked with: ``*dynamic_grid_sizes, *index, *inputs``.
  The ``index`` operands are for the scalar prefetch.

  The kernel function is invoked with:
  ``*index, *inputs, *scratch``.

  The index map functions are invoked with:
  ``*program_ids, *index``.

  See the `check_invariants` method for a more precise specification.
  """
  grid: GridMappingGrid
  grid_names: tuple[Hashable, ...] | None

  # Block mappings for: *inputs, *outputs
  block_mappings: tuple[BlockMapping, ...]
  # The inputs for tracing the index map: the tree and the flat avals
  index_map_tree: tree_util.PyTreeDef
  index_map_avals: tuple[jax_core.AbstractValue]
  # Which dimensions in `grid` are vmapped.
  vmapped_dims: tuple[int, ...]

  num_index_operands: int
  num_inputs: int
  num_outputs: int
  num_scratch_operands: int
  get_grid_indices: Callable | None = None
  local_grid_env: Callable | None = None
  # Primarily dictates how much debugging information is printed.
  debug: bool = False

  def check_invariants(self) -> None:
    if not config.enable_checks.value: return
    assert (len(self.block_mappings) == self.num_inputs + self.num_outputs), (
        self.num_inputs, self.num_outputs,
        self.block_mappings
    )
    # index_map_avals = int32[] * len(self.grid) + index_operands
    assert len(self.index_map_avals) == len(self.grid) + self.num_index_operands, (
        self.index_map_avals,
        self.grid,
        self.num_index_operands,
    )
    # Check that we can put together the avals and the tree.
    index_map_args, index_map_kwargs = self.index_map_tree.unflatten(
        self.index_map_avals)
    assert not index_map_kwargs
    assert len(index_map_args) >= len(self.grid)
    for i in range(len(self.grid)):
      index_map_arg = index_map_args[i]
      assert index_map_arg.shape == (), f"index_map_arg: {index_map_arg}"
      assert index_map_arg.dtype == jnp.int32, f"index_map_arg: {index_map_arg}"

    assert len(self.vmapped_dims) <= len(self.grid)
    for i in self.vmapped_dims:
      assert 0 <= i < len(self.grid)

    if self.grid_names is not None:
      assert len(self.grid) == len(self.grid_names), (self.grid, self.grid_names)

    for bm in self.block_mappings:
      bm.check_invariants()
      assert tuple(self.index_map_avals) == tuple(
          bm.index_map_jaxpr.in_avals
      ), (
          self.index_map_avals,
          "|",
          bm.index_map_jaxpr.in_avals,
      )

  def replace(self, **kwargs) -> GridMapping:
    new_self = dataclasses.replace(self, **kwargs)
    new_self.check_invariants()
    return new_self

  @property
  def num_dynamic_grid_bounds(self):
    return sum(b is dynamic_grid_dim for b in self.grid)

  @property
  def static_grid(self) -> StaticGrid:
    if self.num_dynamic_grid_bounds:
      raise ValueError("Expected a grid with fully static bounds")
    return self.grid  # type: ignore

  @contextlib.contextmanager
  def trace_env(self):
    if self.grid_names is None:
      axis_env_ctx = contextlib.nullcontext()
    else:
      axis_env_ctx = jax_core.extend_axis_env_nd(
          zip(self.grid_names, self.grid)
      )
    with tracing_grid_env(self.grid, self.vmapped_dims), axis_env_ctx:
      yield

  @property
  def slice_index_ops(self):
    """Returns a slice object to select the index operands to a kernel.
    This works on a sequence that contains *index, *ins, *outs, *scratch.
    """
    return slice(0, self.num_index_operands)

  @property
  def slice_block_ops(self):
    """Returns a slice to select the block operands to a kernel.

    The block operands are: *ins, *outs, the same for which we
    have `self.block_mappings`.
    This works on a sequence that contains *index, *ins, *outs, *scratch.
    """
    return slice(self.num_index_operands,
                 self.num_index_operands + len(self.block_mappings))

  @property
  def slice_scratch_ops(self):
    """Returns a slice object to select the scratch operands to a kernel.
    This works on a sequence that contains *index, *ins, *outs, *scratch.
    """
    if self.num_scratch_operands:
      return slice(-self.num_scratch_operands, None)
    else:
      return slice(0, 0)

  @property
  def in_shapes(self) -> Iterable[jax.ShapeDtypeStruct]:
    """The shapes of *index, *inputs."""
    index_shapes = (
        jax.ShapeDtypeStruct(ia.shape, ia.dtype)
        for ia in self.index_map_avals[len(self.grid) :]
    )
    inputs_shapes = (
        bm.array_shape_dtype
        for bm in self.block_mappings[:self.num_inputs])
    return itertools.chain(index_shapes, inputs_shapes)

  @property
  def block_mappings_output(self) -> Iterable[BlockMapping]:
    return itertools.islice(
        self.block_mappings,
        self.num_inputs,
        self.num_inputs + self.num_outputs)

  @property
  def out_shapes(self) -> Iterable[jax.ShapeDtypeStruct]:
    return tuple(
        bm.array_shape_dtype for bm in self.block_mappings_output)

  def __repr__(self):
    if self.debug:
      return (
          f"GridMapping(grid={self.grid}, grid_names={self.grid_names}, "
          f"block_mappings={self.block_mappings}, "
          f"index_map_tree={self.index_map_tree}, "
          f"index_map_avals={self.index_map_avals}, "
          f"vmapped_dims={self.vmapped_dims}, "
          f"num_index_operands={self.num_index_operands}, "
          f"num_inputs={self.num_inputs}, "
          f"num_outputs={self.num_outputs}, "
          f"num_scratch_operands={self.num_scratch_operands}, "
          f"get_grid_indices={self.get_grid_indices}, "
          f"local_grid_env={self.local_grid_env}, "
          f"debug={self.debug})"
      )
    return (
        f"GridMapping(grid={self.grid}, block_mappings={self.block_mappings})"
    )

  def __str__(self):
    return self.__repr__()


def _is_valid_grid_dim(dim: int | jax.Array) -> bool:
  if isinstance(dim, jax.Array):
    return True
  return jax_core.is_dim(dim)


def _max_shape_from_aval(array_aval: jax_core.ShapedArray):
  array_aval_shape = list(array_aval.shape)
  for i, s in enumerate(array_aval.shape):
    try:
      aval = jax_core.get_aval(s)
      if isinstance(aval, jax_core.DShapedArray):
        array_aval_shape[i] = aval.dtype.bound
    except OverflowError as e:
      # Note - there are annoying cases where on 32 bit hardware,
      # a flattened index space may overflow - for these cases,
      # we just take the shape as is.
      # In most places, this is totally sound to do.
      # For ragged/jumble inputs, this will fail downstream.
      return array_aval.shape

  return tuple(array_aval_shape)


def _convert_block_spec_to_block_mapping(
    block_spec: BlockSpec,
    origin: OriginStr,
    array_aval: jax_core.ShapedArray,
    *,
    # Inputs for the index_map
    index_map_avals: Sequence[jax_core.AbstractValue],
    index_map_tree: tree_util.PyTreeDef,
    grid: GridMappingGrid,
    vmapped_dims: tuple[int, ...],
    debug: bool = False,
) -> BlockMapping:
  if block_spec is no_block_spec:
    block_spec = BlockSpec(None, None)
  return block_spec.to_block_mapping(
      origin,
      array_aval,
      index_map_avals=index_map_avals,
      index_map_tree=index_map_tree,
      grid=grid,
      vmapped_dims=vmapped_dims,
      debug=debug,
  )


index_map_grid_aval = jax_core.ShapedArray((), jnp.int32)


class ScratchShape(Protocol):
  def get_array_aval(self) -> jax_core.AbstractValue:
    ...
  def get_ref_aval(self) -> state.AbstractRef | TransformedRef:
    ...


ScratchShapeTree = Sequence[Union[ScratchShape, "ScratchShapeTree"]]


@dataclasses.dataclass(init=False, kw_only=True)
class GridSpec:
  """Encodes the grid parameters for :func:`jax.experimental.pallas.pallas_call`.

  See the documentation for :func:`jax.experimental.pallas.pallas_call`,
  and also :ref:`pallas_grids_and_blockspecs` for a more detailed
  description of the parameters.
  """
  # A canonicalized internal version is in GridMapping.
  grid: TupleGrid
  grid_names: tuple[Hashable, ...] | None
  in_specs: BlockSpecTree
  out_specs: BlockSpecTree
  scratch_shapes: ScratchShapeTree = ()

  def __init__(
      self,
      grid: Grid = (),
      in_specs: BlockSpecTree = no_block_spec,
      out_specs: BlockSpecTree = no_block_spec,
      scratch_shapes: ScratchShapeTree = (),
  ):
    # Be more lenient for in/out_specs
    if isinstance(in_specs, list):
      in_specs = tuple(in_specs)
    elif in_specs is not no_block_spec and not isinstance(in_specs, Sequence):
      raise ValueError(f"`in_specs` must be a tuple or a list. Found: {in_specs}")
    if isinstance(out_specs, list):
      out_specs = tuple(out_specs)

    self.in_specs = in_specs
    self.out_specs = out_specs
    self.scratch_shapes = tuple(scratch_shapes)

    grid_names = None
    if isinstance(grid, int):
      grid = (grid,)
    elif grid and isinstance(grid[0], tuple):  # Check if we have a named grid
      grid_names, grid = util.unzip2(grid)  # type: ignore

    # TODO(b/353730556): allow NumPy scalars in grids
    if not all(_is_valid_grid_dim(g) for g in grid):  # type: ignore
      raise ValueError(
          f"Grid must be a tuple of integers or jax.Array, got {grid}"
      )
    self.grid = grid  # type: ignore
    self.grid_names = grid_names

  def _make_scalar_ref_aval(self, aval):
    assert False  # Not needed in GridSpec


def get_grid_mapping(
    grid_spec: GridSpec,
    in_avals: Sequence[jax_core.AbstractValue],
    in_tree: tree_util.PyTreeDef,
    in_origins: Sequence[OriginStr],
    out_avals: Sequence[jax_core.AbstractValue],
    out_tree: tree_util.PyTreeDef,
    out_origins: Sequence[OriginStr],
    debug: bool = False,
) -> tuple[tuple[jax_core.AbstractValue, ...], GridMapping]:
  if dynamic_shapes_export_enabled():
    dim_check : Any = jax_core.is_dim
  else:
    dim_check : Any = jax_core.is_constant_dim  # type: ignore[no-redef]
  assert all(i is None or dim_check(i) for i in grid_spec.grid)
  grid_mapping_grid = tuple(
      dynamic_grid_dim if d is None else d for d in grid_spec.grid
  )
  # The inputs for the index maps
  index_map_avals = (
      index_map_grid_aval.update(sharding=jax_core.get_cur_mesh_sharding()),
  ) * len(grid_spec.grid)
  index_map_tree = tree_util.tree_structure((index_map_avals, {}))

  num_scalar_prefetch: int = getattr(grid_spec, "num_scalar_prefetch", 0)
  if num_scalar_prefetch:
    all_avals = tree_util.tree_unflatten(in_tree, in_avals)
    scalar_avals, unflat_in_avals = split_list(
        all_avals, [num_scalar_prefetch])
    flat_scalar_avals, scalar_tree = tree_util.tree_flatten(scalar_avals)
    num_flat_scalar_prefetch = len(flat_scalar_avals)
    scalar_ref_avals = [
        grid_spec._make_scalar_ref_aval(aval)
        for aval in flat_scalar_avals]
    jaxpr_scalar_ref_avals = tree_util.tree_unflatten(
        scalar_tree, scalar_ref_avals)
    in_avals, in_tree = tree_util.tree_flatten(tuple(unflat_in_avals))
    index_map_tree = tree_util.tree_structure(((*index_map_avals,
                                                *scalar_avals), {}))
    index_map_avals = (*index_map_avals, *scalar_ref_avals)
    del scalar_ref_avals, flat_scalar_avals, scalar_tree
    del scalar_avals, unflat_in_avals, all_avals
  else:
    num_flat_scalar_prefetch = 0
    jaxpr_scalar_ref_avals = ()
  if grid_spec.scratch_shapes:
    flat_scratch_shapes, scratch_tree = tree_util.tree_flatten(
        grid_spec.scratch_shapes)
    flat_scratch_avals = map(lambda s: s.get_ref_aval(), flat_scratch_shapes)
    num_flat_scratch_operands = len(flat_scratch_avals)
    jaxpr_scratch_avals = tree_util.tree_unflatten(
        scratch_tree, flat_scratch_avals)
    if not isinstance(jaxpr_scratch_avals, (tuple, list)):
      jaxpr_scratch_avals = (jaxpr_scratch_avals,)
    del flat_scratch_avals, flat_scratch_shapes, scratch_tree
  else:
    num_flat_scratch_operands = 0
    jaxpr_scratch_avals = ()

  if grid_spec.in_specs is not no_block_spec:
    flat_in_specs, in_specs_tree = tree_util.tree_flatten(grid_spec.in_specs)
    if in_specs_tree != in_tree:
      raise ValueError(
          pytreedef_mismatch_err_msg("`in_specs`", in_specs_tree,
                                     "`inputs`", in_tree))
  else:
    flat_in_specs = [no_block_spec] * len(in_avals)

  in_block_mappings = map(
      partial(
          _convert_block_spec_to_block_mapping,
          index_map_avals=index_map_avals,
          index_map_tree=index_map_tree,
          grid=grid_mapping_grid,  # type: ignore[arg-type]
          vmapped_dims=(),
          debug=debug,
      ),
      flat_in_specs,
      in_origins[num_flat_scalar_prefetch:],
      in_avals,
  )

  if grid_spec.out_specs is not no_block_spec:
    flat_out_specs, out_specs_tree = tree_util.tree_flatten(grid_spec.out_specs)
    if out_specs_tree != out_tree:
      raise ValueError(
          pytreedef_mismatch_err_msg("`out_specs`", out_specs_tree,
                                     "`out_shape`", out_tree))
  else:
    flat_out_specs = [no_block_spec] * len(out_avals)

  out_block_mappings = map(
      partial(
          _convert_block_spec_to_block_mapping,
          index_map_avals=index_map_avals,
          index_map_tree=index_map_tree,
          grid=grid_mapping_grid,  # type: ignore[arg-type]
          vmapped_dims=(),
          debug=debug,
      ),
      flat_out_specs,
      out_origins,
      out_avals,
  )
  grid_mapping = GridMapping(
      grid=grid_mapping_grid,  # type: ignore[arg-type]
      grid_names=grid_spec.grid_names,
      block_mappings=(*in_block_mappings, *out_block_mappings),
      index_map_avals=index_map_avals,
      index_map_tree=index_map_tree,
      vmapped_dims=(),
      num_index_operands=num_flat_scalar_prefetch,
      num_inputs=len(flat_in_specs),
      num_outputs=len(flat_out_specs),
      num_scratch_operands=num_flat_scratch_operands,
      debug=debug,
  )
  grid_mapping.check_invariants()
  in_ref_avals = [bm.ref_aval for bm in in_block_mappings]
  jaxpr_in_ref_avals = tree_util.tree_unflatten(in_tree, in_ref_avals)
  jaxpr_in_avals = (*jaxpr_scalar_ref_avals,
                    *jaxpr_in_ref_avals)
  out_ref_avals = [bm.ref_aval for bm in out_block_mappings]
  jaxpr_out_avals = tree_util.tree_unflatten(out_tree, out_ref_avals)
  if not isinstance(jaxpr_out_avals, (tuple, list)):
    jaxpr_out_avals = (jaxpr_out_avals,)
  return (*jaxpr_in_avals, *jaxpr_out_avals,
          *jaxpr_scratch_avals), grid_mapping


def unzip_dynamic_grid_bounds(
    grid_spec: GridSpec) -> tuple[GridSpec, tuple[Any, ...]]:
  if dynamic_shapes_export_enabled():
    new_grid : Any = grid_spec.grid
  else:
    new_grid : Any = tuple(d if isinstance(d, int) else None for d in grid_spec.grid)  # type: ignore[no-redef]
  dynamic_bounds = tuple(d for d in grid_spec.grid if not isinstance(d, int))
  # We can't use dataclasses.replace, because our fields are incompatible
  # with __init__'s signature.
  static_self = copy.copy(grid_spec)
  static_self.grid = new_grid
  return static_self, dynamic_bounds


def pytreedef_mismatch_err_msg(
    what1: str, tree1: tree_util.PyTreeDef,
    what2: str, tree2: tree_util.PyTreeDef) -> str:
  errs = list(tree_util.equality_errors_pytreedef(tree1, tree2))
  msg = []
  msg.append(
      f"Pytree for {what1} and {what2} do not match. "
      f"There are {len(errs)} mismatches, including:")
  for path, thing1, thing2, explanation in errs:
    where = f"at {tree_util.keystr(path)}, " if path else ""
    msg.append(
        f"    * {where}{what1} is a {thing1} but"
        f" {what2} is a {thing2}, so {explanation}")
  return "\n".join(msg)


@dataclasses.dataclass(frozen=True)
class CostEstimate:
  flops: int
  transcendentals: int
  bytes_accessed: int

  def __post_init__(self):
    for k, v in dataclasses.asdict(self).items():
      if not isinstance(v, int):
        raise ValueError("All fields in CostEstimate must be ints. "
                         f"{k} is not an int: {type(v)}({v})")

  def to_json(self) -> bytes:
    return (
        f'{{"flops": {self.flops}, "transcendentals": {self.transcendentals},'
        f' "bytes_accessed": {self.bytes_accessed}}}'
    ).encode("ascii")


def get_memory_space_aval(aval: jax_core.AbstractValue) -> Any:
  """Queries the memory space of an array."""
  if isinstance(aval, ShapedArrayWithMemorySpace):
    return aval.memory_space
  if isinstance(aval, state.AbstractRef):
    if aval.memory_space is not None:
      return aval.memory_space
    return get_memory_space_aval(aval.inner_aval)
  return None

def _get_sds(aval: jax_core.AbstractValue):
  match aval:
    case state.AbstractRef(inner_aval=inner_aval):
      if aval.memory_space is not None:
        return aval.memory_space(aval.shape, aval.dtype)
      return _get_sds(inner_aval)
    case ShapedArrayWithMemorySpace():
      return aval.memory_space(aval.shape, aval.dtype)
    case jax_core.ShapedArray():
      return jax.ShapeDtypeStruct(aval.shape, aval.dtype)
    case _:
      raise ValueError(f"Unsupported abstract value: {aval}")


core_map_p = jax_core.Primitive("core_map")
core_map_p.multiple_results = True


def core_map(
    mesh,
    *,
    compiler_params: Any | None = None,
    interpret: bool = False,
    debug: bool = False,
    cost_estimate: CostEstimate | None = None,
    name: str | None = None,
    metadata: dict[str, str] | None = None,
):
  """Runs a function on a mesh, mapping it over the devices in the mesh.

  The function should be stateful in that it takes in no inputs and returns
  no outputs but can mutate closed-over Refs, for example.

  Args:
    mesh: The mesh to run the function on.
    compiler_params: The compiler parameters to pass to the backend.
    interpret: Whether to run the function in interpret mode.
    debug: Whether or not to out helpful debugging information.
    cost_estimate: The cost estimate of the function.
    metadata: Optional dictionary of information about the kernel that will be
      serialized as JSON in the HLO. Can be used for debugging and analysis.
  """
  def wrapped(f):
    flat_args, in_tree = tree_util.tree_flatten(((), {}))
    flat_fun, out_tree_thunk = api_util.flatten_fun(
        lu.wrap_init(f,
                     debug_info=api_util.debug_info("pallas_core_map", f,
                                                    (), {})),
        in_tree)
    with (
        tracing_grid_env(tuple(mesh.shape.values()), mapped_dims=()),
        jax_core.extend_axis_env_nd(mesh.shape.items()),
    ):
      jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, flat_args)
    out = core_map_p.bind(
        *consts,
        jaxpr=jaxpr,
        mesh=mesh,
        compiler_params=compiler_params,
        interpret=(
            config.pallas_tpu_interpret_mode_context_manager.value or interpret
        ),
        debug=debug,
        cost_estimate=cost_estimate,
        name=name or util.fun_name(f),
        metadata=frozen_dict.FrozenDict(metadata)
        if metadata is not None
        else None,
    )
    if out:
      raise ValueError("core_map-ped functions must not return any outputs.")
    return tree_util.tree_unflatten(out_tree_thunk(), out)

  return wrapped


@core_map_p.def_effectful_abstract_eval
def _core_map_abstract_eval(*args, jaxpr, mesh, **kwargs):
  del args
  if jaxpr.outvars:
    raise ValueError("core_map must not return any outputs.")
  interpret = kwargs.get('interpret', False)
  effs = set()
  if interpret:
    try:
      from jax._src.pallas.mosaic import interpret as mosaic_tpu_interpret  # Avoid circular dependency.
      if isinstance(interpret, mosaic_tpu_interpret.InterpretParams):
        effs = mosaic_tpu_interpret.get_interpret_effects()
    except ImportError:
      pass
  for eff in jaxpr.effects:
    if mesh.discharges_effect(eff):
      continue
    if not isinstance(eff, jax_core.NamedAxisEffect):
      effs.add(eff)
      continue
    if eff.name not in mesh.shape:
      effs.add(eff)
  return [], effs


def core_map_lowering_rule(ctx: mlir.LoweringRuleContext,
    *args,
    jaxpr,
    **kwargs
  ):
  del ctx, args, kwargs
  raise ValueError(
      "Attempted to lower core_map without discharging. This can happen if "
      "the core_map body does not modify any Refs or have other observable "
      f"side-effects.\n Jaxpr of the body: {jaxpr}")
mlir.register_lowering(core_map_p, core_map_lowering_rule)


class Mesh(Protocol):

  @property
  def backend(self) -> Backend:
    ...

  @property
  def shape(self) -> collections.OrderedDict[object, int]:
    ...


_core_map_mesh_rules: dict[type[Any], Callable[..., Any]] = {}


with_memory_space_constraint_p = jax_core.Primitive(
    'with_memory_space_constraint')

@with_memory_space_constraint_p.def_impl
def with_memory_space_constraint_impl(x, *, memory_space):
  del x, memory_space
  raise ValueError("Cannot eagerly run with_memory_space_constraint.")


@with_memory_space_constraint_p.def_abstract_eval
def with_memory_space_constraint_abstract_eval(x, *, memory_space):
  if not isinstance(x, jax_core.ShapedArray):
    raise NotImplementedError("with_memory_space_constraint only supports "
                              "arrays.")
  return ShapedArrayWithMemorySpace(
      x.shape, x.dtype, memory_space=memory_space
  )

def with_memory_space_constraint_lowering_rule(ctx, x, *, memory_space):
  del ctx, memory_space
  return [x]
mlir.register_lowering(
    with_memory_space_constraint_p, with_memory_space_constraint_lowering_rule
)


def default_mesh_discharge_rule(
    in_avals,
    out_avals,
    *args,
    mesh,
    compiler_params,
    jaxpr,
    debug,
    interpret,
    cost_estimate,
    name,
    memory_space=MemorySpace.ANY,
    metadata,
):
  """Discharges a ``core_map`` over a mesh to a ``pallas_call``."""
  del out_avals  # Unused.
  default_memory_space = memory_space

  def body(*args):
    # Due to aliasing, ``args`` contains aliased inputs and outputs so we
    # remove outputs.
    in_refs = args[:len(in_avals)]
    jax_core.eval_jaxpr(jaxpr, in_refs)

  assert len(jaxpr.outvars) == 0
  modified_idxs = sorted(
      eff.input_index
      for eff in jaxpr.effects
      if isinstance(eff, state_types.WriteEffect)
  )
  in_memory_spaces = [get_memory_space_aval(aval) for aval in in_avals]
  in_memory_spaces = [
      memory_space if m is None else m for m in in_memory_spaces
  ]
  args = [
      with_memory_space_constraint_p.bind(arg, memory_space=memory_space)
      if memory_space is not None and memory_space is not default_memory_space else arg
      for arg, memory_space in zip(args, in_memory_spaces)
  ]
  in_specs = [
      BlockSpec(memory_space=memory_space) for memory_space in in_memory_spaces
  ]
  out_specs = [in_specs[idx] for idx in modified_idxs]
  out_shapes = [_get_sds(in_avals[idx]) for idx in modified_idxs]
  from jax._src.pallas import pallas_call  # Avoid circular dependency.
  outs = pallas_call._pallas_call(
      body,
      name=name,
      out_shape=out_shapes,
      input_output_aliases={
          in_idx: out_idx for out_idx, in_idx in enumerate(modified_idxs)
      },
      grid_spec=GridSpec(
          grid=tuple(mesh.shape.items()),
          in_specs=in_specs,
          out_specs=out_specs,
      ),
      mesh=mesh,
      compiler_params=compiler_params,
      interpret=interpret,
      debug=debug,
      cost_estimate=cost_estimate,
      metadata=metadata,
  )(*args)
  # ``outs`` lacks the unmodified inputs. Add them back in.
  all_outs = [None] * len(args)
  for out_idx, in_idx in enumerate(modified_idxs):
    all_outs[in_idx] = outs[out_idx]
  return all_outs, ()


@state_discharge.register_discharge_rule(core_map_p)
def _core_map_discharge_rule(in_avals, out_avals, *args_flat, jaxpr, mesh, **kwargs):
  if type(mesh) not in _core_map_mesh_rules:
    raise NotImplementedError(f"Mesh type {type(mesh)} not supported.")
  return _core_map_mesh_rules[type(mesh)](
      in_avals, out_avals, *args_flat, jaxpr=jaxpr, mesh=mesh, **kwargs
  )


def _core_map_typecheck_rule(_, *in_atoms, jaxpr, mesh, **kwargs):
  del in_atoms
  with jax_core.extend_axis_env_nd(tuple(mesh.shape.items())):
    jax_core.check_jaxpr(jaxpr)
  interpret = kwargs.get('interpret', False)
  effs = set()
  if interpret:
    try:
      from jax._src.pallas.mosaic import interpret as mosaic_tpu_interpret  # Avoid circular dependency.
      if isinstance(interpret, mosaic_tpu_interpret.InterpretParams):
        effs = mosaic_tpu_interpret.get_interpret_effects()
    except ImportError:
      pass
  for eff in jaxpr.effects:
    if mesh.discharges_effect(eff):
      continue
    if not isinstance(eff, jax_core.NamedAxisEffect):
      effs.add(eff)
      continue
    if eff.name not in mesh.shape:
      effs.add(eff)
  return [], effs
jax_core.custom_typechecks[core_map_p] = _core_map_typecheck_rule


def lower_as_mlir(
    f,
    *args,
    dynamic_shapes=False,
    device=None,
    static_argnames=(),
    platforms=None,
    **kwargs,
) -> mlir.ir.Module:
  with pallas_export_experimental(dynamic_shapes):
    f = jax.jit(f, device=device, static_argnames=static_argnames)
    if platforms is None:
      platforms = ["tpu"]
    exported = export(f, platforms=platforms)(*args, **kwargs)
    stablehlo = exported.mlir_module()

  return stablehlo  # type: ignore[return-value]


_out_shape_to_aval_mapping: dict[
    type[Any], Callable[[Any], jax_core.AbstractValue]
] = {}
