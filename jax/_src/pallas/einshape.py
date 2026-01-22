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

"""Einshape primitive implementation.

Einshape (see https://github.com/google-deepmind/einshape) is a DSL for various
array transformation operations including reshape, squeeze, expand_dims, and
transpose, using an einsum-like notation.

The DSL consists of an LHS equation and an RHS equation, separated by `->`.
Each side assigns names to dimensions, e.g. `ij` would assign `i` and `j` to the
first and second dimensions of an array.

The DSL uses parentheses `()` to indicate grouping of dimensions:
- On the left-hand side (LHS), parentheses indicate that an existing dimension
  should be split into multiple dimensions.
- On the right-hand side (RHS), parentheses indicate that multiple dimensions
  should be merged into a single dimension.

Dimension reordering in the RHS string (relative to the LHS) specifies a
transpose.

Example equations:
- "n->n": Identity
- "ab->ba": Transposes the 0th and 1st dimensions.
- "nhwc->nchw": Transposes dimensions from (N, H, W, C) to (N, C, H, W).
- "(ab)c->abc": Splits the first dimension into two dimensions (a, b).
- "abc->(ab)c": Merges the first two dimensions (a, b).
- "a(bc)->(ba)c": Splits the second dimension into (b, c), transposes 0 and 1,
  then merges dimensions 0 and 1.

When used inside a Pallas kernel on TPU, `einshape` will attempt to perform a
"tile-preserving" transformation. This is a more efficient implementation that
avoids the overhead of general reshapes or transposes by logically reordering
the underlying TPU vector registers. This is possible if the transformation
preserves the data within the vector registers. If this is not possible,
`einshape` will fall back to a general implementation that will likely involve
vector register relayouts.

As an example, for the equation `a(bc)->bac`, the first step which involves
splitting a(bc)->abc is *not* tile preserving (it changes the sublane dimension
from a to b), but after the transpose to `bac` it is tile preserving. Therefore
the overall `einshape` operation is tile preserving.

Not currently supported:
- Expand dimensions: `a->1a`
- Squeeze dimensions: `a1->a`
"""

import collections
from collections.abc import Sequence
import dataclasses
import functools
import math
from typing import Literal, NamedTuple

from jax._src import api
from jax._src import core as jax_core
from jax._src import dispatch
from jax._src import typing as jax_typing
from jax._src import hijax
from jax._src.interpreters import mlir
from jax._src.lax import lax
from jax._src.pallas.mosaic import lowering as tpu_lowering
from jax._src.pallas.mosaic import tpu_info
import jax.numpy as jnp
import numpy as np


@dataclasses.dataclass(frozen=True)
class SplitDims:
  index: int
  sizes: tuple[int, ...]

  def transform_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
    return (*shape[: self.index], *self.sizes, *shape[self.index + 1 :])


@dataclasses.dataclass(frozen=True)
class MergeDims:
  index: int
  count: int

  def transform_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
    return (
        *shape[: self.index],
        math.prod(shape[self.index : self.index + self.count]),
        *shape[self.index + self.count :],
    )


@dataclasses.dataclass(frozen=True)
class Transpose:
  permutation: tuple[int, ...]

  def transform_shape(self, shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(shape[i] for i in self.permutation)


# TODO(sharadmv): unify this with other Pallas Transforms
Transform = SplitDims | MergeDims | Transpose


def _parse_side(s: str) -> list[list[str]]:
  """Parses one side of an einshape equation into groups of named dimensions.

  Groups are indicated by parentheses. Dimensions outside of parentheses are
  treated as groups of size 1.
  For example:
    "a(bc)d" -> [['a'], ['b', 'c'], ['d']]
    "(ab)c" -> [['a', 'b'], ['c']]

  Args:
    s: One side of an einshape equation string.

  Returns:
    A list of lists of characters, where each inner list represents a group of
    dimensions.
  """
  # Remove spaces
  s = s.replace(" ", "")
  groups = []
  i = 0
  while i < len(s):
    if s[i] == "(":
      # Start of a group
      j = s.find(")", i)
      if j == -1:
        raise ValueError(f"Unmatched parenthesis in {s!r}")
      group = list(s[i + 1 : j])
      groups.append(group)
      i = j + 1
    elif s[i] == ")":
      raise ValueError(f"Unmatched parenthesis in {s!r}")
    else:
      # distinct dimension
      groups.append([s[i]])
      i += 1
  return groups


def _parse_equation(equation: str) -> tuple[list[list[str]], list[list[str]]]:
  """Parses an einshape equation."""
  if equation.count("->") != 1:
    raise ValueError("Equation must contain exactly one '->'")
  lhs_str, rhs_str = equation.split("->")
  return _parse_side(lhs_str), _parse_side(rhs_str)


def get_einshape_transforms(
    equation: str,
    input_shape: tuple[int, ...],
    **sizes: int,
) -> list[Transform]:
  """Parses an einshape equation into a sequence of transforms.

  Args:
    equation: String of the form "ab(cd)->cabd".
    input_shape: The shape of the input array.
    **sizes: Integer sizes for dimensions that are split and cannot be inferred.

  Returns:
    A list of Split, Transpose, and Merge transforms.
  """
  lhs, rhs = _parse_equation(equation)

  # Validate LHS against input shape
  if len(lhs) != len(input_shape):
    raise ValueError(
        f"Equation LHS has {len(lhs)} groups but input has {len(input_shape)}"
        f" dims. LHS: {lhs}, Input shape: {input_shape}"
    )

  dim_sizes: dict[str, int] = {}

  # Populate known sizes from input
  for i, group in enumerate(lhs):
    shape_val = input_shape[i]
    if len(group) == 1:
      name = group[0]
      if name in dim_sizes and dim_sizes[name] != shape_val:
        raise ValueError(
            f"Inconsistent size for {name}: {dim_sizes[name]} vs {shape_val}"
        )
      dim_sizes[name] = shape_val
    else:
      # We have a merged dimension on LHS, need to split
      known_product = 1
      unknown_dims = []
      for name in group:
        if name in sizes:
          dim_sizes[name] = sizes[name]
          known_product *= sizes[name]
        elif name in dim_sizes:
          known_product *= dim_sizes[name]
        else:
          unknown_dims.append(name)

      if not unknown_dims:
        if known_product != shape_val:
          raise ValueError(
              f"Size mismatch for group {group}: expected {shape_val}, got"
              f" {known_product}"
          )
      elif len(unknown_dims) == 1:
        if shape_val % known_product != 0:
          raise ValueError(
              f"Cannot split size {shape_val} with known sizes {known_product}"
          )
        inferred_size = shape_val // known_product
        dim_sizes[unknown_dims[0]] = inferred_size
      else:
        raise ValueError(
            f"Ambiguous split for {group} with size {shape_val}. Unknowns:"
            f" {unknown_dims}. Provide sizes via kwargs."
        )

  # Check if all RHS dims are known
  flat_rhs = [name for group in rhs for name in group]
  for name in flat_rhs:
    if name not in dim_sizes:
      if name in sizes:
        dim_sizes[name] = sizes[name]
      else:
        raise ValueError(f"Unknown dimension {name} in RHS")

  ops: list[Transform] = []

  # 1. Decompose LHS
  current_idx = 0
  for group in lhs:
    if len(group) > 1:
      atomic_sizes = tuple(dim_sizes[name] for name in group)
      ops.append(SplitDims(current_idx, atomic_sizes))
      current_idx += len(group)
    else:
      current_idx += 1

  # 2. Transpose
  lhs_atomic_order = [name for group in lhs for name in group]
  rhs_atomic_order = [name for group in rhs for name in group]

  if set(lhs_atomic_order) != set(rhs_atomic_order):
    raise NotImplementedError(
        "Only reordering/splitting/merging supported (no broadcast yet)."
    )

  if lhs_atomic_order != rhs_atomic_order:
    perm = tuple(lhs_atomic_order.index(name) for name in rhs_atomic_order)
    ops.append(Transpose(perm))

  # 3. Compose RHS
  current_idx = 0
  for group in rhs:
    if len(group) > 1:
      ops.append(MergeDims(current_idx, len(group)))
      current_idx += 1
    else:
      current_idx += 1
  return ops


def _einshape(
    equation: str,
    value: jax_typing.Array,
    **sizes: int,
) -> jax_typing.Array:
  """Reshapes and transposes an array according to an einshape equation.

  Args:
    equation: String of the form "ab(cd)->cabd". Parentheses indicate grouping
      of dimensions. On the LHS, grouped dimensions are split. On the RHS,
      dimensions are merged.
    value: The array to reshape.
    **sizes: Integer sizes for dimensions that are split and cannot be inferred.

  Returns:
    The reshaped and transposed array.
  """
  transforms = get_einshape_transforms(equation, value.shape, **sizes)
  for transform in transforms:
    match transform:
      case SplitDims(_, _):
        new_shape = transform.transform_shape(value.shape)
        value = value.reshape(new_shape)
      case MergeDims(_, _):
        new_shape = transform.transform_shape(value.shape)
        value = value.reshape(new_shape)
      case Transpose(permutation):
        value = lax.transpose(value, permutation)
  return value


einshape_lo_p = jax_core.Primitive("einshape_lo")


def einshape_lo(
    equation: str, x: jax_typing.Array, assert_is_tile_preserving: bool, **sizes: int
) -> jax_typing.Array:
  return einshape_lo_p.bind(
      x,
      equation=equation,
      sizes=tuple(sizes.items()),
      assert_is_tile_preserving=assert_is_tile_preserving,
  )


@einshape_lo_p.def_abstract_eval
def _einshape_lo_abstract_eval(
    x_aval: jax_core.ShapedArray,
    *,
    equation: str,
    sizes: tuple[tuple[str, int], ...],
    assert_is_tile_preserving: bool,
):
  del assert_is_tile_preserving
  out_sds = api.eval_shape(
      functools.partial(_einshape, equation, **dict(sizes)), x_aval  # type: ignore
  )
  return x_aval.update(shape=out_sds.shape, dtype=out_sds.dtype)


def _einshape_lo_lowering(
    ctx: mlir.LoweringRuleContext,
    x,
    *,
    equation: str,
    sizes: tuple[tuple[str, int], ...],
    assert_is_tile_preserving: bool,
):
  del assert_is_tile_preserving

  def f(x):
    return _einshape(equation, x, **dict(sizes))

  return mlir.lower_fun(f, multiple_results=False)(ctx, x)


mlir.register_lowering(einshape_lo_p, _einshape_lo_lowering)
dispatch.simple_impl(einshape_lo_p)


class Einshape(hijax.VJPHiPrimitive):
  """Einshape primitive."""

  def __init__(
      self,
      x_aval: jax_core.ShapedArray,
      *,
      equation: str,
      assert_is_tile_preserving: bool,
      sizes: dict[str, int],
  ):
    self.in_avals = (x_aval,)
    out_type = api.eval_shape(
        functools.partial(_einshape, equation, **sizes), x_aval  # type: ignore
    )
    self.out_aval = hijax.ShapedArray(out_type.shape, out_type.dtype)
    self.equation = equation
    self.sizes = sizes
    self.assert_is_tile_preserving = assert_is_tile_preserving
    self.params = dict(
        x_aval=x_aval,
        equation=equation,
        sizes=sizes,  # type: ignore
        assert_is_tile_preserving=assert_is_tile_preserving,
    )
    super().__init__()

  def expand(self, x: jax_typing.Array) -> jax_typing.Array:
    return einshape_lo(
        self.equation,
        x,
        assert_is_tile_preserving=self.assert_is_tile_preserving,
        **self.sizes,
    )


def einshape(
    equation: str,
    x: jax_typing.Array,
    assert_is_tile_preserving: bool = False,
    **sizes: int,
) -> jax_typing.Array:
  """Reshapes and transposes an array according to an einshape equation.

  Args:
    equation: A string defining the transformation, e.g., "ab(cd)->cabd". -
      Names (e.g., 'a', 'b') represent dimensions. - Parentheses on the LHS,
      like `(cd)`, indicate a dimension that will be split into dimensions `c`
      and `d`. - Parentheses on the RHS, like `(ab)`, indicate dimensions `a`
      and `b` that will be merged.
    x: The input jax_typing.Array to transform.
    assert_is_tile_preserving: If True, assert that the transformation is tile
      preserving. Note that this check only applies inside of Pallas kernels.
    **sizes: Dimension sizes that cannot be inferred from the input shape.
      Required when splitting dimensions unless all but one sub-dimension size
      is known.

  Returns:
    The transformed jax_typing.Array.

  Examples:
    >>> import jax.numpy as jnp
    >>> x = jnp.zeros((10, 20))
    >>> # Split the second dimension (20) into (4, 5)
    >>> y = einshape("a(bc)->abc", x, b=4)
    >>> y.shape
    (10, 4, 5)

    >>> # Transpose and merge the first two dimensions.
    >>> z = einshape("abc->(ba)c", y)
    >>> z.shape
    (40, 5)
  """
  return Einshape(
      jax_core.typeof(x),
      equation=equation,
      sizes=sizes,
      assert_is_tile_preserving=assert_is_tile_preserving,
  )(x)


def _default_einshape_kernel(equation: str, x: jax_typing.Array, **sizes: int):
  return _einshape(equation, x, **sizes)


class Factor(NamedTuple):
  size: int
  kind: Literal["outer", "sublane", "lane"]


def _array_to_2d_tile_array(
    x: jax_typing.Array, tiling: tuple[int, ...]
) -> np.ndarray:
  t1, t2 = tiling[-2:]
  tiled_shape = tuple(x.shape[i] // tiling[i] for i in range(len(x.shape)))
  # Allocate an empty object array to ensure Numpy doesn't coerce JAX tracers
  tiles = np.empty(tiled_shape, dtype=object)
  for idx in np.ndindex(*tiled_shape):
    *leading, i1, i2 = idx
    slices = tuple(leading) + (
        slice(i1 * t1, (i1 + 1) * t1),
        slice(i2 * t2, (i2 + 1) * t2),
    )
    # Standard Integer indexing inherently drops the outer dims -> returns strict 2D array
    tiles[idx] = x[slices]
  return tiles


def _2d_tile_array_to_array(tiles: np.ndarray) -> jax_typing.Array:
  raw_arrays = np.empty(tiles.shape, dtype=object)
  for idx in np.ndindex(*tiles.shape):
    raw_arrays[idx] = tiles[idx]
  return jnp.block(raw_arrays.tolist())


def _consolidate(factors: list[Factor]) -> list[Factor]:
  """Merges contiguous 'outer' factors to allow valid arbitrary outer-dimension reshapes."""
  res: list[Factor] = []
  for f in factors:
    if f.kind == "outer" and res and res[-1].kind == "outer":
      res[-1] = Factor(res[-1].size * f.size, "outer")
    else:
      res.append(f)
  return res


def _init_dims(shape: tuple[int, ...], t1: int, t2: int) -> list[list[Factor]]:
  dims: list[list[Factor]] = []
  for i, s in enumerate(shape):
    if i == len(shape) - 2:
      kind, t_size = "sublane", t1
    elif i == len(shape) - 1:
      kind, t_size = "lane", t2
    else:
      kind, t_size = "outer", 1

    current_dim = []
    assert s % t_size == 0
    if s // t_size > 1:
      current_dim.append(Factor(s // t_size, "outer"))
    if t_size > 1 or kind != "outer":
      current_dim.append(Factor(t_size, kind))  # type: ignore
    dims.append(_consolidate(current_dim))
  return dims


def _apply_split(
    factors: list[Factor], targets: tuple[int, ...]
) -> list[list[Factor]] | None:
  factors = _consolidate(factors)
  queue = collections.deque(factors)
  result = []

  for i, needed in enumerate(targets):
    new_dim = []
    current_size = 1

    # Consume factors iteratively until the required shape volume is met
    while current_size < needed:
      if not queue:
        return None
      b = queue.popleft()

      # Case A: Perfect match or consume smaller outer block
      if needed % (current_size * b.size) == 0:
        new_dim.append(b)
        current_size *= b.size

      # Case B: Split a larger block (only allowed over logical outer limits)
      elif (current_size * b.size) % needed == 0:
        if b.kind != "outer":
          return None  # Illegal splitting of hardware tile limit
        take = needed // current_size
        new_dim.append(Factor(take, "outer"))
        queue.appendleft(Factor(b.size // take, "outer"))
        current_size *= take
      else:
        return None

    # Sweep any trailing physical size-1 markers exactly into the right-most split dimension
    if i == len(targets) - 1:
      while queue and queue[0].size == 1:
        new_dim.append(queue.popleft())

    result.append(_consolidate(new_dim))

  if queue:
    return None
  return result


def _tile_preserving_einshape_kernel(
    equation: str, x: jax_typing.Array, **size_vars: int
):
  tiling = tpu_info.infer_tiling(jax_core.typeof(x))
  assert tiling is not None
  t1, t2 = tiling[-2:]

  assert isinstance(t1, int)
  assert isinstance(t2, int)
  dims = _init_dims(x.shape, t1, t2)
  tiles = _array_to_2d_tile_array(x, tiling)  # type: ignore
  transforms = get_einshape_transforms(equation, x.shape, **size_vars)

  def get_outer_shape(dims_list: list[list[Factor]]) -> tuple[int, ...]:
    return tuple(
        math.prod([f.size for f in d if f.kind == "outer"]) for d in dims_list
    )

  for t in transforms:
    match t:
      case Transpose(permutation):
        tiles = np.transpose(tiles, permutation)
        dims = [dims[i] for i in permutation]
      case SplitDims(index, sizes):
        new_dims = _apply_split(dims[index], sizes)
        assert (
            new_dims is not None
        ), "Tile preserving check passed but split failed."
        dims = dims[:index] + new_dims + dims[index + 1 :]
        tiles = tiles.reshape(get_outer_shape(dims))
      case MergeDims(index, count):
        merged = [b for d in dims[index : index + count] for b in d]
        dims = dims[:index] + [_consolidate(merged)] + dims[index + count :]
        tiles = tiles.reshape(get_outer_shape(dims))

  return _2d_tile_array_to_array(tiles)


def _is_tile_preserving(
    shape: tuple[int, ...],
    transforms: Sequence[Transform],
    tiling: tuple[int, int] | None = None,
) -> bool:
  if not tiling or len(shape) < 2:
    return False

  t1, t2 = tiling
  if shape[-2] % t1 != 0 or shape[-1] % t2 != 0:
    return False

  dims = _init_dims(shape, t1, t2)

  for t in transforms:
    match t:
      case SplitDims(index, sizes):
        if (new_dims := _apply_split(dims[index], sizes)) is None:
          return False
        dims[index : index + 1] = new_dims
      case MergeDims(index, count):
        merged = [b for d in dims[index : index + count] for b in d]
        dims[index : index + count] = [_consolidate(merged)]
      case Transpose(permutation):
        dims = [dims[i] for i in permutation]

  if len(dims) < 2:
    return False

  # Check that the last two dimensions are tiled along (sublane, lane).
  y_dim = dims[-2]
  if not y_dim or y_dim[-1] != Factor(t1, "sublane"):
    return False

  x_dim = dims[-1]
  if not x_dim or x_dim[-1] != Factor(t2, "lane"):
    return False

  return True


def _einshape_kernel(
    equation: str,
    x: jax_typing.Array,
    assert_is_tile_preserving: bool,
    **size_vars: int,
):
  transforms = get_einshape_transforms(equation, x.shape, **dict(size_vars))
  if len(transforms) <= 1:
    return _default_einshape_kernel(equation, x, **size_vars)
  tiling = tpu_info.infer_tiling(jax_core.ShapedArray(x.shape, x.dtype))
  if tiling is not None and _is_tile_preserving(
      x.shape, transforms, tiling[-2:]  # type: ignore
  ):
    return _tile_preserving_einshape_kernel(equation, x, **size_vars)
  elif assert_is_tile_preserving:
    raise ValueError(
        "Tile preserving check failed for einshape kernel with equation:"
        f" {equation} and shape {x.shape} and tiling {tiling}."
    )
  return _default_einshape_kernel(equation, x, **size_vars)


@tpu_lowering.register_lowering_rule(einshape_lo_p)
def _einshape_lo_lowering_rule(
    ctx: tpu_lowering.LoweringRuleContext,
    x,
    *,
    equation: str,
    sizes: tuple[tuple[str, int], ...],
    assert_is_tile_preserving: bool,
):
  return tpu_lowering.lower_fun(
      lambda x: _einshape_kernel(
          equation,
          x,
          assert_is_tile_preserving=assert_is_tile_preserving,
          **dict(sizes),
      ),
      multiple_results=False,
  )(ctx, x)
