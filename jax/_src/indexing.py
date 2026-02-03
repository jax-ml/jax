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

import dataclasses

from jax._src import core
from jax._src import tree_util
from jax._src.typing import Array


@tree_util.register_pytree_node_class
@dataclasses.dataclass
class Slice:
  """A slice with a start index and a size.

  Both start index and size can either be static, i.e. known at tracing
  and compilation time, or dynamic.
  """

  start: int | Array
  size: int | Array
  stride: int = 1

  def __post_init__(self):
    if self.stride < 0:
      raise ValueError("`stride` must be >= 0.")

  @property
  def is_dynamic_start(self):
    return not core.is_dim(self.start)

  @property
  def is_dynamic_size(self):
    return not core.is_dim(self.size)

  def tree_flatten(self):
    # If `start` is statically known, we treat it as static information
    xs = ()
    data = ()
    xs += (self.start,) if self.is_dynamic_start else (None,)
    data += (None,) if self.is_dynamic_start else (self.start,)
    xs += (self.size,) if self.is_dynamic_size else (None,)
    data += (None,) if self.is_dynamic_size else (self.size,)
    data += (self.stride,)
    return xs, data

  @classmethod
  def tree_unflatten(cls, aux_data, children) -> Slice:
    start, size = (
        a if a is not None else b for a, b in zip(children, aux_data[:2])
    )
    return cls(start, size, aux_data[2])

  @classmethod
  def from_slice(cls, slc: slice, size: int) -> Slice:
    start, step, size = core.canonicalize_slice(slc, size)
    if step < 1:
      raise ValueError(f"slice must have a step >= 1 (found: {step})")
    return cls(start, size, step)


def dslice(
    start: int | Array | None,
    size: int | Array | None = None,
    stride: int | None = None,
) -> slice | Slice:
  """Constructs a ``Slice`` from a start index and a size.

  The semantics of ``dslice`` mirror those of the builtin ``slice`` type:

  * ``dslice(None)`` is ``:``
  * ``dslice(j)`` is ``:j``
  * ``dslice(i, j)`` is ``i:i+j``
  * ``dslice(i, j, stride)`` is ``i:i+j:stride``

  Examples:

    >>> x = jax.numpy.arange(10)
    >>> i = 4
    >>> x[i: i + 2]  # standard indexing requires i to be static
    Array([4, 5], dtype=int32)
    >>> x[jax.ds(i, 2)]  # equivalent which allows i to be dynamic
    Array([4, 5], dtype=int32)

    Here is an explicit example of slicing with a dynamic start index:

    >>> @jax.jit(static_argnames='size')
    ... def f(x, i, size):  # example of when `
    ...   return x[jax.ds(i, size)]
    ...
    >>> f(x, i, 2)
    Array([4, 5], dtype=int32)
  """
  if start is None:
    return slice(None)
  if stride is None:
    stride = 1
  if not isinstance(stride, int):
    raise ValueError("Non-static stride in `dslice`")
  if size is None:
    if not isinstance(start, int):
      raise ValueError("Non-static `dslice`")
    return Slice(0, start, stride)
  return Slice(start, size, stride)


ds = dslice  # Handy alias.
