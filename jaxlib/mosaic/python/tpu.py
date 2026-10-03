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

"""Python bindings for the MLIR TPU dialect."""

# ruff: noqa: F401
# ruff: noqa: F403

from ._tpu_enum_gen import *
from . import _tpu_ops_gen
from ._tpu_ops_gen import *
from ._tpu_ops_gen import _Dialect, VectorLoadOp, VectorStoreOp
from jaxlib.mlir._mlir_libs._tpu_ext import *
try:
  from jaxlib.mlir.dialects._ods_common import _cext
except ImportError:
  from mlir.dialects._ods_common import _cext


_cext.globals.append_dialect_search_prefix("jax.jaxlib.mosaic.python")


@_cext.register_operation(_Dialect, replace=True)
class TraceOp(_tpu_ops_gen.TraceOp):  # noqa: F405
  """An extension to the automatically generated TraceOp bindings."""

  def __init__(self, results, message, level, *, loc=None, ip=None):
    super().__init__(results, message, level, loc=loc, ip=ip)
    self.regions[0].blocks.append(*[])  # Append the block.

  @property
  def body(self):
    return self.regions[0].blocks[0]


@_cext.register_operation(_Dialect, replace=True)
class RegionOp(_tpu_ops_gen.RegionOp):  # noqa: F405
  """An extension to the automatically generated RegionOp bindings."""

  def __init__(self, results, *, loc=None, ip=None):
    super().__init__(results, loc=loc, ip=ip)
    self.regions[0].blocks.append()  # Append the block.

  @property
  def body(self):
    return self.regions[0].blocks[0]


def vector_load(
    result,
    base,
    indices,
    *,
    strides=None,
    mask=None,
    loc=None,
    ip=None,
):
  if strides is None:
    strides = []
  return VectorLoadOp(
      result, base, indices, strides, mask=mask, loc=loc, ip=ip
  ).result


def vector_store(
    value_to_store,
    base,
    indices,
    *,
    strides=None,
    add=False,
    mask=None,
    loc=None,
    ip=None,
):
  if strides is None:
    strides = []
  return VectorStoreOp(
      value_to_store, base, indices, strides, mask=mask, add=add, loc=loc, ip=ip
  )


def reinterpret_cast(
    result,
    input,
    dynamic_sizes=None,
    *,
    dynamic_offset=None,
    dynamic_strides=None,
    loc=None,
    ip=None,
):
  if dynamic_sizes is None:
    dynamic_sizes = []
  return _tpu_ops_gen.ReinterpretCastOp(
      result,
      input,
      dynamic_offset=dynamic_offset,
      dynamic_sizes=dynamic_sizes,
      dynamic_strides=dynamic_strides,
      loc=loc,
      ip=ip,
  ).result


def shared_memref_slice(
    result,
    source,
    *,
    loc=None,
    ip=None,
):
  """Slices a VMEM_SHARED memref into a per-subcore VMEM memref.

  Extracts the executing vector subcore's local VMEM slice from a shared VMEM
  memref along the final dimension.

  Requirements:
    - The source memref must reside in `#tpu.memory_space<vmem_shared>`.
    - Slicing occurs strictly along the last dimension (`dim[-1]`).
    - The last dimension of `source` must adhere to the following constraints:
      - Non-packed layouts (e.g. `i32`, `f32`, or unpacked `bf16`):
        `source.shape[-1] == num_subcores * spmem_stripe_size_words * (
            word_size // itemsize)`
      - Packed layouts (e.g. `bf16` with 2 elements packed per 32-bit word):
          `source.shape[-1] == num_subcores * spmem_stripe_size_words`
    - `result` must have memory space `#tpu.memory_space<vmem>` and match
      `source` in rank, element type, and all leading dimensions, with its
      last dimension size equal to `source.shape[-1] // num_subcores`.

  Examples of valid input layouts in VMEM_SHARED, assuming 16 subcores, 4-byte
    words, and 8-word stripes:
    - `8x128xi32, #tpu.tiled<(8,128),[1,1]>`
    - `8x256xbf16, #tpu.tiled<(16),[16,1]>` (no packing, 16 elements/subcore)
    - `8x128xbf16, #tpu.tiled<(8,128)(2,1),[1,1]>` (2 packed elements/word)

  Args:
    result: The resulting subcore-local VMEM memref type or value.
    source: The input VMEM_SHARED memref value to slice.
    loc: Optional MLIR source location.
    ip: Optional insertion point.

  Returns:
    The sliced subcore-local VMEM memref.
  """
  return _tpu_ops_gen.SharedMemRefSliceOp(
      result, source, loc=loc, ip=ip
  ).result
