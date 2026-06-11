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

"""ExtendedDType support for bcomplex32 (complex<bfloat16>).

bcomplex32 is represented physically as a bfloat16 array with an extra
trailing dimension of size 2: shape (..., N) -> physical shape (..., N, 2).
The trailing dim stores [real_part, imaginary_part].

At the MLIR level, bcomplex32 arrays appear as `tensor<...x2xbf16>`,
which is fully compatible with StableHLO/XLA.
"""

from __future__ import annotations


import numpy as np

from jax._src import core
from jax._src import dtypes
from jax._src.interpreters import mlir
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo

_bfloat16_dtype = dtypes._bfloat16_dtype
_bcomplex32_edtype = dtypes.bcomplex32_edtype


# ============================================================
# BComplex32Rules -- required by ExtendedDType
# ============================================================


class BComplex32Rules:
  """Rules for bcomplex32 ExtendedDType."""

  # Allow convert_element_type to/from the physical bf16 representation.
  allow_conversion: bool = True

  @staticmethod
  def physical_element_aval(dtype):
    """Physical element: a pair of bf16 values -> ShapedArray((2,), bf16)."""
    return core.ShapedArray((2,), _bfloat16_dtype)

  @staticmethod
  def tangent_dtype(dtype):
    """Tangent dtype for bcomplex32 is bcomplex32 itself (complex tangents)."""
    return _bcomplex32_edtype

  @staticmethod
  def result_handler(sticky_device, aval):
    """Create a result handler for bcomplex32 arrays."""
    from jax._src.interpreters import pxla
    from jax._src.lax.lax_bcomplex32_array import BComplex32Array

    phys_aval = core.physical_aval(aval)
    phys_handler = pxla.local_result_handlers[core.ShapedArray](
      sticky_device, phys_aval
    )

    def handler(buf):
      buf.aval = core.ShapedArray(buf.shape, buf.dtype)
      return BComplex32Array(aval, buf)

    return phys_handler.wrap(handler)

  @staticmethod
  def global_sharded_result_handler(aval, out_sharding, committed):
    """Create a global sharded result handler for bcomplex32 arrays."""
    from jax._src.interpreters import pxla
    from jax._src.lax.lax_bcomplex32_array import BComplex32Array
    from jax._src.sharding_impls import physical_sharding

    phys_aval = core.physical_aval(aval)
    phys_handler_maker = pxla.global_result_handlers[core.ShapedArray]
    phys_sharding = physical_sharding(aval, out_sharding)
    phys_handler = phys_handler_maker(phys_aval, phys_sharding, committed)

    def handler(phys_array):
      return BComplex32Array(aval, phys_array)

    return phys_handler.wrap(handler)

  @staticmethod
  def make_sharded_array(aval, sharding, arrays, committed):
    """Create a sharded bcomplex32 array from physical arrays."""
    from jax._src.interpreters import pxla
    from jax._src.lax.lax_bcomplex32_array import BComplex32Array
    from jax._src.sharding_impls import physical_sharding

    phys_aval = core.physical_aval(aval)
    phys_handler_maker = pxla.global_result_handlers[core.ShapedArray]
    phys_sharding = physical_sharding(aval, sharding)
    phys_handler = phys_handler_maker(phys_aval, phys_sharding, committed)
    phys_result = phys_handler(arrays)
    return BComplex32Array(aval, phys_result)

  @staticmethod
  def device_put_sharded(vals, aval, sharding, devices):
    """Handle device_put_sharded for bcomplex32."""
    from jax._src import api
    from jax._src.lax import lax as lax_module

    physical_buffers = [
      lax_module.from_edtype_p.bind(v, dtype=_bfloat16_dtype) for v in vals
    ]
    physical_result = api.device_put_sharded(physical_buffers, list(devices))
    return lax_module.to_edtype_p.bind(
      physical_result, edtype=_bcomplex32_edtype
    )

  @staticmethod
  def device_put_replicated(val, aval, sharding, devices):
    """Handle device_put_replicated for bcomplex32."""
    from jax._src.interpreters import pxla
    from jax._src.lax import lax as lax_module
    from jax._src.sharding_impls import physical_sharding

    physical_buf = lax_module.from_edtype_p.bind(val, dtype=_bfloat16_dtype)
    phys_aval = core.physical_aval(aval)
    phys_sharding = physical_sharding(aval, sharding)
    physical_result = pxla.batched_device_put(
      phys_aval, phys_sharding, [physical_buf] * len(devices), devices
    )
    return lax_module.to_edtype_p.bind(
      physical_result, edtype=_bcomplex32_edtype
    )

  @staticmethod
  def full(shape, fill_value, dtype):
    """Create a bcomplex32 array filled with `fill_value`."""
    from jax._src.lax import lax as lax_module

    # Split fill_value into real/imag parts. Must avoid np.asarray because
    # fill_value may be a Tracer (e.g. under jit), and np.asarray would raise
    # TracerArrayConversionError.
    if isinstance(fill_value, complex):
      re_val, im_val = fill_value.real, fill_value.imag
    elif hasattr(fill_value, "dtype") and dtypes.issubdtype(
      fill_value.dtype, np.complexfloating
    ):
      re_val = lax_module.real(fill_value)
      im_val = lax_module.imag(fill_value)
    elif hasattr(fill_value, "imag") and hasattr(fill_value, "real") and not isinstance(
      fill_value, (int, float, np.integer, np.floating)
    ):
      re_val, im_val = fill_value.real, fill_value.imag
    else:
      re_val, im_val = fill_value, 0

    # convert_element_type + broadcast are Tracer-safe (np.asarray is not).
    re_bf16 = lax_module.convert_element_type(re_val, _bfloat16_dtype)
    im_bf16 = lax_module.convert_element_type(im_val, _bfloat16_dtype)
    re_arr = lax_module.broadcast(re_bf16, shape)
    im_arr = lax_module.broadcast(im_bf16, shape)
    physical = lax_module.concatenate(
      [re_arr[..., np.newaxis], im_arr[..., np.newaxis]], dimension=len(shape)
    )
    return lax_module.to_edtype_p.bind(physical, edtype=_bcomplex32_edtype)

  @staticmethod
  def physical_const(val):
    """Return the physical representation of a bcomplex32 constant."""
    from jax._src.lax.lax_bcomplex32_array import BComplex32Array

    if isinstance(val, BComplex32Array):
      return val._base_array
    return val

  @staticmethod
  def zero(dtype):
    """Zero value for bcomplex32: a bf16 scalar zero."""
    return np.zeros((), _bfloat16_dtype)

  @staticmethod
  def add(dtype, x, y):
    """Add two bcomplex32 arrays (used by AD gradient accumulation)."""
    from jax._src.lax import lax as lax_module

    return lax_module.add(x, y)


def ml_dtypes_bfloat16():
  """Return the bfloat16 numpy dtype, importing ml_dtypes lazily."""
  import ml_dtypes

  return ml_dtypes.bfloat16


# Set the rules on the ExtendedDType singleton
_bcomplex32_edtype._rules = BComplex32Rules


# ============================================================
# Helper utilities
# ============================================================


def _is_bcomplex32_edtype(dtype):
  return dtype is _bcomplex32_edtype


def _physical_slice_real(ctx, x_physical, logical_shape):
  """Extract real part from physical bf16 tensor with trailing dim 2.

  x_physical: MLIR value of type tensor<...x2xbf16>
  Returns: MLIR value of type tensor<...xbf16> (the real part)
  """
  rank = len(logical_shape) + 1  # +1 for the trailing dim

  # Build start_indices and limit_indices for the slice
  start_indices = [0] * rank
  start_indices[-1] = 0  # Take element 0 (real)
  limit_indices = list(logical_shape) + [1]  # Slice to get shape (*, 1)
  strides = [1] * rank

  sliced = hlo.slice(
    x_physical,
    start_indices=start_indices,
    limit_indices=limit_indices,
    strides=strides,
  )
  # Remove the trailing singleton dimension: reshape from (*, 1) to (*)
  result_type = ir.RankedTensorType.get(logical_shape, ir.BF16Type.get())
  return hlo.reshape(result_type, sliced)


def _physical_slice_imag(ctx, x_physical, logical_shape):
  """Extract imaginary part from physical bf16 tensor with trailing dim 2."""
  rank = len(logical_shape) + 1

  start_indices = [0] * rank
  start_indices[-1] = 1  # Take element 1 (imag)
  limit_indices = list(logical_shape) + [2]
  strides = [1] * rank

  sliced = hlo.slice(
    x_physical,
    start_indices=start_indices,
    limit_indices=limit_indices,
    strides=strides,
  )
  # Remove the trailing singleton dimension: reshape from (*, 1) to (*)
  result_type = ir.RankedTensorType.get(logical_shape, ir.BF16Type.get())
  return hlo.reshape(result_type, sliced)


def _physical_stack_real_imag(ctx, re_physical, im_physical, logical_shape):
  """Stack real and imag bf16 tensors into a physical bcomplex32 tensor.

  re_physical, im_physical: MLIR values of type tensor<...xbf16>
  Returns: MLIR value of type tensor<...x2xbf16>
  """
  (aval_out,) = ctx.avals_out
  # Add trailing dimension to each
  re_shape = list(logical_shape) + [1]
  im_shape = list(logical_shape) + [1]

  re_type = ir.RankedTensorType.get(re_shape, ir.BF16Type.get())
  im_type = ir.RankedTensorType.get(im_shape, ir.BF16Type.get())

  re_expanded = hlo.reshape(re_type, re_physical)
  im_expanded = hlo.reshape(im_type, im_physical)

  # Concatenate along the last dimension
  return hlo.concatenate(
    [re_expanded, im_expanded], dimension=len(logical_shape)
  )


# ============================================================
# Custom lowerings for real_p, imag_p, complex_p with bcomplex32
# ============================================================


def _real_lower_bcomplex32(ctx, x, **kw):
  """Custom lowering for real_p when input is bcomplex32 ExtendedDType.

  The MLIR input x is the physical bf16 tensor with trailing dim 2.
  We need to extract element 0 (real part).
  """
  (aval_in,) = ctx.avals_in
  # The physical aval of the input has shape (*logical, 2) and dtype bf16.
  # But ctx.avals_in has the LOGICAL aval (shape, bcomplex32_edtype).
  logical_shape = aval_in.shape
  return [_physical_slice_real(ctx, x, logical_shape)]


def _imag_lower_bcomplex32(ctx, x, **kw):
  """Custom lowering for imag_p when input is bcomplex32 ExtendedDType."""
  (aval_in,) = ctx.avals_in
  logical_shape = aval_in.shape
  return [_physical_slice_imag(ctx, x, logical_shape)]


def _complex_lower_bcomplex32(ctx, re, im, **kw):
  """Custom lowering for complex_p when constructing bcomplex32.

  re, im are bf16 MLIR tensors. We need to stack them into (*shape, 2).
  """
  (aval_out,) = ctx.avals_out
  logical_shape = aval_out.shape
  return [_physical_stack_real_imag(ctx, re, im, logical_shape)]


def _neg_lower_bcomplex32(ctx, x, **kw):
  """Custom lowering for neg_p on bcomplex32.

  Negate both real and imag in the physical bf16 representation.
  """
  (aval_in,) = ctx.avals_in
  logical_shape = aval_in.shape
  re = _physical_slice_real(ctx, x, logical_shape)
  im = _physical_slice_imag(ctx, x, logical_shape)
  neg_re = hlo.negate(re)
  neg_im = hlo.negate(im)
  return [_physical_stack_real_imag(ctx, neg_re, neg_im, logical_shape)]


def _conj_lower_bcomplex32(ctx, x, **kw):
  """Custom lowering for conj_p on bcomplex32: conj(re + im*j) = re - im*j."""
  (aval_in,) = ctx.avals_in
  logical_shape = aval_in.shape
  re = _physical_slice_real(ctx, x, logical_shape)
  im = _physical_slice_imag(ctx, x, logical_shape)
  neg_im = hlo.negate(im)
  return [_physical_stack_real_imag(ctx, re, neg_im, logical_shape)]


def _add_lower_bcomplex32(ctx, x, y, **kw):
  """Custom lowering for add_p on bcomplex32."""
  # Element-wise addition on physical bf16 pairs works directly.
  # Since both x and y are physical (*, 2) bf16 tensors, we can just add.
  # The sharding/broadcast is handled by the generic lowering path.
  return [hlo.add(x, y)]


def _sub_lower_bcomplex32(ctx, x, y, **kw):
  """Custom lowering for sub_p on bcomplex32."""
  return [hlo.subtract(x, y)]


def _mul_lower_bcomplex32(ctx, x, y, **kw):
  """Custom lowering for mul_p on bcomplex32.

  Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
  x = [..., 0: re_x, ..., 1: im_x]
  y = [..., 0: re_y, ..., 1: im_y]
  """
  aval_in_x, aval_in_y = ctx.avals_in
  (aval_out,) = ctx.avals_out
  logical_shape = aval_out.shape

  re_x = _physical_slice_real(ctx, x, logical_shape)
  im_x = _physical_slice_imag(ctx, x, logical_shape)
  re_y = _physical_slice_real(ctx, y, logical_shape)
  im_y = _physical_slice_imag(ctx, y, logical_shape)

  # real = re_x * re_y - im_x * im_y
  re_part = hlo.subtract(hlo.multiply(re_x, re_y), hlo.multiply(im_x, im_y))
  # imag = re_x * im_y + im_x * re_y
  im_part = hlo.add(hlo.multiply(re_x, im_y), hlo.multiply(im_x, re_y))

  return [_physical_stack_real_imag(ctx, re_part, im_part, logical_shape)]


def _abs_lower_bcomplex32(ctx, x, **kw):
  """Custom lowering for abs_p on bcomplex32.

  |a + bi| = sqrt(a^2 + b^2)  computed in bf16.
  """
  (aval_in,) = ctx.avals_in
  logical_shape = aval_in.shape
  re = _physical_slice_real(ctx, x, logical_shape)
  im = _physical_slice_imag(ctx, x, logical_shape)
  re_sq = hlo.multiply(re, re)
  im_sq = hlo.multiply(im, im)
  sum_sq = hlo.add(re_sq, im_sq)
  return [hlo.sqrt(sum_sq)]


def _div_lower_bcomplex32(ctx, x, y, **kw):
  """Custom lowering for div_p on bcomplex32.

  Smith's method: (a+bi)/(c+di) = ((ac+bd) + (bc-ad)i) / (c^2 + d^2)
  """
  aval_in_x, aval_in_y = ctx.avals_in
  (aval_out,) = ctx.avals_out
  logical_shape = aval_out.shape

  re_x = _physical_slice_real(ctx, x, logical_shape)
  im_x = _physical_slice_imag(ctx, x, logical_shape)
  re_y = _physical_slice_real(ctx, y, logical_shape)
  im_y = _physical_slice_imag(ctx, y, logical_shape)

  # d = c^2 + d^2
  d = hlo.add(hlo.multiply(re_y, re_y), hlo.multiply(im_y, im_y))
  # real = (a*c + b*d) / d
  re_part = hlo.divide(
    hlo.add(hlo.multiply(re_x, re_y), hlo.multiply(im_x, im_y)), d
  )
  # imag = (b*c - a*d) / d
  im_part = hlo.divide(
    hlo.subtract(hlo.multiply(im_x, re_y), hlo.multiply(re_x, im_y)), d
  )

  return [_physical_stack_real_imag(ctx, re_part, im_part, logical_shape)]


def _dot_general_lower_bcomplex32(
  ctx,
  lhs,
  rhs,
  *,
  dimension_numbers,
  precision,
  preferred_element_type,
  out_sharding,
  **kw,
):
  """Custom lowering for dot_general_p on bcomplex32.

  Decompose complex matmul into 4 real bf16 matmuls:
    C_re = X_re @ Y_re - X_im @ Y_im
    C_im = X_re @ Y_im + X_im @ Y_re
  Each sub-matmul operates on pure bf16 arrays, hitting GPU tensor cores.
  """
  from jax._src.lax.lax import precision_attr

  aval_lhs, aval_rhs = ctx.avals_in
  (aval_out,) = ctx.avals_out
  logical_shape = aval_out.shape

  # Extract real and imag parts from both lhs and rhs
  re_lhs = _physical_slice_real(ctx, lhs, aval_lhs.shape)
  im_lhs = _physical_slice_imag(ctx, lhs, aval_lhs.shape)
  re_rhs = _physical_slice_real(ctx, rhs, aval_rhs.shape)
  im_rhs = _physical_slice_imag(ctx, rhs, aval_rhs.shape)

  # Build the dot dimension numbers for real (bf16) arrays.
  # The logical shape matches, so we reuse the same dimension_numbers.
  (lhs_contracting, rhs_contracting), (lhs_batch, rhs_batch) = dimension_numbers
  dot_dnums = hlo.DotDimensionNumbers.get(
    lhs_batching_dimensions=list(lhs_batch),
    rhs_batching_dimensions=list(rhs_batch),
    lhs_contracting_dimensions=list(lhs_contracting),
    rhs_contracting_dimensions=list(rhs_contracting),
  )

  # Compute output shape for the real-valued dot products.
  # The output logical shape may differ from the real-part shape by
  # having no trailing dim-2, so we derive the real output shape.
  re_out_shape = logical_shape  # real/imag parts have this shape

  # bf16 accumulation type
  bf16_type = ir.BF16Type.get()
  acc_type = ir.RankedTensorType.get(re_out_shape, bf16_type)

  # 4 real matmuls
  c_re_re = hlo.dot_general(
    acc_type,
    re_lhs,
    re_rhs,
    dot_dnums,
    precision_config=precision_attr(precision),
  )
  c_im_im = hlo.dot_general(
    acc_type,
    im_lhs,
    im_rhs,
    dot_dnums,
    precision_config=precision_attr(precision),
  )
  c_re_im = hlo.dot_general(
    acc_type,
    re_lhs,
    im_rhs,
    dot_dnums,
    precision_config=precision_attr(precision),
  )
  c_im_re = hlo.dot_general(
    acc_type,
    im_lhs,
    re_rhs,
    dot_dnums,
    precision_config=precision_attr(precision),
  )

  # C_re = X_re @ Y_re - X_im @ Y_im
  c_re = hlo.subtract(c_re_re, c_im_im)
  # C_im = X_re @ Y_im + X_im @ Y_re
  c_im = hlo.add(c_re_im, c_im_re)

  return [_physical_stack_real_imag(ctx, c_re, c_im, logical_shape)]


def _make_transcendental_lower_bcomplex32(lax_fn_name):
  """Create a lowering for a transcendental function on bcomplex32.

  Strategy: promote to complex64, compute, demote back.
  We use mlir.lower_fun which traces through the Python function,
  producing a jaxpr that uses the already-registered lowerings for
  real, imag, complex, convert_element_type on bcomplex32.
  """

  def lower(ctx, x, **kw):
    def impl(x):
      from jax._src.lax import lax as lax_mod

      # Extract real/imag as bf16, promote to float32
      re_bf16 = lax_mod.real(x)
      im_bf16 = lax_mod.imag(x)
      re_f32 = lax_mod.convert_element_type(re_bf16, np.float32)
      im_f32 = lax_mod.convert_element_type(im_bf16, np.float32)
      # Construct complex64
      x64 = lax_mod.complex(re_f32, im_f32)
      # Compute the transcendental
      result64 = getattr(lax_mod, lax_fn_name)(x64)
      # Demote back to bf16 and reconstruct bcomplex32
      re_out = lax_mod.convert_element_type(
        lax_mod.real(result64), _bfloat16_dtype
      )
      im_out = lax_mod.convert_element_type(
        lax_mod.imag(result64), _bfloat16_dtype
      )
      return lax_mod.complex(re_out, im_out)

    return mlir.lower_fun(impl, multiple_results=False)(ctx, x)

  return lower


def _register_lowerings():
  """Register all bcomplex32 custom lowerings.

  This must be called after the primitives are defined in lax.py.
  We use the mlir_lowering override mechanism.
  """
  from jax._src.lax import lax as lax_module

  # Helper to get the original lowering rule function for a primitive.
  def _get_original_rule(prim):
    entry = mlir._lowerings.get(prim)
    return entry.rule if entry is not None else None

  # For real_p: override lowering when input is bcomplex32_edtype
  original_real_lowering = _get_original_rule(lax_module.real_p)

  def _real_lowering(ctx, x, **kw):
    (aval_in,) = ctx.avals_in
    if _is_bcomplex32_edtype(aval_in.dtype):
      return _real_lower_bcomplex32(ctx, x, **kw)
    return original_real_lowering(ctx, x, **kw)

  mlir.register_lowering(lax_module.real_p, _real_lowering)

  # For imag_p
  original_imag_lowering = _get_original_rule(lax_module.imag_p)

  def _imag_lowering(ctx, x, **kw):
    (aval_in,) = ctx.avals_in
    if _is_bcomplex32_edtype(aval_in.dtype):
      return _imag_lower_bcomplex32(ctx, x, **kw)
    return original_imag_lowering(ctx, x, **kw)

  mlir.register_lowering(lax_module.imag_p, _imag_lowering)

  # For complex_p
  original_complex_lowering = _get_original_rule(lax_module.complex_p)

  def _complex_lowering(ctx, re, im, **kw):
    (aval_out,) = ctx.avals_out
    if _is_bcomplex32_edtype(aval_out.dtype):
      return _complex_lower_bcomplex32(ctx, re, im, **kw)
    return original_complex_lowering(ctx, re, im, **kw)

  mlir.register_lowering(lax_module.complex_p, _complex_lowering)

  # For neg_p
  original_neg_lowering = _get_original_rule(lax_module.neg_p)

  def _neg_lowering(ctx, x, **kw):
    (aval_in,) = ctx.avals_in
    if _is_bcomplex32_edtype(aval_in.dtype):
      return _neg_lower_bcomplex32(ctx, x, **kw)
    return original_neg_lowering(ctx, x, **kw)

  mlir.register_lowering(lax_module.neg_p, _neg_lowering)

  # For conj_p
  original_conj_lowering = _get_original_rule(lax_module.conj_p)

  def _conj_lowering(ctx, x, **kw):
    (aval_in,) = ctx.avals_in
    if _is_bcomplex32_edtype(aval_in.dtype):
      return _conj_lower_bcomplex32(ctx, x, **kw)
    return original_conj_lowering(ctx, x, **kw)

  mlir.register_lowering(lax_module.conj_p, _conj_lowering)

  # For add_p
  original_add_lowering = _get_original_rule(lax_module.add_p)

  def _add_lowering(ctx, x, y, **kw):
    avals_in = ctx.avals_in
    if any(_is_bcomplex32_edtype(a.dtype) for a in avals_in):
      return _add_lower_bcomplex32(ctx, x, y, **kw)
    return original_add_lowering(ctx, x, y, **kw)

  mlir.register_lowering(lax_module.add_p, _add_lowering)

  # For sub_p
  original_sub_lowering = _get_original_rule(lax_module.sub_p)

  def _sub_lowering(ctx, x, y, **kw):
    avals_in = ctx.avals_in
    if any(_is_bcomplex32_edtype(a.dtype) for a in avals_in):
      return _sub_lower_bcomplex32(ctx, x, y, **kw)
    return original_sub_lowering(ctx, x, y, **kw)

  mlir.register_lowering(lax_module.sub_p, _sub_lowering)

  # For mul_p
  original_mul_lowering = _get_original_rule(lax_module.mul_p)

  def _mul_lowering(ctx, x, y, **kw):
    avals_in = ctx.avals_in
    if any(_is_bcomplex32_edtype(a.dtype) for a in avals_in):
      return _mul_lower_bcomplex32(ctx, x, y, **kw)
    return original_mul_lowering(ctx, x, y, **kw)

  mlir.register_lowering(lax_module.mul_p, _mul_lowering)

  # For abs_p
  original_abs_lowering = _get_original_rule(lax_module.abs_p)

  def _abs_lowering(ctx, x, **kw):
    (aval_in,) = ctx.avals_in
    if _is_bcomplex32_edtype(aval_in.dtype):
      return _abs_lower_bcomplex32(ctx, x, **kw)
    return original_abs_lowering(ctx, x, **kw)

  mlir.register_lowering(lax_module.abs_p, _abs_lowering)

  # For div_p
  original_div_lowering = _get_original_rule(lax_module.div_p)

  def _div_lowering(ctx, x, y, **kw):
    avals_in = ctx.avals_in
    if any(_is_bcomplex32_edtype(a.dtype) for a in avals_in):
      return _div_lower_bcomplex32(ctx, x, y, **kw)
    return original_div_lowering(ctx, x, y, **kw)

  mlir.register_lowering(lax_module.div_p, _div_lowering)

  # For dot_general_p
  # We need to override both the default and platform-specific lowerings.
  original_dot_general_lowering = _get_original_rule(lax_module.dot_general_p)

  # Also capture platform-specific original lowerings
  from jax._src.interpreters import mlir as mlir_mod

  platform_specific_dot_general = {}
  for plat in ["cpu", "tpu", "gpu", "cuda", "rocm"]:
    plat_rules = mlir_mod._platform_specific_lowerings.get(plat, {})
    if lax_module.dot_general_p in plat_rules:
      platform_specific_dot_general[plat] = plat_rules[
        lax_module.dot_general_p
      ].rule

  def _dot_general_lowering_wrapper(ctx, lhs, rhs, **params):
    avals_in = ctx.avals_in
    if any(_is_bcomplex32_edtype(a.dtype) for a in avals_in):
      return _dot_general_lower_bcomplex32(ctx, lhs, rhs, **params)
    return original_dot_general_lowering(ctx, lhs, rhs, **params)

  def _dot_general_lowering_wrapper_platform(ctx, lhs, rhs, **params):
    avals_in = ctx.avals_in
    if any(_is_bcomplex32_edtype(a.dtype) for a in avals_in):
      return _dot_general_lower_bcomplex32(ctx, lhs, rhs, **params)
    platform = params.get("platform", "default")
    orig = platform_specific_dot_general.get(
      platform,
      platform_specific_dot_general.get("cpu", original_dot_general_lowering),
    )
    return orig(ctx, lhs, rhs, **params)

  mlir.register_lowering(
    lax_module.dot_general_p, _dot_general_lowering_wrapper
  )
  # Also register for CPU and TPU platforms where dot_general has
  # platform-specific lowerings
  for plat in platform_specific_dot_general:
    mlir.register_lowering(
      lax_module.dot_general_p,
      _dot_general_lowering_wrapper_platform,
      platform=plat,
    )

  # For transcendental unary ops: exp, log, sin, cos, sqrt, tanh
  transcendental_prims = [
    ("exp_p", "exp"),
    ("log_p", "log"),
    ("sin_p", "sin"),
    ("cos_p", "cos"),
    ("sqrt_p", "sqrt"),
    ("tanh_p", "tanh"),
  ]
  for prim_attr, fn_name in transcendental_prims:
    prim = getattr(lax_module, prim_attr)
    original_lowering = _get_original_rule(prim)
    bcomplex32_lower = _make_transcendental_lower_bcomplex32(fn_name)

    def _make_wrapped_lowering(orig, bc32_lower):
      def wrapped(ctx, x, **kw):
        (aval_in,) = ctx.avals_in
        if _is_bcomplex32_edtype(aval_in.dtype):
          return bc32_lower(ctx, x, **kw)
        return orig(ctx, x, **kw)

      return wrapped

    mlir.register_lowering(
      prim, _make_wrapped_lowering(original_lowering, bcomplex32_lower)
    )
