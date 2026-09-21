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

"""Precision testing utilities for elementary floating-point functions."""

import collections
from collections.abc import Callable
import concurrent.futures
import os

from absl import flags
import jax
from jax._src import test_util as jtu
from jax._src import tpu_info
import jax.numpy as jnp
import mpmath
import numpy as np


def _default_num_workers() -> int:
  nproc = os.environ.get("NPROC")
  if nproc is not None:
    return min(int(nproc), 8)
  return min(os.cpu_count() or 1, 8)


NUM_WORST_CASES = flags.DEFINE_integer(
    "jax_numerics_num_worst_cases",
    20,
    "Number of worst-case error inputs to display on precision failure.",
)

MAX_SAMPLES = flags.DEFINE_integer(
    "jax_numerics_max_samples",
    2**20,
    "Maximum number of samples to test per dtype (for bf16, f16, f32). If total"
    " elements <= this, tests exhaustively.",
)

MAX_F64_SAMPLES = flags.DEFINE_integer(
    "jax_numerics_max_f64_samples",
    10000,
    "Maximum number of samples to test for float64 (evaluated using mpmath).",
)

NUM_WORKERS = flags.DEFINE_integer(
    "jax_numerics_num_workers",
    _default_num_workers(),
    "Number of worker threads for parallel chunk evaluation.",
)

# Set high precision for mpmath reference evaluations.
mpmath.mp.prec = 100


def get_hardware_variant() -> str:
  """Returns hardware platform group, e.g. 'cpu', 'gpu', or specific TPU 'tpu_v4i'."""
  dut = jtu.device_under_test()
  if dut == "tpu":
    device_kind = jax.devices()[0].device_kind
    chip = tpu_info.chip_version_from_device_kind(device_kind)
    if chip is not None:
      return f"tpu_{chip.value}"
    return "tpu"
  return dut


def _resolve_override(spec, variant: str, dtype, default):
  """Resolves a per-platform/per-dtype configuration override.

  `spec` may be either a scalar value (returned directly) or a list of
  `(variants, overrides)` pairs, where `variants` is a string or list of
  strings matching either the specific hardware variant (e.g. 'tpu_v5p') or the
  broad device type (e.g. 'tpu', 'cpu', 'gpu'), and `overrides` is either a
  `{dtype: value}` dict or a single value applying to all dtypes on `variants`.
  """
  if not isinstance(spec, list):
    return spec if spec is not None else default
  dut = jtu.device_under_test()
  for variants, overrides in spec:
    if isinstance(variants, str):
      variants = [variants]
    if variant in variants or dut in variants:
      if isinstance(overrides, dict):
        if dtype in overrides:
          return overrides[dtype]
      else:
        return overrides
  return default


def resolve_ignore_inputs(
    ignore_inputs: list | None, variant: str, dtype
) -> np.ndarray:
  """Resolves list of ignored inputs as an array of unsigned integer bit patterns."""
  udt = np.dtype(f"u{np.dtype(dtype).itemsize}")
  vals = _resolve_override(ignore_inputs, variant, dtype, None)
  if not vals:
    return np.array([], dtype=udt)
  res = [
      int(v) if isinstance(v, (int, np.integer))
      else int(np.array(v, dtype=dtype).view(udt))
      for v in vals
  ]
  return np.asarray(res, dtype=udt)


@jax.jit(static_argnames=("dtype", "ftz"))
def _ulp_diff_jax(computed, reference, dtype, ftz: bool = True):
  """Computes signed real float64 ULP difference (computed - reference) in JAX for < float64 dtypes.

  Returns:
    float64 array of signed real ULP differences between `computed` and `reference`.
    Positive values indicate `computed > reference`. Mismatches return +inf or -inf.
  """
  comp_f64 = jnp.asarray(computed, dtype=jnp.float64)
  ref_f64 = jnp.asarray(reference, dtype=jnp.float64)

  # Target dtype precision parameters.
  finfo = np.finfo(dtype)
  p = finfo.nmant + 1  # Total significand bits (including implicit leading 1).
  emin = finfo.minexp
  emax = finfo.maxexp - 1
  tiny = float(np.ldexp(1.0, emin))  # Smallest positive normal number.
  ulp_tiny = float(np.ldexp(1.0, emin - (p - 1)))  # LSB weight for subnormals.
  # Span of the subnormal region [0, tiny). Under FTZ, this region collapses to 0.
  subnormal_span = tiny - ulp_tiny
  # In round-to-nearest-even (RNE), reference values at or beyond this threshold
  # (halfway between max_float and 2^(emax+1)) round to infinity in target dtype.
  overflow_thresh = float(
      np.ldexp(1.0, emax + 1) - np.ldexp(0.5, emax - (p - 1))
  )

  # Check NaN and Inf conditions.
  nan_comp = jnp.isnan(comp_f64)
  nan_ref = jnp.isnan(ref_f64)
  both_nan = nan_comp & nan_ref

  inf_comp = jnp.isinf(comp_f64)
  # A reference value that overflows the target precision is treated as Inf.
  inf_ref = jnp.isinf(ref_f64) | (jnp.abs(ref_f64) >= overflow_thresh)
  sign_comp = jnp.signbit(comp_f64)
  sign_ref = jnp.signbit(ref_f64)
  same_sign = sign_comp == sign_ref

  both_inf = inf_comp & inf_ref
  both_inf_same = both_inf & same_sign
  # Mismatch occurs when NaN/Inf status differs, or infinities have opposite signs.
  mismatch = (nan_comp != nan_ref) | (inf_comp != inf_ref) | (both_inf & ~same_sign)

  abs_comp = jnp.abs(comp_f64)
  abs_ref = jnp.abs(ref_f64)

  if ftz:
    both_subnormal = (abs_comp < tiny) & (abs_ref < tiny)
    both_normal_same_sign = (abs_comp >= tiny) & (abs_ref >= tiny) & same_sign
    # Under FTZ, the subnormal range (-tiny, +tiny) is not representable and
    # flushes to 0. To avoid an artificial (2^(p-1) - 1)-ULP gap when comparing
    # values across the subnormal region, we apply a piecewise-linear
    # contraction `collapse(v) = sign(v) * max(0, |v| - subnormal_span)` where
    # `subnormal_span = tiny - ulp_tiny`. This maps [-subnormal_span,
    # +subnormal_span] to 0 and shifts `±tiny` to `±ulp_tiny`, making the
    # adjacent FTZ floats (-tiny, 0.0, +tiny) spaced 1 ULP apart on the
    # collapsed number line.
    comp_collapsed = jnp.sign(comp_f64) * jnp.maximum(
        0.0, abs_comp - subnormal_span
    )
    ref_collapsed = jnp.sign(ref_f64) * jnp.maximum(0.0, abs_ref - subnormal_span)
    delta_collapsed = jnp.where(
        both_subnormal, 0.0, comp_collapsed - ref_collapsed
    )
    # Normal numbers of the same sign use standard difference without collapsing.
    delta = jnp.where(
        both_normal_same_sign, comp_f64 - ref_f64, delta_collapsed
    )
  else:
    delta = comp_f64 - ref_f64

  # Compute the ULP size corresponding to the reference value.
  # For normal numbers, ulp(ref) = 2^(floor(log2(|ref|)) - (p - 1)).
  # Clamping |ref| from below at `tiny` fixes subnormal/zero ulp(ref) to ulp_tiny.
  _, exp2_ref = jnp.frexp(jnp.where(abs_ref >= tiny, abs_ref, tiny))
  ulp_size = jnp.ldexp(1.0, jnp.minimum(exp2_ref - 1, emax) - (p - 1))

  # Scale difference by ULP size; handle NaN/Inf identity and mismatches.
  signed_ulp = delta / ulp_size
  signed_ulp = jnp.where(both_nan | both_inf_same, 0.0, signed_ulp)
  mismatch_inf = jnp.where(~sign_comp, jnp.inf, -jnp.inf)
  return jnp.where(mismatch, mismatch_inf, signed_ulp)


def _ulp_diff_mpmath(
    computed: np.ndarray, reference: np.ndarray, dtype, ftz: bool = True
) -> np.ndarray:
  """Computes signed real ULP differences (computed - reference) / ulp(reference) with mpmath."""
  comp_arr = np.asarray(computed, dtype=np.float64).ravel()
  ref_arr = np.asarray(reference).ravel()

  # Target dtype precision parameters.
  finfo = np.finfo(dtype)
  p = finfo.nmant + 1  # Total significand bits (including implicit leading 1).
  emin = finfo.minexp
  emax = finfo.maxexp - 1
  tiny = mpmath.ldexp(1, emin)  # Smallest positive normal number.
  ulp_tiny = mpmath.ldexp(1, emin - (p - 1))  # LSB weight for subnormals.
  # Span of the subnormal region [0, tiny). Under FTZ, this region collapses to 0.
  subnormal_span = tiny - ulp_tiny
  # In round-to-nearest-even (RNE), reference values at or beyond this threshold
  # (halfway between max_float and 2^(emax+1)) round to infinity in target dtype.
  overflow_thresh = mpmath.ldexp(1, emax + 1) - mpmath.ldexp(1, emax - p)

  def _scalar_diff(c: float, r) -> float:
    ref = (
        r
        if isinstance(r, mpmath.mpf)
        else (mpmath.nan if np.isnan(r) else mpmath.mpf(float(r)))
    )
    nan_comp = bool(np.isnan(c))
    nan_ref = bool(mpmath.isnan(ref))
    if nan_comp and nan_ref:
      return 0.0

    inf_comp = bool(np.isinf(c))
    # A reference value that overflows the target precision is treated as Inf.
    inf_ref = bool(mpmath.isinf(ref)) or (
        not nan_ref and abs(ref) >= overflow_thresh
    )

    sign_comp = bool(np.signbit(c))
    sign_ref = bool(ref < 0) if not nan_ref else False
    same_sign = sign_comp == sign_ref

    if inf_comp and inf_ref and same_sign:
      return 0.0
    # Mismatch occurs when NaN/Inf status differs, or infinities have opposite signs.
    if (
        (nan_comp != nan_ref)
        or (inf_comp != inf_ref)
        or (inf_comp and inf_ref and not same_sign)
    ):
      return float("-inf") if sign_comp else float("inf")

    mp_comp = mpmath.mpf(c)
    abs_comp = abs(mp_comp)
    abs_ref = abs(ref)

    if ftz:
      if abs_comp < tiny and abs_ref < tiny:
        return 0.0

      # Normal numbers of the same sign use standard difference without collapsing.
      if abs_comp >= tiny and abs_ref >= tiny and (mp_comp > 0) == (ref > 0):
        delta = mp_comp - ref
      else:
        # When bridging across the subnormal boundary, collapse the subnormal range
        # by subtracting `subnormal_span` so normal numbers meeting at zero do not
        # incur an artificial (2^p - 1) ULP discontinuity.
        comp_col = (1 if mp_comp >= 0 else -1) * max(
            mpmath.mpf(0.0), abs_comp - subnormal_span
        )
        ref_col = (1 if ref >= 0 else -1) * max(
            mpmath.mpf(0.0), abs_ref - subnormal_span
        )
        delta = comp_col - ref_col
    else:
      delta = mp_comp - ref

    # Compute the ULP size corresponding to the reference value.
    # For normal numbers, ulp(ref) = 2^(floor(log2(|ref|)) - (p - 1)).
    # Clamping |ref| from below at `tiny` fixes subnormal/zero ulp(ref) to ulp_tiny.
    _, exp2 = mpmath.frexp(max(abs_ref, tiny))
    ulp_size = mpmath.ldexp(1, min(exp2 - 1, emax) - (p - 1))

    return float(delta / ulp_size)

  res = [_scalar_diff(float(ci), ri) for ci, ri in zip(comp_arr, ref_arr)]
  return np.asarray(res, dtype=np.float64).reshape(np.shape(computed))


_MAX_ULP_BIN = 100000.0

_POS_BIN_LABELS = (
    "(0, +0.5] ULP",
    "(+0.5, +1] ULP",
    *(f"(+{i - 1}, +{i}] ULP" for i in range(2, 11)),
    "(+10, +100] ULP",
    "(+100, +1000] ULP",
    "(+1000, +10000] ULP",
    "(+10000, +100000] ULP",
    ">=+100000 ULP",
)
_NEG_BIN_LABELS = (
    "<=-100000 ULP",
    "[-100000, -10000) ULP",
    "[-10000, -1000) ULP",
    "[-1000, -100) ULP",
    "[-100, -10) ULP",
    *(f"[-{i}, -{i - 1}) ULP" for i in range(10, 1, -1)),
    "[-1, -0.5) ULP",
    "[-0.5, 0) ULP",
)
_BIN_LABELS = (*_NEG_BIN_LABELS, "0 ULP", *_POS_BIN_LABELS)


def _map_to_bins(signed_ulp: jax.Array) -> jax.Array:
  """Maps signed real ULP distance to compact histogram bin indices."""
  ulp = jnp.abs(signed_ulp)
  decade = (
      12
      + (ulp > 100.0).astype(jnp.int32)
      + (ulp > 1000.0).astype(jnp.int32)
      + (ulp > 10000.0).astype(jnp.int32)
      + (ulp >= _MAX_ULP_BIN).astype(jnp.int32)
  )
  # Offset 0: exact 0.0
  # Offset 1: (0, 0.5] ULP
  # Offset 2..11: (0.5, 1], (1, 2], ..., (9, 10] ULP
  # Offset 12..16: decades (10, 100], ..., >=100000 ULP
  small_offset = jnp.where(
      ulp == 0.0,
      0,
      jnp.where(ulp <= 0.5, 1, jnp.ceil(ulp).astype(jnp.int32) + 1),
  )
  offset = jnp.where(ulp <= 10.0, small_offset, decade)
  pos = signed_ulp > 0.0
  zero_bin = len(_NEG_BIN_LABELS)
  return jnp.where(pos, zero_bin + offset, zero_bin - offset)


def ulp_diff(
    computed: np.ndarray,
    reference: np.ndarray,
    dtype,
    ftz: bool = True,
) -> np.ndarray:
  """Computes real float64 ULP distance between computed (in dtype) and reference."""
  if np.dtype(dtype) == np.float64:
    return np.abs(_ulp_diff_mpmath(computed, reference, dtype, ftz=ftz))
  cpu_dev = jax.devices("cpu")[0]
  with jax.enable_x64(True), jax.default_device(cpu_dev):
    signed_ulp = _ulp_diff_jax(
        np.asarray(computed, dtype=dtype).astype(np.float64),
        np.asarray(reference, dtype=np.float64),
        dtype,
        ftz,
    )
    return np.asarray(jnp.abs(signed_ulp), dtype=np.float64)


def _flush_subnormals(x: np.ndarray, dtype) -> np.ndarray:
  """Flushes subnormal values of dtype to signed zero."""
  x = np.asarray(x, dtype=dtype)
  tiny = np.finfo(dtype).tiny
  mask = np.abs(x) < tiny
  if not np.any(mask):
    return x
  return np.where(mask, np.where(np.signbit(x), dtype(-0.0), dtype(0.0)), x)


def _eval_mpmath(mpmath_fn, val, dtype=None, input_ftz: bool = True):
  """Evaluates scalar mpmath function at current mpmath precision."""
  if input_ftz and dtype is not None:
    val = _flush_subnormals(np.array(val, dtype=dtype), dtype).item()
  if np.isnan(val):
    return mpmath.nan
  fval = float(val)
  try:
    res = mpmath_fn(mpmath.mpf(fval))
  except ZeroDivisionError:
    return -mpmath.inf if np.signbit(val) else mpmath.inf
  except (ValueError, OverflowError):
    return mpmath.nan
  if isinstance(res, mpmath.mpc):
    return mpmath.nan
  return res


@jax.jit(static_argnames=("dtype", "ftz", "k"))
def _eval_chunk_ulp_stats_pruned(computed, ref, dtype, ftz, k):
  """JIT-compiled exact k-block pruned top_k + vmap(bincount) kernel.

  Reshapes inputs into `n_blocks` rows so that:
  1. Histogram counts are accumulated in parallel across rows via `vmap(bincount)`
     into compact bins rather than a single atomic bincount.
  2. By the pigeonhole principle, the global top `k` elements across the entire
     chunk can reside in at most `k` distinct rows. Finding the `k` rows with the
     largest row-maximums (`jnp.max(ulp_2d, axis=1)`) and running a second
     `top_k` over only those `k * block_size` elements yields the exact global
     top `k` worst cases while avoiding sorting the full array.
  """
  n_blocks = 16384
  block_size = computed.shape[0] // n_blocks
  comp_2d = computed.reshape(n_blocks, block_size)
  ref_2d = ref.reshape(n_blocks, block_size)
  signed_ulp_2d = _ulp_diff_jax(comp_2d, ref_2d, dtype, ftz)
  ulp_2d = jnp.abs(signed_ulp_2d)

  bins_2d = _map_to_bins(signed_ulp_2d)
  chunk_counts = jax.vmap(
      lambda b: jnp.bincount(b, length=len(_BIN_LABELS))
  )(bins_2d).sum(axis=0)

  block_max_ulp = jnp.max(ulp_2d, axis=1)
  _, top_blocks = jax.lax.top_k(block_max_ulp, k)
  cand_ulps = ulp_2d[top_blocks].reshape(-1)
  top_ulps, win_flat = jax.lax.top_k(cand_ulps, k)
  win_block = top_blocks[win_flat // block_size]
  win_col = win_flat % block_size
  top_indices = (
      win_block.astype(jnp.int64) * block_size + win_col.astype(jnp.int64)
  )
  return chunk_counts, top_ulps, top_indices


@jax.jit(static_argnames=("dtype", "ftz", "k"))
def _eval_chunk_ulp_stats_small(computed, ref, dtype, ftz, k):
  """JIT-compiled ULP stats kernel for smaller arrays."""
  signed_ulp = _ulp_diff_jax(computed, ref, dtype, ftz)
  ulp = jnp.abs(signed_ulp)
  bins = _map_to_bins(signed_ulp)
  chunk_counts = jnp.bincount(bins, length=len(_BIN_LABELS))
  top_k_count = min(k, ulp.shape[0])
  top_ulps, top_indices = jax.lax.top_k(ulp, top_k_count)
  return chunk_counts, top_ulps, top_indices


def eval_ulp_stats(
    inputs,
    computed,
    reference,
    dtype,
    ftz: bool = True,
    k: int = 20,
) -> tuple[dict[str, int], list[tuple[float, float, float, float]]]:
  """Computes signed ULP histogram counts and top-k worst cases for a chunk."""
  n = len(inputs)
  if n == 0:
    return {}, []
  in_arr = np.asarray(inputs, dtype=dtype).ravel()
  comp_arr = np.asarray(computed, dtype=dtype).ravel()
  if np.dtype(dtype) == np.float64:
    ref_arr = np.asarray(reference).ravel()
    signed_ulps = _ulp_diff_mpmath(comp_arr, ref_arr, dtype, ftz=ftz)
    ref_f64 = np.array([float(r) for r in ref_arr], dtype=np.float64)
    cpu_dev = jax.devices("cpu")[0]
    with jax.enable_x64(True), jax.default_device(cpu_dev):
      bins = np.asarray(_map_to_bins(jnp.asarray(signed_ulps)))
    counts = np.bincount(bins, minlength=len(_BIN_LABELS))
    counts_dict = {
        label: int(counts[b])
        for b, label in enumerate(_BIN_LABELS)
        if counts[b] > 0
    }
    ulps = np.abs(signed_ulps)
    order = sorted(
        range(n), key=lambda idx: (np.isnan(ulps[idx]), ulps[idx]), reverse=True
    )[: min(k, n)]
    top_k = [
        (float(ulps[i]), float(in_arr[i]), float(comp_arr[i]),
         float(ref_f64[i]))
        for i in order
    ]
    return counts_dict, top_k

  ref_f64 = np.asarray(reference, dtype=np.float64).ravel()
  cpu_dev = jax.devices("cpu")[0]
  with jax.enable_x64(True), jax.default_device(cpu_dev):
    j_comp = jax.device_put(comp_arr.astype(np.float64), cpu_dev)
    j_ref = jax.device_put(ref_f64, cpu_dev)
    if k <= 16384 and n >= 16384 * max(k, 20) and n % 16384 == 0:
      chunk_counts, top_ulps, top_indices = (
          _eval_chunk_ulp_stats_pruned(j_comp, j_ref, dtype, ftz, k)
      )
    else:
      chunk_counts, top_ulps, top_indices = (
          _eval_chunk_ulp_stats_small(j_comp, j_ref, dtype, ftz, k)
      )
  counts = np.asarray(chunk_counts, dtype=np.int64)
  counts_dict = {
      label: int(counts[b])
      for b, label in enumerate(_BIN_LABELS)
      if counts[b] > 0
  }
  top_k = [
      (float(u), in_arr[idx].item(), comp_arr[idx].item(), ref_f64[idx].item())
      for u, idx in zip(np.asarray(top_ulps), np.asarray(top_indices))
  ]
  return counts_dict, top_k


def _format_worst_cases(
    top_k, udt, mpmath_fn, dtype, input_ftz: bool = True,
) -> str:
  lines = [
      f"Top {len(top_k)} worst cases:",
      (f"{'Rank':<4} | {'ULP (real)':<11} | {'Input x':<16} |"
       f" {'x (hex)':<18} | {'Computed y':<24} | {'Nearest y*':<24} |"
       f" {'mpmath'}"),
      "-" * 146,
  ]
  with np.errstate(all="ignore"):
    for rank, (d, x, y, y_ref) in enumerate(top_k, 1):
      x_hex = hex(int(np.array(x, dtype=dtype).view(udt)))
      y_hex = hex(int(np.array(y, dtype=dtype).view(udt)))
      ref_dt = np.array(y_ref, dtype=dtype)
      ref_hex = hex(int(ref_dt.view(udt)))

      mp_val = _eval_mpmath(mpmath_fn, x, dtype=dtype, input_ftz=input_ftz)
      mp_exact_str = mpmath.nstr(mp_val, 30)

      comp_str = f"{y} ({y_hex})"
      nearest_str = f"{ref_dt.item()} ({ref_hex})"
      lines.append(
          f"{rank:<4} | {d:<11.4f} | {str(x):<16} | {x_hex:<18} | {comp_str:<24}"
          f" | {nearest_str:<24} | {mp_exact_str}")
  return "\n".join(lines)


def _fail_precision(
    test_case, jax_fn, dtype, max_ulp, max_diff, worst_cases_str, label="",
):
  variant = get_hardware_variant()
  suffix = f" [{label}]" if label else ""
  header = (
      f"Max real ULP error for {jax_fn.__name__} on {variant}"
      f" ({np.dtype(dtype).name}){suffix} exceeded bound: {max_diff:.4f} >"
      f" {max_ulp}")
  test_case.fail(f"{header}\n{worst_cases_str}")


def _fail_signed_zero(
    test_case,
    jax_fn: Callable,
    dtype,
    signed_zero_errors: list[tuple[object, object, object]],
    udt: np.dtype,
    itemsize: int,
    label: str = "",
) -> None:
  """Fails test_case when a function returns +0.0 instead of -0.0 or vice versa."""
  variant = get_hardware_variant()
  suffix = f" [{label}]" if label else ""
  header = (
      f"Signed zero mismatch for {jax_fn.__name__} on {variant}"
      f" ({np.dtype(dtype).name}){suffix}:"
  )
  rows = []
  for in_val, comp_val, ref_val in signed_zero_errors[:10]:
    comp_sign = "-" if np.signbit(comp_val) else "+"
    ref_sign = "-" if np.signbit(ref_val) else "+"
    in_b = int(np.array(in_val, dtype=dtype).view(udt))
    rows.append(
        f"    x = {in_val!r} ({in_b:#0{itemsize * 2 + 2}x}): expected"
        f" {ref_sign}0.0, got {comp_sign}0.0"
    )
  if len(signed_zero_errors) > 10:
    rows.append(f"    ... and {len(signed_zero_errors) - 10} more")
  test_case.fail(f"{header}\n" + "\n".join(rows))


def render_histogram_from_counts(
    counts_dict: dict[str, int], total: int, width: int = 40
) -> str:
  """Renders a text histogram from a dictionary mapping bin label to count."""
  non_empty = [
      i for i, label in enumerate(_BIN_LABELS) if counts_dict.get(label, 0) > 0
  ]
  if not non_empty:
    return ""
  min_idx, max_idx = non_empty[0], non_empty[-1]
  rows = [
      (_BIN_LABELS[i], counts_dict.get(_BIN_LABELS[i], 0))
      for i in range(min_idx, max_idx + 1)
  ]
  max_count = max(count for _, count in rows)
  lines = []
  for label, count in rows:
    pct = (count / total) * 100
    bar_len = int(round((count / max_count) * width)) if max_count > 0 else 0
    bar = "█" * bar_len
    lines.append(f"    {label:>16}: {count:>6} ({pct:>6.2f}%) {bar}")
  return "\n".join(lines)


def check_unary_precision(
    test_case, jax_fn: Callable, ref_fn: Callable, mpmath_fn: Callable, dtype,
    bounds: float | tuple[float, float] | list | None = None,
    input_ftz: bool | list = True, output_ftz: bool | list = True,
    ignore_inputs: list | None = None,
    check_signed_zeros: bool | list = True,
):
  """Checks unary precision of `jax_fn` against reference implementations.

  Evaluates `jax_fn` across either all possible bit patterns of `dtype` (when
  `total_elements <= MAX_SAMPLES`, e.g. `bfloat16` and `float16` by default, or
  `float32` when `--jax_numerics_max_samples=4294967296`) or a uniform random
  sample of `MAX_SAMPLES` (`MAX_F64_SAMPLES` for `float64`) bit patterns.

  Args:
    test_case: The `jtu.JaxTestCase` instance running the test.
    jax_fn: The JAX unary function under test (e.g. `jnp.sin`).
    ref_fn: The vectorized NumPy/SciPy reference function operating on float64
      arrays (used when `dtype != float64`).
    mpmath_fn: The corresponding `mpmath` reference function used to compute
      high-precision reference values and real ULP errors with mpmath (used for
      `float64` and printed as part of the logging for the top K worst cases).
    dtype: Floating-point dtype to test (`bfloat16`, `float16`, `float32`,
      `float64`).
    bounds: Scalar `max_ulp`, `(min_ulp, max_ulp)` tuple, or list of
      `(variants, bound | {dtype: bound})` override rules. Bounds are quantized
      to upward-rounded multiples of `0.5` ULP
      (`expected_bound = max(0.5, ceil(worst_ulp * 2) / 2)`), where `0.5` ULP
      corresponds to faithful round-to-nearest rounding. In exhaustive runs, the
      observed error bound is checked for tightness against
      `min_ulp <= expected_bound <= max_ulp` (where a scalar `max_ulp` sets
      `min_ulp = max_ulp`). A range allows accommodating host architecture or
      vendor differences (e.g. AMD vs Intel) where different machines produce
      different tight bounds. Defaults to 0.5 ULP (correctly rounded) if a
      platform/dtype combination is not listed.
    input_ftz: Whether subnormal inputs are flushed to zero before reference
      evaluation (bool or per-variant override list).
    output_ftz: Whether subnormal outputs are flushed to zero when computing ULP
      distances (bool or per-variant override list).
    ignore_inputs: Optional per-variant list of specific input values or uint
      bit patterns to exclude from error checking.
    check_signed_zeros: Whether to verify that the sign of zero matches the
      reference when the computed and reference values are both zero (bool or
      per-variant override list).
  """
  if ((dtype == jnp.float64 or dtype == np.float64)
      and jtu.device_under_test() == "tpu"):
    test_case.skipTest("float64 on TPU is ef57 double-double")

  variant = get_hardware_variant()
  in_ftz = _resolve_override(input_ftz, variant, dtype, True)
  out_ftz = _resolve_override(output_ftz, variant, dtype, True)
  chk_signed_zeros = _resolve_override(check_signed_zeros, variant, dtype, True)
  ignored_bits = resolve_ignore_inputs(ignore_inputs, variant, dtype)
  raw_bound = _resolve_override(bounds, variant, dtype, 0.5)
  if isinstance(raw_bound, tuple):
    min_ulp, max_ulp = float(raw_bound[0]), float(raw_bound[1])
  else:
    min_ulp = max_ulp = float(raw_bound)

  is_f64 = dtype == jnp.float64 or dtype == np.float64
  itemsize = np.dtype(dtype).itemsize
  udt = np.dtype(f"u{itemsize}")
  total_elements = 1 << (itemsize * 8)
  max_samples = MAX_F64_SAMPLES.value if is_f64 else MAX_SAMPLES.value
  is_exhaustive = (not is_f64) and (total_elements <= max_samples)
  total_points = min(total_elements, max_samples)

  jitted_jax_fn = jax.jit(jax_fn)
  k = NUM_WORST_CASES.value
  eval_k = max(k, 1)

  def _compute_reference(in_arr: np.ndarray):
    if is_f64:
      return np.array(
          [
              _eval_mpmath(mpmath_fn, val.item(), dtype=dtype, input_ftz=in_ftz)
              for val in in_arr
          ],
          dtype=object,
      )
    ref_in = _flush_subnormals(in_arr, dtype) if in_ftz else in_arr
    return ref_fn(ref_in.astype(np.float64))

  # Process in chunks of at most 64 MB per array to bound concurrent memory
  # usage across worker threads during exhaustive (2**32 element) runs.
  chunk_size = (64 * 1024 * 1024) // itemsize

  if is_exhaustive:
    label = "exhaustive"
    chunks = [
        lambda s=start, c=min(chunk_size, total_points - start): np.arange(
            s, s + c, dtype=udt
        ).view(dtype)
        for start in range(0, total_points, chunk_size)
    ]
  else:
    label = f"sampled {total_points} points"
    base_rng = test_case.rng()
    chunks = []
    for start in range(0, total_points, chunk_size):
      count = min(chunk_size, total_points - start)
      seed = int(base_rng.randint(0, 1 << 31))
      chunks.append(
          lambda c=count, s=seed: jtu.rand_fullrange(np.random.RandomState(s))(
              (c,), dtype
          )
      )

  def _eval_chunk(make_chunk):
    with np.errstate(all="ignore"):
      chunk_inputs = make_chunk()
      chunk_computed = np.asarray(jitted_jax_fn(chunk_inputs))
      chunk_ref = _compute_reference(chunk_inputs)

      if len(ignored_bits) > 0:
        if is_exhaustive:
          # In exhaustive mode, chunk_inputs is a contiguous ascending range of
          # uint bit patterns starting at `start_b`, allowing O(1) index lookup.
          start_b = int(chunk_inputs[0].view(udt))
          end_b = int(chunk_inputs[-1].view(udt))
          idxs = [
              int(b - start_b) for b in ignored_bits if start_b <= b <= end_b
          ]
        else:
          idxs = np.flatnonzero(np.isin(chunk_inputs.view(udt), ignored_bits))
        if len(idxs) > 0:
          chunk_computed = chunk_computed.copy()
          chunk_inputs[idxs] = np.nan
          chunk_computed[idxs] = np.nan
          chunk_ref[idxs] = mpmath.nan if is_f64 else np.nan

      signed_zero_errors = []
      if chk_signed_zeros:
        if is_f64:
          # mpmath.mpf has no signed zero (-0.0) representation, so fall back
          # to ref_fn to check the sign of zero.
          comp_zeros = np.flatnonzero(chunk_computed == 0.0)
          if len(comp_zeros) > 0:
            in_z = chunk_inputs[comp_zeros]
            ref_in_z = _flush_subnormals(in_z, dtype) if in_ftz else in_z
            ref_at_zeros = ref_fn(ref_in_z.astype(np.float64))
            bad_mask = (ref_at_zeros == 0.0) & (
                np.signbit(chunk_computed[comp_zeros])
                != np.signbit(ref_at_zeros)
            )
            for z_idx in np.flatnonzero(bad_mask):
              idx = comp_zeros[z_idx]
              signed_zero_errors.append(
                  (chunk_inputs[idx], chunk_computed[idx], ref_at_zeros[z_idx])
              )
        else:
          mismatches = np.flatnonzero(
              (chunk_computed == 0.0)
              & (chunk_ref == 0.0)
              & (np.signbit(chunk_computed) != np.signbit(chunk_ref))
          )
          for idx in mismatches:
            signed_zero_errors.append(
                (chunk_inputs[idx], chunk_computed[idx], chunk_ref[idx])
            )

      chunk_counts, chunk_top_k = eval_ulp_stats(
          chunk_inputs,
          chunk_computed,
          chunk_ref,
          dtype=dtype,
          ftz=out_ftz,
          k=eval_k,
      )
      return chunk_counts, chunk_top_k, signed_zero_errors

  with jtu.ignore_warning(category=RuntimeWarning):
    if len(chunks) == 1:
      counts_dict, top_k, signed_zero_errors = _eval_chunk(chunks[0])
    else:
      counts_dict = collections.Counter()
      all_top_k = []
      signed_zero_errors = []
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=NUM_WORKERS.value
      ) as executor:
        for chunk_counts, chunk_top_k, chunk_sz_errors in executor.map(
            _eval_chunk, chunks
        ):
          counts_dict.update(chunk_counts)
          all_top_k.extend(chunk_top_k)
          signed_zero_errors.extend(chunk_sz_errors)
      top_k = sorted(
          all_top_k, key=lambda item: (np.isnan(item[0]), item[0]), reverse=True
      )[:eval_k]

  if signed_zero_errors:
    _fail_signed_zero(
        test_case, jax_fn, dtype, signed_zero_errors, udt, itemsize, label=label
    )

  max_diff = float(top_k[0][0]) if top_k else 0.0
  top_k = top_k[:k]

  ignored_str = (
      f", ignored {len(ignored_bits)} inputs" if len(ignored_bits) > 0 else ""
  )
  hist_str = render_histogram_from_counts(counts_dict, total_points)
  worst_cases_str = _format_worst_cases(
      top_k, udt, mpmath_fn, dtype, input_ftz=in_ftz
  )
  output = (
      f"[{variant}] {jax_fn.__name__} ({np.dtype(dtype).name}): max real ULP"
      f" error = {max_diff:.4f} (bound = {max_ulp},"
      f" {label}{ignored_str})\n{hist_str}\n{worst_cases_str}\n")
  print(output, end="", flush=True)

  expected_bound = (
      max(0.5, float(np.ceil(max_diff * 2.0) / 2.0))
      if not np.isinf(max_diff)
      else np.inf
  )
  if max_diff > max_ulp:
    _fail_precision(
        test_case, jax_fn, dtype, max_ulp, max_diff, worst_cases_str,
        label=label)
  elif is_exhaustive and not (min_ulp <= expected_bound <= max_ulp):
    bound_str = (
        f"({min_ulp}, {max_ulp})" if min_ulp != max_ulp else f"{max_ulp}"
    )
    test_case.fail(
        f"ULP bound for {jax_fn.__name__} on {variant} ({np.dtype(dtype).name})"
        " is not tight in exhaustive run: observed max real ULP error"
        f" {max_diff:.4f} (expected bound {expected_bound}, got {bound_str}).")
