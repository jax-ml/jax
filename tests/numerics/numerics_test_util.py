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
    "Maximum number of samples to test per dtype. If total elements <= this,"
    " tests exhaustively.",
)

MAX_ULP_BIN = flags.DEFINE_integer(
    "jax_numerics_max_bincount",
    100000,
    "Maximum ULP difference tracked in bincount histogram. Differences larger than this are clipped.",
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
  `(variants, {dtype: value})` pairs, where `variants` is a string or list of
  strings matching either the specific hardware variant (e.g. 'tpu_v5p') or the
  broad device type (e.g. 'tpu', 'cpu', 'gpu').
  """
  if not isinstance(spec, list):
    return spec if spec is not None else default
  dut = jtu.device_under_test()
  for variants, overrides in spec:
    if isinstance(variants, str):
      variants = [variants]
    if (variant in variants or dut in variants) and dtype in overrides:
      return overrides[dtype]
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
def _ulp_diff_and_sign_jax(x, y, dtype, ftz: bool = True):
  """Computes unsigned uint64 ULP distance and sign (x > y) in JAX.

  Returns:
    ulp: uint64 array of ULP distances between `x` and `y`.
    pos: bool array, True where `x > y` (signed ULP difference is positive).
  """
  x = jnp.asarray(x, dtype=dtype)
  y = jnp.asarray(y, dtype=dtype)
  itemsize = np.dtype(dtype).itemsize
  udt = jnp.dtype(f"u{itemsize}")
  ux = jax.lax.bitcast_convert_type(x, udt)
  uy = jax.lax.bitcast_convert_type(y, udt)

  nan_x = jnp.isnan(x)
  nan_y = jnp.isnan(y)
  both_nan = nan_x & nan_y

  inf_x = jnp.isinf(x)
  inf_y = jnp.isinf(y)

  sign_x = jnp.signbit(x)
  sign_y = jnp.signbit(y)
  same_sign = sign_x == sign_y

  both_inf = inf_x & inf_y
  both_inf_same = both_inf & same_sign

  # Any mismatch between finite/NaN/Inf or between +Inf and -Inf is maximal error.
  mismatch = (nan_x != nan_y) | (inf_x != inf_y) | (both_inf & ~same_sign)

  # Extract unsigned integer magnitude (all bits except the sign bit).
  # In IEEE-754 sign-magnitude encoding, incrementing the magnitude bits by 1
  # steps to the next adjacent floating-point number away from zero (1 ULP).
  mag_mask = udt.type((1 << (itemsize * 8 - 1)) - 1)
  mag_x = (ux & mag_mask).astype(jnp.uint64)
  mag_y = (uy & mag_mask).astype(jnp.uint64)
  if ftz:
    # Subnormal numbers have exponent 0, so their magnitude bits are <= mant_mask.
    # Under Flush-To-Zero (FTZ), all subnormals collapse to magnitude 0, and
    # normal numbers shift down by `mant_mask` so the smallest normal number
    # (`tiny`) sits at distance 1 ULP from zero.
    mant_mask = jnp.uint64((1 << np.finfo(dtype).nmant) - 1)
    mag_x = jnp.where(mag_x <= mant_mask, jnp.uint64(0), mag_x - mant_mask)
    mag_y = jnp.where(mag_y <= mant_mask, jnp.uint64(0), mag_y - mant_mask)

  # For same sign, ULP distance is |mag_x - mag_y|.
  # For opposite signs, ULP distance includes the step across +0.0 and -0.0,
  # so +0.0 (mag=0, sign=0) and -0.0 (mag=0, sign=1) are 1 ULP apart.
  diff_same = jnp.maximum(mag_x, mag_y) - jnp.minimum(mag_x, mag_y)
  diff_opp = mag_x + mag_y + jnp.uint64(1)
  ulp = jnp.where(same_sign, diff_same, diff_opp)
  ulp = jnp.where(both_nan | both_inf_same, jnp.uint64(0), ulp)
  ulp = jnp.where(mismatch, jnp.uint64(np.iinfo(np.uint64).max), ulp)

  # `pos` is True when `x` is strictly greater than `y` on the ordered real line
  # (where +0.0 > -0.0).
  pos_same = jnp.where(sign_x, mag_x < mag_y, mag_x > mag_y)
  pos = jnp.where(same_sign, pos_same, ~sign_x)
  return ulp, pos


def _map_to_31_bins(ulp: jax.Array, pos: jax.Array) -> jax.Array:
  """Maps unsigned ULP distance and sign to 31 compact histogram bin indices."""
  decade = (
      11
      + (ulp >= 100).astype(jnp.int32)
      + (ulp >= 1000).astype(jnp.int32)
      + (ulp >= 10000).astype(jnp.int32)
      + (ulp >= 100000).astype(jnp.int32)
  )
  offset = jnp.where(ulp <= 10, ulp.astype(jnp.int32), decade)
  return jnp.where(pos, 15 + offset, 15 - offset)


# Representative signed ULP integer for each of the 31 bins produced by
# `_map_to_31_bins`, chosen so `_histogram_bin(val)` maps each bin index to its
# corresponding bucket in `render_histogram_from_counts`.
_BIN_REPRESENTATIVES = (
    -100000,
    -10000,
    -1000,
    -100,
    -11,
    *range(-10, 11),
    11,
    100,
    1000,
    10000,
    100000,
)


def ulp_diff(
    x: np.ndarray, y: np.ndarray, dtype, ftz: bool = True
) -> np.ndarray:
  """Computes integer ULP distance between x and y.

  When x and y have opposite signs, the distance includes the signed zero
  transition, so +0.0 and -0.0 differ by 1 ULP.
  """
  cpu_dev = jax.devices("cpu")[0]
  with jax.enable_x64(True), jax.default_device(cpu_dev):
    ulp, _ = _ulp_diff_and_sign_jax(
        np.asarray(x, dtype=dtype), np.asarray(y, dtype=dtype), dtype, ftz
    )
    return np.asarray(ulp, dtype=np.uint64)


def _flush_subnormals(x: np.ndarray, dtype) -> np.ndarray:
  """Flushes subnormal values of dtype to signed zero."""
  x = np.asarray(x, dtype=dtype)
  tiny = np.finfo(dtype).tiny
  mask = np.abs(x) < tiny
  if not np.any(mask):
    return x
  return np.where(mask, np.where(np.signbit(x), dtype(-0.0), dtype(0.0)), x)


def _round_mpmath_to_dtype(mp_val, dtype) -> np.ndarray:
  """Rounds an mpmath.mpf directly to dtype (RNE) without double rounding.

  Converting a 100-bit `mpmath.mpf` via Python `float` (`float64`) before
  casting to `float32`, `float16`, or `bfloat16` can suffer from double
  rounding when the true value lies within 2^-53 of a target precision midpoint.
  This function scales `mp_val` directly to the target least-significant bit
  (accounting for subnormal gradual underflow) and rounds to nearest-even.
  """
  if mpmath.isnan(mp_val):
    return np.array(np.nan, dtype=dtype)
  if mpmath.isinf(mp_val):
    return np.array(np.inf if mp_val > 0 else -np.inf, dtype=dtype)
  if mp_val == 0:
    return np.array(0.0, dtype=dtype)

  finfo = np.finfo(dtype)
  p = finfo.nmant + 1
  emin = finfo.minexp
  _, exp2 = mpmath.frexp(mp_val)
  # Exponent of the unit in the last place (clamp to subnormal minimum exponent).
  lsb_exp = max(exp2 - p, emin - (p - 1))
  scaled = mpmath.ldexp(mp_val, -lsb_exp)
  q = int(mpmath.floor(scaled))
  r = scaled - q
  # Round to nearest, ties to even (RNE).
  if r > 0.5 or (r == 0.5 and (q & 1)):
    q += 1
  if q == 0 and mp_val < 0:
    return np.array(-0.0, dtype=dtype)
  return np.array(float(mpmath.ldexp(q, lsb_exp)), dtype=dtype)


def _eval_mpmath(mpmath_fn, val, dtype=None, input_ftz: bool = True):
  """Evaluates scalar mpmath function at current mpmath precision."""
  if input_ftz and dtype is not None:
    val = _flush_subnormals(np.array(val, dtype=dtype), dtype).item()
  res = mpmath_fn(mpmath.mpf(float(val)))
  if isinstance(res, mpmath.mpc):
    return mpmath.nan
  return res


@jax.jit(static_argnames=("dtype", "ftz", "k"))
def _eval_chunk_ulp_stats_pruned(inputs, computed, reference, dtype, ftz, k):
  """JIT-compiled exact k-block pruned top_k + 31-bin vmap(bincount) kernel.

  Reshapes inputs into `n_blocks` rows so that:
  1. Histogram counts are accumulated in parallel across rows via `vmap(bincount)`
     into 31 compact bins rather than a single 200k-bin atomic bincount.
  2. By the pigeonhole principle, the global top `k` elements across the entire
     chunk can reside in at most `k` distinct rows. Finding the `k` rows with the
     largest row-maximums (`jnp.max(ulp_2d, axis=1)`) and running a second
     `top_k` over only those `k * block_size` elements yields the exact global
     top `k` worst cases while avoiding sorting the full array.
  """
  n_blocks = 16384
  block_size = inputs.shape[0] // n_blocks
  comp_2d = computed.reshape(n_blocks, block_size)
  ref_2d = reference.reshape(n_blocks, block_size)
  ulp_2d, pos_2d = _ulp_diff_and_sign_jax(comp_2d, ref_2d, dtype, ftz)

  bins_2d = _map_to_31_bins(ulp_2d, pos_2d)
  chunk_counts = jax.vmap(lambda b: jnp.bincount(b, length=31))(bins_2d).sum(
      axis=0
  )

  block_max_ulp = jnp.max(ulp_2d, axis=1)
  _, top_blocks = jax.lax.top_k(block_max_ulp, k)
  cand_ulps = ulp_2d[top_blocks].reshape(-1)
  top_ulps, win_flat = jax.lax.top_k(cand_ulps, k)
  win_block = top_blocks[win_flat // block_size]
  win_col = win_flat % block_size
  top_indices = (
      win_block.astype(jnp.int64) * block_size + win_col.astype(jnp.int64)
  )
  top_x = jnp.asarray(inputs, dtype=dtype)[top_indices]
  top_y = jnp.asarray(computed, dtype=dtype)[top_indices]
  top_y_ref = jnp.asarray(reference, dtype=dtype)[top_indices]
  return chunk_counts, top_ulps, top_x, top_y, top_y_ref


@jax.jit(static_argnames=("dtype", "ftz", "k"))
def _eval_chunk_ulp_stats_small(inputs, computed, reference, dtype, ftz, k):
  """JIT-compiled ULP stats kernel for smaller arrays."""
  ulp, pos = _ulp_diff_and_sign_jax(computed, reference, dtype, ftz)
  bins = _map_to_31_bins(ulp, pos)
  chunk_counts = jnp.bincount(bins, length=31)
  top_k_count = min(k, ulp.shape[0])
  top_ulps, top_indices = jax.lax.top_k(ulp, top_k_count)
  top_x = jnp.asarray(inputs, dtype=dtype)[top_indices]
  top_y = jnp.asarray(computed, dtype=dtype)[top_indices]
  top_y_ref = jnp.asarray(reference, dtype=dtype)[top_indices]
  return chunk_counts, top_ulps, top_x, top_y, top_y_ref


def eval_ulp_stats(
    inputs,
    computed,
    reference,
    dtype,
    ftz: bool = True,
    max_bincount: int = 100000,
    k: int = 20,
) -> tuple[dict[int, int], list[tuple[int, float, float, float]]]:
  """Computes signed ULP histogram counts and top-k worst cases for a chunk."""
  del max_bincount  # Unused; 31 compact bins are always used.
  n = len(inputs)
  if n == 0:
    return {}, []
  cpu_dev = jax.devices("cpu")[0]
  with jax.enable_x64(True), jax.default_device(cpu_dev):
    j_in = jax.device_put(np.asarray(inputs, dtype=dtype).ravel(), cpu_dev)
    j_comp = jax.device_put(np.asarray(computed, dtype=dtype).ravel(), cpu_dev)
    j_ref = jax.device_put(np.asarray(reference, dtype=dtype).ravel(), cpu_dev)
    if n >= 16384 * 20 and n % 16384 == 0:
      chunk_counts, top_ulps, top_x, top_y, top_y_ref = (
          _eval_chunk_ulp_stats_pruned(j_in, j_comp, j_ref, dtype, ftz, k)
      )
    else:
      chunk_counts, top_ulps, top_x, top_y, top_y_ref = (
          _eval_chunk_ulp_stats_small(j_in, j_comp, j_ref, dtype, ftz, k)
      )
  counts = np.asarray(chunk_counts, dtype=np.int64)
  counts_dict = {
      _BIN_REPRESENTATIVES[b]: int(counts[b])
      for b in range(31)
      if counts[b] > 0
  }
  top_k = [
      (int(u), x.item(), y.item(), yr.item())
      for u, x, y, yr in zip(
          np.asarray(top_ulps),
          np.asarray(top_x),
          np.asarray(top_y),
          np.asarray(top_y_ref),
      )
  ]
  return counts_dict, top_k


def _fail_precision(
    test_case, jax_fn, mpmath_fn, dtype, max_ulp, top_k, udt, label="",
    input_ftz: bool = True, output_ftz: bool = True
):
  variant = get_hardware_variant()
  max_diff = int(top_k[0][0])
  suffix = f" [{label}]" if label else ""

  lines = [
      f"Max integer ULP error for {jax_fn.__name__} on {variant} "
      f"({np.dtype(dtype).name}){suffix} exceeded bound: {max_diff} > {max_ulp}",
      f"Top {len(top_k)} worst cases:",
      f"{'Rank':<4} | {'ULP (ref)':<9} | {'ULP (mp)':<8} | {'Input x':<16} | {'x (hex)':<18} | {'Computed y':<24} | {'Reference y*':<24} | {'mpmath exact'}",
      "-" * 155,
  ]

  for rank, (d, x, y, y_ref) in enumerate(top_k, 1):
    x_hex = hex(int(np.array(x, dtype=dtype).view(udt)))
    y_hex = hex(int(np.array(y, dtype=dtype).view(udt)))
    ref_hex = hex(int(np.array(y_ref, dtype=dtype).view(udt)))
    mp_val = _eval_mpmath(mpmath_fn, x, dtype=dtype, input_ftz=input_ftz)
    y_mp = _round_mpmath_to_dtype(mp_val, dtype)
    ulp_mp = int(ulp_diff(np.array([y]), np.array([y_mp]), dtype, ftz=output_ftz)[0])
    mp_exact_str = mpmath.nstr(mp_val, 30)

    comp_str = f"{y} ({y_hex})"
    ref_str = f"{y_ref} ({ref_hex})"
    lines.append(
        f"{rank:<4} | {int(d):<9} | {ulp_mp:<8} | {str(x):<16} | {x_hex:<18} | {comp_str:<24} | {ref_str:<24} | {mp_exact_str}"
    )

  test_case.fail("\n".join(lines))


def _fmt_signed(v: int) -> str:
  return "0" if v == 0 else f"{v:+d}"


def _histogram_bin(v: int) -> tuple[int, str]:
  """Maps a signed ULP error integer to a `(sort_key, label)` bucket.

  Errors in [-10, +10] are mapped to individual 1-ULP bins; larger errors are
  grouped into power-of-10 decade intervals up to `+-MAX_ULP_BIN`.
  """
  max_bin = MAX_ULP_BIN.value
  if v <= -max_bin:
    return (-max_bin, f"<={_fmt_signed(-max_bin)} ULP")
  if v >= max_bin:
    return (max_bin, f">={_fmt_signed(max_bin)} ULP")
  if -10 <= v <= 10:
    return (v, f"{_fmt_signed(v)} ULP")
  low = 10 ** int(np.floor(np.log10(abs(v))))
  high = low * 10
  if v > 0:
    return (low, f"[{_fmt_signed(low)}, {_fmt_signed(high)}) ULP")
  return (-high + 1, f"({_fmt_signed(-high)}, {_fmt_signed(-low)}] ULP")


def render_histogram_from_counts(
    counts_dict: dict[int, int], total: int, width: int = 40
) -> str:
  """Renders a text histogram from a dictionary mapping signed ULP error to count."""
  if not counts_dict:
    return ""
  bins: dict[tuple[int, str], int] = collections.defaultdict(int)
  for val, count in counts_dict.items():
    bins[_histogram_bin(val)] += count

  sorted_bins = sorted(bins.items(), key=lambda item: item[0][0])
  max_count = max(count for _, count in sorted_bins)
  lines = []
  for (_, label), count in sorted_bins:
    pct = (count / total) * 100
    bar_len = int(round((count / max_count) * width)) if max_count > 0 else 0
    bar = "█" * bar_len
    lines.append(f"    {label:>16}: {count:>6} ({pct:>6.2f}%) {bar}")
  return "\n".join(lines)


def check_unary_precision(
    test_case, jax_fn, mpmath_fn, dtype, max_ulp: int | None = None,
    bounds: list | None = None, input_ftz: bool | list = True,
    output_ftz: bool | list = True,
    ignore_inputs: list | None = None,
):
  """Checks unary precision of `jax_fn` against a higher-precision reference.

  Evaluates `jax_fn` across either all possible bit patterns of `dtype` (when
  `total_elements <= MAX_SAMPLES`, e.g. `bfloat16` and `float16` by default, or
  `float32` when `--jax_numerics_max_samples=4294967296`) or a uniform random
  sample of `MAX_SAMPLES` bit patterns.

  Args:
    test_case: The `jtu.JaxTestCase` instance running the test.
    jax_fn: The JAX unary function under test (e.g. `jnp.sin`).
    mpmath_fn: The corresponding `mpmath` reference function used to format
      exact values on failure.
    dtype: Floating-point dtype to test (`bfloat16`, `float16`, `float32`, `float64`).
    max_ulp: Explicit maximum allowed ULP error (overrides `bounds` if set).
    bounds: List of `(variants, {dtype: max_ulp})` override rules. Defaults to 0
      ULP if a platform/dtype combination is not listed.
    input_ftz: Whether subnormal inputs are flushed to zero before reference
      evaluation (bool or per-variant override list).
    output_ftz: Whether subnormal outputs are flushed to zero when computing
      ULP distances (bool or per-variant override list).
    ignore_inputs: Optional per-variant list of specific input values or uint
      bit patterns to exclude from error checking.
  """
  variant = get_hardware_variant()
  in_ftz = _resolve_override(input_ftz, variant, dtype, True)
  out_ftz = _resolve_override(output_ftz, variant, dtype, True)
  ignored_bits = resolve_ignore_inputs(ignore_inputs, variant, dtype)
  if max_ulp is None:
    if bounds is None:
      raise ValueError("Either bounds or max_ulp must be provided.")
    max_ulp = _resolve_override(bounds, variant, dtype, 0)

  itemsize = np.dtype(dtype).itemsize
  udt = np.dtype(f"u{itemsize}")
  total_elements = 1 << (itemsize * 8)
  max_samples = MAX_SAMPLES.value
  is_exhaustive = total_elements <= max_samples
  total_points = min(total_elements, max_samples)

  jitted_jax_fn = jax.jit(jax_fn)
  np_fn = getattr(np, jax_fn.__name__)
  k = NUM_WORST_CASES.value
  max_bin = MAX_ULP_BIN.value

  def _compute_reference(in_arr: np.ndarray) -> np.ndarray:
    ref_in = _flush_subnormals(in_arr, dtype) if in_ftz else in_arr
    return np_fn(ref_in.astype(np.float64)).astype(dtype)

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
      chunk_reference = _compute_reference(chunk_inputs)

      if len(ignored_bits) > 0:
        if is_exhaustive:
          # In exhaustive mode, chunk_inputs is a contiguous ascending range of
          # uint bit patterns starting at `start_b`, allowing O(1) index lookup.
          start_b = int(chunk_inputs[0].view(udt))
          end_b = int(chunk_inputs[-1].view(udt))
          for b in ignored_bits:
            if start_b <= b <= end_b:
              if not chunk_computed.flags.writeable:
                chunk_computed = chunk_computed.copy()
              idx = int(b - start_b)
              chunk_computed[idx] = chunk_reference[idx]
        else:
          mask = np.isin(chunk_inputs.view(udt), ignored_bits)
          chunk_computed = np.where(mask, chunk_reference, chunk_computed)

      return eval_ulp_stats(
          chunk_inputs,
          chunk_computed,
          chunk_reference,
          dtype=dtype,
          ftz=out_ftz,
          max_bincount=max_bin,
          k=k,
      )

  with jtu.ignore_warning(category=RuntimeWarning):
    if len(chunks) == 1:
      counts_dict, top_k = _eval_chunk(chunks[0])
    else:
      counts_dict = collections.Counter()
      all_top_k = []
      with concurrent.futures.ThreadPoolExecutor(
          max_workers=NUM_WORKERS.value
      ) as executor:
        for chunk_counts, chunk_top_k in executor.map(_eval_chunk, chunks):
          counts_dict.update(chunk_counts)
          all_top_k.extend(chunk_top_k)
      top_k = sorted(all_top_k, key=lambda item: item[0], reverse=True)[:k]

  max_diff = int(top_k[0][0]) if top_k else 0

  ignored_str = (
      f", ignored {len(ignored_bits)} inputs" if len(ignored_bits) > 0 else ""
  )
  hist_str = render_histogram_from_counts(counts_dict, total_points)
  output = (
      f"[{variant}] {jax_fn.__name__} ({np.dtype(dtype).name}): "
      f"max ULP error = {max_diff} (bound = {max_ulp}, {label}{ignored_str})\n"
      f"{hist_str}\n"
  )
  print(output, end="", flush=True)

  if max_diff > max_ulp:
    _fail_precision(
        test_case, jax_fn, mpmath_fn, dtype, max_ulp, top_k, udt,
        label=label, input_ftz=in_ftz, output_ftz=out_ftz
    )
  elif is_exhaustive and max_diff < max_ulp:
    test_case.fail(
        f"ULP bound for {jax_fn.__name__} on {variant} ({np.dtype(dtype).name}) "
        f"is not tight in exhaustive run: observed max ULP error {max_diff} < bound {max_ulp}."
    )
