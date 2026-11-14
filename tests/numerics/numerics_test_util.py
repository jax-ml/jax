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
from collections.abc import Callable, Iterable, Iterator, Sequence
import concurrent.futures
import functools
import math
import os
import sys
from typing import Any

from absl import flags
from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src import tpu_info
import jax.numpy as jnp
import mpmath
import numpy as np

try:
  import google_benchmark
except ImportError:
  google_benchmark = None


class ClassShardedTestLoader(jtu.JaxTestLoader):
  """Test loader that assigns each TestCase class to its own shard."""

  def shardTestCaseNames(
      self,
      iterator: Iterator[Any],
      ordered_names: Sequence[str],
      shard_index: int,
  ) -> Sequence[str]:
    if not ordered_names:
      return []
    bucket = next(iterator)
    if bucket == shard_index:
      return ordered_names
    return []



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


def _default_ftz(dtype) -> bool:
  """Returns whether subnormals flush to zero by default for `dtype`."""
  if jtu.device_under_test() == "gpu":
    return False
  if dtype is not None and np.dtype(dtype) == np.float16:
    return False
  return True


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
) -> Callable[..., np.ndarray] | None:
  """Resolves per-variant/per-dtype ignore_inputs into a callable predicate.

  The callable must have the signature:
    `ignore_fn(*args: np.ndarray) -> np.ndarray[bool]`
  where each argument in `*args` is a 1D NumPy array corresponding to an input
  argument to `jax_fn(*args)`, and the return value is a boolean mask where
  True indicates that the input should be excluded from accuracy checking.
  """
  vals = _resolve_override(ignore_inputs, variant, dtype, None)
  if vals is None:
    return None
  if callable(vals):
    return vals
  if isinstance(vals, (list, tuple, set)):
    udt = np.dtype(f"u{np.dtype(dtype).itemsize}")
    res = [
        int(v) if isinstance(v, (int, np.integer))
        else int(np.array(v, dtype=dtype).view(udt))
        for v in vals
    ]
    ignored_arr = np.asarray(res, dtype=udt)
    return lambda *args: np.isin(args[0].view(udt), ignored_arr)
  raise TypeError(
      f"ignore_inputs for variant {variant!r} and dtype {dtype} must be a"
      f" callable (*args: np.ndarray) -> np.ndarray[bool] or sequence, got {type(vals)}."
  )



@jax.jit(static_argnames=("dtype", "ftz"))
def _ulp_diff_jax(computed, reference, dtype, ftz: bool = True):
  """Computes signed real float64 ULP difference `(computed - reference) / ulp(reference)` in JAX for < float64 dtypes.

  For a target `dtype` with `p = nmant + 1` significand bits, minimum normal
  exponent `emin = minexp`, and maximum normal exponent `emax = maxexp - 1`,
  `ulp(reference)` is `2^(min(max(floor(log2(|reference|)), emin), emax) - (p - 1))`,
  clamped below to the subnormal quantum `ulp_tiny = 2^(emin - (p - 1))` for
  `|reference| < 2^emin` and capped above at the top normal binade's ULP
  `2^(emax - (p - 1))`.

  Behavior at infinity and the overflow boundary:
    - Matching `NaN`s, or matching infinities of the same sign (treating finite
      `|reference| >= 2^(emax + 1) - 0.5 * ulp(max_float)` as overflowing to
      `±inf` in round-to-nearest-even), return `0.0`.
    - When `reference` is finite with `|reference| >= 2^emax` (in or above the
      top normal binade) and `computed` has the same sign, rounding across the
      overflow threshold—either `computed = ±inf` below the threshold or finite
      `computed` (such as `±max_float`) at or above the threshold—is measured
      continuously by placing `±inf` at `±2^(emax + 1) = ±(max_float + 1 ULP)`.
    - `NaN` mismatches (one `NaN` and one non-`NaN`) return `+inf`.
    - All other infinity mismatches (such as opposite-signed infinities, finite
      `computed` for an infinite `reference`, or infinite `computed` when
      `|reference| < 2^emax`) return `+inf` if `computed > reference` and `-inf`
      if `computed < reference`.

  Behavior with respect to subnormals:
    - When `ftz=False` (gradual underflow), subnormal values use the uncollapsed
      difference `computed - reference` divided by `ulp(reference)` (which is
      `ulp_tiny` throughout `|reference| < 2^emin`).
    - When `ftz=True` (flush-to-zero), subnormal magnitudes `|v| < tiny = 2^emin`
      are treated as flushing to `0.0`:
      * If both `|computed| < tiny` and `|reference| < tiny`, returns `0.0`.
      * If both are normal (`>= tiny`) with the same sign, returns the standard
        `(computed - reference) / ulp(reference)`.
      * When comparing across the subnormal region (one normal and one
        subnormal/zero, or normal values of opposite signs), applies the
        piecewise-linear contraction
        `collapse(v) = sign(v) * max(0, |v| - (tiny - ulp_tiny))` to both
        operands before subtracting, so that the adjacent FTZ-representable
        values `(-tiny, 0.0, +tiny)` are spaced `1 ULP` apart without an
        artificial `(2^(p - 1) - 1)`-ULP gap across the subnormal range.

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
  overflow_val = float(np.ldexp(1.0, emax + 1))
  top_binade = float(np.ldexp(1.0, emax))

  # Check NaN and Inf conditions.
  computed_is_nan = jnp.isnan(comp_f64)
  ref_is_nan = jnp.isnan(ref_f64)
  both_nan = computed_is_nan & ref_is_nan

  computed_is_inf = jnp.isinf(comp_f64)
  # A reference value that overflows the target precision is treated as Inf.
  ref_is_inf = jnp.isinf(ref_f64) | (jnp.abs(ref_f64) >= overflow_thresh)
  computed_is_neg = jnp.signbit(comp_f64)
  ref_is_neg = jnp.signbit(ref_f64)
  same_sign = computed_is_neg == ref_is_neg

  abs_comp = jnp.abs(comp_f64)
  abs_ref = jnp.abs(ref_f64)

  both_inf = computed_is_inf & ref_is_inf
  both_inf_same = both_inf & same_sign
  # When `ref` is finite in the top binade (`abs_ref >= 2^emax`) and `comp` has
  # the same sign, rounding across `overflow_thresh` (either `comp = ±inf` for
  # `abs_ref < overflow_thresh`, or finite `comp` for `abs_ref >= overflow_thresh`)
  # is measured continuously by placing `±inf` at `±2^(emax+1) = ±(max_float + 1 ULP)`.
  near_overflow = (
      ~jnp.isinf(ref_f64)
      & ~ref_is_nan
      & ~computed_is_nan
      & same_sign
      & (abs_ref >= top_binade)
  )

  comp_eff = jnp.where(
      computed_is_inf & near_overflow,
      jnp.where(computed_is_neg, -overflow_val, overflow_val),
      comp_f64,
  )

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
        both_normal_same_sign, comp_eff - ref_f64, delta_collapsed
    )
  else:
    delta = comp_eff - ref_f64

  # Compute the ULP size corresponding to the reference value.
  # For normal numbers, ulp(ref) = 2^(floor(log2(|ref|)) - (p - 1)).
  # Clamping |ref| from below at `tiny` fixes subnormal/zero ulp(ref) to ulp_tiny.
  _, exp2_ref = jnp.frexp(jnp.where(abs_ref >= tiny, abs_ref, tiny))
  ulp_size = jnp.ldexp(1.0, jnp.minimum(exp2_ref - 1, emax) - (p - 1))

  # Scale difference by ULP size; handle NaN/Inf identity and mismatches.
  signed_ulp = delta / ulp_size
  signed_ulp = jnp.where(both_nan | both_inf_same, 0.0, signed_ulp)
  # Return +inf on NaN disagreements, and ±inf matching sign(computed - reference)
  # on infinity mismatches (treating overflowing references as ±inf).
  inf_mismatch = ((computed_is_inf != ref_is_inf) & ~near_overflow) | (
      both_inf & ~same_sign
  )
  inf_mismatch_val = jnp.where(
      computed_is_inf,
      jnp.where(computed_is_neg, -jnp.inf, jnp.inf),
      jnp.where(ref_is_neg, jnp.inf, -jnp.inf),
  )
  return jnp.where(
      computed_is_nan != ref_is_nan,
      jnp.inf,
      jnp.where(inf_mismatch, inf_mismatch_val, signed_ulp),
  )


def _ulp_diff_mpmath(
    computed: np.ndarray, reference: np.ndarray, dtype, ftz: bool = True
) -> np.ndarray:
  """Computes signed real ULP differences `(computed - reference) / ulp(reference)` with mpmath.

  Follows the exact same semantics as `_ulp_diff_jax` for `ulp(reference)`,
  infinity/overflow-boundary handling (`±inf` placed at `±2^(emax + 1)` when
  `reference` is finite with `|reference| >= 2^emax` and `computed` has the same
  sign), and subnormal handling (`ftz=False` gradual underflow vs. `ftz=True`
  subnormal-range contraction), evaluated in arbitrary-precision `mpmath`
  arithmetic so `float64` inputs and overflowing reference values do not suffer
  double-precision rounding or overflow.
  """
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
  overflow_val = mpmath.ldexp(1, emax + 1)
  top_binade = mpmath.ldexp(1, emax)

  def _scalar_diff(c: float, r) -> float:
    ref = (
        r
        if isinstance(r, mpmath.mpf)
        else (mpmath.nan if np.isnan(r) else mpmath.mpf(float(r)))
    )
    computed_is_nan = bool(np.isnan(c))
    ref_is_nan = bool(mpmath.isnan(ref))
    if computed_is_nan and ref_is_nan:
      return 0.0

    computed_is_inf = bool(np.isinf(c))
    abs_ref = abs(ref) if not ref_is_nan else mpmath.nan
    # A reference value that overflows the target precision is treated as Inf.
    ref_is_inf = bool(mpmath.isinf(ref)) or (
        not ref_is_nan and abs_ref >= overflow_thresh
    )

    computed_is_neg = bool(np.signbit(c))
    ref_is_neg = bool(ref < 0) if not ref_is_nan else False
    same_sign = computed_is_neg == ref_is_neg

    if computed_is_inf and ref_is_inf and same_sign:
      return 0.0
    near_overflow = (
        not bool(mpmath.isinf(ref))
        and not ref_is_nan
        and not computed_is_nan
        and same_sign
        and abs_ref >= top_binade
    )
    if computed_is_nan != ref_is_nan:
      return math.inf
    if ((computed_is_inf != ref_is_inf) and not near_overflow) or (
        computed_is_inf and ref_is_inf and not same_sign
    ):
      # Infinity mismatch: ±inf matching sign(computed - reference), treating
      # overflowing references as ±inf.
      if computed_is_inf:
        return -math.inf if computed_is_neg else math.inf
      return math.inf if ref_is_neg else -math.inf

    if computed_is_inf and near_overflow:
      mp_comp = -overflow_val if computed_is_neg else overflow_val
    else:
      mp_comp = mpmath.mpf(c)
    abs_comp = abs(mp_comp)

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
    "(+1, +1.5] ULP",
    "(+1.5, +2] ULP",
    *(f"(+{i - 1}, +{i}] ULP" for i in range(3, 11)),
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
    *(f"[-{i}, -{i - 1}) ULP" for i in range(10, 2, -1)),
    "[-2, -1.5) ULP",
    "[-1.5, -1) ULP",
    "[-1, -0.5) ULP",
    "[-0.5, 0) ULP",
)
_BIN_LABELS = (*_NEG_BIN_LABELS, "0 ULP", *_POS_BIN_LABELS)


def _map_to_bins(signed_ulp: jax.Array) -> jax.Array:
  """Maps signed real ULP distance to compact histogram bin indices."""
  ulp = jnp.abs(signed_ulp)
  decade = (
      13
      + (ulp > 100.0).astype(jnp.int32)
      + (ulp > 1000.0).astype(jnp.int32)
      + (ulp > 10000.0).astype(jnp.int32)
      + (ulp >= _MAX_ULP_BIN).astype(jnp.int32)
  )
  # Offset 0: exact 0.0
  # Offset 1: (0, 0.5] ULP
  # Offset 2: (0.5, 1] ULP
  # Offset 3: (1, 1.5] ULP
  # Offset 4..12: (1.5, 2], (2, 3], ..., (9, 10] ULP
  # Offset 13..17: decades (10, 100], ..., >=100000 ULP
  small_offset = jnp.select(
      [ulp == 0.0, ulp <= 0.5, ulp <= 1.0, ulp <= 1.5],
      [0, 1, 2, 3],
      default=jnp.ceil(ulp).astype(jnp.int32) + 2,
  )
  offset = jnp.where(ulp <= 10.0, small_offset, decade)
  pos = signed_ulp > 0.0
  zero_bin = len(_NEG_BIN_LABELS)
  return jnp.where(pos, zero_bin + offset, zero_bin - offset)


def ulp_diff(
    computed: np.ndarray,
    reference: np.ndarray,
    dtype,
    ftz: bool | None = None,
) -> np.ndarray:
  """Computes signed real float64 ULP error `(computed - reference) / ulp(reference)`.

  Positive values indicate `computed > reference`; see `_ulp_diff_jax` for
  subnormal/FTZ, overflow-boundary, and NaN/Inf mismatch conventions. For
  `float64` or object-dtype references (such as `mpmath.mpf` arrays), evaluates
  the error directly in `mpmath` without rounding `reference` to `float64`.
  """
  if ftz is None:
    ftz = _default_ftz(dtype)
  dt = np.dtype(dtype)
  ref_arr = np.asarray(reference)
  if dt == np.float64 or ref_arr.dtype == object:
    return _ulp_diff_mpmath(computed, ref_arr, dt, ftz=ftz)
  cpu_dev = jax.devices("cpu")[0]
  with jax.enable_x64(True), jax.default_device(cpu_dev):
    return np.asarray(
        _ulp_diff_jax(
            np.asarray(computed, dtype=dt).astype(np.float64),
            np.asarray(ref_arr, dtype=np.float64),
            dt,
            ftz,
        ),
        dtype=np.float64,
    )


def _flush_subnormals(x: np.ndarray, dtype) -> np.ndarray:
  """Flushes subnormal values of dtype to signed zero."""
  x = np.asarray(x, dtype=dtype)
  tiny = np.finfo(dtype).tiny
  mask = np.abs(x) < tiny
  if not np.any(mask):
    return x
  return np.where(mask, np.where(np.signbit(x), dtype(-0.0), dtype(0.0)), x)


class _FloatMpf(mpmath.mpf):
  """mpmath.mpf subclass that preserves IEEE-754 signbit of -0.0 and NaN on float()."""

  def __new__(cls, val: float):
    f = float(val)
    obj = super().__new__(cls, mpmath.nan if np.isnan(f) else f)
    obj._orig_float = f
    return obj

  def __float__(self) -> float:
    return self._orig_float


def eval_mpmath(mpmath_fn, *vals, dtype=None, input_ftz: bool | None = None):
  """Evaluates scalar mpmath function at current mpmath precision."""
  if input_ftz is None:
    input_ftz = _default_ftz(dtype)
  if input_ftz and dtype is not None:
    vals = tuple(
        _flush_subnormals(np.array(v, dtype=dtype), dtype).item()
        if jnp.issubdtype(dtype, jnp.floating)
        else v
        for v in vals
    )
  res = mpmath_fn(*(_FloatMpf(float(v)) for v in vals))
  if isinstance(res, mpmath.mpc):
    return mpmath.nan
  if isinstance(res, mpmath.ctx_mp_python.mpnumeric):
    return +res
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
    ftz: bool | None = None,
    k: int = 20,
) -> tuple[dict[str, int], list[tuple[float, ...]]]:
  """Computes signed ULP histogram counts and top-k worst cases for a chunk."""
  if ftz is None:
    ftz = _default_ftz(dtype)
  is_tuple_inputs = isinstance(inputs, (tuple, list))
  n = len(inputs[0]) if is_tuple_inputs else len(inputs)
  if n == 0:
    return {}, []
  if is_tuple_inputs:
    in_arrs = [np.asarray(a, dtype=dtype).ravel() for a in inputs]
  else:
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
    if is_tuple_inputs:
      top_k = [
          (float(ulps[i]), tuple(a[i].item() for a in in_arrs), float(comp_arr[i]),
           float(ref_f64[i]))
          for i in order
      ]
    else:
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
  if is_tuple_inputs:
    top_k = [
        (float(u), tuple(a[idx].item() for a in in_arrs), comp_arr[idx].item(), ref_f64[idx].item())
        for u, idx in zip(np.asarray(top_ulps), np.asarray(top_indices))
    ]
  else:
    top_k = [
        (float(u), in_arr[idx].item(), comp_arr[idx].item(), ref_f64[idx].item())
        for u, idx in zip(np.asarray(top_ulps), np.asarray(top_indices))
    ]
  return counts_dict, top_k


def _format_worst_cases(
    top_k, udt, mpmath_fn, dtype, input_ftz: bool = True,
) -> str:
  is_tuple = len(top_k) > 0 and isinstance(top_k[0][1], (tuple, list))
  ref_header = "mpmath" if mpmath_fn is not None else "Reference y*"
  if is_tuple:
    lines = [
        f"Top {len(top_k)} worst cases:",
        (f"{'Rank':<4} | {'ULP (real)':<11} | {'Inputs':<28} |"
         f" {'Inputs (hex)':<36} | {'Computed y':<24} | {'Nearest y*':<24} |"
         f" {ref_header}"),
        "-" * 170,
    ]
    with np.errstate(all="ignore"):
      for rank, (d, in_vals, y, y_ref) in enumerate(top_k, 1):
        in_str = "(" + ", ".join(f"{float(v):.6g}" if isinstance(v, (float, np.floating)) else str(v) for v in in_vals) + ")"
        hex_str = "(" + ", ".join(hex(int(np.array(v, dtype=dtype).view(udt))) for v in in_vals) + ")"
        y_hex = hex(int(np.array(y, dtype=dtype).view(udt)))
        ref_dt = np.array(y_ref, dtype=dtype)
        ref_hex = hex(int(ref_dt.view(udt)))

        if mpmath_fn is not None:
          mp_val = eval_mpmath(mpmath_fn, *in_vals, dtype=dtype, input_ftz=input_ftz)
          mp_exact_str = mpmath.nstr(mp_val, 30)
        else:
          mp_exact_str = str(y_ref)

        comp_str = f"{y} ({y_hex})"
        nearest_str = f"{ref_dt.item()} ({ref_hex})"
        lines.append(
            f"{rank:<4} | {d:<11.4f} | {in_str:<28} | {hex_str:<36} | {comp_str:<24}"
            f" | {nearest_str:<24} | {mp_exact_str}")
    return "\n".join(lines)

  lines = [
      f"Top {len(top_k)} worst cases:",
      (f"{'Rank':<4} | {'ULP (real)':<11} | {'Input x':<16} |"
       f" {'x (hex)':<18} | {'Computed y':<24} | {'Nearest y*':<24} |"
       f" {ref_header}"),
      "-" * 146,
  ]
  with np.errstate(all="ignore"):
    for rank, (d, x, y, y_ref) in enumerate(top_k, 1):
      x_hex = hex(int(np.array(x, dtype=dtype).view(udt)))
      y_hex = hex(int(np.array(y, dtype=dtype).view(udt)))
      ref_dt = np.array(y_ref, dtype=dtype)
      ref_hex = hex(int(ref_dt.view(udt)))

      if mpmath_fn is not None:
        mp_val = eval_mpmath(mpmath_fn, x, dtype=dtype, input_ftz=input_ftz)
        mp_exact_str = mpmath.nstr(mp_val, 30)
      else:
        mp_exact_str = str(y_ref)

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
  fn_name = getattr(jax_fn, "__name__", str(jax_fn))
  header = (
      f"Max real ULP error for {fn_name} on {variant}"
      f" ({np.dtype(dtype).name}){suffix} exceeded bound: {max_diff:.4f} >"
      f" {max_ulp}"
  )
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
  fn_name = getattr(jax_fn, "__name__", str(jax_fn))
  header = (
      f"Signed zero mismatch for {fn_name} on {variant}"
      f" ({np.dtype(dtype).name}){suffix}:"
  )
  rows = []
  for in_val, comp_val, ref_val in signed_zero_errors[:10]:
    comp_sign = "-" if np.signbit(float(comp_val)) else "+"
    ref_sign = "-" if np.signbit(float(ref_val)) else "+"
    if isinstance(in_val, (tuple, list)):
      in_str = "(" + ", ".join(repr(x) for x in in_val) + ")"
      in_b_str = (
          "("
          + ", ".join(
              f"{int(np.array(x, dtype=dtype).view(udt)):#0{itemsize * 2 + 2}x}"
              for x in in_val
          )
          + ")"
      )
      rows.append(
          f"    inputs = {in_str} {in_b_str}: expected"
          f" {ref_sign}0.0, got {comp_sign}0.0"
      )
    else:
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


def _dedup_preserving_order(arr: np.ndarray, dtype) -> np.ndarray:
  """Deduplicates `arr` by unsigned bit pattern while preserving first-seen order."""
  udt = np.dtype(f"u{np.dtype(dtype).itemsize}")
  u = arr.view(udt)
  _, first_idx = np.unique(u, return_index=True)
  return u[np.sort(first_idx)].view(dtype)


@functools.cache
def _common_interesting_points(dtype) -> tuple[float, ...]:
  """Returns common interesting floating-point domain points for `dtype`."""
  dt = np.dtype(dtype).type
  finfo = np.finfo(dtype)
  with np.errstate(all="ignore"):
    min_subnormal = float(np.nextafter(dt(0.0), dt(1.0)))
    max_subnormal = float(np.nextafter(dt(finfo.tiny), dt(0.0)))
    tiny = float(finfo.tiny)
    fmax = float(finfo.max)

  pos_magnitudes = [
      # Smallest and largest subnormal and normal values:
      min_subnormal,
      max_subnormal,
      tiny,
      fmax,
      # Overflow/underflow boundaries for x^2, x^3, hypot, and roots:
      math.sqrt(tiny),
      math.sqrt(fmax),
      math.cbrt(tiny),
      math.cbrt(fmax),
      # Overflow/underflow boundaries for exp and exp2:
      math.log(fmax),
      -math.log(tiny),
      -math.log(min_subnormal),
      math.log2(fmax),
      -math.log2(tiny),
      -math.log2(min_subnormal),
      # Small integers and simple dyadic fractions:
      0.25,
      0.5,
      0.75,
      1.0,
      1.5,
      2.0,
      2.5,
      3.0,
      4.0,
      5.0,
      6.0,
      7.0,
      8.0,
      9.0,
      10.0,
      # Common mathematical constants:
      math.e,
      1.0 / math.e,
      0.5 * math.log(2.0),
      math.log(2.0),
      math.log(10.0),
      math.log2(math.e),
      math.log10(math.e),
      math.sqrt(2.0),
      math.sqrt(0.5),
      math.sqrt(3.0),
      math.sqrt(math.pi),
      2.0 / math.sqrt(math.pi),
  ]

  # Points approaching 1 from below and above (1 +- 2^-k):
  for k in range(1, finfo.nmant + 1):
    pos_magnitudes.append(1.0 - math.ldexp(1.0, -k))
    pos_magnitudes.append(1.0 + math.ldexp(1.0, -k))

  # Multiples of pi (quarter- and sixth-multiples near zero, plus large
  # multiples for trigonometric range reduction).
  for k in range(1, 17):
    pos_magnitudes.append(k * (math.pi / 4.0))
  for k in (1, 2, 4, 5):
    pos_magnitudes.append(k * (math.pi / 6.0))
  for k in range(1, 7):
    pos_magnitudes.append((10.0**k) * math.pi)
  for k in (10, 20, 30, 50):
    pos_magnitudes.append(math.ldexp(math.pi, k))

  # Powers of two: all exponents in [-32, 32], plus exponents across the
  # full normal and subnormal range of dtype.
  min_exp = finfo.minexp - finfo.nmant
  max_exp = finfo.maxexp - 1
  exp_step = max(1, (max_exp - min_exp) // 32)
  exponents = (
      set(range(max(-32, min_exp), min(32, max_exp) + 1))
      | set(range(min_exp, max_exp + 1, exp_step))
      | {
          min_exp,
          -finfo.maxexp,
          finfo.minexp - 1,
          finfo.minexp,
          finfo.minexp + 1,
          -(max_exp // 2),
          -finfo.nmant - 1,
          -finfo.nmant,
          -(finfo.nmant // 2),
          finfo.nmant // 2,
          finfo.nmant,
          finfo.nmant + 1,
          max_exp // 2,
          max_exp - 1,
          max_exp,
      }
  )
  for k in sorted(exponents):
    if min_exp <= k <= max_exp:
      pos_magnitudes.append(math.ldexp(1.0, k))

  pts = [0.0, -0.0, math.inf, -math.inf, math.nan]
  for v in pos_magnitudes:
    if v <= fmax:
      pts.append(v)
      pts.append(-v)
  return tuple(pts)


def _with_ulp_neighbors(
    vals: Sequence[float],
    dtype,
    radius: int = 2,
    max_points: int | None = None,
    include_specials: bool = True,
) -> np.ndarray:
  """Returns `vals` and the floats within `radius` ULPs of each value."""
  fmax = float(np.finfo(dtype).max)
  valid = [
      fv
      for v in vals
      if not math.isnan(fv := float(v)) and (math.isinf(fv) or abs(fv) <= fmax)
  ]
  with np.errstate(all="ignore"):
    arr = _dedup_preserving_order(np.asarray(valid, dtype=dtype), dtype)
    neg_inf = np.asarray(-np.inf, dtype=dtype)
    pos_inf = np.asarray(np.inf, dtype=dtype)
    if len(arr) > 0:
      cols = [arr]
      lo = hi = arr
      for _ in range(radius):
        lo = np.nextafter(lo, neg_inf)
        hi = np.nextafter(hi, pos_inf)
        cols.extend([lo, hi])
      expanded = np.column_stack(cols).ravel()
    else:
      expanded = arr
    if include_specials:
      specials = np.array([0.0, -0.0, np.inf, -np.inf, np.nan], dtype=dtype)
      expanded = np.concatenate([specials, expanded])
    out = _dedup_preserving_order(expanded, dtype)
  if max_points is not None and len(out) > max_points:
    out = out[:max_points]
  return out


class Domain:
  """A set of floating-point bit patterns of a given `dtype`."""

  # Floating-point dtype of the values in this domain (e.g. float16, float32).
  dtype: np.dtype
  # Unsigned integer dtype of the same bit width, used to view bit patterns.
  bits_dtype: np.dtype
  # Sorted, disjoint half-open `[start, stop)` unsigned bit-pattern intervals.
  ranges: tuple[tuple[int, int], ...]

  def __init__(self, dtype, ranges: Iterable[tuple[int, int]]):
    self.dtype = np.dtype(dtype)
    self.bits_dtype = np.dtype(f"u{self.dtype.itemsize}")
    limit = 1 << (self.dtype.itemsize * 8)
    merged: list[tuple[int, int]] = []
    for s, e in sorted((s, e) for s, e in ranges if s < e):
      if s < 0 or e > limit:
        raise ValueError(
            f"Bit range ({s}, {e}) out of bounds for {self.dtype.name}."
        )
      if merged and s <= merged[-1][1]:
        merged[-1] = (merged[-1][0], max(merged[-1][1], e))
      else:
        merged.append((s, e))
    if not merged:
      raise ValueError(f"Domain contains no {self.dtype.name} bit patterns.")
    self.ranges = tuple(merged)

  @property
  def size(self) -> int:
    """Returns the total number of distinct bit patterns in this domain."""
    return sum(e - s for s, e in self.ranges)

  @property
  def is_all(self) -> bool:
    """Returns True if this domain covers all bit patterns of `self.dtype`."""
    return self.ranges == ((0, 1 << (self.dtype.itemsize * 8)),)

  def __eq__(self, other: object) -> bool:
    return (
        isinstance(other, Domain)
        and self.dtype == other.dtype
        and self.ranges == other.ranges
    )

  def __or__(self, other: "Domain") -> "Domain":
    """Returns a `Domain` representing the union of `self` and `other`."""
    if self.dtype != other.dtype:
      raise ValueError(
          f"Cannot union Domains with dtypes {self.dtype} and {other.dtype}."
      )
    return Domain(self.dtype, (*self.ranges, *other.ranges))

  def contains(self, a: np.ndarray) -> np.ndarray:
    """Returns a boolean mask indicating which elements of `a` are in `self`."""
    u = np.asarray(a, dtype=self.dtype).view(self.bits_dtype)
    return np.logical_or.reduce([(u >= s) & (u < e) for s, e in self.ranges])

  def at(self, idx: np.ndarray) -> np.ndarray:
    """Maps 0-based indices `idx` into `self.ranges` as a `self.dtype` array."""
    idx_u = np.asarray(idx, dtype=np.uint64)
    out = np.empty(idx_u.shape, dtype=self.bits_dtype)
    offset = 0
    for s, e in self.ranges:
      n = e - s
      lo = np.uint64(offset)
      hi = np.uint64(offset + n - 1)
      m = (idx_u >= lo) & (idx_u <= hi)
      out[m] = (np.uint64(s) + (idx_u[m] - lo)).astype(self.bits_dtype)
      offset += n
    return out.view(self.dtype)

  def sample(self, rng: np.random.RandomState, n: int) -> np.ndarray:
    """Uniformly samples `n` values of `self.dtype` from `self.ranges`."""
    if self.is_all:
      return jtu.rand_fullrange(rng)((n,), self.dtype)
    counts = np.array([e - s for s, e in self.ranges], dtype=np.float64)
    which = rng.choice(len(self.ranges), size=n, p=counts / counts.sum())
    out = np.empty(n, dtype=self.bits_dtype)
    for r_idx, (s, e) in enumerate(self.ranges):
      m = which == r_idx
      k = rng.randint(0, np.uint64(e - s), size=int(m.sum()), dtype=np.uint64)
      out[m] = (np.uint64(s) + k).astype(self.bits_dtype)
    return out.view(self.dtype)

  @staticmethod
  def all(dtype) -> "Domain":
    """Returns a `Domain` covering all bit patterns of `dtype`."""
    dt = np.dtype(dtype)
    return Domain(dt, [(0, 1 << (dt.itemsize * 8))])

  @staticmethod
  def interval(dtype, lo: float, hi: float) -> "Domain":
    """Returns a `Domain` of all `dtype` floats `x` with `lo <= x <= hi`.

    Includes both `-0.0` and `+0.0` whenever `0` lies in `[lo, hi]`.
    """
    lo, hi = float(lo), float(hi)
    if math.isnan(lo) or math.isnan(hi) or lo > hi:
      raise ValueError(f"Invalid interval [{lo}, {hi}]: expected lo <= hi.")
    dt = np.dtype(dtype)
    bits_dtype = np.dtype(f"u{dt.itemsize}")
    fmax = float(np.finfo(dt).max)
    to_bits = lambda x: int(np.asarray(x, dtype=dt).view(bits_dtype))
    to_float = lambda u: float(np.asarray(u, dtype=bits_dtype).view(dt))
    ranges = []
    # Clamp finite bounds to [-fmax, fmax] and step by 1 ULP if rounding
    # to `dt` stepped outside [lo, hi].
    if hi >= 0:
      lo_pos = np.inf if lo > fmax else max(0.0, lo)
      hi_pos = hi if math.isinf(hi) else min(hi, fmax)
      if lo_pos <= hi_pos:
        u_lo = to_bits(lo_pos)
        u_hi = to_bits(0.0 if hi_pos == 0 else hi_pos)
        u_lo += to_float(u_lo) < lo
        u_hi -= to_float(u_hi) > hi
        if u_lo <= u_hi:
          ranges.append((u_lo, u_hi + 1))
    if lo <= 0:
      hi_neg = -np.inf if hi < -fmax else (-0.0 if hi >= 0 else hi)
      lo_neg = lo if math.isinf(lo) else max(lo, -fmax)
      if lo_neg <= hi_neg:
        u_hi = to_bits(hi_neg)
        u_lo = to_bits(-0.0 if lo_neg == 0 else lo_neg)
        u_hi += to_float(u_hi) > hi
        u_lo -= to_float(u_lo) < lo
        if u_hi <= u_lo:
          ranges.append((u_hi, u_lo + 1))
    return Domain(dt, ranges)

  @staticmethod
  def points(dtype, *values: float) -> "Domain":
    """Returns a `Domain` containing specific `dtype` values."""
    dt = np.dtype(dtype)
    bits_dtype = np.dtype(f"u{dt.itemsize}")
    ranges = []
    with np.errstate(over="ignore"):
      for v in values:
        b = int(np.asarray(v, dtype=dt).view(bits_dtype))
        ranges.append((b, b + 1))
    return Domain(dt, ranges)

  @staticmethod
  def infs(dtype) -> "Domain":
    """Returns a `Domain` containing `-inf` and `+inf` of `dtype`."""
    return Domain.points(dtype, -np.inf, np.inf)

  @staticmethod
  def nans(dtype) -> "Domain":
    """Returns a `Domain` containing all IEEE NaN bit patterns of `dtype`."""
    dt = np.dtype(dtype)
    sign = 1 << (dt.itemsize * 8 - 1)
    bits_dtype = np.dtype(f"u{dt.itemsize}")
    pos_inf = int(np.asarray(np.inf, dtype=dt).view(bits_dtype))
    return Domain(dt, [(pos_inf + 1, sign), (sign + pos_inf + 1, sign << 1)])

  @staticmethod
  def infs_and_nans(dtype) -> "Domain":
    """Returns a `Domain` containing `±inf` and all NaN bit patterns."""
    return Domain.infs(dtype) | Domain.nans(dtype)


DomainLike = Domain | Callable[[Any], Domain]


def _make_interesting_samples(
    nargs: int,
    dtype,
    rng: np.random.RandomState,
    max_points: int,
    interesting_points: Sequence[Any] | None = None,
    domains: Sequence[Domain] = (),
) -> tuple[np.ndarray, ...]:
  """Generates up to `max_points` test inputs at and near interesting values.

  For a 1-argument function, this returns the common interesting values for
  `dtype` (such as 0, +-1, +-inf, nan, and smallest/largest normal and
  subnormal values), any extra values from `interesting_points`, and the floats
  within 3 ULPs of each.

  For a multi-argument function, testing every combination of 1D interesting
  values would produce too many inputs. Instead, we combine:
    1. Any explicit input tuples in `interesting_points` (and values within 1
       ULP of each coordinate).
    2. All combinations of a small list of key boundary values (0,
       +-min_subnormal, +-tiny, +-0.5, +-1, +-2, +-max_float, +-inf, nan, and up
       to 8 custom scalars).
    3. Tuples where arguments are equal or 1 ULP apart, up to sign (`y = +-x`
       or `y = +-(x +- 1 ULP)`).
    4. Tuples where one argument is an interesting 1D value and the other
       arguments are random floats.

  Args:
    nargs: Number of input arguments to the function under test.
    dtype: Floating-point dtype of the generated arrays.
    rng: Random number generator.
    max_points: Maximum number of test inputs to return.
    interesting_points: Optional extra test values. Each entry can be a scalar
      float or a tuple/list of `nargs` floats.
    domains: Optional sequence of `nargs` `Domain`s restricting the generated
      values for each argument.

  Returns:
    A tuple of `nargs` 1D arrays of `dtype`, each of length at most
    `max_points`.
  """
  if max_points <= 0:
    return tuple(np.empty((0,), dtype=dtype) for _ in range(nargs))

  custom_scalars: list[float] = []
  custom_tuples: list[tuple[float, ...]] = []
  for item in interesting_points or ():
    if isinstance(item, (tuple, list)):
      if len(item) != nargs:
        raise ValueError(
            f"Tuple in interesting_points must have length nargs={nargs}, got"
            f" {item!r}."
        )
      custom_tuples.append(tuple(float(x) for x in item))
    else:
      custom_scalars.append(float(item))

  # Include the finite start and end values of each domain range so sampled runs
  # probe at and within 3 ULPs of domain boundaries.
  domains = domains or (Domain.all(dtype),) * nargs
  for d in domains:
    boundary_bits = [b for start, stop in d.ranges for b in (start, stop - 1)]
    boundary_vals = np.array(boundary_bits, dtype=d.bits_dtype).view(d.dtype)
    custom_scalars.extend(float(x) for x in boundary_vals if np.isfinite(x))

  # Put caller-supplied scalars first so they are kept if `max_points` is small.
  vals_1d = _with_ulp_neighbors(
      (*custom_scalars, *_common_interesting_points(dtype)),
      dtype,
      radius=3,
      max_points=None if nargs == 1 else max(256, max_points // (2 * nargs)),
  )
  if nargs == 1:
    return (vals_1d[domains[0].contains(vals_1d)][:max_points],)

  finfo = np.finfo(dtype)
  fmax = float(finfo.max)
  dt = np.dtype(dtype).type
  with np.errstate(all="ignore"):
    min_sub = float(np.nextafter(dt(0.0), dt(1.0)))
    tiny = float(finfo.tiny)
    neg_inf = dt(-np.inf)
    pos_inf = dt(np.inf)

  coords: list[list[np.ndarray]] = [[] for _ in range(nargs)]

  def _add_grid(axes_1d: Sequence[np.ndarray]):
    for i, g in enumerate(np.meshgrid(*axes_1d, indexing="ij")):
      coords[i].append(g.ravel())

  # 1. Explicit tuples from interesting_points, plus +-1 ULP on each coordinate.
  for tup in custom_tuples:
    if all(math.isnan(x) or math.isinf(x) or abs(x) <= fmax for x in tup):
      axes = [
          np.asarray([x], dtype=dtype)
          if math.isnan(x)
          else _with_ulp_neighbors([x], dtype, radius=1, include_specials=False)
          for x in tup
      ]
      _add_grid(axes)

  # 2. All combinations of a small set of boundary values.
  max_cart = max(4, int(round((max_points // 2) ** (1.0 / nargs))))
  core_1d = _with_ulp_neighbors(
      custom_scalars[:8]
      + [0.0, min_sub, tiny, 0.5, 1.0, 2.0, fmax, -min_sub, -tiny, -0.5, -1.0, -2.0, -fmax],
      dtype,
      radius=1,
      max_points=max_cart,
  )
  _add_grid([core_1d] * nargs)
  tier1_len = sum(len(c) for c in coords[0])

  # 3. Tuples where arguments are equal or 1 ULP apart (up to sign).
  non_nan = vals_1d[~np.isnan(vals_1d)]
  with np.errstate(all="ignore"):
    for shifted in (non_nan, np.nextafter(non_nan, pos_inf), np.nextafter(non_nan, neg_inf)):
      for sign in (1, -1):
        coords[0].append(non_nan)
        for i in range(1, nargs):
          coords[i].append(sign * shifted)

  # 4. Tuples where one argument is an interesting 1D value and the rest are random.
  for fixed_axis in range(nargs):
    fixed_vals = vals_1d[domains[fixed_axis].contains(vals_1d)]
    for i in range(nargs):
      coords[i].append(
          fixed_vals
          if i == fixed_axis
          else domains[i].sample(rng, len(fixed_vals))
      )

  out = [np.concatenate(c) for c in coords]
  valid = np.logical_and.reduce([d.contains(a) for a, d in zip(out, domains)])
  tier1_len = int(np.sum(valid[:tier1_len]))
  out = [a[valid] for a in out]
  if len(out[0]) > max_points:
    if tier1_len < max_points:
      perm = tier1_len + rng.permutation(len(out[0]) - tier1_len)
      keep = np.concatenate([np.arange(tier1_len), perm[: max_points - tier1_len]])
      out = [a[keep] for a in out]
    else:
      out = [a[:max_points] for a in out]
  return tuple(out)


def _make_exhaustive_chunk(
    start: int,
    count: int,
    nargs: int,
    dtype,
    domains: Sequence[Domain] = (),
) -> tuple[np.ndarray, ...]:
  """Generates `count` exhaustive n-ary bit-pattern tuples from `start`."""
  domains = domains or (Domain.all(dtype),) * nargs
  radices = tuple(d.size for d in domains)
  r0 = radices[0]
  if nargs == 2 and start % r0 == 0 and count % r0 == 0:
    n_rows = count // r0
    row0 = domains[0].at(np.arange(r0, dtype=np.intp))
    col1 = domains[1].at(
        np.arange(start // r0, start // r0 + n_rows, dtype=np.intp)
    )
    return (np.tile(row0, n_rows), np.repeat(col1, r0))
  coords = np.unravel_index(
      np.arange(start, start + count, dtype=np.intp), radices, order="F"
  )
  return tuple(d.at(idx) for idx, d in zip(coords, domains))


def check_nary_precision(
    test_case,
    jax_fn: Callable,
    ref_fn: Callable,
    mpmath_fn: Callable | None = None,
    dtype=None,
    nargs: int = 2,
    bounds: float | tuple[float, float] | list | None = None,
    input_ftz: bool | list | None = None,
    output_ftz: bool | list | None = None,
    ignore_inputs: list | None = None,
    check_signed_zeros: bool | list = True,
    ref_dtype: object | None = None,
    max_samples: int | None = None,
    interesting_points: Sequence[Any] | None = None,
    domains: Sequence[DomainLike | None] | None = None,
):
  """Checks precision of n-ary `jax_fn` against reference implementations.

  Evaluates `jax_fn` across either all possible bit-pattern combinations of
  `dtype` (when `total_elements <= MAX_SAMPLES`, e.g. unary `bfloat16` and
  `float16` by default, or unary `float32` and binary `bfloat16`/`float16` when
  `--jax_numerics_max_samples=4294967296`) or a sample of `MAX_SAMPLES`
  (`MAX_F64_SAMPLES` for `float64`) inputs consisting of values at and near
  interesting points (such as 0, +-1, +-inf, smallest/largest normal and
  subnormal values, poles, and approximation thresholds) together with random
  floating-point values across all exponents.

  Args:
    test_case: The `jtu.JaxTestCase` instance running the test.
    jax_fn: The JAX function under test (e.g. `jnp.atan2`).
    ref_fn: The vectorized NumPy/SciPy reference function operating on float64
      arrays (or `ref_dtype` arrays if `ref_dtype` is specified).
    mpmath_fn: Optional `mpmath` reference function used to compute
      high-precision reference values and real ULP errors with mpmath (used for
      `float64` and printed as part of the logging for the top K worst cases).
      If omitted or None, `ref_fn` is used as the reference across all dtypes
      including float64.
    dtype: Floating-point dtype to test (`bfloat16`, `float16`, `float32`,
      `float64`).
    nargs: Number of input arguments to `jax_fn`.
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
      evaluation (bool or per-variant override list). Defaults to False on GPU
      and for float16 (which does not flush subnormals on CPU, GPU, or TPU),
      and True otherwise.
    output_ftz: Whether subnormal outputs are flushed to zero when computing ULP
      distances (bool or per-variant override list). Defaults to False on GPU
      and for float16, and True otherwise.
    ignore_inputs: Optional per-variant list of callable predicates
      `ignore_fn(*args: np.ndarray) -> np.ndarray[bool]` where True indicates
      that the input should be excluded from accuracy checking.
    check_signed_zeros: Whether to verify that the sign of zero matches the
      reference when the computed and reference values are both zero (bool or
      per-variant override list).
    ref_dtype: Optional dtype for evaluating `ref_fn` before casting to float64
      (useful for discrete operations like `nextafter` where rounding must occur
      in the target dtype).
    max_samples: Optional override for the maximum number of sample inputs to
      test. Defaults to `MAX_F64_SAMPLES.value` for float64 and
      `MAX_SAMPLES.value` otherwise.
    interesting_points: Optional sequence of additional function-specific
      test points (scalars, or `nargs`-tuples for n-ary functions) to sample at
      and near during non-exhaustive runs.
    domains: Optional sequence of `nargs` `Domain`s, `dtype -> Domain`
      callables, or `None` (for `Domain.all(dtype)`) restricting the bit
      patterns tested for each argument.
  """
  if dtype is None:
    dtype = mpmath_fn
    mpmath_fn = None

  if ((dtype == jnp.float64 or dtype == np.float64)
      and jtu.device_under_test() == "tpu"):
    test_case.skipTest("float64 on TPU is ef57 double-double")

  variant = get_hardware_variant()
  default_ftz = _default_ftz(dtype)
  in_ftz = _resolve_override(input_ftz, variant, dtype, default_ftz)
  out_ftz = _resolve_override(output_ftz, variant, dtype, default_ftz)
  chk_signed_zeros = _resolve_override(check_signed_zeros, variant, dtype, True)
  ignore_predicate = resolve_ignore_inputs(ignore_inputs, variant, dtype)
  raw_bound = _resolve_override(bounds, variant, dtype, 0.5)
  if isinstance(raw_bound, tuple):
    min_ulp, max_ulp = float(raw_bound[0]), float(raw_bound[1])
  else:
    min_ulp = max_ulp = float(raw_bound)

  is_f64 = dtype == jnp.float64 or dtype == np.float64
  itemsize = np.dtype(dtype).itemsize
  udt = np.dtype(f"u{itemsize}")

  def _resolve_domain(d: DomainLike | None) -> Domain:
    if d is None:
      return Domain.all(dtype)
    resolved = d(dtype) if callable(d) else d
    if resolved.dtype != np.dtype(dtype):
      raise ValueError(
          f"Domain dtype {resolved.dtype} does not match {np.dtype(dtype)}."
      )
    return resolved

  if domains is not None and len(domains) != nargs:
    raise ValueError(
        f"Expected len(domains) == nargs ({nargs}), got {len(domains)}."
    )
  resolved_domains = tuple(
      _resolve_domain(d) for d in (domains or (None,) * nargs)
  )

  if max_samples is None:
    max_samples = MAX_F64_SAMPLES.value if is_f64 else MAX_SAMPLES.value

  total_elements = math.prod(d.size for d in resolved_domains)
  is_exhaustive = total_elements <= max_samples
  total_points = min(total_elements, max_samples)

  k = NUM_WORST_CASES.value
  eval_k = max(k, 1)

  def _compute_reference(chunk_inputs: tuple[np.ndarray, ...]):
    if is_f64 and mpmath_fn is not None:
      return np.array(
          [
              eval_mpmath(mpmath_fn, *vals, dtype=dtype, input_ftz=in_ftz)
              for vals in zip(*chunk_inputs)
          ],
          dtype=object,
      )
    ref_inputs = tuple(
        _flush_subnormals(a, dtype) if in_ftz else a for a in chunk_inputs
    )
    if ref_dtype is not None:
      return np.asarray(
          ref_fn(*(a.astype(ref_dtype) for a in ref_inputs)), dtype=np.float64
      )
    return ref_fn(*(a.astype(np.float64) for a in ref_inputs))

  # Process in power-of-two chunks of at most 64 MB across all input arrays to
  # bound concurrent memory usage across worker threads during exhaustive runs
  # while keeping chunk sizes divisible by 16384 for the pruned top_k kernel.
  raw_chunk_size = (64 * 1024 * 1024) // (itemsize * max(1, nargs))
  chunk_size = 1 << max(14, raw_chunk_size.bit_length() - 1)

  if is_exhaustive:
    label = "exhaustive"
    chunks = [
        lambda s=start, c=min(chunk_size, total_points - start): (
            _make_exhaustive_chunk(s, c, nargs, dtype, resolved_domains)
        )
        for start in range(0, total_points, chunk_size)
    ]
  else:
    label = f"sampled {total_points} points"
    base_rng = test_case.rng()
    max_interesting_cap = 4096 if is_f64 else 32768
    chunks = []
    for start in range(0, total_points, chunk_size):
      count = min(chunk_size, total_points - start)
      seed = int(base_rng.randint(0, 1 << 31))
      is_first_chunk = start == 0

      def _make_chunk(c=count, s=seed, first=is_first_chunk):
        rng = np.random.RandomState(s)
        if first:
          interesting_budget = min(max_interesting_cap, c // 2)
          int_samples = _make_interesting_samples(
              nargs,
              dtype,
              rng,
              interesting_budget,
              interesting_points,
              resolved_domains,
          )
          n_rand = c - len(int_samples[0])
          rand_samples = tuple(
              d.sample(rng, n_rand) for d in resolved_domains
          )
          return tuple(
              np.concatenate([int_a, rand_a])
              for int_a, rand_a in zip(int_samples, rand_samples)
          )
        return tuple(d.sample(rng, c) for d in resolved_domains)

      chunks.append(_make_chunk)

  def _eval_chunk(make_chunk):
    with np.errstate(all="ignore"):
      chunk_inputs = make_chunk()
      if ignore_predicate is not None:
        mask = np.asarray(ignore_predicate(*chunk_inputs), dtype=bool)
        if (frac_ignored := mask.mean()) > 0.5:
          raise ValueError(
              f"ignore_predicate ignores too many inputs: {frac_ignored:.2f}."
              " tests/numerics is designed to validate functions across the"
              " full input domain, and is inefficient when a large fraction of"
              " inputs are ignored. Please reduce the ignored domain, or avoid"
              " tests/numerics if this is not possible."
          )
        idxs = np.flatnonzero(mask)
        if len(idxs) > 0:
          chunk_inputs = tuple(a.copy() for a in chunk_inputs)
          for a in chunk_inputs:
            a[idxs] = np.nan
      else:
        idxs = ()

      chunk_computed = np.asarray(jax_fn(*chunk_inputs))
      chunk_ref = _compute_reference(chunk_inputs)

      if len(idxs) > 0:
        chunk_computed = chunk_computed.copy()
        chunk_computed[idxs] = np.nan
        chunk_ref[idxs] = (
            mpmath.nan if (is_f64 and mpmath_fn is not None) else np.nan
        )

      signed_zero_errors = []
      if chk_signed_zeros:
        if is_f64 and mpmath_fn is not None:
          # mpmath.mpf has no signed zero (-0.0) representation, so fall back
          # to ref_fn to check the sign of zero.
          comp_zeros = np.flatnonzero(chunk_computed == 0.0)
          if len(comp_zeros) > 0:
            in_z = tuple(a[comp_zeros] for a in chunk_inputs)
            ref_in_z = tuple(
                _flush_subnormals(a, dtype) if in_ftz else a for a in in_z
            )
            ref_at_zeros = ref_fn(*(a.astype(np.float64) for a in ref_in_z))
            bad_mask = (ref_at_zeros == 0.0) & (
                np.signbit(chunk_computed[comp_zeros])
                != np.signbit(ref_at_zeros)
            )
            for z_idx in np.flatnonzero(bad_mask):
              idx = comp_zeros[z_idx]
              err_input = (
                  chunk_inputs[0][idx]
                  if nargs == 1
                  else tuple(a[idx] for a in chunk_inputs)
              )
              signed_zero_errors.append(
                  (err_input, chunk_computed[idx], ref_at_zeros[z_idx])
              )
        else:
          mismatches = np.flatnonzero(
              (chunk_computed == 0.0)
              & (chunk_ref == 0.0)
              & (np.signbit(chunk_computed) != np.signbit(chunk_ref))
          )
          for idx in mismatches:
            err_input = (
                chunk_inputs[0][idx]
                if nargs == 1
                else tuple(a[idx] for a in chunk_inputs)
            )
            signed_zero_errors.append(
                (err_input, chunk_computed[idx], chunk_ref[idx])
            )

      chunk_counts, chunk_top_k = eval_ulp_stats(
          chunk_inputs[0] if nargs == 1 else chunk_inputs,
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

  ignored_str = ", ignored custom inputs" if ignore_predicate is not None else ""
  hist_str = render_histogram_from_counts(counts_dict, total_points)
  worst_cases_str = _format_worst_cases(
      top_k, udt, mpmath_fn, dtype, input_ftz=in_ftz
  )
  fn_name = getattr(jax_fn, "__name__", str(jax_fn))
  output = (
      f"[{variant}] {fn_name} ({np.dtype(dtype).name}): max real ULP"
      f" error = {max_diff:.4f} (bound = {max_ulp},"
      f" {label}{ignored_str})\n{hist_str}\n{worst_cases_str}\n"
  )
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
  elif (
      is_exhaustive
      and not np.isinf(max_ulp)
      and not (min_ulp <= expected_bound <= max_ulp)
  ):
    bound_str = (
        f"({min_ulp}, {max_ulp})" if min_ulp != max_ulp else f"{max_ulp}"
    )
    test_case.fail(
        f"ULP bound for {fn_name} on {variant} ({np.dtype(dtype).name})"
        " is not tight in exhaustive run: observed max real ULP error"
        f" {max_diff:.4f} (expected bound {expected_bound}, got {bound_str})."
    )


def check_unary_precision(
    test_case,
    jax_fn: Callable,
    ref_fn: Callable,
    mpmath_fn: Callable | None = None,
    dtype=None,
    bounds: float | tuple[float, float] | list | None = None,
    input_ftz: bool | list | None = None,
    output_ftz: bool | list | None = None,
    ignore_inputs: list | None = None,
    check_signed_zeros: bool | list = True,
    max_samples: int | None = None,
    interesting_points: Sequence[Any] | None = None,
    domain: DomainLike | None = None,
):
  """Checks unary precision of `jax_fn` against reference implementations.

  This is a convenience wrapper for `check_nary_precision(..., nargs=1)`.
  See `check_nary_precision` for detailed parameter documentation, bounds
  quantization semantics, and tightness checks.
  """
  if dtype is None:
    dtype = mpmath_fn
    mpmath_fn = None
  check_nary_precision(
      test_case,
      jax_fn,
      ref_fn,
      mpmath_fn,
      dtype,
      nargs=1,
      bounds=bounds,
      input_ftz=input_ftz,
      output_ftz=output_ftz,
      ignore_inputs=ignore_inputs,
      check_signed_zeros=check_signed_zeros,
      max_samples=max_samples,
      interesting_points=interesting_points,
      domains=(domain,) if domain is not None else None,
  )


def check_precision_at_points(
    test_case,
    jax_fn: Callable,
    mpmath_fn: Callable,
    *args: jax.Array,
    max_ulp: float,
    input_ftz: bool | None = None,
    output_ftz: bool | None = None,
):
  """Asserts `jax_fn(*args)` is within `max_ulp` of `mpmath_fn(*args)` elementwise.

  Args:
    test_case: The `jtu.JaxTestCase` instance running the test.
    jax_fn: The JAX function under test, vectorized to accept array inputs.
    mpmath_fn: The scalar `mpmath` reference function.
    *args: Array inputs to `jax_fn`, all of the same shape and floating-point
      dtype. `mpmath_fn` is evaluated elementwise over the zipped flattened
      elements of these arrays.
    max_ulp: The maximum allowed real ULP distance between the computed JAX
      output and the `mpmath` reference.
    input_ftz: Whether to flush subnormal inputs to zero before evaluating
      `mpmath_fn`. Defaults to False on GPU and for float16, and True otherwise.
    output_ftz: Whether to flush subnormal outputs to zero when computing ULP
      distances. Defaults to False on GPU and for float16, and True otherwise.
  """
  res = np.asarray(jax_fn(*args))
  dtype = res.dtype.type
  is_f64 = dtype == np.float64
  in_dtype = args[0].dtype.type if args else dtype
  np_args = [np.asarray(a).ravel() for a in args]
  with mpmath.extraprec(80):
    ref = np.array(
        [
            eval_mpmath(
                mpmath_fn,
                *(v.item() for v in vals),
                dtype=in_dtype,
                input_ftz=input_ftz,
            )
            for vals in zip(*np_args)
        ],
        dtype=object if is_f64 else np.float64,
    ).reshape(res.shape)
  ulps = np.abs(ulp_diff(res, ref, dtype, ftz=output_ftz))
  max_diff = float(np.max(ulps))
  if max_diff > max_ulp or np.isnan(max_diff):
    idx = int(np.nanargmax(ulps) if not np.all(np.isnan(ulps)) else 0)
    pt = tuple(a[idx].item() for a in np_args)
    pt_str = pt[0] if len(pt) == 1 else pt
    test_case.fail(
        f"ULP error {max_diff} exceeds bound {max_ulp} at input {pt_str}: "
        f"computed {res.ravel()[idx]}, reference {ref.ravel()[idx]}"
    )


def register_benchmark(
    jax_fn: Callable,
    *,
    name: str | None = None,
    dtypes: Sequence[Any] = (
        jnp.bfloat16,
        jnp.float16,
        jnp.float32,
        jnp.float64,
    ),
    nargs: int = 1,
    size: int = 10**6,
) -> None:
  """Registers google_benchmark execution and compilation benchmarks for `jax_fn` across `dtypes`."""
  if google_benchmark is None:
    return
  fn_name = name or getattr(jax_fn, "__name__", str(jax_fn)).lstrip("_")
  for dtype in dtypes:
    dtype = np.dtype(dtype)

    def _bench_exec(state, dtype=dtype):
      is_f64 = dtype == np.float64
      if is_f64 and jtu.device_under_test() == "tpu":
        state.skip_with_error("float64 on TPU is ef57 double-double")
        return
      with jax.enable_x64(is_f64):
        rng = jtu.rand_fullrange(np.random.RandomState(0))
        args = tuple(jax.device_put(rng((size,), dtype)) for _ in range(nargs))
        f = jax.jit(jax_fn)
        f(*args).block_until_ready()
        while state:
          f(*args).block_until_ready()

    def _bench_compile(state, dtype=dtype):
      is_f64 = dtype == np.float64
      if is_f64 and jtu.device_under_test() == "tpu":
        state.skip_with_error("float64 on TPU is ef57 double-double")
        return
      with jax.enable_x64(is_f64):
        rng = jtu.rand_fullrange(np.random.RandomState(0))
        args = tuple(jax.device_put(rng((size,), dtype)) for _ in range(nargs))
        f = jax.jit(jax_fn)
        f.lower(*args).compile()
        while state:
          state.pause_timing()
          jax.clear_caches()
          state.resume_timing()
          f.lower(*args).compile()

    google_benchmark.register(_bench_exec, name=f"{fn_name}_{dtype.name}")
    google_benchmark.register(
        _bench_compile, name=f"{fn_name}_{dtype.name}_compile"
    )


def main() -> None:
  """Runs google_benchmark if `--benchmark*` flags are passed, else absltest."""
  if google_benchmark is not None and any(
      arg.startswith("--benchmark") for arg in sys.argv[1:]
  ):
    shard_status_file = os.environ.get("TEST_SHARD_STATUS_FILE")
    if shard_status_file:
      with open(shard_status_file, "w"):
        pass
    if int(os.environ.get("TEST_SHARD_INDEX", "0")) != 0:
      return
    google_benchmark.main()
  else:
    absltest.main(testLoader=ClassShardedTestLoader())
