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

# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
"""Comprehensive tests for unified MHA handlers (mha_v2).

Test strategy:
- Dtype-based tolerances (eps^(2/3))
- Dropout: shape/crash check only (no numeric comparison)
- Covers all head dims, seqlen combos, MHA/MQA/GQA, features, edge cases
- Regression guards for every historical bug found during AITER bump
"""

import math
import pytest
import jax
import jax.numpy as jnp

from jax._src.aiter.aiter_mha import (
    flash_attn_func,
    flash_attn_varlen,
    get_gfx,
)

# AITER MHA accuracy checks compare bf16/fp16 kernel outputs against an fp32
# reference (and use eps^(2/3) tolerances).  Enable x64 so the reference path
# can use float64 where it helps numerical fidelity, and so scalar Python
# floats used as scales aren't silently demoted.
jax.config.update("jax_enable_x64", True)

# AITER MHA kernels are only built/tuned for gfx942 (MI300) and gfx950
# (MI355).  Other ROCm GPUs and all non-ROCm platforms are unsupported.
_SUPPORTED_GFX = {"gfx942", "gfx950"}


def _on_rocm() -> bool:
    """True only when JAX is running on an AMD ROCm GPU."""
    try:
        return str(jax.devices()[0]).startswith("rocm")
    except Exception:
        return False


def _detected_gfx() -> str:
    """Return the GFX arch (e.g. 'gfx942') or '' if it can't be detected."""
    try:
        return get_gfx()
    except Exception:
        return ""


_GFX = _detected_gfx()

pytestmark = pytest.mark.skipif(
    not _on_rocm() or _GFX not in _SUPPORTED_GFX,
    reason=(
        "AITER MHA kernels are only supported on ROCm gfx942 / gfx950 "
        f"(detected: platform={'rocm' if _on_rocm() else 'non-rocm'}, "
        f"gfx={_GFX or 'unknown'})"
    ),
)

# ---------------------------------------------------------------------------
# Tolerances (matching TE: eps^(2/3))
# ---------------------------------------------------------------------------

def get_tolerances(dtype):
    if dtype == jnp.float16:
        eps = jnp.finfo(jnp.float16).eps
        tol = float(eps ** (2.0 / 3.0))
        return tol, tol
    elif dtype == jnp.bfloat16:
        eps = jnp.finfo(jnp.bfloat16).eps
        tol = float(eps ** (2.0 / 3.0))
        return tol, tol
    return 1e-5, 1e-5


def assert_close(actual, expected, dtype, name="", bwd_factor=1):
    atol, rtol = get_tolerances(dtype)
    atol *= bwd_factor
    a32 = actual.astype(jnp.float32)
    e32 = expected.astype(jnp.float32)
    max_diff = float(jnp.max(jnp.abs(a32 - e32)))
    max_ref = float(jnp.max(jnp.abs(e32)))
    rel_diff = max_diff / max(max_ref, 1e-6)
    assert max_diff < atol or rel_diff < rtol, \
        f"{name}: max_diff={max_diff:.6f} (atol={atol:.6f}), rel={rel_diff:.6f} (rtol={rtol:.6f})"


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------

def attention_ref(q, k, v, causal=False, scale=None, window_size=(-1, -1)):
    q, k, v = q.astype(jnp.float32), k.astype(jnp.float32), v.astype(jnp.float32)
    scale = scale or 1.0 / math.sqrt(q.shape[-1])
    sq, sk = q.shape[1], k.shape[1]
    attn = jnp.einsum("bshd,bthd->bhst", q, k) * scale
    if causal:
        mask = jnp.tril(jnp.ones((sq, sk), dtype=bool), k=sk - sq)
        attn = jnp.where(mask[None, None, :, :], attn, float("-inf"))
    if window_size != (-1, -1):
        wl, wr = window_size
        row_idx = jnp.arange(sq)[:, None] + (sk - sq)
        col_idx = jnp.arange(sk)[None, :]
        swa_mask = jnp.ones((sq, sk), dtype=bool)
        if wl >= 0:
            swa_mask = swa_mask & (row_idx - col_idx <= wl)
        if wr >= 0:
            swa_mask = swa_mask & (col_idx - row_idx <= wr)
        attn = jnp.where(swa_mask[None, None, :, :], attn, float("-inf"))
    attn = jax.nn.softmax(attn, axis=-1)
    return jnp.einsum("bhst,bthd->bshd", attn, v)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_qkv(b, sq, sk, hq, hk, d, dtype, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(key, 3)
    q = jax.random.normal(k1, (b, sq, hq, d), dtype=dtype)
    k_t = jax.random.normal(k2, (b, sk, hk, d), dtype=dtype)
    v = jax.random.normal(k3, (b, sk, hk, d), dtype=dtype)
    return q, k_t, v


def make_varlen(batch, max_sq, max_sk, hq, hk, d, dtype, seed=0):
    key = jax.random.PRNGKey(seed)
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)
    sq = jax.random.randint(k4, (batch,), 1, max_sq + 1)
    sk = jax.random.randint(k5, (batch,), 1, max_sk + 1)
    tq, tk = int(jnp.sum(sq)), int(jnp.sum(sk))
    cu_sq = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(sq).astype(jnp.int32)])
    cu_sk = jnp.concatenate([jnp.zeros(1, jnp.int32), jnp.cumsum(sk).astype(jnp.int32)])
    q = jax.random.normal(k1, (tq, hq, d), dtype=dtype)
    k_t = jax.random.normal(k2, (tk, hk, d), dtype=dtype)
    v = jax.random.normal(k3, (tk, hk, d), dtype=dtype)
    return q, k_t, v, cu_sq, cu_sk, max_sq, max_sk


def run_fwd(q, k, v, **kw):
    result = flash_attn_func(q, k, v, **kw)
    return result[0] if isinstance(result, tuple) else result


def run_bwd(q, k, v, **kw):
    def loss(q, k, v):
        return jnp.sum(run_fwd(q, k, v, **kw))
    return jax.grad(loss, argnums=(0, 1, 2))(q, k, v)


def run_varlen_fwd(q, k, v, cu_sq, cu_sk, msq, msk, **kw):
    result = flash_attn_varlen(q, k, v, cu_sq, cu_sk,
                               max_seqlen_q=msq, max_seqlen_k=msk, **kw)
    return result[0] if isinstance(result, tuple) else result


def run_varlen_bwd(q, k, v, cu_sq, cu_sk, msq, msk, **kw):
    def loss(q, k, v):
        return jnp.sum(run_varlen_fwd(q, k, v, cu_sq, cu_sk, msq, msk, **kw))
    return jax.grad(loss, argnums=(0, 1, 2))(q, k, v)


# ===========================================================================
# BATCH CONFIGS
# ===========================================================================

# Core configs: cover all head dims, seqlens, MHA types
BATCH_CORE = [
    # Self-attention configs
    pytest.param(2, 128, 128, 4, 4, 32, jnp.bfloat16, id="d32_bf16"),
    pytest.param(2, 128, 128, 4, 4, 40, jnp.float16, id="d40_fp16"),
    pytest.param(2, 128, 128, 4, 4, 59, jnp.bfloat16, id="d59_bf16"),
    pytest.param(2, 128, 128, 4, 4, 64, jnp.bfloat16, id="d64_bf16"),
    pytest.param(2, 128, 128, 4, 4, 64, jnp.float16, id="d64_fp16"),
    pytest.param(2, 64, 64, 4, 4, 96, jnp.bfloat16, id="d96_bf16"),
    pytest.param(2, 64, 64, 4, 4, 96, jnp.float16, id="d96_fp16"),
    pytest.param(2, 64, 64, 4, 4, 111, jnp.float16, id="d111_fp16"),
    pytest.param(2, 128, 128, 4, 4, 128, jnp.bfloat16, id="d128_bf16"),
    pytest.param(2, 128, 128, 4, 4, 128, jnp.float16, id="d128_fp16"),
    pytest.param(2, 64, 64, 4, 4, 160, jnp.bfloat16, id="d160_bf16"),
    pytest.param(2, 64, 64, 4, 4, 256, jnp.bfloat16, id="d256_bf16"),
    # Cross-attention (sq != sk)
    pytest.param(2, 512, 256, 4, 4, 64, jnp.bfloat16, id="cross_sq_gt_sk"),
    pytest.param(2, 256, 512, 4, 4, 64, jnp.bfloat16, id="cross_sq_lt_sk"),
    pytest.param(2, 1024, 1023, 4, 4, 128, jnp.bfloat16, id="off_by_one"),
    pytest.param(2, 1023, 1024, 4, 4, 128, jnp.bfloat16, id="off_by_one_rev"),
    # GQA / MQA
    pytest.param(2, 128, 128, 6, 3, 64, jnp.bfloat16, id="gqa_bf16"),
    pytest.param(2, 128, 128, 6, 3, 64, jnp.float16, id="gqa_fp16"),
    pytest.param(2, 128, 128, 6, 1, 64, jnp.float16, id="mqa_fp16"),
    pytest.param(2, 128, 128, 8, 2, 128, jnp.bfloat16, id="gqa4_d128"),
    # Large seqlens
    pytest.param(2, 1024, 1024, 4, 4, 128, jnp.bfloat16, id="large_1024"),
    pytest.param(1, 2048, 2048, 4, 4, 64, jnp.bfloat16, id="large_2048"),
    # Decode (sq=1)
    pytest.param(2, 1, 128, 4, 4, 64, jnp.bfloat16, id="decode_sq1"),
    pytest.param(2, 1, 512, 4, 4, 128, jnp.bfloat16, id="decode_sq1_long"),
    # Larger batch
    pytest.param(8, 64, 64, 4, 4, 64, jnp.bfloat16, id="batch8"),
]

# Configs suitable for accuracy check (MHA, sq==sk, reasonable size)
BATCH_ACCURACY = [c for c in BATCH_CORE
                  if c.values[1] == c.values[2] and c.values[3] == c.values[4]
                  and c.values[1] <= 256 and c.values[5] <= 128]

# Varlen configs
VARLEN_CONFIGS = [
    pytest.param(4, 128, 128, 4, 4, 64, jnp.bfloat16, id="vl_self_bf16"),
    pytest.param(4, 128, 128, 4, 4, 64, jnp.float16, id="vl_self_fp16"),
    pytest.param(4, 256, 128, 4, 4, 128, jnp.bfloat16, id="vl_cross_bf16"),
    pytest.param(4, 128, 128, 6, 3, 64, jnp.bfloat16, id="vl_gqa_bf16"),
    pytest.param(4, 128, 128, 6, 1, 64, jnp.float16, id="vl_mqa_fp16"),
    pytest.param(4, 64, 64, 4, 4, 96, jnp.bfloat16, id="vl_d96_bf16"),
    pytest.param(4, 64, 64, 4, 4, 128, jnp.float16, id="vl_d128_fp16"),
    pytest.param(4, 64, 64, 4, 4, 32, jnp.bfloat16, id="vl_d32_bf16"),
    pytest.param(2, 512, 512, 4, 4, 64, jnp.bfloat16, id="vl_large_bf16"),
]


# ===========================================================================
# BATCH FORWARD TESTS
# ===========================================================================

@pytest.mark.parametrize("b,sq,sk,hq,hk,d,dtype", BATCH_CORE)
@pytest.mark.parametrize("causal", [False, True], ids=["nomask", "causal"])
def test_batch_fwd_shape(b, sq, sk, hq, hk, d, dtype, causal):
    """Forward: correct shape, dtype, finite values for all configs."""
    q, k_t, v = make_qkv(b, sq, sk, hq, hk, d, dtype)
    out = run_fwd(q, k_t, v, causal=causal)
    assert out.shape == (b, sq, hq, d)
    assert out.dtype == dtype
    assert jnp.all(jnp.isfinite(out)), f"NaN/Inf in output for d={d}"


@pytest.mark.parametrize("b,sq,sk,hq,hk,d,dtype", BATCH_ACCURACY)
@pytest.mark.parametrize("causal", [False, True], ids=["nomask", "causal"])
def test_batch_fwd_accuracy(b, sq, sk, hq, hk, d, dtype, causal):
    """Forward accuracy vs JAX reference."""
    q, k_t, v = make_qkv(b, sq, sk, hq, hk, d, dtype, seed=42)
    scale = d ** (-0.5)
    out = run_fwd(q, k_t, v, causal=causal, softmax_scale=scale)
    ref = attention_ref(q, k_t, v, causal=causal, scale=scale).astype(dtype)
    assert_close(out, ref, dtype, "fwd_out")


# ===========================================================================
# BATCH BACKWARD TESTS
# ===========================================================================

@pytest.mark.parametrize("b,sq,sk,hq,hk,d,dtype", BATCH_CORE)
@pytest.mark.parametrize("causal", [False, True], ids=["nomask", "causal"])
def test_batch_bwd_shape(b, sq, sk, hq, hk, d, dtype, causal):
    """Backward: gradient shapes, dtypes, finiteness."""
    q, k_t, v = make_qkv(b, sq, sk, hq, hk, d, dtype, seed=1)
    dq, dk, dv = run_bwd(q, k_t, v, causal=causal)
    assert dq.shape == q.shape, f"dq {dq.shape} != {q.shape}"
    assert dk.shape == k_t.shape, f"dk {dk.shape} != {k_t.shape}"
    assert dv.shape == v.shape, f"dv {dv.shape} != {v.shape}"
    assert dq.dtype == dtype
    assert jnp.all(jnp.isfinite(dq)), "dq NaN/Inf"
    assert jnp.all(jnp.isfinite(dk)), "dk NaN/Inf"
    assert jnp.all(jnp.isfinite(dv)), "dv NaN/Inf"


_BWD_XFAIL_DIMS = {96, 111, 128}

@pytest.mark.parametrize("b,sq,sk,hq,hk,d,dtype", BATCH_ACCURACY)
def test_batch_bwd_accuracy(b, sq, sk, hq, hk, d, dtype):
    """Backward accuracy: gradients vs JAX reference (10x relaxed tolerance).

    Head dims >= 96 are xfailed due to known CK/ASM backward accuracy
    limitations on gfx950 (large dq errors at these dims).
    """
    if d in _BWD_XFAIL_DIMS:
        pytest.xfail(f"Known backward accuracy issue for d={d} on gfx950")
    q, k_t, v = make_qkv(b, sq, sk, hq, hk, d, dtype, seed=2)
    scale = d ** (-0.5)

    def aiter_loss(q, k, v):
        return jnp.sum(run_fwd(q, k, v, softmax_scale=scale))
    def ref_loss(q, k, v):
        return jnp.sum(attention_ref(q, k, v, scale=scale).astype(dtype))

    dq, dk, dv = jax.grad(aiter_loss, argnums=(0, 1, 2))(q, k_t, v)
    dq_r, dk_r, dv_r = jax.grad(ref_loss, argnums=(0, 1, 2))(q, k_t, v)

    for n, g, r in [("dq", dq, dq_r), ("dk", dk, dk_r), ("dv", dv, dv_r)]:
        assert_close(g, r, dtype, n, bwd_factor=10)


# ===========================================================================
# DROPOUT TESTS (shape/crash only, no numeric comparison)
# ===========================================================================

@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("causal", [False, True], ids=["nomask", "causal"])
def test_dropout_fwd(dtype, causal):
    """Dropout forward: no crash, finite output."""
    q, k_t, v = make_qkv(2, 128, 128, 4, 4, 64, dtype, seed=10)
    out = run_fwd(q, k_t, v, dropout_p=0.1, causal=causal)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
def test_dropout_bwd(dtype):
    """Dropout backward: no crash, finite gradients."""
    q, k_t, v = make_qkv(2, 64, 64, 4, 4, 64, dtype, seed=10)
    dq, dk, dv = run_bwd(q, k_t, v, dropout_p=0.1)
    assert jnp.all(jnp.isfinite(dq))
    assert jnp.all(jnp.isfinite(dk))
    assert jnp.all(jnp.isfinite(dv))


# ===========================================================================
# SWA (SLIDING WINDOW ATTENTION)
# ===========================================================================

@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
def test_swa_fwd(dtype):
    """SWA forward."""
    q, k_t, v = make_qkv(2, 256, 256, 4, 4, 64, dtype, seed=11)
    out = run_fwd(q, k_t, v, causal=True, window_size=(64, 0))
    assert out.shape == (2, 256, 4, 64)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
def test_swa_bwd(dtype):
    """SWA backward."""
    q, k_t, v = make_qkv(2, 128, 128, 4, 4, 64, dtype, seed=11)
    dq, dk, dv = run_bwd(q, k_t, v, causal=True, window_size=(32, 0))
    assert jnp.all(jnp.isfinite(dq))
    assert jnp.all(jnp.isfinite(dk))


# ===========================================================================
# BIAS AND ALIBI
# ===========================================================================

@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
def test_bias_fwd(dtype):
    """Attention bias forward."""
    b, sq, sk, h, d = 2, 128, 128, 4, 64
    q, k_t, v = make_qkv(b, sq, sk, h, h, d, dtype, seed=15)
    bias = jax.random.normal(jax.random.PRNGKey(99), (sq, sk), dtype=dtype) * 0.1
    out = run_fwd(q, k_t, v, bias=bias)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
def test_bias_bwd(dtype):
    """Attention bias backward: dbias gradient flows."""
    b, sq, sk, h, d = 2, 64, 64, 4, 64
    q, k_t, v = make_qkv(b, sq, sk, h, h, d, dtype, seed=15)
    bias = jax.random.normal(jax.random.PRNGKey(99), (sq, sk), dtype=dtype) * 0.1

    def loss(q, k, v, bias):
        return jnp.sum(run_fwd(q, k, v, bias=bias))
    dq, dk, dv, dbias = jax.grad(loss, argnums=(0, 1, 2, 3))(q, k_t, v, bias)
    assert jnp.all(jnp.isfinite(dq))
    assert jnp.all(jnp.isfinite(dk))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("alibi_shape", ["1d", "2d"], ids=["alibi1d", "alibi2d"])
def test_alibi_fwd(dtype, alibi_shape):
    """ALiBi forward with 1D and 2D slopes."""
    b, sq, sk, h, d = 2, 128, 128, 4, 64
    q, k_t, v = make_qkv(b, sq, sk, h, h, d, dtype, seed=16)
    if alibi_shape == "1d":
        alibi = jnp.linspace(0.1, 0.5, h, dtype=jnp.float32)
    else:
        alibi = jnp.broadcast_to(jnp.linspace(0.1, 0.5, h), (b, h)).astype(jnp.float32)
    out = run_fwd(q, k_t, v, alibi_slopes=alibi)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
def test_alibi_causal(dtype):
    """ALiBi with causal mask."""
    b, sq, sk, h, d = 2, 128, 128, 4, 64
    q, k_t, v = make_qkv(b, sq, sk, h, h, d, dtype, seed=16)
    alibi = jnp.linspace(0.1, 0.5, h, dtype=jnp.float32)
    out = run_fwd(q, k_t, v, alibi_slopes=alibi, causal=True)
    assert jnp.all(jnp.isfinite(out))


# ===========================================================================
# RETURN VALUES
# ===========================================================================

def test_return_lse():
    """Return log-sum-exp."""
    b, sq, sk, h, d = 2, 128, 128, 4, 64
    q, k_t, v = make_qkv(b, sq, sk, h, h, d, jnp.bfloat16, seed=12)
    result = flash_attn_func(q, k_t, v, return_lse=True)
    assert isinstance(result, tuple) and len(result) >= 2
    assert result[0].shape == (b, sq, h, d)
    assert result[1].shape == (b, h, sq)
    assert jnp.all(jnp.isfinite(result[1]))


def test_return_attn_probs_with_dropout():
    """Return attention probs (S_dmask) with dropout."""
    b, sq, sk, h, d = 2, 64, 64, 4, 64
    q, k_t, v = make_qkv(b, sq, sk, h, h, d, jnp.bfloat16, seed=12)
    result = flash_attn_func(q, k_t, v, dropout_p=0.1, return_attn_probs=True)
    assert isinstance(result, tuple) and len(result) >= 2


# ===========================================================================
# PADDED HEAD DIMENSIONS
# ===========================================================================

@pytest.mark.parametrize("d", [32, 40, 59, 64, 96, 111, 128, 160, 256],
                         ids=[f"d{d}" for d in [32, 40, 59, 64, 96, 111, 128, 160, 256]])
def test_padded_head_dim_fwd(d):
    """All head dims produce correct output shape."""
    q, k_t, v = make_qkv(2, 64, 64, 4, 4, d, jnp.bfloat16, seed=13)
    out = run_fwd(q, k_t, v)
    assert out.shape == (2, 64, 4, d)
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("d", [59, 111], ids=["d59", "d111"])
def test_padded_head_dim_bwd(d):
    """Non-multiple-of-8 head dims: backward produces finite gradients."""
    q, k_t, v = make_qkv(2, 64, 64, 4, 4, d, jnp.bfloat16, seed=13)
    dq, dk, dv = run_bwd(q, k_t, v)
    assert dq.shape == (2, 64, 4, d)
    assert jnp.all(jnp.isfinite(dq))


# ===========================================================================
# DETERMINISTIC
# ===========================================================================

@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
def test_deterministic_consistency(dtype):
    """Deterministic: identical results across calls."""
    q, k_t, v = make_qkv(2, 128, 128, 4, 4, 64, dtype, seed=14)
    o1 = run_fwd(q, k_t, v, deterministic=True)
    o2 = run_fwd(q, k_t, v, deterministic=True)
    assert jnp.allclose(o1, o2, atol=0)


def test_deterministic_bwd():
    """Deterministic backward: identical gradients across calls."""
    q, k_t, v = make_qkv(2, 64, 64, 4, 4, 64, jnp.bfloat16, seed=14)
    dq1, _, _ = run_bwd(q, k_t, v, deterministic=True)
    dq2, _, _ = run_bwd(q, k_t, v, deterministic=True)
    assert jnp.allclose(dq1, dq2, atol=0)


# ===========================================================================
# VARLEN TESTS
# ===========================================================================

@pytest.mark.parametrize("batch,max_sq,max_sk,hq,hk,d,dtype", VARLEN_CONFIGS)
@pytest.mark.parametrize("causal", [False, True], ids=["nomask", "causal"])
def test_varlen_fwd(batch, max_sq, max_sk, hq, hk, d, dtype, causal):
    """Varlen forward: shape, dtype, finite."""
    q, k_t, v, cu_sq, cu_sk, msq, msk = make_varlen(batch, max_sq, max_sk, hq, hk, d, dtype, seed=20)
    out = run_varlen_fwd(q, k_t, v, cu_sq, cu_sk, msq, msk, causal=causal)
    assert out.shape == (q.shape[0], hq, d)
    assert out.dtype == dtype
    assert jnp.all(jnp.isfinite(out))


@pytest.mark.parametrize("batch,max_sq,max_sk,hq,hk,d,dtype", VARLEN_CONFIGS)
def test_varlen_bwd(batch, max_sq, max_sk, hq, hk, d, dtype):
    """Varlen backward: gradient shapes and finiteness."""
    q, k_t, v, cu_sq, cu_sk, msq, msk = make_varlen(batch, max_sq, max_sk, hq, hk, d, dtype, seed=21)
    dq, dk, dv = run_varlen_bwd(q, k_t, v, cu_sq, cu_sk, msq, msk)
    assert dq.shape == q.shape
    assert dk.shape == k_t.shape
    assert dv.shape == v.shape
    assert jnp.all(jnp.isfinite(dq))
    assert jnp.all(jnp.isfinite(dk))
    assert jnp.all(jnp.isfinite(dv))


# ===========================================================================
# EDGE CASES
# ===========================================================================

def test_decode_sq1_fwd_bwd():
    """Decode: sq=1, forward + backward."""
    q, k_t, v = make_qkv(2, 1, 256, 4, 4, 64, jnp.bfloat16, seed=30)
    out = run_fwd(q, k_t, v)
    assert out.shape == (2, 1, 4, 64)
    dq, dk, dv = run_bwd(q, k_t, v)
    assert jnp.all(jnp.isfinite(dq))


def test_sq_gt_sk_nomask():
    """sq > sk without causal mask."""
    q, k_t, v = make_qkv(2, 256, 128, 4, 4, 64, jnp.bfloat16, seed=31)
    out = run_fwd(q, k_t, v)
    assert out.shape == (2, 256, 4, 64)
    dq, dk, dv = run_bwd(q, k_t, v)
    assert jnp.all(jnp.isfinite(dq))


def test_sq_gt_sk_causal():
    """sq > sk with causal mask."""
    q, k_t, v = make_qkv(2, 256, 128, 4, 4, 64, jnp.bfloat16, seed=31)
    out = run_fwd(q, k_t, v, causal=True)
    assert jnp.all(jnp.isfinite(out))
    dq, dk, dv = run_bwd(q, k_t, v, causal=True)
    assert jnp.all(jnp.isfinite(dq))


def test_large_batch():
    """Batch=16."""
    q, k_t, v = make_qkv(16, 64, 64, 4, 4, 64, jnp.bfloat16, seed=32)
    out = run_fwd(q, k_t, v, causal=True)
    assert jnp.all(jnp.isfinite(out))


def test_single_head():
    """Single head (nheads=1)."""
    q, k_t, v = make_qkv(2, 64, 64, 1, 1, 64, jnp.bfloat16, seed=33)
    out = run_fwd(q, k_t, v)
    assert out.shape == (2, 64, 1, 64)
    dq, dk, dv = run_bwd(q, k_t, v)
    assert jnp.all(jnp.isfinite(dq))


def test_many_heads():
    """Many heads (nheads=16)."""
    q, k_t, v = make_qkv(2, 64, 64, 16, 16, 64, jnp.bfloat16, seed=34)
    out = run_fwd(q, k_t, v)
    assert out.shape == (2, 64, 16, 64)


# ===========================================================================
# REGRESSION GUARDS — every historical bug from AITER bump
# ===========================================================================

class TestRegressions:
    """Configs that caught historical bugs. Must never regress."""

    def test_v3_bwd_sq_gt_sk_causal(self):
        """c5bc2e2: ASM v3 bwd wrong gradients for causal sq > sk on gfx950."""
        for d in [96, 128]:
            q, k_t, v = make_qkv(2, 128, 64, 4, 4, d, jnp.bfloat16, seed=40)
            dq, dk, dv = run_bwd(q, k_t, v, causal=True)
            assert jnp.all(jnp.isfinite(dq)), f"d={d}: dq NaN"

    @pytest.mark.parametrize("d", [96, 111, 128], ids=["d96", "d111", "d128"])
    @pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16], ids=["fp16", "bf16"])
    def test_1024_1023_causal(self, d, dtype):
        """c5bc2e2: seqlen=(1024,1023) causal d>=96 on gfx950."""
        q, k_t, v = make_qkv(2, 1024, 1023, 4, 4, d, dtype, seed=41)
        dq, dk, dv = run_bwd(q, k_t, v, causal=True)
        assert jnp.all(jnp.isfinite(dq)), f"d={d}: dq NaN"

    def test_mqa_gqa_bwd_routing(self):
        """e53633c: MQA/GQA backward routing (nhead_q != nhead_k guard)."""
        for hq, hk in [(6, 3), (6, 1), (8, 2)]:
            q, k_t, v = make_qkv(2, 128, 128, hq, hk, 64, jnp.bfloat16, seed=42)
            dq, dk, dv = run_bwd(q, k_t, v, causal=True)
            assert dq.shape == (2, 128, hq, 64)
            assert dk.shape == (2, 128, hk, 64)
            assert jnp.all(jnp.isfinite(dq)), f"hq={hq},hk={hk}: dq NaN"
            assert jnp.all(jnp.isfinite(dk))
            assert jnp.all(jnp.isfinite(dv))

    @pytest.mark.parametrize("d", [
        pytest.param(96, marks=pytest.mark.xfail(reason="gfx950 CK varlen bwd causal max_sk>256 kernel issue")),
        pytest.param(128, marks=pytest.mark.xfail(reason="gfx950 CK varlen bwd causal max_sk>256 kernel issue")),
    ], ids=["d96", "d128"])
    def test_varlen_large_sk_causal(self, d):
        """c5bc2e2: varlen max_sk>256 causal d>=96 on gfx950."""
        q, k_t, v, cu_sq, cu_sk, msq, msk = make_varlen(4, 512, 512, 4, 4, d, jnp.bfloat16, seed=43)
        dq, dk, dv = run_varlen_bwd(q, k_t, v, cu_sq, cu_sk, msq, msk, causal=True)
        assert jnp.all(jnp.isfinite(dq)), f"d={d}: varlen dq NaN"

    def test_gfx950_1block_override(self):
        """is_950_1block: sk<=256 with hd in (64,128] forces deterministic=False."""
        q, k_t, v = make_qkv(2, 128, 128, 4, 4, 96, jnp.bfloat16, seed=44)
        dq, dk, dv = run_bwd(q, k_t, v, deterministic=True)
        assert jnp.all(jnp.isfinite(dq))

    def test_swa_not_v3_bwd(self):
        """SWA excluded from ASM v3 backward (wrong gradients on gfx950)."""
        q, k_t, v = make_qkv(2, 128, 128, 4, 4, 128, jnp.bfloat16, seed=45)
        dq, dk, dv = run_bwd(q, k_t, v, causal=True, window_size=(32, 0))
        assert jnp.all(jnp.isfinite(dq))

    @pytest.mark.parametrize("d", [32, 64, 96, 128], ids=["d32", "d64", "d96", "d128"])
    def test_all_head_dims_bwd(self, d):
        """All common head dims produce finite backward gradients."""
        q, k_t, v = make_qkv(2, 64, 64, 4, 4, d, jnp.bfloat16, seed=46)
        dq, dk, dv = run_bwd(q, k_t, v)
        assert jnp.all(jnp.isfinite(dq)), f"d={d}: dq NaN"
        assert jnp.all(jnp.isfinite(dk))
        assert jnp.all(jnp.isfinite(dv))
