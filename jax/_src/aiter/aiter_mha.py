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
"""Simplified MHA using unified AITER entry point.

Calls aiter::mha_fwd / aiter::mha_bwd through a single FFI handler per
direction. CK vs ASM v3 dispatch is handled internally by AITER based on
the use_asm_v3 flag. No Python-side dispatch logic.

GSPMD sharding: custom_partitioning tells XLA how to partition the FFI
calls for multi-GPU FSDP.  For batch-mode attention every dimension
except the batch axis is replicated, so each device runs independently
on its local batch shard (output sharding = Q input sharding, no
collectives).  custom_partitioning wraps the raw FFI calls;
custom_vjp sits on the outer public API -- they compose because they
are on different levels of the call stack.
"""

from __future__ import annotations

import functools
import logging
import os
import shutil
import subprocess
from collections import namedtuple
from typing import Optional
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.custom_partitioning import (
    ArrayMapping,
    SdyShardingRule,
    custom_partitioning,
)
from jax.sharding import NamedSharding, PartitionSpec as P


# -------------- Register AITER ------------

from jax._src.lib import gpu_aiter

logger = logging.getLogger(__name__)


if gpu_aiter is not None:
    for platform, targets in gpu_aiter.registrations().items():
        for name, value in targets:
            logger.debug(
                "Registering AITER FFI target: %s for platform: %s",
                name, platform,
            )
            jax.ffi.register_ffi_target(
                name, value, platform=platform, api_version=1
            )
else:
    logger.debug("AITER FFI targets not found (gpu_aiter unavailable)")
# ------------------------------------------------------------

@functools.lru_cache(maxsize=1)
def get_gfx() -> str:
    """Return the GFX architecture string (e.g. 'gfx950', 'gfx942')."""
    gfx = os.getenv("GPU_ARCHS", "native")
    if gfx == "native":
        rocminfo = shutil.which("rocminfo")
        if rocminfo is None:
            raise RuntimeError(
                "rocminfo not found on PATH; set GPU_ARCHS to specify the "
                "target architecture (e.g. GPU_ARCHS=gfx942)"
            )
        try:
            result = subprocess.run(
                [os.path.realpath(rocminfo)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            for line in result.stdout.split("\n"):
                if "gfx" in line.lower():
                    return line.split(":")[-1].strip()
        except Exception as e:
            raise RuntimeError(f"Get GPU arch from rocminfo failed: {e}")
        raise RuntimeError("rocminfo did not report a gfx architecture")
    if ";" in gfx:
        gfx = gfx.split(";")[-1]
    return gfx


# Hashable config bundles for custom_partitioning static args.
MhaFwdConfig = namedtuple("MhaFwdConfig", [
    "dropout_p", "softmax_scale", "is_causal", "wl", "wr",
    "return_lse", "return_randval", "use_asm_v3", "how_v3_bf16_cvt",
    "max_seqlen_q", "max_seqlen_k", "min_seqlen_q",
    "logits_soft_cap", "zero_tensors",
    "cp_axis", "cp_size", "cp_load_balanced",
])

MhaBwdConfig = namedtuple("MhaBwdConfig", [
    "dropout_p", "softmax_scale", "is_causal", "wl", "wr",
    "deterministic", "use_asm_v3", "is_v3_atomic_fp32", "how_v3_bf16_cvt",
    "max_seqlen_q", "max_seqlen_k", "zero_tensors",
    "cp_axis", "cp_size", "cp_load_balanced",
])


def _empty(dtype):
    return jnp.zeros((0,), dtype=dtype)


def _sf(x) -> np.float32:
    return np.float32(x)


def _si(x) -> np.int32:
    return np.int32(x)


# ---------------------------------------------------------------------------
# Unified forward FFI wrapper
# ---------------------------------------------------------------------------

def _make_fwd_call(out_shape, lse_shape, p_shape, rng_shape, dtype):
    call = jax.ffi.ffi_call(
        "hip_mha_fwd_ffi",
        (
            jax.ShapeDtypeStruct(out_shape, dtype),
            jax.ShapeDtypeStruct(lse_shape, jnp.float32),
            jax.ShapeDtypeStruct(p_shape, jnp.uint8),
            jax.ShapeDtypeStruct(rng_shape, jnp.int64),
        ),
        vmap_method="broadcast_all",
        input_layouts=[None] * 9,
        output_layouts=[None] * 4,
        has_side_effect=False,
    )

    def _invoke(q, k, v, cu_sq, cu_skv, out_prov, bias, alibi, gen, *,
                dropout_p, softmax_scale, is_causal, wl, wr,
                return_lse, return_randval, use_asm_v3, how_v3_bf16_cvt,
                max_seqlen_q_attr, max_seqlen_k_attr, min_seqlen_q,
                logits_soft_cap, zero_tensors):
        return call(q, k, v, cu_sq, cu_skv, out_prov, bias, alibi, gen,
                    dropout_p=dropout_p, softmax_scale=softmax_scale,
                    is_causal=is_causal, window_size_left=wl, window_size_right=wr,
                    return_softmax_lse=return_lse,
                    return_dropout_randval=return_randval,
                    use_asm_v3=use_asm_v3, how_v3_bf16_cvt=how_v3_bf16_cvt,
                    max_seqlen_q_attr=max_seqlen_q_attr,
                    max_seqlen_k_attr=max_seqlen_k_attr,
                    min_seqlen_q=min_seqlen_q,
                    logits_soft_cap=logits_soft_cap,
                    zero_tensors=zero_tensors)

    return jax.jit(_invoke, static_argnames=(
        "dropout_p", "softmax_scale", "is_causal", "wl", "wr",
        "return_lse", "return_randval", "use_asm_v3", "how_v3_bf16_cvt",
        "max_seqlen_q_attr", "max_seqlen_k_attr", "min_seqlen_q",
        "logits_soft_cap", "zero_tensors"))


# ---------------------------------------------------------------------------
# Unified backward FFI wrapper
# ---------------------------------------------------------------------------

def _make_bwd_call(dq_shape, dk_shape, dv_shape, sd_shape, dbias_shape, dtype):
    call = jax.ffi.ffi_call(
        "hip_mha_bwd_ffi",
        (
            jax.ShapeDtypeStruct(dq_shape, dtype),
            jax.ShapeDtypeStruct(dk_shape, dtype),
            jax.ShapeDtypeStruct(dv_shape, dtype),
            jax.ShapeDtypeStruct(sd_shape, jnp.float32),
            jax.ShapeDtypeStruct(dbias_shape, dtype),
        ),
        vmap_method="broadcast_all",
        input_layouts=[None] * 15,
        output_layouts=[None] * 5,
        has_side_effect=False,
    )

    def _invoke(dout, q, k, v, out, lse, cu_sq, cu_sk,
                dq, dk, dv, bias, alibi, rng, gen, *,
                dropout_p, softmax_scale, is_causal, wl, wr,
                deterministic, use_asm_v3, is_v3_atomic_fp32, how_v3_bf16_cvt,
                max_seqlen_q_attr, max_seqlen_k_attr, zero_tensors):
        return call(dout, q, k, v, out, lse, cu_sq, cu_sk,
                    dq, dk, dv, bias, alibi, rng, gen,
                    dropout_p=dropout_p, softmax_scale=softmax_scale,
                    is_causal=is_causal, window_size_left=wl, window_size_right=wr,
                    deterministic=deterministic, use_asm_v3=use_asm_v3,
                    is_v3_atomic_fp32=is_v3_atomic_fp32,
                    how_v3_bf16_cvt=how_v3_bf16_cvt,
                    max_seqlen_q_attr=max_seqlen_q_attr,
                    max_seqlen_k_attr=max_seqlen_k_attr,
                    zero_tensors=zero_tensors)

    return jax.jit(_invoke, static_argnames=(
        "dropout_p", "softmax_scale", "is_causal", "wl", "wr",
        "deterministic", "use_asm_v3", "is_v3_atomic_fp32", "how_v3_bf16_cvt",
        "max_seqlen_q_attr", "max_seqlen_k_attr", "zero_tensors"))


# ---------------------------------------------------------------------------
# Sharding helpers
# ---------------------------------------------------------------------------

def _get_padded_spec(arg_info):
    """Pad a PartitionSpec to match ndim, filling with None."""
    if arg_info.sharding is None:
        return (None,) * arg_info.ndim
    spec = arg_info.sharding.spec
    return spec + (None,) * (arg_info.ndim - len(spec))


def _get_rank(t):
    """Get tensor rank from either JAX ShapeDtypeStruct or MLIR RankedTensorType."""
    if hasattr(t, 'ndim'):
        return t.ndim
    if hasattr(t, 'rank'):
        return t.rank
    return len(t.shape)


# ---------------------------------------------------------------------------
# Raw forward/backward FFI helpers (no partitioning, shape-driven)
# ---------------------------------------------------------------------------

def _mha_fwd_raw(q, k, v, cu_sq, cu_skv, out_prov, bias, alibi, gen,
                 config):

    is_varlen = (q.ndim == 3)
    if is_varlen:
        total_q, hq, dq = q.shape
        _, hk, dv = v.shape
        out_shape = (total_q, hq, dv)
        lse_shape = (hq, config.max_seqlen_q) if config.return_lse else (0,)
        p_shape = (0,)
    else:
        b, sq, hq, dq = q.shape
        _, sk, hk, dv = v.shape
        out_shape = (b, sq, hq, dv)
        lse_shape = (b, hq, sq) if config.return_lse else (0,)
        p_shape = (b, hq, sq, sk) if config.return_randval else (0,)
    rng_shape = (2,)
    fn = _make_fwd_call(out_shape, lse_shape, p_shape,
                                  rng_shape, q.dtype)
    return fn(q, k, v, cu_sq, cu_skv, out_prov, bias, alibi, gen,
              dropout_p=_sf(config.dropout_p),
              softmax_scale=_sf(config.softmax_scale),
              is_causal=config.is_causal,
              wl=_si(config.wl), wr=_si(config.wr),
              return_lse=config.return_lse,
              return_randval=config.return_randval,
              use_asm_v3=config.use_asm_v3,
              how_v3_bf16_cvt=_si(config.how_v3_bf16_cvt),
              max_seqlen_q_attr=_si(config.max_seqlen_q),
              max_seqlen_k_attr=_si(config.max_seqlen_k),
              min_seqlen_q=_si(config.min_seqlen_q),
              logits_soft_cap=_sf(config.logits_soft_cap),
              zero_tensors=config.zero_tensors)


def _mha_bwd_raw(dout, q, k, v, out, lse, cu_sq, cu_sk,
                 dq_ws, dk_ws, dv_ws, bias, alibi, rng, gen,
                 config):

    is_varlen = (q.ndim == 3)
    if is_varlen:
        total_q, hq, dq_dim = q.shape
        _, hk, _ = k.shape
        dv_dim = v.shape[-1]
        total_k = k.shape[0]
        dq_shape = (total_q, hq, dq_dim)
        dk_shape = (total_k, hk, dq_dim)
        dv_shape = (total_k, hk, dv_dim)
        sd_shape = (hq, config.max_seqlen_q)
        dbias_shape = (0,)
    else:
        b, sq, hq, dq_dim = q.shape
        _, sk, hk, _ = k.shape
        dv_dim = v.shape[-1]
        dq_shape = (b, sq, hq, dq_dim)
        dk_shape = (b, sk, hk, dq_dim)
        dv_shape = (b, sk, hk, dv_dim)
        sd_shape = (b, hq, sq)
        dbias_shape = (b, sq, hq, sk) if (bias.size > 0) else (0,)
    fn = _make_bwd_call(dq_shape, dk_shape, dv_shape,
                                  sd_shape, dbias_shape, q.dtype)
    return fn(dout, q, k, v, out, lse, cu_sq, cu_sk,
              dq_ws, dk_ws, dv_ws, bias, alibi, rng, gen,
              dropout_p=_sf(config.dropout_p),
              softmax_scale=_sf(config.softmax_scale),
              is_causal=config.is_causal,
              wl=_si(config.wl), wr=_si(config.wr),
              deterministic=config.deterministic,
              use_asm_v3=config.use_asm_v3,
              is_v3_atomic_fp32=config.is_v3_atomic_fp32,
              how_v3_bf16_cvt=_si(config.how_v3_bf16_cvt),
              max_seqlen_q_attr=_si(config.max_seqlen_q),
              max_seqlen_k_attr=_si(config.max_seqlen_k),
              zero_tensors=config.zero_tensors)


# ---------------------------------------------------------------------------
# custom_partitioning: forward
# ---------------------------------------------------------------------------

@partial(custom_partitioning, static_argnums=(9,))
def _mha_fwd_partitioned(q, k, v, cu_sq, cu_skv, out_prov,
                         bias, alibi, gen, config):
    return _mha_fwd_raw(q, k, v, cu_sq, cu_skv, out_prov,
                        bias, alibi, gen, config)


def _mha_fwd_infer_sharding(config, mesh, arg_shapes, result_shapes):
    q_spec = _get_padded_spec(arg_shapes[0])
    is_varlen = (arg_shapes[0].ndim == 3)

    out_sharding = NamedSharding(mesh, P(*q_spec))

    if is_varlen:
        lse_sh = NamedSharding(mesh, P(*((None,) * result_shapes[1].ndim)))
    else:
        if result_shapes[1].ndim == 3:
            lse_sh = NamedSharding(mesh, P(q_spec[0], q_spec[2], q_spec[1]))
        else:
            lse_sh = NamedSharding(mesh, P(None))

    p_sh = NamedSharding(mesh, P(*((None,) * result_shapes[2].ndim)))
    rng_sh = NamedSharding(mesh, P(*((None,) * result_shapes[3].ndim)))
    return (out_sharding, lse_sh, p_sh, rng_sh)


def _mha_fwd_partition(config, mesh, arg_shapes, result_shapes):
    out_shardings = _mha_fwd_infer_sharding(config, mesh,
                                            arg_shapes, result_shapes)
    q_spec = _get_padded_spec(arg_shapes[0])
    cp_axis = config.cp_axis
    cp_active = cp_axis and config.cp_size > 1

    shardings = []
    for i, a in enumerate(arg_shapes):
        if a.shape[0] == 0:
            shardings.append(NamedSharding(mesh, P(*((None,) * a.ndim))))
        elif i == 6 and a.ndim == 2:
            shardings.append(NamedSharding(mesh, P(q_spec[1], None)))
        elif cp_active and i in (1, 2) and a.ndim == 4:
            s = _get_padded_spec(a)
            shardings.append(NamedSharding(mesh, P(s[0], None, s[2], s[3])))
        else:
            shardings.append(a.sharding)
    arg_shardings = tuple(shardings)

    def _lowered(q, k, v, cu_sq, cu_skv, out_prov, bias, alibi, gen):
        return _mha_fwd_raw(q, k, v, cu_sq, cu_skv, out_prov,
                            bias, alibi, gen, config)

    return mesh, _lowered, out_shardings, arg_shardings


def _mha_fwd_shardy_rule(config, mesh, in_types, out_types):
    """Shardy sharding rule: batch dims passthrough, rest placeholders."""
    is_4d = (_get_rank(in_types[0]) == 4)
    if is_4d:
        q_spec = ("…0", "sq", "hq", "dq")
        k_spec = ("…0", "sk", "hk", "dq")
        v_spec = ("…0", "sk", "hk", "dv")
    else:
        q_spec = ("…0", "hq", "dq")
        k_spec = ("…1", "hk", "dq")
        v_spec = ("…1", "hk", "dv")
    in_spec: list[ArrayMapping] = [
        ArrayMapping(*q_spec),
        ArrayMapping(*k_spec),
        ArrayMapping(*v_spec),
    ]
    fid = 10
    for i in range(3, len(in_types)):
        if i == 6 and _get_rank(in_types[i]) == 2:
            in_spec.append(ArrayMapping("sq", "sk"))
        else:
            in_spec.append(ArrayMapping(f"…{fid}"))
            fid += 1

    out_spec: list[ArrayMapping] = []
    if is_4d:
        out_spec.append(ArrayMapping("…0", "sq", "hq", "dv"))
        if _get_rank(out_types[1]) == 3:
            out_spec.append(ArrayMapping("…0", "hq", "sq"))
        else:
            out_spec.append(ArrayMapping(f"…{fid}"))
    else:
        out_spec.append(ArrayMapping("…0", "hq", "dv"))
        out_spec.append(ArrayMapping(f"…{fid}"))
    fid += 1
    for j in range(2, len(out_types)):
        out_spec.append(ArrayMapping(f"…{fid}"))
        fid += 1
    return SdyShardingRule(tuple(in_spec), tuple(out_spec))


_mha_fwd_partitioned.def_partition(
    _mha_fwd_partition,
    infer_sharding_from_operands=_mha_fwd_infer_sharding,
    sharding_rule=_mha_fwd_shardy_rule,
)


# ---------------------------------------------------------------------------
# custom_partitioning: backward
# ---------------------------------------------------------------------------

@partial(custom_partitioning, static_argnums=(15,))
def _mha_bwd_partitioned(dout, q, k, v, out, lse, cu_sq, cu_sk,
                         dq_ws, dk_ws, dv_ws, bias, alibi, rng, gen,
                         config):
    return _mha_bwd_raw(dout, q, k, v, out, lse, cu_sq, cu_sk,
                        dq_ws, dk_ws, dv_ws, bias, alibi, rng, gen,
                        config)


def _mha_bwd_infer_sharding(config, mesh, arg_shapes, result_shapes):
    q_spec = _get_padded_spec(arg_shapes[1])
    k_spec = _get_padded_spec(arg_shapes[2])
    v_spec = _get_padded_spec(arg_shapes[3])

    dq_sh = NamedSharding(mesh, P(*q_spec))
    dk_sh = NamedSharding(mesh, P(*k_spec))
    dv_sh = NamedSharding(mesh, P(*v_spec))
    sd_sh = NamedSharding(mesh, P(*((None,) * result_shapes[3].ndim)))
    dbias_sh = NamedSharding(mesh, P(*((None,) * result_shapes[4].ndim)))

    if result_shapes[3].ndim == 3:
        sd_sh = NamedSharding(mesh, P(q_spec[0], q_spec[2], q_spec[1]))
    if result_shapes[4].ndim == 4:
        dbias_sh = NamedSharding(mesh, P(q_spec[0], q_spec[1], q_spec[2], None))

    return (dq_sh, dk_sh, dv_sh, sd_sh, dbias_sh)


def _mha_bwd_partition(config, mesh, arg_shapes, result_shapes):
    out_shardings = _mha_bwd_infer_sharding(config, mesh,
                                            arg_shapes, result_shapes)
    q_spec = _get_padded_spec(arg_shapes[1])
    cp_axis = config.cp_axis
    cp_active = cp_axis and config.cp_size > 1

    shardings = []
    for i, a in enumerate(arg_shapes):
        if a.shape[0] == 0:
            shardings.append(NamedSharding(mesh, P(*((None,) * a.ndim))))
        elif i == 11 and a.ndim == 2:
            shardings.append(NamedSharding(mesh, P(q_spec[1], None)))
        elif cp_active and i in (2, 3) and a.ndim == 4:
            s = _get_padded_spec(a)
            shardings.append(NamedSharding(mesh, P(s[0], None, s[2], s[3])))
        else:
            shardings.append(a.sharding)
    arg_shardings = tuple(shardings)

    def _lowered(dout, q, k, v, out, lse, cu_sq, cu_sk,
                 dq_ws, dk_ws, dv_ws, bias, alibi, rng, gen):
        return _mha_bwd_raw(dout, q, k, v, out, lse, cu_sq, cu_sk,
                            dq_ws, dk_ws, dv_ws, bias, alibi, rng, gen,
                            config)

    return mesh, _lowered, out_shardings, arg_shardings


def _mha_bwd_shardy_rule(config, mesh, in_types, out_types):
    """Shardy sharding rule for backward: all independent placeholders."""
    fid = 0
    in_spec: list[ArrayMapping] = []
    for i in range(len(in_types)):
        in_spec.append(ArrayMapping(f"…{fid}"))
        fid += 1
    out_spec: list[ArrayMapping] = []
    for i in range(len(out_types)):
        out_spec.append(ArrayMapping(f"…{fid}"))
        fid += 1
    return SdyShardingRule(tuple(in_spec), tuple(out_spec))


_mha_bwd_partitioned.def_partition(
    _mha_bwd_partition,
    infer_sharding_from_operands=_mha_bwd_infer_sharding,
    sharding_rule=_mha_bwd_shardy_rule,
)


# ---------------------------------------------------------------------------
# Forward: single call to aiter::mha_fwd (AITER handles CK vs ASM)
# ---------------------------------------------------------------------------

def mha_fwd_unified(q, k, v, dropout_p, softmax_scale, causal,
                    wl, wr, return_lse, return_softmax,
                    bias=None, alibi_slopes=None,
                    cu_seqlens_q=None, cu_seqlens_kv=None, gen=None,
                    max_seqlen_q=-1, max_seqlen_k=-1, min_seqlen_q=0,
                    logits_soft_cap=0.0, zero_tensors=False,
                    cp_axis=None, cp_size=1, cp_load_balanced=True):
    """Unified forward for both batch (4D q) and varlen (3D q)."""
    if cu_seqlens_q is None:
        cu_seqlens_q = _empty(jnp.int32)
    if cu_seqlens_kv is None:
        cu_seqlens_kv = _empty(jnp.int32)
    if bias is None:
        bias = _empty(q.dtype)
    if alibi_slopes is None:
        alibi_slopes = _empty(jnp.float32)
    if gen is None:
        gen = _empty(jnp.int64)

    bf16_cvt = 0 if get_gfx() == "gfx950" else 1
    dq = q.shape[-1]
    use_v3_fwd = not (get_gfx() == "gfx950" and dq >= 96)

    config = MhaFwdConfig(
        dropout_p=float(dropout_p),
        softmax_scale=float(softmax_scale),
        is_causal=causal,
        wl=int(wl), wr=int(wr),
        return_lse=return_lse,
        return_randval=bool(return_softmax and dropout_p > 0),
        use_asm_v3=use_v3_fwd,
        how_v3_bf16_cvt=int(bf16_cvt),
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        min_seqlen_q=int(min_seqlen_q),
        logits_soft_cap=float(logits_soft_cap),
        zero_tensors=zero_tensors,
        cp_axis=cp_axis,
        cp_size=int(cp_size) if cp_size else 1,
        cp_load_balanced=cp_load_balanced,
    )
    return _mha_fwd_partitioned(q, k, v, cu_seqlens_q, cu_seqlens_kv,
                                _empty(q.dtype), bias, alibi_slopes, gen,
                                config)


def mha_bwd_unified(dout, q, k, v, out, lse, dropout_p, softmax_scale,
                    causal, wl, wr, deterministic,
                    use_asm_v3, is_v3_atomic_fp32, how_v3_bf16_cvt,
                    bias=None, alibi_slopes=None, rng_state=None,
                    cu_seqlens_q=None, cu_seqlens_k=None,
                    max_seqlen_q=-1, max_seqlen_k=-1, zero_tensors=False,
                    cp_axis=None, cp_size=1, cp_load_balanced=True):
    """Unified backward for both batch (4D q) and varlen (3D q)."""
    if cu_seqlens_q is None:
        cu_seqlens_q = _empty(jnp.int32)
    if cu_seqlens_k is None:
        cu_seqlens_k = _empty(jnp.int32)
    if bias is None:
        bias = _empty(q.dtype)
    if alibi_slopes is None:
        alibi_slopes = _empty(jnp.float32)
    if rng_state is None:
        rng_state = _empty(jnp.int64)

    config = MhaBwdConfig(
        dropout_p=float(dropout_p),
        softmax_scale=float(softmax_scale),
        is_causal=causal,
        wl=int(wl), wr=int(wr),
        deterministic=deterministic,
        use_asm_v3=use_asm_v3,
        is_v3_atomic_fp32=is_v3_atomic_fp32,
        how_v3_bf16_cvt=int(how_v3_bf16_cvt),
        max_seqlen_q=int(max_seqlen_q),
        max_seqlen_k=int(max_seqlen_k),
        zero_tensors=zero_tensors,
        cp_axis=cp_axis,
        cp_size=int(cp_size) if cp_size else 1,
        cp_load_balanced=cp_load_balanced,
    )
    results = _mha_bwd_partitioned(
        dout, q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k,
        _empty(q.dtype), _empty(q.dtype), _empty(q.dtype),
        bias, alibi_slopes, rng_state, _empty(jnp.int64),
        config)

    dq_out, dk_out, dv_out, sd_out, dbias_expanded = results
    is_varlen = (q.ndim == 3)
    if not is_varlen and bias.size > 0:
        dbias_out = jnp.sum(dbias_expanded, axis=(0, 2))
    else:
        dbias_out = dbias_expanded
    return [dq_out, dk_out, dv_out, sd_out, dbias_out]


# ---------------------------------------------------------------------------
# Simplified forward/backward dispatch (no can_impl_* logic)
# ---------------------------------------------------------------------------

def _flash_attn_forward(q, k, v, dropout_p, softmax_scale, causal,
                        wl, wr, bias, alibi_slopes,
                        return_lse, return_softmax,
                        cu_seqlens_q=None, cu_seqlens_kv=None):
    _, sk, _, _ = v.shape
    if wl >= sk: wl = -1
    if wr >= sk: wr = -1

    result = mha_fwd_unified(
        q, k, v, dropout_p, softmax_scale, causal, wl, wr,
        return_lse, return_softmax,
        bias=bias, alibi_slopes=alibi_slopes,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv)
    return result


def _flash_attn_backward(dout, q, k, v, out, lse,
                         dropout_p, softmax_scale, causal, wl, wr,
                         bias, alibi_slopes, deterministic,
                         rng_state=None):
    _, sq, hq, dq = q.shape
    _, sk, hk, _ = k.shape

    bf16_cvt = 0 if get_gfx() == "gfx950" else 1

    # v3 eligibility: exclude known-broken configs.
    swa = (wl > 0) or (wr >= 0 and wr != -1)
    use_v3 = True
    if dropout_p > 0:
        use_v3 = False
    if hq != hk:
        use_v3 = False
    if bias is not None and bias.size > 0:
        use_v3 = False
    if swa:
        use_v3 = False
    if causal and get_gfx() == "gfx950" and sq > sk:
        use_v3 = False
    if get_gfx() == "gfx950" and dq >= 96:
        use_v3 = False

    # gfx950 1-block override: sk<=256 with hd in (64,128]
    is_950_1block = (
        get_gfx() == "gfx950" and sk <= 256
        and dq > 64 and dq <= 128 and dq % 8 == 0
    )
    bwd_det = False if is_950_1block else deterministic
    use_v3_bwd = False if is_950_1block else use_v3
    bwd_atomic = False if is_950_1block else use_v3_bwd

    results = mha_bwd_unified(
        dout, q, k, v, out, lse,
        dropout_p, softmax_scale, causal, wl, wr,
        bwd_det, use_v3_bwd, bwd_atomic, bf16_cvt,
        bias=bias, alibi_slopes=alibi_slopes, rng_state=rng_state)

    return results[0], results[1], results[2], results[3], results[4]


# ---------------------------------------------------------------------------
# Public API: flash_attn_func with custom_vjp
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(3, 4, 5, 6, 9, 10, 11, 12, 13))
def flash_attn_func(
    q: jnp.ndarray,
    k: jnp.ndarray,
    v: jnp.ndarray,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    bias: Optional[jnp.ndarray] = None,
    alibi_slopes: Optional[jnp.ndarray] = None,
    deterministic: bool = True,
    return_lse: bool = False,
    return_attn_probs: bool = False,
    cu_seqlens_q: Optional[jnp.ndarray] = None,
    cu_seqlens_kv: Optional[jnp.ndarray] = None,
) -> tuple[jnp.ndarray, ...]:
    """Flash attention with automatic CK/ASM v3 dispatch via AITER.

    Args:
        q: (batch, seqlen_q, nheads, headdim_q)
        k: (batch, seqlen_k, nheads_k, headdim_q)
        v: (batch, seqlen_k, nheads_k, headdim_v)
        dropout_p: Dropout probability (0.0 during eval).
        softmax_scale: Scaling factor (default: 1/sqrt(headdim_q)).
        causal: Apply causal mask (bottom-right aligned).
        window_size: (left, right) for sliding window attention.
        bias: (seqlen_q, seqlen_k) attention bias.
        alibi_slopes: (nheads,) or (batch, nheads) ALiBi slopes.
        deterministic: Use deterministic backward (slower, more memory).
        return_lse: Return log-sum-exp values.
        return_attn_probs: Return attention probabilities (testing only).
    Returns:
        out: (batch, seqlen_q, nheads, headdim_v), or tuple if return_lse/return_attn_probs.
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    hd_q_og = q.shape[3]
    hd_v_og = v.shape[3]

    q_p, k_p, v_p = q, k, v
    if hd_q_og % 8 != 0:
        pad = 8 - hd_q_og % 8
        q_p = jnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, pad)))
        k_p = jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, pad)))
    if hd_v_og % 8 != 0:
        pad = 8 - hd_v_og % 8
        v_p = jnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, pad)))

    sk = k_p.shape[1]
    wl = -1 if window_size[0] >= sk else window_size[0]
    wr = -1 if window_size[1] >= sk else window_size[1]

    out_p, lse, s_dmask, _ = _flash_attn_forward(
        q_p, k_p, v_p, dropout_p, softmax_scale,
        causal=causal, wl=wl, wr=wr,
        bias=bias, alibi_slopes=alibi_slopes,
        return_lse=return_lse,
        return_softmax=return_attn_probs and dropout_p > 0,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv)

    out = out_p[..., :hd_v_og]
    result = [out]
    if return_lse:
        result.append(lse)
    if return_attn_probs:
        result.append(s_dmask)
    return tuple(result)


def _flash_attn_func_fwd(q, k, v,
                         dropout_p=0.0, softmax_scale=None, causal=False,
                         window_size=(-1, -1), bias=None, alibi_slopes=None,
                         deterministic=True, return_lse=False,
                         return_attn_probs=False,
                         cu_seqlens_q=None, cu_seqlens_kv=None):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    hd_q_og = q.shape[3]
    hd_v_og = v.shape[3]

    q_p, k_p, v_p = q, k, v
    if hd_q_og % 8 != 0:
        pad = 8 - hd_q_og % 8
        q_p = jnp.pad(q, ((0, 0), (0, 0), (0, 0), (0, pad)))
        k_p = jnp.pad(k, ((0, 0), (0, 0), (0, 0), (0, pad)))
    if hd_v_og % 8 != 0:
        pad = 8 - hd_v_og % 8
        v_p = jnp.pad(v, ((0, 0), (0, 0), (0, 0), (0, pad)))

    sk = k_p.shape[1]
    wl = -1 if window_size[0] >= sk else window_size[0]
    wr = -1 if window_size[1] >= sk else window_size[1]

    out_p, lse, s_dmask, rng_state = _flash_attn_forward(
        q_p, k_p, v_p, dropout_p, softmax_scale,
        causal=causal, wl=wl, wr=wr,
        bias=bias, alibi_slopes=alibi_slopes,
        return_lse=True, return_softmax=return_attn_probs and dropout_p > 0,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_kv)

    out = out_p[..., :hd_v_og]
    result = [out]
    if return_lse:
        result.append(lse)
    if return_attn_probs:
        result.append(s_dmask)
    result = tuple(result)

    residuals = (q_p, k_p, v_p, out_p, lse, rng_state,
                 dropout_p, softmax_scale, causal, (wl, wr),
                 bias, alibi_slopes, deterministic, hd_q_og, hd_v_og)
    return result, residuals


def _flash_attn_func_bwd(dropout_p, softmax_scale, causal, window_size,
                         deterministic, return_lse, return_attn_probs,
                         cu_seqlens_q, cu_seqlens_kv,
                         residuals, grad_outputs):
    (q_p, k_p, v_p, out_p, lse, rng_state,
     res_dp, res_scale, res_causal, res_ws,
     res_bias, res_alibi, res_det, hd_q_og, hd_v_og) = residuals

    dout = grad_outputs[0] if isinstance(grad_outputs, tuple) else grad_outputs
    if dout.shape[-1] != out_p.shape[-1]:
        pad = out_p.shape[-1] - dout.shape[-1]
        dout = jnp.pad(dout, ((0, 0), (0, 0), (0, 0), (0, pad)))

    dq_p, dk_p, dv_p, _, dbias = _flash_attn_backward(
        dout, q_p, k_p, v_p, out_p, lse,
        res_dp, res_scale, res_causal, res_ws[0], res_ws[1],
        res_bias, res_alibi, res_det, rng_state)

    dq = dq_p[..., :hd_q_og]
    dk = dk_p[..., :hd_q_og]
    dv = dv_p[..., :hd_v_og]

    return (dq, dk, dv, dbias, None)


flash_attn_func.defvjp(_flash_attn_func_fwd, _flash_attn_func_bwd)


# ---------------------------------------------------------------------------
# Varlen public API: flash_attn_varlen with custom_vjp
# ---------------------------------------------------------------------------

@partial(jax.custom_vjp, nondiff_argnums=(5, 6, 7, 8, 9, 10, 11, 12))
def flash_attn_varlen(
    q: jnp.ndarray,              # [total_q, nheads, headdim]
    k: jnp.ndarray,              # [total_k, nheads_k, headdim]
    v: jnp.ndarray,              # [total_k, nheads_k, headdim_v]
    cu_seqlens_q: jnp.ndarray,   # [batch_size + 1]
    cu_seqlens_k: jnp.ndarray,   # [batch_size + 1]
    max_seqlen_q: int = 0,
    max_seqlen_k: int = 0,
    dropout_p: float = 0.0,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    deterministic: bool = False,
    return_lse: bool = False,
) -> tuple[jnp.ndarray, ...]:
    """Variable-length flash attention using packed sequences.

    Args:
        q: [total_q, nheads, headdim] packed query tokens.
        k: [total_k, nheads_k, headdim] packed key tokens.
        v: [total_k, nheads_k, headdim_v] packed value tokens.
        cu_seqlens_q: [batch_size+1] cumulative sequence lengths for Q.
        cu_seqlens_k: [batch_size+1] cumulative sequence lengths for K.
        max_seqlen_q: Maximum query sequence length.
        max_seqlen_k: Maximum key sequence length.
    Returns:
        out: [total_q, nheads, headdim_v].
    """
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    hd_q_og = q.shape[-1]
    hd_v_og = v.shape[-1]

    q_p, k_p, v_p = q, k, v
    if hd_q_og % 8 != 0:
        pad = 8 - hd_q_og % 8
        q_p = jnp.pad(q, ((0, 0), (0, 0), (0, pad)))
        k_p = jnp.pad(k, ((0, 0), (0, 0), (0, pad)))
    if hd_v_og % 8 != 0:
        pad = 8 - hd_v_og % 8
        v_p = jnp.pad(v, ((0, 0), (0, 0), (0, pad)))

    wl = window_size[0]
    wr = window_size[1]

    out_p, lse, _, _ = mha_fwd_unified(
        q_p, k_p, v_p, dropout_p, softmax_scale, causal, wl, wr,
        return_lse=return_lse, return_softmax=False,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k)

    out = out_p[..., :hd_v_og]
    if return_lse:
        return (out, lse)
    return (out,)


def _flash_attn_varlen_fwd(q, k, v, cu_seqlens_q, cu_seqlens_k,
                           max_seqlen_q, max_seqlen_k, dropout_p,
                           softmax_scale, causal, window_size,
                           deterministic, return_lse):
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    hd_q_og = q.shape[-1]
    hd_v_og = v.shape[-1]

    q_p, k_p, v_p = q, k, v
    if hd_q_og % 8 != 0:
        pad = 8 - hd_q_og % 8
        q_p = jnp.pad(q, ((0, 0), (0, 0), (0, pad)))
        k_p = jnp.pad(k, ((0, 0), (0, 0), (0, pad)))
    if hd_v_og % 8 != 0:
        pad = 8 - hd_v_og % 8
        v_p = jnp.pad(v, ((0, 0), (0, 0), (0, pad)))

    wl, wr = window_size

    out_p, lse, _, rng_state = mha_fwd_unified(
        q_p, k_p, v_p, dropout_p, softmax_scale, causal, wl, wr,
        return_lse=True, return_softmax=False,
        cu_seqlens_q=cu_seqlens_q, cu_seqlens_kv=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q, max_seqlen_k=max_seqlen_k)

    out = out_p[..., :hd_v_og]
    result = (out, lse) if return_lse else (out,)

    residuals = (q_p, k_p, v_p, out_p, lse, rng_state,
                 cu_seqlens_q, cu_seqlens_k,
                 dropout_p, softmax_scale, causal, (wl, wr),
                 deterministic, hd_q_og, hd_v_og,
                 max_seqlen_q, max_seqlen_k)
    return result, residuals


def _flash_attn_varlen_bwd(max_seqlen_q, max_seqlen_k, dropout_p,
                           softmax_scale, causal, window_size,
                           deterministic, return_lse,
                           residuals, grad_outputs):
    (q_p, k_p, v_p, out_p, lse, rng_state,
     cu_sq, cu_sk, res_dp, res_scale, res_causal, res_ws,
     res_det, hd_q_og, hd_v_og,
     res_max_sq, res_max_sk) = residuals

    dout = grad_outputs[0] if isinstance(grad_outputs, tuple) else grad_outputs
    if dout.shape[-1] != out_p.shape[-1]:
        pad = out_p.shape[-1] - dout.shape[-1]
        dout = jnp.pad(dout, ((0, 0), (0, 0), (0, pad)))

    _, _, hq, dq = q_p.shape if q_p.ndim == 4 else (None, None, q_p.shape[1], q_p.shape[2])
    hk = k_p.shape[1] if k_p.ndim == 3 else k_p.shape[2]

    bf16_cvt = 0 if get_gfx() == "gfx950" else 1

    swa = (window_size[0] > 0) or (window_size[1] >= 0 and window_size[1] != -1)
    use_v3 = True
    if res_dp > 0:
        use_v3 = False
    if hq != hk:
        use_v3 = False
    if swa:
        use_v3 = False
    if causal and get_gfx() == "gfx950" and max_seqlen_k > 256:
        use_v3 = False
    if get_gfx() == "gfx950" and dq >= 96:
        use_v3 = False

    bwd_atomic = use_v3

    results = mha_bwd_unified(
        dout, q_p, k_p, v_p, out_p, lse,
        res_dp, res_scale, res_causal, res_ws[0], res_ws[1],
        res_det, use_v3, bwd_atomic, bf16_cvt,
        rng_state=rng_state,
        cu_seqlens_q=cu_sq, cu_seqlens_k=cu_sk,
        max_seqlen_q=res_max_sq, max_seqlen_k=res_max_sk)

    dq = results[0][..., :hd_q_og]
    dk = results[1][..., :hd_q_og]
    dv = results[2][..., :hd_v_og]

    return (dq, dk, dv, None, None)


flash_attn_varlen.defvjp(_flash_attn_varlen_fwd, _flash_attn_varlen_bwd)
