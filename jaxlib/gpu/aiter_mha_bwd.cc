// Copyright 2026 The JAX Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Unified MHA backward FFI handler for both batch and varlen modes.
// Detects mode from tensor rank: 4D = batch [b,s,h,d], 3D = varlen [total,h,d].
// Calls aiter::mha_bwd(args, stream) which handles CK vs ASM v3 internally.
//
// Multi-GPU note: AITER's fmha_v3_bwd impl_ptr_map is now protected by a
// mutex (Xinya's fix cherry-picked onto third_party/aiter), so ASM v3
// kernels can be used on all devices concurrently.

#include <hip/hip_runtime.h>
#include <cstdint>
#include <exception>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "aiter/mha_bwd.h"
#include "aiter_mha_common_utils.h"

namespace ffi = xla::ffi;

namespace {

  // Stream-ordered RAII workspace.
  //
  // hipMallocAsync / hipFreeAsync use the HIP runtime's per-device memory
  // pool, so allocations are cheap (cached) but still stream-ordered: the
  // allocation only becomes visible after the issuing stream reaches the
  // malloc, and the free happens after the stream reaches it.
  //
  // AsyncWorkspace replaces all of that.  Each handler invocation creates
  // its own AsyncWorkspaces; their destructors enqueue hipFreeAsync on
  // the same stream that issued the consuming kernel, so the buffer is
  // only released after the kernel has finished.  HIP errors are checked
  // and surfaced as ffi::Error rather than aborting.

  struct AsyncWorkspace {
    void *ptr_ = nullptr;
    size_t bytes_ = 0;
    hipStream_t stream_ = nullptr;

    AsyncWorkspace() = default;
    AsyncWorkspace(const AsyncWorkspace &) = delete;
    AsyncWorkspace &operator=(const AsyncWorkspace &) = delete;

    ~AsyncWorkspace() noexcept {
      if (ptr_ != nullptr) {
        // Best-effort: enqueue an async free on the consuming stream and
        // ignore any error here -- destructors must not throw across an
        // FFI boundary, and there is no recovery we can do anyway.
        (void)hipFreeAsync(ptr_, stream_);
      }
    }

    // Allocate `bytes` of device workspace ordered on `stream`.  When
    // `zero` is true the buffer is zero-initialised on the same stream.
    // Returns ffi::Error on failure with the workspace left empty.
    ffi::Error allocate(size_t bytes, hipStream_t stream, bool zero = true) {
      ptr_ = nullptr;
      bytes_ = 0;
      stream_ = stream;
      if (bytes == 0) return ffi::Error::Success();
      hipError_t err = hipMallocAsync(&ptr_, bytes, stream);
      if (err != hipSuccess) {
        ptr_ = nullptr;
        return ffi::Error(ffi::ErrorCode::kResourceExhausted,
                          std::string("hipMallocAsync(") +
                              std::to_string(bytes) +
                              ") failed: " + hipGetErrorString(err));
      }
      bytes_ = bytes;
      if (zero) {
        err = hipMemsetAsync(ptr_, 0, bytes, stream);
        if (err != hipSuccess) {
          (void)hipFreeAsync(ptr_, stream);
          ptr_ = nullptr;
          bytes_ = 0;
          return ffi::Error(ffi::ErrorCode::kInternal,
                            std::string("hipMemsetAsync failed: ") +
                                hipGetErrorString(err));
        }
      }
      return ffi::Error::Success();
    }

    void *ptr() const { return ptr_; }
  };

  size_t compute_dq_acc_size_unified(
      bool is_varlen, int64_t batch_size, int64_t seqlen_q_or_total,
      int64_t seqlen_k_or_max, int64_t num_heads, int64_t head_size,
      bool deterministic, bool use_asm_v3, bool is_v3_atomic_fp32,
      ffi::DataType q_dtype, std::vector<int64_t> &out_shape) {

    size_t elem_sz = 4;

    if (is_varlen) {
      // Varlen: 4D layout [split, total_q, nheads, head_size]
      if (!deterministic) {
        out_shape = {1, seqlen_q_or_total, num_heads, head_size};
      } else {
        int64_t kN0 = head_size <= 128 ? 128 : 64;
        int64_t nsplits = (seqlen_k_or_max + kN0 - 1) / kN0;
        out_shape = {nsplits, seqlen_q_or_total, num_heads, head_size};
      }
    } else {
      // Batch: 5D layout depends on path
      if (!deterministic) {
        if (use_asm_v3 && is_v3_atomic_fp32) {
          out_shape = {1, batch_size, num_heads, seqlen_q_or_total, head_size};
        } else if (use_asm_v3 && !is_v3_atomic_fp32) {
          int64_t sq_pad = ((seqlen_q_or_total + 15) / 16) * 16;
          int64_t pd = (head_size == 192) ? 192 : 128;
          out_shape = {1, batch_size, num_heads, sq_pad, pd};
          elem_sz = (q_dtype == ffi::DataType::F16 || q_dtype == ffi::DataType::BF16) ? 2 : 4;
        } else {
          out_shape = {1, batch_size, seqlen_q_or_total, num_heads, head_size};
        }
      } else {
        int64_t kN0 = head_size <= 128 ? 128 : 64;
        int64_t nsplits = (seqlen_k_or_max + kN0 - 1) / kN0;
        if (use_asm_v3) {
          out_shape = {nsplits, batch_size, num_heads, seqlen_q_or_total, head_size};
        } else {
          out_shape = {nsplits, batch_size, seqlen_q_or_total, num_heads, head_size};
        }
      }
    }

    size_t total = 1;
    for (auto d : out_shape) total *= d;
    return total * elem_sz;
  } // compute_dq_acc_size_unified
} // namespace

namespace jax_aiter {

ffi::Error aiter_mha_bwd_impl(
    hipStream_t stream,
    ffi::AnyBuffer dout, ffi::AnyBuffer q, ffi::AnyBuffer k, ffi::AnyBuffer v,
    ffi::AnyBuffer out, ffi::AnyBuffer softmax_lse,
    std::optional<ffi::AnyBuffer> cu_seqlens_q,
    std::optional<ffi::AnyBuffer> cu_seqlens_k,
    std::optional<ffi::AnyBuffer> dq, std::optional<ffi::AnyBuffer> dk,
    std::optional<ffi::AnyBuffer> dv,
    std::optional<ffi::AnyBuffer> bias, std::optional<ffi::AnyBuffer> alibi_slopes,
    std::optional<ffi::AnyBuffer> rng_state, std::optional<ffi::AnyBuffer> gen,
    ffi::Result<ffi::AnyBuffer> dq_ret, ffi::Result<ffi::AnyBuffer> dk_ret,
    ffi::Result<ffi::AnyBuffer> dv_ret, ffi::Result<ffi::AnyBuffer> softmax_d_ret,
    ffi::Result<ffi::AnyBuffer> dbias_ret,
    float dropout_p, float softmax_scale, bool is_causal,
    int window_size_left, int window_size_right, bool deterministic,
    bool use_asm_v3, bool is_v3_atomic_fp32, int how_v3_bf16_cvt,
    int max_seqlen_q_attr, int max_seqlen_k_attr, bool zero_tensors) {
  // Outermost try/catch: AITER and a few utility helpers throw
  // std::runtime_error.  XLA's FFI handlers are extern "C" thunks, and
  // unwinding past an extern "C" boundary is undefined behaviour, so we
  // must convert any exception into an ffi::Error here.
  try {

  if (!q.untyped_data() || !k.untyped_data() || !v.untyped_data() ||
      !out.untyped_data() || !softmax_lse.untyped_data() || !dout.untyped_data()) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "Required input buffer is null");
  }

  const int dev_idx = ::jax_aiter::device_from_ptr(q.untyped_data());
  if (dev_idx < 0)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "bad device from q");

  // Pin the HIP device context to the data device BEFORE any device
  // allocations or kernel launches happen.  Previously hipSetDevice was
  // called only just before aiter::mha_bwd, after several workspace
  // allocations had already executed -- which on multi-GPU setups would
  // place those workspaces on the wrong device.
  HIP_CHECK(hipSetDevice(dev_idx));

  auto q_dims = q.dimensions();
  auto k_dims = k.dimensions();
  auto v_dims = v.dimensions();
  auto dout_dims = dout.dimensions();
  auto out_dims = out.dimensions();
  auto lse_dims = softmax_lse.dimensions();

  const bool is_varlen = (q_dims.size() == 3);

  int64_t batch_size, seqlen_q, num_heads, head_size_q;
  int64_t seqlen_k, num_heads_k, head_size_v;
  int64_t max_sq, max_sk;

  if (is_varlen) {
    seqlen_q = q_dims[0]; // total_q
    num_heads = q_dims[1];
    head_size_q = q_dims[2];
    seqlen_k = k_dims[0]; // total_k
    num_heads_k = k_dims[1];
    head_size_v = v_dims[2];
    if (!cu_seqlens_q.has_value() || !mha_utils::is_valid_buffer(*cu_seqlens_q))
      return ffi::Error(ffi::ErrorCode::kInvalidArgument, "varlen requires cu_seqlens_q");
    batch_size = cu_seqlens_q->dimensions()[0] - 1;
    max_sq = max_seqlen_q_attr;
    max_sk = max_seqlen_k_attr;
  } else {
    batch_size = q_dims[0];
    seqlen_q = q_dims[1];
    num_heads = q_dims[2];
    head_size_q = q_dims[3];
    seqlen_k = k_dims[1];
    num_heads_k = k_dims[2];
    head_size_v = v_dims[3];
    max_sq = seqlen_q;
    max_sk = seqlen_k;
  }

  auto q_dtype = q.element_type();
  if (q_dtype != ffi::DataType::F16 && q_dtype != ffi::DataType::BF16) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "FlashAttention backward only supports fp16 and bf16");
  }
  if (k.element_type() != q_dtype || v.element_type() != q_dtype ||
      out.element_type() != q_dtype || dout.element_type() != q_dtype) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "Q, K, V, OUT, DOUT must have the same dtype");
  }

  if (max_sq == 0) {
    if (dq_ret->size_bytes() > 0)
      HIP_CHECK(hipMemsetAsync(dq_ret->untyped_data(), 0, dq_ret->size_bytes(), stream));
    if (dk_ret->size_bytes() > 0)
      HIP_CHECK(hipMemsetAsync(dk_ret->untyped_data(), 0, dk_ret->size_bytes(), stream));
    if (dv_ret->size_bytes() > 0)
      HIP_CHECK(hipMemsetAsync(dv_ret->untyped_data(), 0, dv_ret->size_bytes(), stream));
    if (softmax_d_ret->size_bytes() > 0)
      HIP_CHECK(hipMemsetAsync(softmax_d_ret->untyped_data(), 0, softmax_d_ret->size_bytes(), stream));
    if (dbias_ret->size_bytes() > 0)
      HIP_CHECK(hipMemsetAsync(dbias_ret->untyped_data(), 0, dbias_ret->size_bytes(), stream));
    return ffi::Error::Success();
  }

  if (!(dropout_p >= 0.0f && dropout_p < 1.0f)) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      "dropout_p must be in [0, 1)");
  }
  if (num_heads % num_heads_k != 0)
    return ffi::Error(ffi::ErrorCode::kInvalidArgument, "num_heads_q must be divisible by num_heads_k");

  bool is_mqa_gqa = (num_heads != num_heads_k);
  std::string dtype_str(mha_utils::dtype_to_string(q_dtype));

  int ref_sk = is_varlen ? max_sk : seqlen_k;
  if (window_size_left >= ref_sk) window_size_left = -1;
  if (window_size_right >= ref_sk) window_size_right = -1;
  int ref_sq = is_varlen ? max_sq : seqlen_q;

  auto mask = mha_utils::create_mask_info(is_causal, window_size_left, window_size_right, ref_sq, ref_sk);

  // Bias handling
  const void *bias_ptr = nullptr;
  ck_tile::index_t stride_bias = 0;
  bool has_bias = bias.has_value() && mha_utils::is_valid_buffer(*bias);
  bool has_alibi = alibi_slopes.has_value() && mha_utils::is_valid_buffer(*alibi_slopes);

  if (has_bias) {
    bias_ptr = bias->untyped_data();
    auto bd = bias->dimensions();
    stride_bias = bd.size() >= 2 ? mha_utils::calculate_stride(bd, 0) : 0;
  } else if (has_alibi) {
    bias_ptr = alibi_slopes->untyped_data();
    auto ad = alibi_slopes->dimensions();
    stride_bias = ad.size() >= 2 ? mha_utils::calculate_stride(ad, 0) : 0;
  }
  bias_enum bias_type = mha_utils::get_bias_type(has_bias, has_alibi);

  bool has_dbias = has_bias && (dbias_ret->size_bytes() > 0) && !is_varlen;
  AsyncWorkspace dbias_ws;
  void *dbias_expanded_ptr = nullptr;
  ck_tile::index_t stride_dbias = 0, nhead_stride_dbias = 0, batch_stride_dbias = 0;

  if (has_dbias) {
    size_t dbias_sz = batch_size * seqlen_q * num_heads * seqlen_k * mha_utils::dtype_size(q.element_type());
    if (auto err = dbias_ws.allocate(dbias_sz, stream); !err.success())
      return err;
    dbias_expanded_ptr = dbias_ws.ptr();
    stride_dbias = num_heads * seqlen_k;
    nhead_stride_dbias = seqlen_k;
    batch_stride_dbias = seqlen_q * num_heads * seqlen_k;
  }

  // RNG — use void* to avoid forming uint64_t* from untyped storage.
  AsyncWorkspace dummy_rng_ws;
  void *seed_ptr = nullptr, *offset_ptr = nullptr;
  if (dropout_p > 0.0f) {
    if (!rng_state.has_value() || !mha_utils::is_valid_buffer(*rng_state))
      return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                        "rng_state must be a valid buffer when dropout_p > 0");
    auto [s, o] = mha_utils::get_rng_seed_offset_ptrs(rng_state, dropout_p);
    seed_ptr = s; offset_ptr = o;
  } else {
    if (auto err = dummy_rng_ws.allocate(2 * sizeof(uint64_t), stream,
                                         /*zero=*/false);
        !err.success())
      return err;
    void *dummy_rng = dummy_rng_ws.ptr();
    seed_ptr = dummy_rng;
    offset_ptr = static_cast<char *>(dummy_rng) + sizeof(uint64_t);
  }

  // dq_acc
  std::vector<int64_t> dq_acc_shape;
  size_t dq_acc_bytes = compute_dq_acc_size_unified(
      is_varlen, batch_size, seqlen_q, is_varlen ? max_sk : seqlen_k,
      num_heads, head_size_q, deterministic, use_asm_v3, is_v3_atomic_fp32,
      q.element_type(), dq_acc_shape);

  AsyncWorkspace dq_acc_ws;
  if (auto err = dq_acc_ws.allocate(dq_acc_bytes, stream); !err.success())
    return err;
  void *dq_acc_ptr = dq_acc_ws.ptr();

  // dq_acc strides
  ck_tile::index_t split_stride_dq_acc = 1, batch_stride_dq_acc = 0;
  ck_tile::index_t nhead_stride_dq_acc = 1, stride_dq_acc = 1;

  int rank = dq_acc_shape.size();
  if (rank >= 4) {
    std::vector<ck_tile::index_t> strides(rank);
    strides[rank - 1] = 1;
    for (int i = rank - 2; i >= 0; i--)
      strides[i] = strides[i + 1] * dq_acc_shape[i + 1];

    split_stride_dq_acc = strides[0];
    if (is_varlen) {
      // [split, total_q, nheads, head] → strides[1]=total_q stride, strides[2]=nhead stride
      stride_dq_acc = strides[1];
      nhead_stride_dq_acc = strides[2];
    } else {
      batch_stride_dq_acc = strides[1];
      if (use_asm_v3) {
        // ASM v3: [split, batch, nheads, seqlen_q, head]
        nhead_stride_dq_acc = strides[2];
        stride_dq_acc = strides[3];
      } else {
        // CK: [split, batch, seqlen_q, nheads, head]
        stride_dq_acc = strides[2];
        nhead_stride_dq_acc = strides[3];
      }
    }
  }

  // MQA/GQA expansion
  auto dq_dims = dq_ret->dimensions();
  auto dk_dims = dk_ret->dimensions();
  auto dv_dims = dv_ret->dimensions();

  AsyncWorkspace dk_exp_ws, dv_exp_ws;
  void *dk_expanded_ptr = nullptr, *dv_expanded_ptr = nullptr;
  void *dk_final = dk_ret->untyped_data(), *dv_final = dv_ret->untyped_data();

  if (is_mqa_gqa) {
    size_t dk_sz = (is_varlen ? seqlen_k : batch_size * seqlen_k) * num_heads * head_size_q * mha_utils::dtype_size(q.element_type());
    size_t dv_sz = (is_varlen ? seqlen_k : batch_size * seqlen_k) * num_heads * head_size_v * mha_utils::dtype_size(v.element_type());
    if (auto err = dk_exp_ws.allocate(dk_sz, stream); !err.success())
      return err;
    if (auto err = dv_exp_ws.allocate(dv_sz, stream); !err.success())
      return err;
    dk_expanded_ptr = dk_exp_ws.ptr();
    dv_expanded_ptr = dv_exp_ws.ptr();
    dk_final = dk_expanded_ptr; dv_final = dv_expanded_ptr;
  }

  // Zero tensors (varlen)
  if (zero_tensors) {
    HIP_CHECK(hipMemsetAsync(dq_ret->untyped_data(), 0, dq_ret->size_bytes(), stream));
    HIP_CHECK(hipMemsetAsync(dk_final, 0, is_mqa_gqa ? ((is_varlen ? seqlen_k : batch_size * seqlen_k) * num_heads * head_size_q * mha_utils::dtype_size(q.element_type())) : dk_ret->size_bytes(), stream));
    HIP_CHECK(hipMemsetAsync(dv_final, 0, is_mqa_gqa ? ((is_varlen ? seqlen_k : batch_size * seqlen_k) * num_heads * head_size_v * mha_utils::dtype_size(v.element_type())) : dv_ret->size_bytes(), stream));
    HIP_CHECK(hipMemsetAsync(softmax_d_ret->untyped_data(), 0, softmax_d_ret->size_bytes(), stream));
  }

  float p_undrop = mha_utils::calculate_p_undrop(dropout_p);

  // Strides based on rank
  ck_tile::index_t stride_q, stride_k, stride_v, stride_o, stride_do, stride_dq, stride_dk, stride_dv;
  ck_tile::index_t nhs_q, nhs_k, nhs_v, nhs_o, nhs_do, nhs_lse, nhs_dq, nhs_dk, nhs_dv;
  ck_tile::index_t bs_q = 0, bs_k = 0, bs_v = 0, bs_o = 0, bs_do = 0, bs_lse = 0, bs_dq = 0, bs_dk = 0, bs_dv = 0;

  if (is_varlen) {
    stride_q = mha_utils::calculate_stride(q_dims, 0);
    stride_k = mha_utils::calculate_stride(k_dims, 0);
    stride_v = mha_utils::calculate_stride(v_dims, 0);
    stride_o = mha_utils::calculate_stride(out_dims, 0);
    stride_do = mha_utils::calculate_stride(dout_dims, 0);
    stride_dq = mha_utils::calculate_stride(dq_dims, 0);
    stride_dk = is_mqa_gqa ? (num_heads * head_size_q) : mha_utils::calculate_stride(dk_dims, 0);
    stride_dv = is_mqa_gqa ? (num_heads * head_size_v) : mha_utils::calculate_stride(dv_dims, 0);
    nhs_q = mha_utils::calculate_stride(q_dims, 1);
    nhs_k = mha_utils::calculate_stride(k_dims, 1);
    nhs_v = mha_utils::calculate_stride(v_dims, 1);
    nhs_o = mha_utils::calculate_stride(out_dims, 1);
    nhs_do = mha_utils::calculate_stride(dout_dims, 1);
    nhs_lse = mha_utils::calculate_stride(lse_dims, 0);
    nhs_dq = mha_utils::calculate_stride(dq_dims, 1);
    nhs_dk = is_mqa_gqa ? head_size_q : mha_utils::calculate_stride(dk_dims, 1);
    nhs_dv = is_mqa_gqa ? head_size_v : mha_utils::calculate_stride(dv_dims, 1);
  } else {
    stride_q = mha_utils::calculate_stride(q_dims, 1);
    stride_k = mha_utils::calculate_stride(k_dims, 1);
    stride_v = mha_utils::calculate_stride(v_dims, 1);
    stride_o = mha_utils::calculate_stride(out_dims, 1);
    stride_do = mha_utils::calculate_stride(dout_dims, 1);
    stride_dq = mha_utils::calculate_stride(dq_dims, 1);
    stride_dk = is_mqa_gqa ? (num_heads * head_size_q) : mha_utils::calculate_stride(dk_dims, 1);
    stride_dv = is_mqa_gqa ? (num_heads * head_size_v) : mha_utils::calculate_stride(dv_dims, 1);
    nhs_q = mha_utils::calculate_stride(q_dims, 2);
    nhs_k = mha_utils::calculate_stride(k_dims, 2);
    nhs_v = mha_utils::calculate_stride(v_dims, 2);
    nhs_o = mha_utils::calculate_stride(out_dims, 2);
    nhs_do = mha_utils::calculate_stride(dout_dims, 2);
    nhs_lse = mha_utils::calculate_stride(lse_dims, 1);
    nhs_dq = mha_utils::calculate_stride(dq_dims, 2);
    nhs_dk = is_mqa_gqa ? head_size_q : mha_utils::calculate_stride(dk_dims, 2);
    nhs_dv = is_mqa_gqa ? head_size_v : mha_utils::calculate_stride(dv_dims, 2);
    bs_q = mha_utils::calculate_stride(q_dims, 0);
    bs_k = mha_utils::calculate_stride(k_dims, 0);
    bs_v = mha_utils::calculate_stride(v_dims, 0);
    bs_o = mha_utils::calculate_stride(out_dims, 0);
    bs_do = mha_utils::calculate_stride(dout_dims, 0);
    bs_lse = mha_utils::calculate_stride(lse_dims, 0);
    bs_dq = mha_utils::calculate_stride(dq_dims, 0);
    bs_dk = is_mqa_gqa ? (seqlen_k * num_heads * head_size_q) : mha_utils::calculate_stride(dk_dims, 0);
    bs_dv = is_mqa_gqa ? (seqlen_k * num_heads * head_size_v) : mha_utils::calculate_stride(dv_dims, 0);
  }

  // Seqstart pointers
  const void *seqstart_q_ptr = nullptr, *seqstart_k_ptr = nullptr;
  if (is_varlen) {
    seqstart_q_ptr = cu_seqlens_q->untyped_data();
    if (cu_seqlens_k.has_value() && mha_utils::is_valid_buffer(*cu_seqlens_k))
      seqstart_k_ptr = cu_seqlens_k->untyped_data();
  }

  auto args = aiter::mha_bwd_args{
      .use_asm_v3 = use_asm_v3,
      .v3_atomic_fp32 = is_v3_atomic_fp32,
      .v3_bf16_cvt = how_v3_bf16_cvt,
      .v3_api_check = false,
      .hdim_q = static_cast<int>(head_size_q),
      .hdim_v = static_cast<int>(head_size_v),
      .data_type = std::move(dtype_str),
      .is_group_mode = is_varlen,
      .mask_type = static_cast<int>(mask.type),
      .bias_type = static_cast<int>(bias_type),
      .has_dbias = has_dbias,
      .has_dropout = (dropout_p > 0.0f),
      .is_store_randval = false,
      .is_deterministic = deterministic,
      .q_ptr = q.untyped_data(), .k_ptr = k.untyped_data(),
      .v_ptr = v.untyped_data(), .bias_ptr = bias_ptr,
      .o_ptr = out.untyped_data(), .lse_ptr = softmax_lse.untyped_data(),
      .do_ptr = dout.untyped_data(), .d_ptr = softmax_d_ret->untyped_data(),
      .rand_val_ptr = nullptr,
      .dq_ptr = dq_ret->untyped_data(), .dk_ptr = dk_final, .dv_ptr = dv_final,
      .dbias_ptr = dbias_expanded_ptr, .dq_acc_ptr = dq_acc_ptr,
      .seqstart_q_ptr = seqstart_q_ptr, .seqstart_k_ptr = seqstart_k_ptr,
      .seqlen_q = static_cast<int>(seqlen_q), .seqlen_k = static_cast<int>(seqlen_k),
      .batch = static_cast<int>(batch_size),
      .max_seqlen_q = static_cast<int>(max_sq), .max_seqlen_k = static_cast<int>(max_sk),
      .nhead_q = static_cast<int>(num_heads), .nhead_k = static_cast<int>(num_heads_k),
      .scale = softmax_scale,
      .stride_q = static_cast<int>(stride_q), .stride_k = static_cast<int>(stride_k),
      .stride_v = static_cast<int>(stride_v), .stride_bias = static_cast<int>(stride_bias),
      .stride_o = static_cast<int>(stride_o), .stride_randval = 0,
      .stride_do = static_cast<int>(stride_do),
      .stride_dq_acc = static_cast<int>(stride_dq_acc),
      .stride_dq = static_cast<int>(stride_dq), .stride_dk = static_cast<int>(stride_dk),
      .stride_dv = static_cast<int>(stride_dv), .stride_dbias = static_cast<int>(stride_dbias),
      .nhead_stride_q = static_cast<int>(nhs_q), .nhead_stride_k = static_cast<int>(nhs_k),
      .nhead_stride_v = static_cast<int>(nhs_v), .nhead_stride_bias = 0,
      .nhead_stride_o = static_cast<int>(nhs_o), .nhead_stride_randval = 0,
      .nhead_stride_do = static_cast<int>(nhs_do),
      .nhead_stride_lsed = static_cast<int>(nhs_lse),
      .nhead_stride_dq_acc = static_cast<int64_t>(nhead_stride_dq_acc),
      .nhead_stride_dq = static_cast<int>(nhs_dq),
      .nhead_stride_dk = static_cast<int>(nhs_dk), .nhead_stride_dv = static_cast<int>(nhs_dv),
      .nhead_stride_dbias = static_cast<int>(nhead_stride_dbias),
      .batch_stride_q = static_cast<int>(bs_q), .batch_stride_k = static_cast<int>(bs_k),
      .batch_stride_v = static_cast<int>(bs_v), .batch_stride_bias = 0,
      .batch_stride_o = static_cast<int>(bs_o), .batch_stride_randval = 0,
      .batch_stride_do = static_cast<int>(bs_do),
      .batch_stride_lsed = static_cast<int>(bs_lse),
      .batch_stride_dq_acc = static_cast<int64_t>(batch_stride_dq_acc),
      .batch_stride_dq = static_cast<int>(bs_dq),
      .batch_stride_dk = static_cast<int>(bs_dk), .batch_stride_dv = static_cast<int>(bs_dv),
      .batch_stride_dbias = static_cast<int>(batch_stride_dbias),
      .split_stride_dq_acc = static_cast<int>(split_stride_dq_acc),
      .window_size_left = static_cast<int>(mask.left),
      .window_size_right = static_cast<int>(mask.right),
      .p_drop = dropout_p, .p_undrop = p_undrop,
      .drop_seed_offset = std::pair<const void *, const void *>(
          seed_ptr, offset_ptr)
  };

  // Note: hipSetDevice(dev_idx) was already called at the top of the
  // handler so any AITER kernel loads / launches target the data device.

  auto stream_config = mha_utils::create_stream_config(stream);
  float runtime = aiter::mha_bwd(args, stream_config);

  if (runtime < 0) {
    return ffi::Error(ffi::ErrorCode::kInternal, "aiter::mha_bwd failed");
  }

  // MQA/GQA reduction
  if (is_mqa_gqa) {
    int64_t groups = num_heads / num_heads_k;
    int64_t total_tokens = is_varlen ? seqlen_k : batch_size * seqlen_k;

    auto dk_err = mha_utils::launch_mqa_gqa_reduction(
        dk_expanded_ptr, dk_ret->untyped_data(),
        is_varlen ? 1 : batch_size,
        is_varlen ? seqlen_k : seqlen_k,
        num_heads, num_heads_k, head_size_q, groups, q.element_type(), stream);
    if (!dk_err.success()) return dk_err;
    auto dv_err = mha_utils::launch_mqa_gqa_reduction(
        dv_expanded_ptr, dv_ret->untyped_data(),
        is_varlen ? 1 : batch_size,
        is_varlen ? seqlen_k : seqlen_k,
        num_heads, num_heads_k, head_size_v, groups, v.element_type(), stream);
    if (!dv_err.success()) return dv_err;

    // dk/dv expanded buffers freed asynchronously by AsyncWorkspace dtors
  }

  if (has_dbias && dbias_expanded_ptr) {
    size_t dbias_size = batch_size * seqlen_q * num_heads * seqlen_k * mha_utils::dtype_size(q.element_type());
    HIP_CHECK(hipMemcpyAsync(dbias_ret->untyped_data(), dbias_expanded_ptr,
                             dbias_size, hipMemcpyDeviceToDevice, stream));
  }
  // All workspace AsyncWorkspaces are freed (stream-ordered) at scope exit.

  return ffi::Error::Success();

  } catch (const std::exception &e) {
    return ffi::Error(ffi::ErrorCode::kInvalidArgument,
                      std::string("aiter_mha_bwd: ") + e.what());
  } catch (...) {
    return ffi::Error(ffi::ErrorCode::kInternal,
                      "aiter_mha_bwd: unknown C++ exception");
  }
} // aiter_mha_bwd_impl

} // namespace jax_aiter

#pragma GCC visibility push(default)

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    aiter_mha_bwd, jax_aiter::aiter_mha_bwd_impl,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<hipStream_t>>()
        .Arg<ffi::AnyBuffer>() // dout
        .Arg<ffi::AnyBuffer>() // q
        .Arg<ffi::AnyBuffer>() // k
        .Arg<ffi::AnyBuffer>() // v
        .Arg<ffi::AnyBuffer>() // out
        .Arg<ffi::AnyBuffer>() // softmax_lse
        .Arg<ffi::AnyBuffer>() // cu_seqlens_q (optional)
        .Arg<ffi::AnyBuffer>() // cu_seqlens_k (optional)
        .Arg<ffi::AnyBuffer>() // dq (optional)
        .Arg<ffi::AnyBuffer>() // dk (optional)
        .Arg<ffi::AnyBuffer>() // dv (optional)
        .Arg<ffi::AnyBuffer>() // bias (optional)
        .Arg<ffi::AnyBuffer>() // alibi_slopes (optional)
        .Arg<ffi::AnyBuffer>() // rng_state (optional)
        .Arg<ffi::AnyBuffer>() // gen (optional)
        .Ret<ffi::AnyBuffer>() // dq_ret
        .Ret<ffi::AnyBuffer>() // dk_ret
        .Ret<ffi::AnyBuffer>() // dv_ret
        .Ret<ffi::AnyBuffer>() // softmax_d_ret
        .Ret<ffi::AnyBuffer>() // dbias_ret
        .Attr<float>("dropout_p")
        .Attr<float>("softmax_scale")
        .Attr<bool>("is_causal")
        .Attr<int>("window_size_left")
        .Attr<int>("window_size_right")
        .Attr<bool>("deterministic")
        .Attr<bool>("use_asm_v3")
        .Attr<bool>("is_v3_atomic_fp32")
        .Attr<int>("how_v3_bf16_cvt")
        .Attr<int>("max_seqlen_q_attr")
        .Attr<int>("max_seqlen_k_attr")
        .Attr<bool>("zero_tensors"),
    {xla::ffi::Traits::kCmdBufferCompatible});

#pragma GCC visibility pop