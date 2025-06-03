/* Copyright 2022 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "jaxlib/gpu/rnn_kernels.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "jaxlib/gpu/ffi_wrapper.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/handle_pool.h"
#include "jaxlib/gpu/vendor.h"
#include "jaxlib/kernel_helpers.h"

namespace jax {

namespace JAX_GPU_NAMESPACE {

std::string ErrorString(gpudnnStatus_t status) {
  return gpudnnGetErrorString(status);
}

template <typename T>
std::string ErrorString(T status, const char* file, std::int64_t line,
                        const char* expr) {
  return absl::StrFormat("%s:%d: operation %s failed: %s", file, line, expr,
                         ErrorString(status));
}

absl::Status AsStatus(gpudnnStatus_t status, const char* file,
                      std::int64_t line, const char* expr) {
  if (status != GPUDNN_STATUS_SUCCESS)
    return absl::InternalError(ErrorString(status, file, line, expr));
  return absl::OkStatus();
}
}  // namespace JAX_GPU_NAMESPACE

using DnnHandlePool = HandlePool<gpudnnHandle_t, gpuStream_t>;

template <>
/*static*/ absl::StatusOr<DnnHandlePool::Handle> DnnHandlePool::Borrow(
    gpuStream_t stream) {
  DnnHandlePool* pool = Instance();
  absl::MutexLock lock(&pool->mu_);
  gpudnnHandle_t handle;
  if (pool->handles_[stream].empty()) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnCreate(&handle)));
  } else {
    handle = pool->handles_[stream].back();
    pool->handles_[stream].pop_back();
  }
  if (stream) {
    JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetStream(handle, stream)));
  }
  return Handle(pool, handle, stream);
}

namespace JAX_GPU_NAMESPACE {

static absl::StatusOr<std::pair<size_t, size_t>>
DoRnnComputeWorkspaceReserveSpaceSizes(int input_size, int hidden_size,
                                       int num_layers, int batch_size,
                                       int max_seq_length, float dropout,
                                       bool bidirectional,
                                       bool cudnn_allow_tf32) {
  auto h = DnnHandlePool::Borrow(/*stream=*/nullptr);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  gpudnnRNNDescriptor_t rnn_desc;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnCreateRNNDescriptor(&rnn_desc)));

  gpudnnDropoutDescriptor_t dropout_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnCreateDropoutDescriptor(&dropout_desc)));
  size_t state_size;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnDropoutGetStatesSize(handle.get(), &state_size)));

#ifdef JAX_GPU_HIP
  void* dropout_states_dev = nullptr;
  // Allocate minimal memory for dropout states (can be very small since it's
  // not used)
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpuMalloc(&dropout_states_dev, state_size)));
  if (!dropout_states_dev) {
    return absl::InternalError(
        "Failed to allocate minimal GPU memory for dropout states.");
  }
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetDropoutDescriptor(
      dropout_desc, handle.get(), dropout, dropout_states_dev, state_size, 123,
      false, false, MIOPEN_RNG_PSEUDO_XORWOW)));
#else   // JAX_GPU_CUDA
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetDropoutDescriptor(
      dropout_desc, handle.get(), dropout, nullptr, state_size, 123)));
#endif  // JAX_GPU_HIP

  // TODO(zhangqiaorjc): Handle other kinds of RNN.
  gpudnnRNNMode_t cell_mode = GPUDNN_LSTM;
  gpudnnRNNBiasMode_t bias_mode = GPUDNN_RNN_DOUBLE_BIAS;
  int num_directions = 1;
  gpudnnDirectionMode_t dir_mode = GPUDNN_UNIDIRECTIONAL;
  if (bidirectional) {
    dir_mode = GPUDNN_BIDIRECTIONAL;
    num_directions = 2;
  }
  gpudnnRNNInputMode_t input_mode = GPUDNN_LINEAR_INPUT;
  gpudnnDataType_t data_type = GPUDNN_DATA_FLOAT;

#ifdef JAX_GPU_HIP
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetRNNDescriptor(
      rnn_desc, hidden_size, num_layers, dropout_desc, input_mode, dir_mode,
      cell_mode, bias_mode, GPUDNN_RNN_ALGO_STANDARD, data_type)));
#else   // JAX_GPU_CUDA
  gpudnnDataType_t math_prec = GPUDNN_DATA_FLOAT;
  gpudnnMathType_t math_type =
      cudnn_allow_tf32 ? GPUDNN_DEFAULT_MATH : GPUDNN_FMA_MATH;
  int32_t proj_size = hidden_size;
  uint32_t aux_flags = GPUDNN_RNN_PADDED_IO_ENABLED;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetRNNDescriptor(
      rnn_desc, GPUDNN_RNN_ALGO_STANDARD, cell_mode, bias_mode, dir_mode,
      input_mode, data_type, math_prec, math_type, input_size, hidden_size,
      proj_size, num_layers, dropout_desc, aux_flags)));
#endif  // JAX_GPU_HIP

  gpudnnForwardMode_t fwdMode = GPUDNN_FWD_MODE_TRAINING;
  gpudnnRNNDataLayout_t layout = GPUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
  float padding = 0.0f;

  std::vector<int32_t> seq_length_vector(batch_size, max_seq_length);
  int32_t* seq_length_array = &seq_length_vector[0];

  gpudnnRNNDataDescriptor_t input_data_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnCreateRNNDataDescriptor(&input_data_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetRNNDataDescriptor(
      input_data_desc, data_type, layout, max_seq_length, batch_size,
      input_size, seq_length_array, &padding)));

  size_t workSpaceSize;
  size_t reserveSpaceSize;
#ifdef JAX_GPU_HIP
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpudnnGetRNNTempSpaceSizes(handle.get(), rnn_desc, input_data_desc,
                                 fwdMode, &workSpaceSize, &reserveSpaceSize)));
#else   // JAX_GPU_CUDA
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnGetRNNTempSpaceSizes(
      handle.get(), rnn_desc, fwdMode, input_data_desc, &workSpaceSize,
      &reserveSpaceSize)));
#endif  // JAX_GPU_HIP
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnDestroyDropoutDescriptor(dropout_desc)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnDestroyRNNDataDescriptor(input_data_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnDestroyRNNDescriptor(rnn_desc)));

  // Round up to nearest multiples of 4 so we can return them as f32 arrays.
  workSpaceSize += (workSpaceSize % 4);
  reserveSpaceSize += (reserveSpaceSize % 4);
  return std::make_pair(workSpaceSize, reserveSpaceSize);
}

absl::StatusOr<std::pair<size_t, size_t>> RnnComputeWorkspaceReserveSpaceSizes(
    int input_size, int hidden_size, int num_layers, int batch_size,
    int max_seq_length, float dropout, bool bidirectional,
    bool cudnn_allow_tf32) {
  return DoRnnComputeWorkspaceReserveSpaceSizes(
      input_size, hidden_size, num_layers, batch_size, max_seq_length, dropout,
      bidirectional, cudnn_allow_tf32);
}

static absl::Status DnnRNNForward_(gpuStream_t stream, void** buffers,
                                   const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<RnnDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const RnnDescriptor& d = **s;
  auto h = DnnHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  gpudnnRNNDescriptor_t rnn_desc;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnCreateRNNDescriptor(&rnn_desc)));

  gpudnnDropoutDescriptor_t dropout_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnCreateDropoutDescriptor(&dropout_desc)));
  size_t state_size;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnDropoutGetStatesSize(handle.get(), &state_size)));

#ifdef JAX_GPU_HIP
  void* dropout_states_dev = nullptr;
  // Allocate minimal memory for dropout states (can be very small since it's
  // not used).
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpuMalloc(&dropout_states_dev, state_size)));
  if (!dropout_states_dev) {
    return absl::InternalError(
        "Failed to allocate minimal GPU memory for dropout states.");
  }
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetDropoutDescriptor(
      dropout_desc, handle.get(), d.dropout, dropout_states_dev, state_size,
      123, false, false, MIOPEN_RNG_PSEUDO_XORWOW)));
#else   // JAX_GPU_CUDA
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetDropoutDescriptor(
      dropout_desc, handle.get(), d.dropout, nullptr, state_size, 123)));
#endif  // JAX_GPU_HIP

  // TODO(zhangqiaorjc): Handle other kinds of RNN.
  gpudnnRNNMode_t cell_mode = GPUDNN_LSTM;
  gpudnnRNNBiasMode_t bias_mode = GPUDNN_RNN_DOUBLE_BIAS;
  int num_directions = 1;
  gpudnnDirectionMode_t dir_mode = GPUDNN_UNIDIRECTIONAL;
  if (d.bidirectional) {
    dir_mode = GPUDNN_BIDIRECTIONAL;
    num_directions = 2;
  }
  gpudnnRNNInputMode_t input_mode = GPUDNN_LINEAR_INPUT;
  gpudnnDataType_t data_type = GPUDNN_DATA_FLOAT;

#ifdef JAX_GPU_HIP
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetRNNDescriptor(
      rnn_desc, d.hidden_size, d.num_layers, dropout_desc, input_mode, dir_mode,
      cell_mode, bias_mode, GPUDNN_RNN_ALGO_STANDARD, data_type)));
#else   // JAX_GPU_CUDA
  gpudnnDataType_t math_prec = GPUDNN_DATA_FLOAT;
  gpudnnMathType_t math_type =
      d.cudnn_allow_tf32 ? GPUDNN_DEFAULT_MATH : GPUDNN_FMA_MATH;
  int32_t proj_size = d.hidden_size;
  uint32_t aux_flags = GPUDNN_RNN_PADDED_IO_ENABLED;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetRNNDescriptor(
      rnn_desc, GPUDNN_RNN_ALGO_STANDARD, cell_mode, bias_mode, dir_mode,
      input_mode, data_type, math_prec, math_type, d.input_size, d.hidden_size,
      proj_size, d.num_layers, dropout_desc, aux_flags)));
#endif  // JAX_GPU_HIP

  gpudnnForwardMode_t fwdMode = GPUDNN_FWD_MODE_TRAINING;
  gpudnnRNNDataLayout_t layout = GPUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
  float padding = 0.0f;

  // TODO(zhangqiaorjc): Avoid this cudaMemcpy if possible.
  auto seq_lengths_buf = buffers[4];
  std::vector<int32_t> seq_length_vector(d.batch_size, 0);
  int32_t* seq_length_array = &seq_length_vector[0];
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpuMemcpyAsync(seq_length_array, seq_lengths_buf,
                                   seq_length_vector.size() * sizeof(int32_t),
                                   gpuMemcpyDeviceToHost, stream)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuStreamSynchronize(stream)));

  gpudnnRNNDataDescriptor_t input_data_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnCreateRNNDataDescriptor(&input_data_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetRNNDataDescriptor(
      input_data_desc, data_type, layout, d.max_seq_length, d.batch_size,
      d.input_size, seq_length_array, &padding)));

  gpudnnRNNDataDescriptor_t output_data_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnCreateRNNDataDescriptor(&output_data_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetRNNDataDescriptor(
      output_data_desc, data_type, layout, d.max_seq_length, d.batch_size,
      d.hidden_size * num_directions, seq_length_array, &padding)));

  // Shape is (num_directions * num_layers, batch_size, hidden_size)
  int dims[3];
  dims[0] = num_directions * d.num_layers;
  dims[1] = d.batch_size;
  dims[2] = d.hidden_size;
  int strides[3];
  strides[0] = dims[1] * dims[2];
  strides[1] = dims[2];
  strides[2] = 1;
  gpudnnTensorDescriptor_t h_desc;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnCreateTensorDescriptor(&h_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpudnnSetTensorNdDescriptor(h_desc, data_type, 3, dims, strides)));

  gpudnnTensorDescriptor_t c_desc;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnCreateTensorDescriptor(&c_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpudnnSetTensorNdDescriptor(c_desc, data_type, 3, dims, strides)));

  size_t weight_space_size;
#ifdef JAX_GPU_HIP
  miopenTensorDescriptor_t input_tensor_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(miopenCreateTensorDescriptor(&input_tensor_desc)));
  int dimsA[2] = {d.batch_size, d.input_size};
  int stridesA[2] = {dimsA[1], 1};  // Row-major order, similar to GPUDNN
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(miopenSetTensorDescriptor(
      input_tensor_desc, data_type, 2, dimsA, stridesA)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpudnnGetRNNWeightSpaceSize(handle.get(), rnn_desc, input_tensor_desc,
                                  &weight_space_size, data_type)));
#else   // JAX_GPU_CUDA
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpudnnGetRNNWeightSpaceSize(handle.get(), rnn_desc, &weight_space_size)));
#endif  // JAX_GPU_HIP

  auto input_buf = buffers[0];
  auto h_0_buf = buffers[1];
  auto c_0_buf = buffers[2];
  auto weights_buf = buffers[3];
  auto output_buf = buffers[5];
  auto h_n_buf = buffers[6];
  auto c_n_buf = buffers[7];
  auto workspace_buf = buffers[8];
  auto reserve_space_buf = buffers[9];

#ifdef JAX_GPU_HIP
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnRNNForward(
      handle.get(), rnn_desc, fwdMode, input_data_desc, input_buf, h_desc,
      h_0_buf, h_n_buf, c_desc, c_0_buf, c_n_buf, output_data_desc, output_buf,
      weights_buf, weight_space_size, workspace_buf, d.workspace_size,
      reserve_space_buf, d.reserve_space_size)));
#else   // JAX_GPU_CUDA
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnRNNForward(
      handle.get(), rnn_desc, fwdMode, (const int32_t*)seq_lengths_buf,
      input_data_desc, input_buf, output_data_desc, output_buf, h_desc, h_0_buf,
      h_n_buf, c_desc, c_0_buf, c_n_buf, weight_space_size, weights_buf,
      d.workspace_size, workspace_buf, d.reserve_space_size,
      reserve_space_buf)));
#endif  // JAX_GPU_HIP

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnDestroyTensorDescriptor(h_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnDestroyTensorDescriptor(c_desc)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnDestroyRNNDataDescriptor(input_data_desc)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnDestroyRNNDataDescriptor(output_data_desc)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnDestroyDropoutDescriptor(dropout_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnDestroyRNNDescriptor(rnn_desc)));

  return absl::OkStatus();
}

static absl::Status DnnRNNBackward_(gpuStream_t stream, void** buffers,
                                    const char* opaque, size_t opaque_len) {
  auto s = UnpackDescriptor<RnnDescriptor>(opaque, opaque_len);
  JAX_RETURN_IF_ERROR(s.status());
  const RnnDescriptor& d = **s;
  auto h = DnnHandlePool::Borrow(stream);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  gpudnnRNNDescriptor_t rnn_desc;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnCreateRNNDescriptor(&rnn_desc)));

  gpudnnDropoutDescriptor_t dropout_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnCreateDropoutDescriptor(&dropout_desc)));
  size_t state_size;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnDropoutGetStatesSize(handle.get(), &state_size)));

#ifdef JAX_GPU_HIP
  void* dropout_states_dev = nullptr;
  // Allocate minimal memory for dropout states (can be very small since it's
  // not used)
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpuMalloc(&dropout_states_dev, state_size)));
  if (!dropout_states_dev) {
    return absl::InternalError(
        "Failed to allocate minimal GPU memory for dropout states.");
  }
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetDropoutDescriptor(
      dropout_desc, handle.get(), d.dropout, dropout_states_dev, state_size,
      123, false, false, MIOPEN_RNG_PSEUDO_XORWOW)));
#else   // JAX_GPU_CUDA
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetDropoutDescriptor(
      dropout_desc, handle.get(), d.dropout, nullptr, state_size, 123)));
#endif  // JAX_GPU_HIP

  // TODO(zhangqiaorjc): Handle other kinds of RNN.
  gpudnnRNNMode_t cell_mode = GPUDNN_LSTM;
  gpudnnRNNBiasMode_t bias_mode = GPUDNN_RNN_DOUBLE_BIAS;
  int num_directions = 1;
  gpudnnDirectionMode_t dir_mode = GPUDNN_UNIDIRECTIONAL;
  if (d.bidirectional) {
    dir_mode = GPUDNN_BIDIRECTIONAL;
    num_directions = 2;
  }
  gpudnnRNNInputMode_t input_mode = GPUDNN_LINEAR_INPUT;
  gpudnnDataType_t data_type = GPUDNN_DATA_FLOAT;

#ifdef JAX_GPU_HIP
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetRNNDescriptor(
      rnn_desc, d.hidden_size, d.num_layers, dropout_desc, input_mode, dir_mode,
      cell_mode, bias_mode, GPUDNN_RNN_ALGO_STANDARD, data_type)));
#else   // JAX_GPU_CUDA
  gpudnnDataType_t math_prec = GPUDNN_DATA_FLOAT;
  gpudnnMathType_t math_type =
      d.cudnn_allow_tf32 ? GPUDNN_DEFAULT_MATH : GPUDNN_FMA_MATH;
  int32_t proj_size = d.hidden_size;
  uint32_t aux_flags = GPUDNN_RNN_PADDED_IO_ENABLED;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetRNNDescriptor(
      rnn_desc, GPUDNN_RNN_ALGO_STANDARD, cell_mode, bias_mode, dir_mode,
      input_mode, data_type, math_prec, math_type, d.input_size, d.hidden_size,
      proj_size, d.num_layers, dropout_desc, aux_flags)));
#endif  // JAX_GPU_HIP

  gpudnnRNNDataLayout_t layout = GPUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
  float padding = 0.0f;

  auto seq_lengths_buf = buffers[10];
  std::vector<int32_t> seq_length_vector(d.batch_size, d.max_seq_length);
  int32_t* seq_length_array = &seq_length_vector[0];
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpuMemcpyAsync(seq_length_array, seq_lengths_buf,
                                   seq_length_vector.size() * sizeof(int32_t),
                                   gpuMemcpyDeviceToHost, stream)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuStreamSynchronize(stream)));

  gpudnnRNNDataDescriptor_t input_data_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnCreateRNNDataDescriptor(&input_data_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetRNNDataDescriptor(
      input_data_desc, data_type, layout, d.max_seq_length, d.batch_size,
      d.input_size, seq_length_array, &padding)));

  gpudnnRNNDataDescriptor_t output_data_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnCreateRNNDataDescriptor(&output_data_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnSetRNNDataDescriptor(
      output_data_desc, data_type, layout, d.max_seq_length, d.batch_size,
      d.hidden_size * num_directions, seq_length_array, &padding)));

  // Shape is (num_directions * num_layers, batch_size, hidden_size)
  int dims[3];
  dims[0] = num_directions * d.num_layers;
  dims[1] = d.batch_size;
  dims[2] = d.hidden_size;
  int strides[3];
  strides[0] = dims[1] * dims[2];
  strides[1] = dims[2];
  strides[2] = 1;
  gpudnnTensorDescriptor_t h_desc;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnCreateTensorDescriptor(&h_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpudnnSetTensorNdDescriptor(h_desc, data_type, 3, dims, strides)));

  gpudnnTensorDescriptor_t c_desc;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnCreateTensorDescriptor(&c_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpudnnSetTensorNdDescriptor(c_desc, data_type, 3, dims, strides)));

  size_t weight_space_size;
#ifdef JAX_GPU_HIP
  miopenTensorDescriptor_t input_tensor_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(miopenCreateTensorDescriptor(&input_tensor_desc)));
  int input_dims[2] = {d.batch_size, d.input_size};
  int input_strides[2] = {input_dims[1], 1};  // row-major order
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(miopenSetTensorDescriptor(
      input_tensor_desc, data_type, 2, input_dims, input_strides)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpudnnGetRNNWeightSpaceSize(handle.get(), rnn_desc, input_tensor_desc,
                                  &weight_space_size, data_type)));
#else   // JAX_GPU_CUDA
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      gpudnnGetRNNWeightSpaceSize(handle.get(), rnn_desc, &weight_space_size)));
#endif  // JAX_GPU_HIP

  auto dy_buf = buffers[0];
  auto dh_n_buf = buffers[1];
  auto dc_n_buf = buffers[2];
  auto x_buf = buffers[3];
  auto h_0_buf = buffers[4];
  auto c_0_buf = buffers[5];
  auto w_buf = buffers[6];
  auto y_buf = buffers[7];
  auto reserve_space_buf = buffers[8];
  auto zeroed_dw_buf = buffers[9];
  // auto seq_lengths_buf = buffers[10];

  auto dx_buf = buffers[11];
  auto dh_0_buf = buffers[12];
  auto dc_0_buf = buffers[13];
  // auto dw_buf = buffers[14];
  auto workspace_buf = buffers[15];

#ifdef JAX_GPU_HIP
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnRNNBackwardData(
      handle.get(), rnn_desc, output_data_desc, y_buf, dy_buf, h_desc, h_0_buf,
      dh_n_buf, dh_0_buf, c_desc, c_0_buf, dc_n_buf, dc_0_buf, input_data_desc,
      dx_buf, w_buf, weight_space_size, workspace_buf, d.workspace_size,
      reserve_space_buf, d.reserve_space_size)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnRNNBackwardWeights(
      handle.get(), rnn_desc, input_data_desc, x_buf, h_desc, h_0_buf,
      output_data_desc, y_buf, zeroed_dw_buf, weight_space_size, workspace_buf,
      d.workspace_size, reserve_space_buf, d.reserve_space_size)));
#else   // JAX_GPU_CUDA
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnRNNBackwardData(
      handle.get(), rnn_desc, (const int32_t*)seq_lengths_buf, output_data_desc,
      y_buf, dy_buf, input_data_desc, dx_buf, h_desc, h_0_buf, dh_n_buf,
      dh_0_buf, c_desc, c_0_buf, dc_n_buf, dc_0_buf, weight_space_size, w_buf,
      d.workspace_size, workspace_buf, d.reserve_space_size,
      reserve_space_buf)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnRNNBackwardWeights(
      handle.get(), rnn_desc, GPUDNN_WGRAD_MODE_ADD,
      (const int32_t*)seq_lengths_buf, input_data_desc, x_buf, h_desc, h_0_buf,
      output_data_desc, y_buf, weight_space_size, zeroed_dw_buf,
      d.workspace_size, workspace_buf, d.reserve_space_size,
      reserve_space_buf)));
#endif  // JAX_GPU_HIP

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnDestroyTensorDescriptor(h_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnDestroyTensorDescriptor(c_desc)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnDestroyRNNDataDescriptor(input_data_desc)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnDestroyRNNDataDescriptor(output_data_desc)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpudnnDestroyDropoutDescriptor(dropout_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpudnnDestroyRNNDescriptor(rnn_desc)));

  return absl::OkStatus();
}

JAX_GPU_REGISTER_WRAPPED_LEGACY_KERNEL(RNNForwardFfi, DnnRNNForward_);
JAX_GPU_REGISTER_WRAPPED_LEGACY_KERNEL(RNNBackwardFfi, DnnRNNBackward_);

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
