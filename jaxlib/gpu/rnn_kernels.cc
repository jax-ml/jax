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

#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/handle_pool.h"
#include "jaxlib/kernel_helpers.h"
#include "xla/service/custom_call_status.h"

namespace jax {

namespace JAX_GPU_NAMESPACE {

std::string ErrorString(gpudnnStatus_t status) {
  return cudnnGetErrorString(status);
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

static absl::StatusOr<std::pair<int, int>>
DoRnnComputeWorkspaceReserveSpaceSizes(int input_size, int hidden_size,
                                       int num_layers, int batch_size,
                                       int max_seq_length, float dropout,
                                       bool bidirectional,
				       bool cudnn_allow_tf32) {
  auto h = DnnHandlePool::Borrow(/*stream=*/nullptr);
  JAX_RETURN_IF_ERROR(h.status());
  auto& handle = *h;

  cudnnRNNDescriptor_t rnn_desc;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnCreateRNNDescriptor(&rnn_desc)));

  cudnnDropoutDescriptor_t dropout_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnCreateDropoutDescriptor(&dropout_desc)));
  size_t state_size;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnDropoutGetStatesSize(handle.get(), &state_size)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnSetDropoutDescriptor(
      dropout_desc, handle.get(), dropout, nullptr, state_size, 123)));

  // TODO(zhangqiaorjc): Handle other kinds of RNN.
  cudnnRNNMode_t cell_mode = CUDNN_LSTM;
  cudnnRNNBiasMode_t bias_mode = CUDNN_RNN_DOUBLE_BIAS;
  int num_directions = 1;
  cudnnDirectionMode_t dir_mode = CUDNN_UNIDIRECTIONAL;
  if (bidirectional) {
    dir_mode = CUDNN_BIDIRECTIONAL;
    num_directions = 2;
  }
  cudnnRNNInputMode_t input_mode = CUDNN_LINEAR_INPUT;
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
  cudnnDataType_t math_prec = CUDNN_DATA_FLOAT;
  cudnnMathType_t math_type = cudnn_allow_tf32? CUDNN_DEFAULT_MATH: CUDNN_FMA_MATH;
  int32_t proj_size = hidden_size;
  uint32_t aux_flags = CUDNN_RNN_PADDED_IO_ENABLED;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnSetRNNDescriptor_v8(
      rnn_desc, CUDNN_RNN_ALGO_STANDARD, cell_mode, bias_mode, dir_mode,
      input_mode, data_type, math_prec, math_type, input_size, hidden_size,
      proj_size, num_layers, dropout_desc, aux_flags)));

  cudnnForwardMode_t fwdMode = CUDNN_FWD_MODE_TRAINING;
  cudnnRNNDataLayout_t layout = CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
  float padding = 0.0f;

  std::vector<int32_t> seq_length_vector(batch_size, max_seq_length);
  int32_t* seq_length_array = &seq_length_vector[0];

  cudnnRNNDataDescriptor_t input_data_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnCreateRNNDataDescriptor(&input_data_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnSetRNNDataDescriptor(
      input_data_desc, data_type, layout, max_seq_length, batch_size,
      input_size, seq_length_array, &padding)));

  size_t workSpaceSize;
  size_t reserveSpaceSize;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnGetRNNTempSpaceSizes(
      handle.get(), rnn_desc, fwdMode, input_data_desc, &workSpaceSize,
      &reserveSpaceSize)));

  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnDestroyDropoutDescriptor(dropout_desc)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnDestroyRNNDataDescriptor(input_data_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnDestroyRNNDescriptor(rnn_desc)));

  // Round up to nearest multiples of 4 so we can return them as f32 arrays.
  workSpaceSize += (workSpaceSize % 4);
  reserveSpaceSize += (reserveSpaceSize % 4);
  return std::make_pair(workSpaceSize, reserveSpaceSize);
}

absl::StatusOr<std::pair<int, int>> RnnComputeWorkspaceReserveSpaceSizes(
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

  cudnnRNNDescriptor_t rnn_desc;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnCreateRNNDescriptor(&rnn_desc)));

  cudnnDropoutDescriptor_t dropout_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnCreateDropoutDescriptor(&dropout_desc)));
  size_t state_size;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnDropoutGetStatesSize(handle.get(), &state_size)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnSetDropoutDescriptor(
      dropout_desc, handle.get(), d.dropout, nullptr, state_size, 123)));

  // TODO(zhangqiaorjc): Handle other kinds of RNN.
  cudnnRNNMode_t cell_mode = CUDNN_LSTM;
  cudnnRNNBiasMode_t bias_mode = CUDNN_RNN_DOUBLE_BIAS;
  int num_directions = 1;
  cudnnDirectionMode_t dir_mode = CUDNN_UNIDIRECTIONAL;
  if (d.bidirectional) {
    dir_mode = CUDNN_BIDIRECTIONAL;
    num_directions = 2;
  }
  cudnnRNNInputMode_t input_mode = CUDNN_LINEAR_INPUT;
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
  cudnnDataType_t math_prec = CUDNN_DATA_FLOAT;
  cudnnMathType_t math_type = d.cudnn_allow_tf32? CUDNN_DEFAULT_MATH: CUDNN_FMA_MATH;
  int32_t proj_size = d.hidden_size;
  uint32_t aux_flags = CUDNN_RNN_PADDED_IO_ENABLED;

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnSetRNNDescriptor_v8(
      rnn_desc, CUDNN_RNN_ALGO_STANDARD, cell_mode, bias_mode, dir_mode,
      input_mode, data_type, math_prec, math_type, d.input_size, d.hidden_size,
      proj_size, d.num_layers, dropout_desc, aux_flags)));

  cudnnForwardMode_t fwdMode = CUDNN_FWD_MODE_TRAINING;
  cudnnRNNDataLayout_t layout = CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
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

  cudnnRNNDataDescriptor_t input_data_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnCreateRNNDataDescriptor(&input_data_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnSetRNNDataDescriptor(
      input_data_desc, data_type, layout, d.max_seq_length, d.batch_size,
      d.input_size, seq_length_array, &padding)));

  cudnnRNNDataDescriptor_t output_data_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnCreateRNNDataDescriptor(&output_data_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnSetRNNDataDescriptor(
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
  cudnnTensorDescriptor_t h_desc;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnCreateTensorDescriptor(&h_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      cudnnSetTensorNdDescriptor(h_desc, data_type, 3, dims, strides)));

  cudnnTensorDescriptor_t c_desc;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnCreateTensorDescriptor(&c_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      cudnnSetTensorNdDescriptor(c_desc, data_type, 3, dims, strides)));

  size_t weight_space_size;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      cudnnGetRNNWeightSpaceSize(handle.get(), rnn_desc, &weight_space_size)));

  auto input_buf = buffers[0];
  auto h_0_buf = buffers[1];
  auto c_0_buf = buffers[2];
  auto weights_buf = buffers[3];
  auto output_buf = buffers[5];
  auto h_n_buf = buffers[6];
  auto c_n_buf = buffers[7];
  auto workspace_buf = buffers[8];
  auto reserve_space_buf = buffers[9];
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnRNNForward(
      handle.get(), rnn_desc, fwdMode, (const int32_t*)seq_lengths_buf,
      input_data_desc, input_buf, output_data_desc, output_buf, h_desc, h_0_buf,
      h_n_buf, c_desc, c_0_buf, c_n_buf, weight_space_size, weights_buf,
      d.workspace_size, workspace_buf, d.reserve_space_size,
      reserve_space_buf)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnDestroyTensorDescriptor(h_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnDestroyTensorDescriptor(c_desc)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnDestroyDropoutDescriptor(dropout_desc)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnDestroyRNNDataDescriptor(input_data_desc)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnDestroyRNNDataDescriptor(output_data_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnDestroyRNNDescriptor(rnn_desc)));

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

  cudnnRNNDescriptor_t rnn_desc;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnCreateRNNDescriptor(&rnn_desc)));

  cudnnDropoutDescriptor_t dropout_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnCreateDropoutDescriptor(&dropout_desc)));
  size_t state_size;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnDropoutGetStatesSize(handle.get(), &state_size)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnSetDropoutDescriptor(
      dropout_desc, handle.get(), d.dropout, nullptr, state_size, 123)));

  // TODO(zhangqiaorjc): Handle other kinds of RNN.
  cudnnRNNMode_t cell_mode = CUDNN_LSTM;
  cudnnRNNBiasMode_t bias_mode = CUDNN_RNN_DOUBLE_BIAS;
  int num_directions = 1;
  cudnnDirectionMode_t dir_mode = CUDNN_UNIDIRECTIONAL;
  if (d.bidirectional) {
    dir_mode = CUDNN_BIDIRECTIONAL;
    num_directions = 2;
  }
  cudnnRNNInputMode_t input_mode = CUDNN_LINEAR_INPUT;
  cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
  cudnnDataType_t math_prec = CUDNN_DATA_FLOAT;
  cudnnMathType_t math_type = d.cudnn_allow_tf32? CUDNN_DEFAULT_MATH: CUDNN_FMA_MATH;
  int32_t proj_size = d.hidden_size;
  uint32_t aux_flags = CUDNN_RNN_PADDED_IO_ENABLED;

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnSetRNNDescriptor_v8(
      rnn_desc, CUDNN_RNN_ALGO_STANDARD, cell_mode, bias_mode, dir_mode,
      input_mode, data_type, math_prec, math_type, d.input_size, d.hidden_size,
      proj_size, d.num_layers, dropout_desc, aux_flags)));

  cudnnRNNDataLayout_t layout = CUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
  float padding = 0.0f;

  auto seq_lengths_buf = buffers[10];
  std::vector<int32_t> seq_length_vector(d.batch_size, d.max_seq_length);
  int32_t* seq_length_array = &seq_length_vector[0];
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(gpuMemcpyAsync(seq_length_array, seq_lengths_buf,
                                   seq_length_vector.size() * sizeof(int32_t),
                                   gpuMemcpyDeviceToHost, stream)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(gpuStreamSynchronize(stream)));

  cudnnRNNDataDescriptor_t input_data_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnCreateRNNDataDescriptor(&input_data_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnSetRNNDataDescriptor(
      input_data_desc, data_type, layout, d.max_seq_length, d.batch_size,
      d.input_size, seq_length_array, &padding)));

  cudnnRNNDataDescriptor_t output_data_desc;
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnCreateRNNDataDescriptor(&output_data_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnSetRNNDataDescriptor(
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
  cudnnTensorDescriptor_t h_desc;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnCreateTensorDescriptor(&h_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      cudnnSetTensorNdDescriptor(h_desc, data_type, 3, dims, strides)));

  cudnnTensorDescriptor_t c_desc;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnCreateTensorDescriptor(&c_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      cudnnSetTensorNdDescriptor(c_desc, data_type, 3, dims, strides)));

  size_t weight_space_size;
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(
      cudnnGetRNNWeightSpaceSize(handle.get(), rnn_desc, &weight_space_size)));

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

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnRNNBackwardData_v8(
      handle.get(), rnn_desc, (const int32_t*)seq_lengths_buf, output_data_desc,
      y_buf, dy_buf, input_data_desc, dx_buf, h_desc, h_0_buf, dh_n_buf,
      dh_0_buf, c_desc, c_0_buf, dc_n_buf, dc_0_buf, weight_space_size, w_buf,
      d.workspace_size, workspace_buf, d.reserve_space_size,
      reserve_space_buf)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnRNNBackwardWeights_v8(
      handle.get(), rnn_desc, CUDNN_WGRAD_MODE_ADD,
      (const int32_t*)seq_lengths_buf, input_data_desc, x_buf, h_desc, h_0_buf,
      output_data_desc, y_buf, weight_space_size, zeroed_dw_buf,
      d.workspace_size, workspace_buf, d.reserve_space_size,
      reserve_space_buf)));

  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnDestroyTensorDescriptor(h_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnDestroyTensorDescriptor(c_desc)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnDestroyDropoutDescriptor(dropout_desc)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnDestroyRNNDataDescriptor(input_data_desc)));
  JAX_RETURN_IF_ERROR(
      JAX_AS_STATUS(cudnnDestroyRNNDataDescriptor(output_data_desc)));
  JAX_RETURN_IF_ERROR(JAX_AS_STATUS(cudnnDestroyRNNDescriptor(rnn_desc)));

  return absl::OkStatus();
}

void RNNForward(gpuStream_t stream, void** buffers, const char* opaque,
                size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = DnnRNNForward_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

void RNNBackward(gpuStream_t stream, void** buffers, const char* opaque,
                 size_t opaque_len, XlaCustomCallStatus* status) {
  auto s = DnnRNNBackward_(stream, buffers, opaque, opaque_len);
  if (!s.ok()) {
    XlaCustomCallStatusSetFailure(status, std::string(s.message()).c_str(),
                                  s.message().length());
  }
}

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
