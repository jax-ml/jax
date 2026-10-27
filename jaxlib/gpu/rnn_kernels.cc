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
#include "absl/synchronization/mutex.h"
#include "jaxlib/ffi_helpers.h"
#include "jaxlib/gpu/gpu_kernel_helpers.h"
#include "jaxlib/gpu/handle_pool.h"
#include "jaxlib/gpu/vendor.h"
#include "xla/ffi/api/ffi.h"

#define JAX_FFI_RETURN_IF_GPU_ERROR(...) \
  FFI_RETURN_IF_ERROR_STATUS(JAX_AS_STATUS(__VA_ARGS__))

namespace ffi = ::xla::ffi;

namespace jax {

namespace JAX_GPU_NAMESPACE {
#ifdef JAX_GPU_CUDA
std::string ErrorString(cudnnStatus_t status) {
  return cudnnGetErrorString(status);
}
#elif defined(JAX_GPU_HIP)
std::string ErrorString(miopenStatus_t status) {
  return miopenGetErrorString(status);
}
#endif
}  // namespace JAX_GPU_NAMESPACE

using DnnHandlePool = HandlePool<gpudnnHandle_t, gpuStream_t>;

template <>
/*static*/ absl::StatusOr<DnnHandlePool::Handle> DnnHandlePool::Borrow(
    gpuStream_t stream) {
  DnnHandlePool* pool = Instance();
  absl::MutexLock lock(pool->mu_);
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

static ffi::Error RNNForwardFfiV2(
    gpuStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer x,
    ffi::AnyBuffer h_0, ffi::AnyBuffer c_0, ffi::AnyBuffer weights,
    ffi::AnyBuffer seq_lengths, ffi::Result<ffi::AnyBuffer> y,
    ffi::Result<ffi::AnyBuffer> h_n, ffi::Result<ffi::AnyBuffer> c_n,
    ffi::Result<ffi::AnyBuffer> reserve_space, float dropout,
    bool bidirectional, bool cudnn_allow_tf32) {
  int batch_size = x.dimensions()[0];
  int max_seq_length = x.dimensions()[1];
  int input_size = x.dimensions()[2];
  int num_directions = bidirectional ? 2 : 1;
  int num_layers = h_0.dimensions()[0] / num_directions;
  int hidden_size = h_0.dimensions()[2];
  size_t reserve_space_size = reserve_space->size_bytes();

  FFI_ASSIGN_OR_RETURN(auto handle, DnnHandlePool::Borrow(stream));

  gpudnnRNNDescriptor_t rnn_desc;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnCreateRNNDescriptor(&rnn_desc));

  gpudnnDropoutDescriptor_t dropout_desc;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnCreateDropoutDescriptor(&dropout_desc));
  size_t state_size;
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpudnnDropoutGetStatesSize(handle.get(), &state_size));

#ifdef JAX_GPU_HIP
  void* dropout_states_dev = nullptr;
  // Allocate minimal memory for dropout states (can be very small since it's
  // not used).
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuMalloc(&dropout_states_dev, state_size));
  if (!dropout_states_dev) {
    return ffi::Error::Internal(
        "Failed to allocate minimal GPU memory for dropout states.");
  }
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnSetDropoutDescriptor(
      dropout_desc, handle.get(), dropout, dropout_states_dev, state_size, 123,
      false, false, MIOPEN_RNG_PSEUDO_XORWOW));
#else   // JAX_GPU_CUDA
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnSetDropoutDescriptor(
      dropout_desc, handle.get(), dropout, nullptr, state_size, 123));
#endif  // JAX_GPU_HIP

  // TODO(zhangqiaorjc): Handle other kinds of RNN.
  gpudnnRNNMode_t cell_mode = GPUDNN_LSTM;
  gpudnnRNNBiasMode_t bias_mode = GPUDNN_RNN_DOUBLE_BIAS;
  gpudnnDirectionMode_t dir_mode =
      bidirectional ? GPUDNN_BIDIRECTIONAL : GPUDNN_UNIDIRECTIONAL;
  gpudnnRNNInputMode_t input_mode = GPUDNN_LINEAR_INPUT;
  gpudnnDataType_t data_type = GPUDNN_DATA_FLOAT;

#ifdef JAX_GPU_HIP
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnSetRNNDescriptor(
      rnn_desc, hidden_size, num_layers, dropout_desc, input_mode, dir_mode,
      cell_mode, bias_mode, GPUDNN_RNN_ALGO_STANDARD, data_type));
#else   // JAX_GPU_CUDA
  gpudnnDataType_t math_prec = GPUDNN_DATA_FLOAT;
  gpudnnMathType_t math_type =
      cudnn_allow_tf32 ? GPUDNN_DEFAULT_MATH : GPUDNN_FMA_MATH;
  int32_t proj_size = hidden_size;
  uint32_t aux_flags = GPUDNN_RNN_PADDED_IO_ENABLED;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnSetRNNDescriptor(
      rnn_desc, GPUDNN_RNN_ALGO_STANDARD, cell_mode, bias_mode, dir_mode,
      input_mode, data_type, math_prec, math_type, input_size, hidden_size,
      proj_size, num_layers, dropout_desc, aux_flags));
#endif  // JAX_GPU_HIP

  gpudnnForwardMode_t fwdMode = GPUDNN_FWD_MODE_TRAINING;
  gpudnnRNNDataLayout_t layout = GPUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
  float padding = 0.0f;

  // TODO(zhangqiaorjc): Avoid this cudaMemcpy if possible.
  std::vector<int32_t> seq_length_vector(batch_size, 0);
  int32_t* seq_length_array = &seq_length_vector[0];
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpuMemcpyAsync(seq_length_array, seq_lengths.untyped_data(),
                     seq_length_vector.size() * sizeof(int32_t),
                     gpuMemcpyDeviceToHost, stream));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuStreamSynchronize(stream));

  gpudnnRNNDataDescriptor_t input_data_desc;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnCreateRNNDataDescriptor(&input_data_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnSetRNNDataDescriptor(
      input_data_desc, data_type, layout, max_seq_length, batch_size,
      input_size, seq_length_array, &padding));

  gpudnnRNNDataDescriptor_t output_data_desc;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnCreateRNNDataDescriptor(&output_data_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnSetRNNDataDescriptor(
      output_data_desc, data_type, layout, max_seq_length, batch_size,
      hidden_size * num_directions, seq_length_array, &padding));

  // Shape is (num_directions * num_layers, batch_size, hidden_size)
  int dims[3];
  dims[0] = num_directions * num_layers;
  dims[1] = batch_size;
  dims[2] = hidden_size;
  int strides[3];
  strides[0] = dims[1] * dims[2];
  strides[1] = dims[2];
  strides[2] = 1;
  gpudnnTensorDescriptor_t h_desc;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnCreateTensorDescriptor(&h_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpudnnSetTensorNdDescriptor(h_desc, data_type, 3, dims, strides));

  gpudnnTensorDescriptor_t c_desc;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnCreateTensorDescriptor(&c_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpudnnSetTensorNdDescriptor(c_desc, data_type, 3, dims, strides));

  size_t weight_space_size;
#ifdef JAX_GPU_HIP
  miopenTensorDescriptor_t input_tensor_desc;
  JAX_FFI_RETURN_IF_GPU_ERROR(miopenCreateTensorDescriptor(&input_tensor_desc));
  int dimsA[2] = {batch_size, input_size};
  int stridesA[2] = {dimsA[1], 1};  // Row-major order, similar to GPUDNN
  JAX_FFI_RETURN_IF_GPU_ERROR(miopenSetTensorDescriptor(
      input_tensor_desc, data_type, 2, dimsA, stridesA));
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpudnnGetRNNWeightSpaceSize(handle.get(), rnn_desc, input_tensor_desc,
                                  &weight_space_size, data_type));
#else   // JAX_GPU_CUDA
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpudnnGetRNNWeightSpaceSize(handle.get(), rnn_desc, &weight_space_size));
#endif  // JAX_GPU_HIP

  size_t workspace_size = 0;
  size_t required_reserve_space_size = 0;
#ifdef JAX_GPU_HIP
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnGetRNNTempSpaceSizes(
      handle.get(), rnn_desc, input_data_desc, fwdMode, &workspace_size,
      &required_reserve_space_size));
#else   // JAX_GPU_CUDA
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnGetRNNTempSpaceSizes(
      handle.get(), rnn_desc, fwdMode, input_data_desc, &workspace_size,
      &required_reserve_space_size));
#endif  // JAX_GPU_HIP
  auto maybe_workspace = scratch.Allocate(workspace_size);
  if (!maybe_workspace.has_value()) {
    return ffi::Error::Internal(
        "Unable to allocate workspace for cudnn_rnn_fwd");
  }
  void* workspace_buf = maybe_workspace.value();

#ifdef JAX_GPU_HIP
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnRNNForward(
      handle.get(), rnn_desc, fwdMode, input_data_desc, x.untyped_data(),
      h_desc, h_0.untyped_data(), h_n->untyped_data(), c_desc,
      c_0.untyped_data(), c_n->untyped_data(), output_data_desc,
      y->untyped_data(), weights.untyped_data(), weight_space_size,
      workspace_buf, workspace_size, reserve_space->untyped_data(),
      reserve_space_size));
#else   // JAX_GPU_CUDA
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnRNNForward(
      handle.get(), rnn_desc, fwdMode,
      static_cast<const int32_t*>(seq_lengths.untyped_data()), input_data_desc,
      x.untyped_data(), output_data_desc, y->untyped_data(), h_desc,
      h_0.untyped_data(), h_n->untyped_data(), c_desc, c_0.untyped_data(),
      c_n->untyped_data(), weight_space_size, weights.untyped_data(),
      workspace_size, workspace_buf, reserve_space_size,
      reserve_space->untyped_data()));
#endif  // JAX_GPU_HIP

  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnDestroyTensorDescriptor(h_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnDestroyTensorDescriptor(c_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnDestroyRNNDataDescriptor(input_data_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnDestroyRNNDataDescriptor(output_data_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnDestroyDropoutDescriptor(dropout_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnDestroyRNNDescriptor(rnn_desc));

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kRNNForward, RNNForwardFfiV2,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // x
                                  .Arg<ffi::AnyBuffer>()  // h_0
                                  .Arg<ffi::AnyBuffer>()  // c_0
                                  .Arg<ffi::AnyBuffer>()  // weights
                                  .Arg<ffi::AnyBuffer>()  // seq_lengths
                                  .Ret<ffi::AnyBuffer>()  // y
                                  .Ret<ffi::AnyBuffer>()  // h_n
                                  .Ret<ffi::AnyBuffer>()  // c_n
                                  .Ret<ffi::AnyBuffer>()  // reserve_space
                                  .Attr<float>("dropout")
                                  .Attr<bool>("bidirectional")
                                  .Attr<bool>("cudnn_allow_tf32"));

static ffi::Error RNNBackwardFfiV2(
    gpuStream_t stream, ffi::ScratchAllocator scratch, ffi::AnyBuffer dy,
    ffi::AnyBuffer dh_n, ffi::AnyBuffer dc_n, ffi::AnyBuffer x,
    ffi::AnyBuffer h_0, ffi::AnyBuffer c_0, ffi::AnyBuffer w, ffi::AnyBuffer y,
    ffi::AnyBuffer reserve_space, ffi::AnyBuffer zeroed_dw,
    ffi::AnyBuffer seq_lengths, ffi::Result<ffi::AnyBuffer> dx,
    ffi::Result<ffi::AnyBuffer> dh_0, ffi::Result<ffi::AnyBuffer> dc_0,
    ffi::Result<ffi::AnyBuffer> dw, float dropout, bool bidirectional,
    bool cudnn_allow_tf32) {
  int batch_size = x.dimensions()[0];
  int max_seq_length = x.dimensions()[1];
  int input_size = x.dimensions()[2];
  int num_directions = bidirectional ? 2 : 1;
  int num_layers = h_0.dimensions()[0] / num_directions;
  int hidden_size = h_0.dimensions()[2];
  size_t reserve_space_size = reserve_space.size_bytes();

  FFI_ASSIGN_OR_RETURN(auto handle, DnnHandlePool::Borrow(stream));

  gpudnnRNNDescriptor_t rnn_desc;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnCreateRNNDescriptor(&rnn_desc));

  gpudnnDropoutDescriptor_t dropout_desc;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnCreateDropoutDescriptor(&dropout_desc));
  size_t state_size;
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpudnnDropoutGetStatesSize(handle.get(), &state_size));

#ifdef JAX_GPU_HIP
  void* dropout_states_dev = nullptr;
  // Allocate minimal memory for dropout states (can be very small since it's
  // not used)
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuMalloc(&dropout_states_dev, state_size));
  if (!dropout_states_dev) {
    return ffi::Error::Internal(
        "Failed to allocate minimal GPU memory for dropout states.");
  }
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnSetDropoutDescriptor(
      dropout_desc, handle.get(), dropout, dropout_states_dev, state_size, 123,
      false, false, MIOPEN_RNG_PSEUDO_XORWOW));
#else   // JAX_GPU_CUDA
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnSetDropoutDescriptor(
      dropout_desc, handle.get(), dropout, nullptr, state_size, 123));
#endif  // JAX_GPU_HIP

  // TODO(zhangqiaorjc): Handle other kinds of RNN.
  gpudnnRNNMode_t cell_mode = GPUDNN_LSTM;
  gpudnnRNNBiasMode_t bias_mode = GPUDNN_RNN_DOUBLE_BIAS;
  gpudnnDirectionMode_t dir_mode =
      bidirectional ? GPUDNN_BIDIRECTIONAL : GPUDNN_UNIDIRECTIONAL;
  gpudnnRNNInputMode_t input_mode = GPUDNN_LINEAR_INPUT;
  gpudnnDataType_t data_type = GPUDNN_DATA_FLOAT;

#ifdef JAX_GPU_HIP
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnSetRNNDescriptor(
      rnn_desc, hidden_size, num_layers, dropout_desc, input_mode, dir_mode,
      cell_mode, bias_mode, GPUDNN_RNN_ALGO_STANDARD, data_type));
#else   // JAX_GPU_CUDA
  gpudnnDataType_t math_prec = GPUDNN_DATA_FLOAT;
  gpudnnMathType_t math_type =
      cudnn_allow_tf32 ? GPUDNN_DEFAULT_MATH : GPUDNN_FMA_MATH;
  int32_t proj_size = hidden_size;
  uint32_t aux_flags = GPUDNN_RNN_PADDED_IO_ENABLED;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnSetRNNDescriptor(
      rnn_desc, GPUDNN_RNN_ALGO_STANDARD, cell_mode, bias_mode, dir_mode,
      input_mode, data_type, math_prec, math_type, input_size, hidden_size,
      proj_size, num_layers, dropout_desc, aux_flags));
#endif  // JAX_GPU_HIP

  gpudnnRNNDataLayout_t layout = GPUDNN_RNN_DATA_LAYOUT_BATCH_MAJOR_UNPACKED;
  float padding = 0.0f;

  std::vector<int32_t> seq_length_vector(batch_size, max_seq_length);
  int32_t* seq_length_array = &seq_length_vector[0];
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpuMemcpyAsync(seq_length_array, seq_lengths.untyped_data(),
                     seq_length_vector.size() * sizeof(int32_t),
                     gpuMemcpyDeviceToHost, stream));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpuStreamSynchronize(stream));

  gpudnnRNNDataDescriptor_t input_data_desc;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnCreateRNNDataDescriptor(&input_data_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnSetRNNDataDescriptor(
      input_data_desc, data_type, layout, max_seq_length, batch_size,
      input_size, seq_length_array, &padding));

  gpudnnRNNDataDescriptor_t output_data_desc;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnCreateRNNDataDescriptor(&output_data_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnSetRNNDataDescriptor(
      output_data_desc, data_type, layout, max_seq_length, batch_size,
      hidden_size * num_directions, seq_length_array, &padding));

  // Shape is (num_directions * num_layers, batch_size, hidden_size)
  int dims[3];
  dims[0] = num_directions * num_layers;
  dims[1] = batch_size;
  dims[2] = hidden_size;
  int strides[3];
  strides[0] = dims[1] * dims[2];
  strides[1] = dims[2];
  strides[2] = 1;
  gpudnnTensorDescriptor_t h_desc;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnCreateTensorDescriptor(&h_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpudnnSetTensorNdDescriptor(h_desc, data_type, 3, dims, strides));

  gpudnnTensorDescriptor_t c_desc;
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnCreateTensorDescriptor(&c_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpudnnSetTensorNdDescriptor(c_desc, data_type, 3, dims, strides));

  size_t weight_space_size;
#ifdef JAX_GPU_HIP
  miopenTensorDescriptor_t input_tensor_desc;
  JAX_FFI_RETURN_IF_GPU_ERROR(miopenCreateTensorDescriptor(&input_tensor_desc));
  int input_dims[2] = {batch_size, input_size};
  int input_strides[2] = {input_dims[1], 1};  // row-major order
  JAX_FFI_RETURN_IF_GPU_ERROR(miopenSetTensorDescriptor(
      input_tensor_desc, data_type, 2, input_dims, input_strides));
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpudnnGetRNNWeightSpaceSize(handle.get(), rnn_desc, input_tensor_desc,
                                  &weight_space_size, data_type));
#else   // JAX_GPU_CUDA
  JAX_FFI_RETURN_IF_GPU_ERROR(
      gpudnnGetRNNWeightSpaceSize(handle.get(), rnn_desc, &weight_space_size));
#endif  // JAX_GPU_HIP

  gpudnnForwardMode_t fwdMode = GPUDNN_FWD_MODE_TRAINING;
  size_t workspace_size = 0;
  size_t required_reserve_space_size = 0;
#ifdef JAX_GPU_HIP
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnGetRNNTempSpaceSizes(
      handle.get(), rnn_desc, input_data_desc, fwdMode, &workspace_size,
      &required_reserve_space_size));
#else   // JAX_GPU_CUDA
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnGetRNNTempSpaceSizes(
      handle.get(), rnn_desc, fwdMode, input_data_desc, &workspace_size,
      &required_reserve_space_size));
#endif  // JAX_GPU_HIP
  auto maybe_workspace = scratch.Allocate(workspace_size);
  if (!maybe_workspace.has_value()) {
    return ffi::Error::Internal(
        "Unable to allocate workspace for cudnn_rnn_bwd");
  }
  void* workspace_buf = maybe_workspace.value();

  void* dw_buf = dw->untyped_data();
  void* zeroed_dw_buf = zeroed_dw.untyped_data();
  if (dw_buf != nullptr && dw_buf != zeroed_dw_buf) {
    JAX_FFI_RETURN_IF_GPU_ERROR(
        gpuMemcpyAsync(dw_buf, zeroed_dw_buf, weight_space_size,
                       gpuMemcpyDeviceToDevice, stream));
  }
  void* target_dw_buf = dw_buf != nullptr ? dw_buf : zeroed_dw_buf;

#ifdef JAX_GPU_HIP
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnRNNBackwardData(
      handle.get(), rnn_desc, output_data_desc, y.untyped_data(),
      dy.untyped_data(), h_desc, h_0.untyped_data(), dh_n.untyped_data(),
      dh_0->untyped_data(), c_desc, c_0.untyped_data(), dc_n.untyped_data(),
      dc_0->untyped_data(), input_data_desc, dx->untyped_data(),
      w.untyped_data(), weight_space_size, workspace_buf, workspace_size,
      reserve_space.untyped_data(), reserve_space_size));

  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnRNNBackwardWeights(
      handle.get(), rnn_desc, input_data_desc, x.untyped_data(), h_desc,
      h_0.untyped_data(), output_data_desc, y.untyped_data(), target_dw_buf,
      weight_space_size, workspace_buf, workspace_size,
      reserve_space.untyped_data(), reserve_space_size));
#else   // JAX_GPU_CUDA
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnRNNBackwardData(
      handle.get(), rnn_desc,
      static_cast<const int32_t*>(seq_lengths.untyped_data()), output_data_desc,
      y.untyped_data(), dy.untyped_data(), input_data_desc, dx->untyped_data(),
      h_desc, h_0.untyped_data(), dh_n.untyped_data(), dh_0->untyped_data(),
      c_desc, c_0.untyped_data(), dc_n.untyped_data(), dc_0->untyped_data(),
      weight_space_size, w.untyped_data(), workspace_size, workspace_buf,
      reserve_space_size, reserve_space.untyped_data()));

  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnRNNBackwardWeights(
      handle.get(), rnn_desc, GPUDNN_WGRAD_MODE_ADD,
      static_cast<const int32_t*>(seq_lengths.untyped_data()), input_data_desc,
      x.untyped_data(), h_desc, h_0.untyped_data(), output_data_desc,
      y.untyped_data(), weight_space_size, target_dw_buf, workspace_size,
      workspace_buf, reserve_space_size, reserve_space.untyped_data()));
#endif  // JAX_GPU_HIP

  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnDestroyTensorDescriptor(h_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnDestroyTensorDescriptor(c_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnDestroyRNNDataDescriptor(input_data_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnDestroyRNNDataDescriptor(output_data_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnDestroyDropoutDescriptor(dropout_desc));
  JAX_FFI_RETURN_IF_GPU_ERROR(gpudnnDestroyRNNDescriptor(rnn_desc));

  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(kRNNBackward, RNNBackwardFfiV2,
                              ffi::Ffi::Bind()
                                  .Ctx<ffi::PlatformStream<gpuStream_t>>()
                                  .Ctx<ffi::ScratchAllocator>()
                                  .Arg<ffi::AnyBuffer>()  // dy
                                  .Arg<ffi::AnyBuffer>()  // dh_n
                                  .Arg<ffi::AnyBuffer>()  // dc_n
                                  .Arg<ffi::AnyBuffer>()  // x
                                  .Arg<ffi::AnyBuffer>()  // h_0
                                  .Arg<ffi::AnyBuffer>()  // c_0
                                  .Arg<ffi::AnyBuffer>()  // w
                                  .Arg<ffi::AnyBuffer>()  // y
                                  .Arg<ffi::AnyBuffer>()  // reserve_space
                                  .Arg<ffi::AnyBuffer>()  // zeroed_dw
                                  .Arg<ffi::AnyBuffer>()  // seq_lengths
                                  .Ret<ffi::AnyBuffer>()  // dx
                                  .Ret<ffi::AnyBuffer>()  // dh_0
                                  .Ret<ffi::AnyBuffer>()  // dc_0
                                  .Ret<ffi::AnyBuffer>()  // dw
                                  .Attr<float>("dropout")
                                  .Attr<bool>("bidirectional")
                                  .Attr<bool>("cudnn_allow_tf32"));

}  // namespace JAX_GPU_NAMESPACE
}  // namespace jax
