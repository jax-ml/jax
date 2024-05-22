#include <xla/ffi/api/c_api.h>
#include <xla/ffi/api/api.h>
#include <xla/ffi/api/ffi.h>

// an elementwise product with a grid stride loop
__global__ void MulKernel(const float *a, const float *b, float *c, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride)
    c[i] = a[i] * b[i];
}

// host wrapper function that launches the kernel with hardcoded grid/block size
void LaunchMulKernel(cudaStream_t stream,
                     const float *a, const float *b, float *c, size_t n) {
  const int block_dim = 128;
  const int grid_dim = 1;
  MulKernel<<<grid_dim, block_dim, /*dynamic_shared_mem_bytes=*/0, stream>>>(
      a, b, c, n);
}

namespace ffi = xla::ffi;

// XLA FFI binding wrapper around our host launcher function
extern "C" XLA_FFI_Error *Mul(XLA_FFI_CallFrame *call_frame) {
  static const auto *kImpl =
      ffi::Ffi::Bind()
          .Ctx<ffi::PlatformStream<cudaStream_t>>()
          .Arg<ffi::Buffer<ffi::DataType::F32>>()
          .Arg<ffi::Buffer<ffi::DataType::F32>>()
          .Ret<ffi::Buffer<ffi::DataType::F32>>()
          .Attr<size_t>("n")
          .To([](cudaStream_t stream,
                 ffi::Buffer<ffi::DataType::F32> a,
                 ffi::Buffer<ffi::DataType::F32> b,
                 ffi::Result<ffi::Buffer<ffi::DataType::F32>> c,
                 size_t n) -> ffi::Error {
            LaunchMulKernel(stream, a.data, b.data, c->data, n);
            return ffi::Error::Success();
      }).release();
  return kImpl->Call(call_frame);
}
