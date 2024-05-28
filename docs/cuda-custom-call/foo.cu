#include <xla/ffi/api/c_api.h>
#include <xla/ffi/api/api.h>
#include <xla/ffi/api/ffi.h>

// c = a * (b+1)
// This strawman operation works well for demo purposes because:
// 1. it's simple enough to be quickly understood,
// 2. it's complex enough to require intermediate outputs in grad computation,
//    like many operations in practice do, and
// 3. it does not have a built-in implementation in JAX.
__global__ void FooKernel(const float *a, const float *b, float *c, size_t n) {
  size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  const size_t grid_stride = blockDim.x * gridDim.x;
  for (size_t i = tid; i < n; i += grid_stride)
    c[i] = a[i] * (b[i] + 1.0f);
}

namespace ffi = xla::ffi;

// XLA FFI binding wrapper that launches the kernel
extern "C" XLA_FFI_Error *Foo(XLA_FFI_CallFrame *call_frame) {
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
            // Host function wrapper that launches the kernel with hardcoded
            // grid/block size. Note, it uses types from XLA FFI. The return
            // type must be ffi::Error. Buffer type provides buffer dimensions,
            // so the "n" argument here is not strictly necessary, but it allows
            // us to demonstrate the use of attributes (.Attr in the FFI handler
            // definition above).
            const int block_dim = 128;
            const int grid_dim = 1;
            // Note how we access regular Buffer data vs Result Buffer data:
            FooKernel<<<grid_dim, block_dim, /*shared_mem=*/0, stream>>>(
                a.data, b.data, c->data, n);
            // Check for launch time errors. Note that this function may also
            // return error codes from previous, asynchronous launches. This
            // means that an error status returned here could have been caused
            // by a different kernel previously launched by XLA.
            cudaError_t last_error = cudaGetLastError();
            if (last_error != cudaSuccess) {
              return ffi::Error(XLA_FFI_Error_Code_INTERNAL,
                                std::string("CUDA error: ") +
                                cudaGetErrorString(last_error));
            }
            return ffi::Error::Success();
      }).release();
  return kImpl->Call(call_frame);
}
