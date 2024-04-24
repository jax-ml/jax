#include "nanobind/nanobind.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "jaxlib/mosaic/gpu/integrations/c/passes.h"
#include "xla/service/custom_call_status.h"

namespace jax::cuda {
namespace {

namespace nb = nanobind;
using MosaicHostFunc = void(void**);

void MosaicKernelCall(void* stream, void** buffers, char* opaque,
                      size_t opaque_len, XlaCustomCallStatus* status) {
  void** static_args = *reinterpret_cast<void***>(opaque);
  MosaicHostFunc* func = reinterpret_cast<MosaicHostFunc*>(static_args[0]);
  void* ctx = static_args[1];
  void* args[3] = {&ctx, &stream, &buffers};
  func(args);
}

NB_MODULE(_mosaic_gpu_ext, m) {
  m.def("_custom_call_capsule",
        []() { return EncapsulateFunction(MosaicKernelCall); });
  m.def("register_passes", []() { return mlirMosaicGpuRegisterPasses(); });
}

}  // namespace
}  // namespace jax::cuda
