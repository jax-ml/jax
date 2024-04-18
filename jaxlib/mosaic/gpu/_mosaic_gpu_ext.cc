#include "nanobind/nanobind.h"
#include "jaxlib/kernel_nanobind_helpers.h"
#include "xla/service/custom_call_status.h"

namespace jax::cuda {
namespace {

namespace nb = nanobind;
using MosaicHostFunc = void(void**);

void MosaicKernelCall(void* stream, void** buffers, char* opaque,
                      size_t opaque_len, XlaCustomCallStatus* status) {
  void* args[2] = {&stream, &buffers};
  MosaicHostFunc* func = *reinterpret_cast<MosaicHostFunc**>(opaque);
  func(args);
}

NB_MODULE(_mosaic_gpu_ext, m) {
  m.def("_custom_call_capsule",
        []() { return EncapsulateFunction(MosaicKernelCall); });
}

}  // namespace
}  // namespace jax::cuda
