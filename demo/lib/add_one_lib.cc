#include <iostream>

#include <pybind11/pybind11.h>
#include "jax_custom_call.h" // Header that comes from JAX

namespace py = pybind11;

struct Info {
  float n;
};
// TODO: upgrade XLA CPU custom call to use serialization not pointer (common case is string)

// Cannot use C++ objects in the call!
// Instead of passing a `Status`, we could pass an API object.
extern "C" void add_one(JaxFFI_API* api, JaxFFIStatus* status, void* descriptor, void** inputs, void** outputs) {
  // descriptor could be pointer memory or string?
  // Info* info = (Info*) descriptor;
  // std::cout << "Info: " << info->n << std::endl;
  // *outputs[0] = *inputs[0] + info->n;
  JaxFFIStatusSetFailure(api, status, "bad");
}

PYBIND11_MODULE(add_one_lib, m) {
  m.def("get_function", []() {
    return py::capsule(reinterpret_cast<void*>(add_one), JAX_FFI_CALL_CPU);
  });
  m.def("get_descriptor", [](float n) {
    Info* info = new Info();
    info->n = n;
    return py::capsule(reinterpret_cast<void*>(info), [](void* ptr) {
      delete reinterpret_cast<Info*>(ptr);
    });
  });
}
