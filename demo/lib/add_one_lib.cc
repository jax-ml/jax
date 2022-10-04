#include <iostream>

#include <pybind11/pybind11.h>
#include "jax_custom_call.h" // Header that comes from JAX

namespace py = pybind11;

struct Info {
  float n;
};
// TODO: upgrade XLA CPU custom call to use serialization not pointer (common case is string)

extern "C" void add_one(JaxFFIApi* api, JaxFFIStatus* status, void* descriptor, void** inputs, void** outputs) {
  assert(JaxFFIVersion(api) == 1);
  Info* info = (Info*) descriptor;
  float* input = (float*) inputs[0];
  float* output = (float*) outputs[0];
  *output = *input + info->n;
  if (info->n < 0) {
    JaxFFIStatusSetFailure(api, status, "Info must be >= 0");
  }
}

PYBIND11_MODULE(add_one_lib, m) {
  m.def("get_function", []() {
    return py::capsule(reinterpret_cast<void*>(add_one), JaxFFICallCpu);
  });
  m.def("get_descriptor", [](float n) {
    Info* info = new Info();
    info->n = n;
    return py::capsule(reinterpret_cast<void*>(info), [](void* ptr) {
      delete reinterpret_cast<Info*>(ptr);
    });
  });
}
