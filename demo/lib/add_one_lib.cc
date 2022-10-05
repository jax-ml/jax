#include <iostream>

#include <pybind11/pybind11.h>
#include "jax_custom_call.h" // Header that comes from JAX

namespace py = pybind11;

struct Info {
  float n;
};

extern "C" void add_one(JaxFFIApi* api, JaxFFIStatus* status,
                        void* descriptor, size_t descriptor_size,
                        void** inputs, void** outputs) {
  assert(JaxFFIVersion(api) == 1);
  Info info;
  assert(descriptor_size == sizeof(Info));
  std::memcpy(&info, descriptor, descriptor_size);
  assert(false);
  abort();
  printf("descriptor_size=%d\n", descriptor_size);
  printf("sizeof=%d\n", sizeof(Info));
  printf("info.n=%d\n", info.n);
  fflush(stdout);
  fflush(stderr);
  //if (info.n < 0) {
  if (info.n < 0) {
    JaxFFIStatusSetFailure(api, status, "Info must be >= 0");
  }
  float* input = (float*) inputs[0];
  float* output = (float*) outputs[0];
  *output = *input + info.n;
}

PYBIND11_MODULE(add_one_lib, m) {
  m.def("get_function", []() {
    return py::capsule(reinterpret_cast<void*>(add_one), JaxFFICallCpu);
  });
  m.def("get_descriptor", [](float n) {
    Info info;
    info.n = n;
    // Serialize user-provided static info as a byte string.
    return py::bytes(reinterpret_cast<const char*>(&info), sizeof(Info));
  });
}
