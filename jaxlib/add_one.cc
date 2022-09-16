#include <pybind11/pybind11.h>
//#include "cuda.h"
#include "jax_custom_call.h" // Header that comes from JAX

struct Info {
  float n;
}
// TODO: upgrade XLA CPU custom call to use serialization not pointer (common case is string)

// Cannot use C++ objects in the call!
// Instead of passing a `Status`, we could pass an API object.
//void add_one(JaxCustomCallApi* api, void* descriptor, CUstream* stream, void** inputs, void** outputs) {
void add_one(JaxCustomCallApi* api, void* descriptor, void** inputs, void** outputs) {
  // descriptor could be pointer memory or string?
  Info* info = (Info*) descriptor;
  *outputs[0] = *inputs[0] + info->n;

  const char error[] = "bad";
  JaxCustomCallStatusSetFailure(api->status, error, sizeof(error));
}

PYBIND11_MODULE(add_one_lib, m) {
  m.def("get_function", []() {
    // This string can represent which custom call version
    return py::capsule(&add_one, GPU_CUSTOM_CALL_V1);
  });
  m.def("get_descriptor", [](float n): {
    Info* info = new Info();
    info->n = n;
    return py::capsule(&info, []() {
      delete info;
    });
  });
}
