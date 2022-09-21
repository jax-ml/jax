#ifndef JAX_CUSTOM_CALL_H_
#define JAX_CUSTOM_CALL_H_

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

#define JAX_FFI_CALL_CPU "JAX_FFI_CALL_CPU"

#define JaxFFIVersion() \
        ((*((void (*)())(api[0])))())
#define JaxFFIStatusSetSuccess(api, status) \
        ((*((void (*)(JaxFFIStatus*))(api[1])))(status))
#define JaxFFIStatusSetFailure(api, status, msg) \
        ((*((void (*)(JaxFFIStatus*, const char*))(api[2])))(status, msg))

struct JaxFFIStatus;

typedef void* JaxFFI_API;

/*
void add_one(JaxFFI_API* api, JaxFFIStatus* status, ...) {
  JaxFFIStatusSetFailure(api, status, "...");
}
*/

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // JAX_CUSTOM_CALL_H_
