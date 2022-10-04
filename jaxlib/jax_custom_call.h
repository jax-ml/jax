#ifndef JAX_CUSTOM_CALL_H_
#define JAX_CUSTOM_CALL_H_

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

#define JaxFFICallCpu "JaxFFICallCpu"

#define JaxFFIVersion(api) (api->version)
#define JaxFFIStatusSetSuccess(api, status) \
        ((*((void (*)(JaxFFIStatus*))(api->fn_table[0])(status))))
#define JaxFFIStatusSetFailure(api, status, msg) \
        ((*((void (*)(JaxFFIStatus*, const char*))(api->fn_table[1])))(status, msg))

struct JaxFFIStatus;

struct JaxFFIApi {
  int version;
  void** fn_table;
};

/*
void add_one(JaxFFIApi* api, JaxFFIStatus* status, ...) {
  JaxFFIStatusSetFailure(api, status, "...");
}
*/

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // JAX_CUSTOM_CALL_H_
