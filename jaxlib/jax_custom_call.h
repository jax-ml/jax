#ifndef JAX_CUSTOM_CALL_H_
#define JAX_CUSTOM_CALL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>


struct JaxCustomCallStatus;

struct JaxCustomCallApi {
  JaxCustomCallStatus* status;
};

void JaxCustomCallStatusSetSuccess(JaxCustomCallStatus* status);

void JaxCustomCallStatusSetFailure(JaxCustomCallStatus* status,
                                   const char* message, uint32_t message_len);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // JAX_CUSTOM_CALL_H_
