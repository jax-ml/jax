#include "jax_custom_call.h"

#include <optional>

struct JaxCustomCallStatus {
  std::optional<std::string> message;
}

void JaxCustomCallStatusSetSuccess(JaxCustomCallStatus* status) {
  status->message = std::nullopt;
}

void JaxCustomCallStatusSetFailure(JaxCustomCallStatus* status,
                                   const char* message, size_t message_len) {
  status->message = std::string(message, strnlen(message, message_len));
}

namespace jax {

std::optional<std::string> JaxCustomCallStatusGetMessage(
    JaxCustomCallStatus* status) {
  return status->message;
}

}
