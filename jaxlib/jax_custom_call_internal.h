#ifndef JAX_CUSTOM_CALL_INTERNAL_H_
#define JAX_CUSTOM_CALL_INTERNAL_H_

#include <optional>
#include <string>

#include "jax_custom_call.h"

namespace jax {

std::optional<std::string> JaxCustomCallStatusGetMessage(
    JaxCustomCallStatus* status);

} // namespace jax

#endif  // JAX_CUSTOM_CALL_INTERNAL_H_
