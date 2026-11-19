/* Copyright 2026 The JAX Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef JAXLIB_THUNKY_DIALECT_CAPI_H_
#define JAXLIB_THUNKY_DIALECT_CAPI_H_

#include <stdint.h>

#include "mlir-c/IR.h"
#include "mlir-c/Support.h"

#ifdef __cplusplus
extern "C" {
#endif

MLIR_DECLARE_CAPI_DIALECT_REGISTRATION(Thunky, thunky);

MLIR_CAPI_EXPORTED MlirType mlirThunkyBufferTypeGet(MlirContext ctx,
                                                    int64_t size);
MLIR_CAPI_EXPORTED MlirType mlirThunkyTokenTypeGet(MlirContext ctx);

#ifdef __cplusplus
}
#endif

#endif  // JAXLIB_THUNKY_DIALECT_CAPI_H_
