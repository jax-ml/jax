#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_INFER_VECTOR_LAYOUT_EXTENSIONS_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_INFER_VECTOR_LAYOUT_EXTENSIONS_H_

#include <array>
#include <cstdint>

#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/Support/LLVM.h"

namespace mlir::tpu::extensions {

bool canInferVectorLayout(const Operation &op);

LogicalResult inferVectorLayout(const Operation &op,
                                std::array<int64_t, 2> target_shape);

}  // namespace mlir::tpu::extensions

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_INFER_VECTOR_LAYOUT_EXTENSIONS_H_
