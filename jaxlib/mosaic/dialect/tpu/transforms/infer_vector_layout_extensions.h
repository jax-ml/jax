#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_INFER_VECTOR_LAYOUT_EXTENSIONS_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_INFER_VECTOR_LAYOUT_EXTENSIONS_H_

#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/Support/LLVM.h"

namespace mlir::tpu::ext {

bool canInferVectorLayout(const Operation &op);

LogicalResult inferVectorLayout(const Operation &op);

}  // namespace mlir::tpu::ext

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANSFORMS_INFER_VECTOR_LAYOUT_EXTENSIONS_H_
