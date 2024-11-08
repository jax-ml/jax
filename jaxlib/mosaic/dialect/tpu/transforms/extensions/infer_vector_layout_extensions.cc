#include "jaxlib/mosaic/dialect/tpu/transforms/infer_vector_layout_extensions.h"

#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "mlir/include/mlir/Support/LogicalResult.h"

namespace mlir::tpu::extensions {

bool canInferVectorLayout(const Operation &op) { return false; }

LogicalResult inferVectorLayout(const Operation &op) { return failure(); }

}  // namespace mlir::tpu::extensions