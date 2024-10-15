#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_APPLY_VECTOR_LAYOUT_EXTENSIONS_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_APPLY_VECTOR_LAYOUT_EXTENSIONS_H_

#include <functional>

#include "third_party/llvm/llvm-project/llvm/include/llvm/ADT/StringMap.h"
#include "mlir/include/mlir/IR/Operation.h"
#include "mlir/include/mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu::extensions {

using RewriteContext = ApplyVectorLayoutContext;

using rule_type = std::function<LogicalResult(
    RewriteContext &, Operation &, ArrayRef<Layout>, ArrayRef<Layout>)>;

llvm::StringMap<rule_type> rules();

}  // namespace mlir::tpu::extensions

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_APPLY_VECTOR_LAYOUT_EXTENSIONS_H_
