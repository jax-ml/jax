#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANFORMS_APPLY_VECTOR_LAYOUT_EXTENSIONS_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANFORMS_APPLY_VECTOR_LAYOUT_EXTENSIONS_H_

#include <functional>

#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "jaxlib/mosaic/dialect/tpu/layout.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu::extensions {

const llvm::StringMap<
    std::function<LogicalResult(ApplyVectorLayoutContext &, Operation &,
                                ArrayRef<Layout>, ArrayRef<Layout>)>> &
rules();

}  // namespace mlir::tpu::extensions

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_TRANFORMS_APPLY_VECTOR_LAYOUT_EXTENSIONS_H_
