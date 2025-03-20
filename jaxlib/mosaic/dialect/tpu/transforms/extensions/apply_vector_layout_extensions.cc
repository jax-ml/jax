#include "jaxlib/mosaic/dialect/tpu/transforms/apply_vector_layout_extensions.h"

#include "llvm/ADT/StringMap.h"
#include "mlir/IR/Operation.h"

namespace mlir::tpu::extensions {

using RewriteContext = ApplyVectorLayoutContext;

using rule_type = std::function<LogicalResult(
    RewriteContext &, Operation &, ArrayRef<Layout>, ArrayRef<Layout>)>;

const llvm::StringMap<rule_type> &rules() {
  static const llvm::StringMap<rule_type> *rules =
      new llvm::StringMap<rule_type>{};
  return *rules;
}

}  // namespace mlir::tpu::extensions