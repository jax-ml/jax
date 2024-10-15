#include "jaxlib/mosaic/apply_vector_layout_extensions.h"

#include "third_party/llvm/llvm-project/llvm/include/llvm/ADT/StringMap.h"
#include "mlir/include/mlir/IR/Operation.h"

namespace mlir::tpu::extensions {

llvm::StringMap<rule_type> rules() {
  auto *rules = new llvm::StringMap<rule_type>{};
  return *rules;
}

}  // namespace mlir::tpu::extensions