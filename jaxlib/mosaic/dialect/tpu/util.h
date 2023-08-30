#ifndef THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_UTIL_H_
#define THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_UTIL_H_

#include <cstdint>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Location.h"

namespace mlir::tpu {

inline arith::ConstantOp IdxConst(int64_t idx, OpBuilder &builder,
                                  Location loc) {
  return builder.create<arith::ConstantOp>(loc, builder.getIndexType(),
                                           builder.getIndexAttr(idx));
}

}  // namespace mlir::tpu

#endif  // THIRD_PARTY_PY_JAX_JAXLIB_MOSAIC_DIALECT_TPU_UTIL_H_
