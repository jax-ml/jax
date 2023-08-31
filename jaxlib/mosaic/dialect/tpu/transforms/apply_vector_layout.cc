#include <memory>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"  // NOLINT
#include "mlir/Pass/Pass.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"

namespace mlir::tpu {

#define GEN_PASS_DECL_APPLYVECTORLAYOUTPASS
#define GEN_PASS_DEF_APPLYVECTORLAYOUTPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"

struct ApplyVectorLayoutPass
    : public impl::ApplyVectorLayoutPassBase<ApplyVectorLayoutPass> {
  ApplyVectorLayoutPass(int hardware_generation_, int lane_count_,
                        int sublane_count_) {
    hardware_generation = hardware_generation_;
    sublane_count = sublane_count_;
    lane_count = lane_count_;
  }
  void runOnOperation() override {
    // Fail if hardware_generation has not been set from the default value.
    if (hardware_generation < 0) {
      signalPassFailure();
      return;
    }
    // No-op for now
  }
};

std::unique_ptr<OperationPass<func::FuncOp>> createApplyVectorLayoutPass(
    int hardware_generation, int lane_count, int sublane_count) {
  return std::make_unique<ApplyVectorLayoutPass>(hardware_generation,
                                                 lane_count, sublane_count);
}

}  // namespace mlir::tpu
