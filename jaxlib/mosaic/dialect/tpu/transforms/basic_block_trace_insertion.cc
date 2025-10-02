/* Copyright 2024 The JAX Authors.
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
#include <cstdint>
#include <memory>
#include <deque>
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "absl/strings/str_cat.h"
#include "jaxlib/mosaic/dialect/tpu/tpu_dialect.h"
namespace mlir::tpu {
#define GEN_PASS_DECL_BASICBLOCKTRACEINSERTIONPASS
#define GEN_PASS_DEF_BASICBLOCKTRACEINSERTIONPASS
#include "jaxlib/mosaic/dialect/tpu/tpu_passes.h.inc"
namespace {
struct BasicBlockTraceInsertionPass
    : public impl::BasicBlockTraceInsertionPassBase<
          BasicBlockTraceInsertionPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    std::deque<Region*> queue{&func.getBody()};
    int64_t block_counter = 0;
    Location loc = UnknownLoc::get(func.getContext());
    while (!queue.empty()) {
      Region* region = queue.front();
      queue.pop_front();
      for (auto it = region->begin(); it != region->end(); ++it) {
        Block& block = *it;
        if (block.empty()) {
          continue;
        }
        OpBuilder::atBlockBegin(&block).create<tpu::TraceStartOp>(
            loc, absl::StrCat("__block_", block_counter++), /*level=*/10);
        OpBuilder::atBlockTerminator(&block).create<tpu::TraceStopOp>(loc);
        for (Operation& op : block.without_terminator()) {
          for (Region &region : op.getRegions()) {
            if (!region.empty()) {
              queue.push_back(&region);
            }
          }
        }
      }
    }
  }
};
}  // namespace
std::unique_ptr<OperationPass<func::FuncOp>>
createBasicBlockTraceInsertionPass() {
  return std::make_unique<BasicBlockTraceInsertionPass>();
}
}  // namespace mlir::tpu