/* Copyright 2025 The JAX Authors.

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

#include "jaxlib/mosaic/dialect/tpu/util.h"

#include <functional>
#include <numeric>

#include <gtest/gtest.h>
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Support/LLVM.h"

namespace mlir::tpu {

namespace {

class TreeReduceTest : public ::testing::Test,
                       public ::testing::WithParamInterface</*size=*/int> {};

TEST_P(TreeReduceTest, Test) {
  auto values = llvm::to_vector(llvm::seq<int>(0, GetParam()));
  EXPECT_EQ(treeReduce(ArrayRef<int>(values), std::plus<int>()),
            std::reduce(values.begin(), values.end(), 0, std::plus<int>()));
}

INSTANTIATE_TEST_SUITE_P(TreeReduceTest, TreeReduceTest,
                         ::testing::Range(1, 5));

}  // namespace

}  // namespace mlir::tpu
