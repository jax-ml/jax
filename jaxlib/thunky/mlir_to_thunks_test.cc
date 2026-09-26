/* Copyright 2026 The JAX Authors.

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

#include "jaxlib/thunky/mlir_to_thunks.h"

#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/str_format.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/Parser/Parser.h"
#include "jaxlib/thunky/dialect/thunky_dialect.h"
#include "xla/backends/gpu/codegen/kernels/custom_kernel.h"
#include "xla/backends/gpu/runtime/async_thunk.h"
#include "xla/backends/gpu/runtime/collective_group_thunk.h"
#include "xla/backends/gpu/runtime/conditional_thunk.h"
#include "xla/backends/gpu/runtime/custom_call_thunk.h"
#include "xla/backends/gpu/runtime/custom_kernel_thunk.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/memset_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/backends/gpu/runtime/thunk.pb.h"
#include "xla/backends/gpu/runtime/while_thunk.h"
#include "xla/ffi/ffi.h"
#include "xla/service/buffer_assignment.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

class MlirToThunksTest : public ::testing::Test {
 protected:
  void SetUp() override {
    context_.loadDialect<mlir::func::FuncDialect>();
    context_.loadDialect<mlir::thunky::ThunkyDialect>();
  }

  mlir::OwningOpRef<mlir::ModuleOp> ParseModule(llvm::StringRef source) {
    return mlir::parseSourceString<mlir::ModuleOp>(source, &context_);
  }

  mlir::MLIRContext context_;
};

TEST_F(MlirToThunksTest, LowersMemzeroAndCopy) {
  constexpr llvm::StringLiteral kModule = R"mlir(
    module {
      func.func @main(%arg0: !thunky.buffer<64>, %arg1: !thunky.buffer<64>) {
        thunky.memzero %arg0 : !thunky.buffer<64>
        thunky.copy %arg0, %arg1 : !thunky.buffer<64>, !thunky.buffer<64>
        return
      }
    }
  )mlir";
  auto module = ParseModule(kModule);
  ASSERT_TRUE(module);

  BufferAllocation alloc0(0, 64, 0);
  BufferAllocation alloc1(1, 64, 0);
  BufferAllocation::Slice slice0(&alloc0, 0, 64);
  BufferAllocation::Slice slice1(&alloc1, 0, 64);

  auto status_or_thunks = LowerThunkyModuleToThunkSequence(
      *module, {slice0, slice1}, {}, []() { return Thunk::ThunkInfo(); });
  ASSERT_THAT(status_or_thunks, absl_testing::IsOk());
  const ThunkSequence& thunks = *status_or_thunks;
  ASSERT_EQ(thunks.size(), 2);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kMemzero);
  EXPECT_EQ(thunks[1]->kind(), Thunk::kCopy);

  auto* copy_thunk = static_cast<DeviceToDeviceCopyThunk*>(thunks[1].get());
  EXPECT_EQ(copy_thunk->source().slice, slice0);
  EXPECT_EQ(copy_thunk->destination().slice, slice1);
  EXPECT_EQ(copy_thunk->size_bytes(), 64);
}

TEST_F(MlirToThunksTest, LowersSliceBuffer) {
  constexpr llvm::StringLiteral kModule = R"mlir(
    module {
      func.func @main(%arg0: !thunky.buffer<128>) {
        %0 = thunky.slice_buffer %arg0 offset = 32 : !thunky.buffer<128> -> !thunky.buffer<64>
        thunky.memzero %0 : !thunky.buffer<64>
        return
      }
    }
  )mlir";
  auto module = ParseModule(kModule);
  ASSERT_TRUE(module);

  BufferAllocation alloc(0, 128, 0);
  BufferAllocation::Slice slice(&alloc, 0, 128);

  auto status_or_thunks = LowerThunkyModuleToThunkSequence(
      *module, {slice}, {}, []() { return Thunk::ThunkInfo(); });
  ASSERT_THAT(status_or_thunks, absl_testing::IsOk());
  const ThunkSequence& thunks = *status_or_thunks;
  ASSERT_EQ(thunks.size(), 1);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kMemzero);

  auto* memzero_thunk = static_cast<MemzeroThunk*>(thunks[0].get());
  EXPECT_EQ(memzero_thunk->destination().slice.offset(), 32);
  EXPECT_EQ(memzero_thunk->destination().slice.size(), 64);
}

TEST_F(MlirToThunksTest, LowersWhile) {
  constexpr llvm::StringLiteral kModule = R"mlir(
    module {
      func.func @main(%arg0: !thunky.buffer<4>, %arg1: !thunky.buffer<64>) {
        thunky.while %arg0 : !thunky.buffer<4> cond {
          thunky.memzero %arg0 : !thunky.buffer<4>
        } body {
          thunky.memzero %arg1 : !thunky.buffer<64>
        }
        return
      }
    }
  )mlir";
  auto module = ParseModule(kModule);
  ASSERT_TRUE(module);

  BufferAllocation alloc0(0, 4, 0);
  BufferAllocation alloc1(1, 64, 0);
  BufferAllocation::Slice slice0(&alloc0, 0, 4);
  BufferAllocation::Slice slice1(&alloc1, 0, 64);

  auto status_or_thunks = LowerThunkyModuleToThunkSequence(
      *module, {slice0, slice1}, {}, []() { return Thunk::ThunkInfo(); });
  ASSERT_THAT(status_or_thunks, absl_testing::IsOk());
  const ThunkSequence& thunks = *status_or_thunks;
  ASSERT_EQ(thunks.size(), 1);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kWhile);

  auto* while_thunk = static_cast<WhileThunk*>(thunks[0].get());
  EXPECT_EQ(while_thunk->condition_result_buffer(), slice0);
}

TEST_F(MlirToThunksTest, LowersCond) {
  constexpr llvm::StringLiteral kModule = R"mlir(
    module {
      func.func @main(%arg0: !thunky.buffer<4>, %arg1: !thunky.buffer<64>, %arg2: !thunky.buffer<64>) {
        thunky.cond %arg0 : !thunky.buffer<4> {
          thunky.memzero %arg1 : !thunky.buffer<64>
        }, {
          thunky.memzero %arg2 : !thunky.buffer<64>
        }
        return
      }
    }
  )mlir";
  auto module = ParseModule(kModule);
  ASSERT_TRUE(module);

  BufferAllocation alloc0(0, 4, 0);
  BufferAllocation alloc1(1, 64, 0);
  BufferAllocation alloc2(2, 64, 0);
  BufferAllocation::Slice slice0(&alloc0, 0, 4);
  BufferAllocation::Slice slice1(&alloc1, 0, 64);
  BufferAllocation::Slice slice2(&alloc2, 0, 64);

  auto status_or_thunks =
      LowerThunkyModuleToThunkSequence(*module, {slice0, slice1, slice2}, {},
                                       []() { return Thunk::ThunkInfo(); });
  ASSERT_THAT(status_or_thunks, absl_testing::IsOk());
  const ThunkSequence& thunks = *status_or_thunks;
  ASSERT_EQ(thunks.size(), 1);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kConditional);

  auto* cond_thunk = static_cast<ConditionalThunk*>(thunks[0].get());
  EXPECT_EQ(cond_thunk->branch_index_buffer().slice, slice0);
  EXPECT_EQ(cond_thunk->branch_executors().size(), 2);
}

TEST_F(MlirToThunksTest, LowersAsyncStartAndDone) {
  constexpr llvm::StringLiteral kModule = R"mlir(
    module {
      func.func @main(%arg0: !thunky.buffer<64>) {
        %token = thunky.async_start stream = 1 {
          thunky.memzero %arg0 : !thunky.buffer<64>
        } : !thunky.token
        thunky.async_done %token : !thunky.token
        return
      }
    }
  )mlir";
  auto module = ParseModule(kModule);
  ASSERT_TRUE(module);

  BufferAllocation alloc0(0, 64, 0);
  BufferAllocation::Slice slice0(&alloc0, 0, 64);

  auto status_or_thunks = LowerThunkyModuleToThunkSequence(
      *module, {slice0}, {}, []() { return Thunk::ThunkInfo(); });
  ASSERT_THAT(status_or_thunks, absl_testing::IsOk());
  const ThunkSequence& thunks = *status_or_thunks;
  ASSERT_EQ(thunks.size(), 2);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kAsyncStart);
  EXPECT_EQ(thunks[1]->kind(), Thunk::kAsyncDone);

  auto* async_start = static_cast<AsyncStartThunk*>(thunks[0].get());
  auto* async_done = static_cast<AsyncDoneThunk*>(thunks[1].get());
  EXPECT_EQ(async_start->async_execution(), async_done->async_execution());
  EXPECT_EQ(async_start->thunks().size(), 1);
  EXPECT_EQ(async_start->thunks()[0]->kind(), Thunk::kMemzero);
}

TEST_F(MlirToThunksTest, LowersCollectives) {
  constexpr llvm::StringLiteral kModule = R"mlir(
    module {
      func.func @main(%arg0: !thunky.buffer<64>, %arg1: !thunky.buffer<64>) {
        thunky.all_reduce %arg0, %arg1 {
          element_type = f32,
          group_mode = "cross_replica",
          reduction_kind = "sum",
          replica_groups = [[0, 1]]
        } : !thunky.buffer<64>, !thunky.buffer<64>
        thunky.all_gather %arg0, %arg1 {
          element_type = f32,
          group_mode = "cross_replica",
          replica_groups = [[0, 1]]
        } : !thunky.buffer<64>, !thunky.buffer<64>
        thunky.reduce_scatter %arg0, %arg1 {
          element_type = f32,
          group_mode = "cross_replica",
          reduction_kind = "sum",
          replica_groups = [[0, 1]]
        } : !thunky.buffer<64>, !thunky.buffer<64>
        thunky.all_to_all (%arg0), (%arg1) {
          element_type = f32,
          group_mode = "cross_replica",
          has_split_dimension = true,
          replica_groups = [[0, 1]]
        } : (!thunky.buffer<64>), (!thunky.buffer<64>)
        thunky.collective_permute %arg0, %arg1 {
          element_type = f32,
          group_mode = "cross_replica",
          source_target_pairs = [[0, 1], [1, 0]]
        } : !thunky.buffer<64>, !thunky.buffer<64>
        return
      }
    }
  )mlir";
  auto module = ParseModule(kModule);
  ASSERT_TRUE(module);

  BufferAllocation alloc0(0, 64, 0);
  BufferAllocation alloc1(1, 64, 0);
  BufferAllocation::Slice slice0(&alloc0, 0, 64);
  BufferAllocation::Slice slice1(&alloc1, 0, 64);

  auto status_or_thunks = LowerThunkyModuleToThunkSequence(
      *module, {slice0, slice1}, {}, []() { return Thunk::ThunkInfo(); });
  ASSERT_THAT(status_or_thunks, absl_testing::IsOk());
  const ThunkSequence& thunks = *status_or_thunks;
  ASSERT_EQ(thunks.size(), 5);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kAllReduce);
  EXPECT_EQ(thunks[1]->kind(), Thunk::kAllGather);
  EXPECT_EQ(thunks[2]->kind(), Thunk::kReduceScatter);
  EXPECT_EQ(thunks[3]->kind(), Thunk::kAllToAll);
  EXPECT_EQ(thunks[4]->kind(), Thunk::kCollectivePermute);
}

TEST_F(MlirToThunksTest, LowersCollectiveGroup) {
  constexpr llvm::StringLiteral kModule = R"mlir(
    module {
      func.func @main(%arg0: !thunky.buffer<64>, %arg1: !thunky.buffer<64>) {
        thunky.collective_group {
          thunky.copy %arg0, %arg1 : !thunky.buffer<64>, !thunky.buffer<64>
        }
        return
      }
    }
  )mlir";
  auto module = ParseModule(kModule);
  ASSERT_TRUE(module);

  BufferAllocation alloc0(0, 64, 0);
  BufferAllocation alloc1(1, 64, 0);
  BufferAllocation::Slice slice0(&alloc0, 0, 64);
  BufferAllocation::Slice slice1(&alloc1, 0, 64);

  auto status_or_thunks = LowerThunkyModuleToThunkSequence(
      *module, {slice0, slice1}, {}, []() { return Thunk::ThunkInfo(); });
  ASSERT_THAT(status_or_thunks, absl_testing::IsOk());
  const ThunkSequence& thunks = *status_or_thunks;
  ASSERT_EQ(thunks.size(), 1);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kGroup);

  auto* group_thunk = static_cast<CollectiveGroupThunk*>(thunks[0].get());
  ASSERT_EQ(group_thunk->thunks().size(), 1);
  EXPECT_EQ(group_thunk->thunks()[0]->kind(), Thunk::kCopy);
}

static absl::Status DummyCustomCallHandler() { return absl::OkStatus(); }
XLA_FFI_DEFINE_HANDLER(kDummyCustomCallHandler, DummyCustomCallHandler,
                       xla::ffi::Ffi::Bind());
XLA_FFI_REGISTER_HANDLER(xla::ffi::GetXlaFfiApi(), "dummy_custom_call", "CUDA",
                         kDummyCustomCallHandler);

TEST_F(MlirToThunksTest, LowersMemset32) {
  constexpr llvm::StringLiteral kModule = R"mlir(
    module {
      func.func @main(%arg0: !thunky.buffer<64>) {
        thunky.memset32 %arg0, 42 : !thunky.buffer<64>
        return
      }
    }
  )mlir";
  auto module = ParseModule(kModule);
  ASSERT_TRUE(module);

  BufferAllocation alloc0(0, 64, 0);
  BufferAllocation::Slice slice0(&alloc0, 0, 64);

  auto status_or_thunks = LowerThunkyModuleToThunkSequence(
      *module, {slice0}, {}, []() { return Thunk::ThunkInfo(); });
  ASSERT_THAT(status_or_thunks, absl_testing::IsOk());
  const ThunkSequence& thunks = *status_or_thunks;
  ASSERT_EQ(thunks.size(), 1);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kMemset32BitValue);

  auto* memset_thunk = static_cast<Memset32BitValueThunk*>(thunks[0].get());
  EXPECT_EQ(memset_thunk->destination(), slice0);
  EXPECT_EQ(memset_thunk->value(), 42);
}

TEST_F(MlirToThunksTest, LowersPtxKernel) {
  constexpr llvm::StringLiteral kModule = R"mlir(
    module {
      func.func @main(%arg0: !thunky.buffer<64>) {
        thunky.ptx_kernel %arg0 : !thunky.buffer<64> name = "my_kernel" {
          ptx = ".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry my_kernel(.param .u64 p0) { ret; }",
          grid_dim = array<i64: 1, 1, 1>,
          block_dim = array<i64: 32, 1, 1>,
          written = array<i1: true>,
          shmem_bytes = 0 : i64
        }
        return
      }
    }
  )mlir";
  auto module = ParseModule(kModule);
  ASSERT_TRUE(module);

  BufferAllocation alloc0(0, 64, 0);
  BufferAllocation::Slice slice0(&alloc0, 0, 64);

  auto status_or_thunks = LowerThunkyModuleToThunkSequence(
      *module, {slice0}, {}, []() { return Thunk::ThunkInfo(); });
  ASSERT_THAT(status_or_thunks, absl_testing::IsOk());
  const ThunkSequence& thunks = *status_or_thunks;
  ASSERT_EQ(thunks.size(), 1);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kCustomKernel);

  auto* custom_kernel_thunk = static_cast<CustomKernelThunk*>(thunks[0].get());
  EXPECT_EQ(custom_kernel_thunk->custom_kernel().name(), "my_kernel");
}

TEST_F(MlirToThunksTest, LowersCustomCall) {
  constexpr llvm::StringLiteral kModule = R"mlir(
    module {
      func.func @main(%arg0: !thunky.buffer<64>, %arg1: !thunky.buffer<64>) {
        thunky.custom_call %arg0, %arg1 : !thunky.buffer<64>, !thunky.buffer<64> target = "dummy_custom_call" {
          backend_config = {},
          num_operands = 1 : i64,
          num_results = 1 : i64
        }
        return
      }
    }
  )mlir";
  auto module = ParseModule(kModule);
  ASSERT_TRUE(module);

  BufferAllocation alloc0(0, 64, 0);
  BufferAllocation alloc1(1, 64, 0);
  BufferAllocation::Slice slice0(&alloc0, 0, 64);
  BufferAllocation::Slice slice1(&alloc1, 0, 64);

  auto status_or_thunks = LowerThunkyModuleToThunkSequence(
      *module, {slice0, slice1}, {}, []() { return Thunk::ThunkInfo(); });
  ASSERT_THAT(status_or_thunks, absl_testing::IsOk());
  const ThunkSequence& thunks = *status_or_thunks;
  ASSERT_EQ(thunks.size(), 1);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kCustomCall);

  auto* custom_call_thunk = static_cast<CustomCallThunk*>(thunks[0].get());
  EXPECT_EQ(custom_call_thunk->target_name(), "dummy_custom_call");
  EXPECT_EQ(custom_call_thunk->operands().size(), 1);
  EXPECT_EQ(custom_call_thunk->results().size(), 1);
  EXPECT_EQ(custom_call_thunk->operands()[0]->slice, slice0);
  EXPECT_EQ(custom_call_thunk->results()[0]->slice, slice1);
}

TEST_F(MlirToThunksTest, LowersCallThunkProto) {
  BufferAllocation dummy_alloc(0, 64, 0);
  BufferAllocation::Slice dummy_slice(&dummy_alloc, 0, 64);
  Shape shape = ShapeUtil::MakeShape(U8, {64});
  MemzeroThunk memzero(Thunk::ThunkInfo(), ShapedSlice{dummy_slice, shape});
  ASSERT_OK_AND_ASSIGN(ThunkProto proto, memzero.ToProto());
  std::string proto_bytes = proto.SerializeAsString();

  std::string escaped_proto;
  llvm::raw_string_ostream os(escaped_proto);
  llvm::printEscapedString(proto_bytes, os);

  std::string module_str = absl::StrFormat(
      R"mlir(
    module {
      func.func @main(%%arg0: !thunky.buffer<64>) {
        thunky.call_thunk_proto %%arg0 : !thunky.buffer<64> name = "test" {
          thunk_proto = "%s",
          asm_text = "",
          binary = ""
        }
        return
      }
    }
  )mlir",
      escaped_proto);

  auto module = ParseModule(module_str);
  ASSERT_TRUE(module);

  BufferAllocation alloc0(0, 64, 0);
  BufferAllocation::Slice slice0(&alloc0, 0, 64);

  auto status_or_thunks = LowerThunkyModuleToThunkSequence(
      *module, {slice0}, {}, []() { return Thunk::ThunkInfo(); });
  ASSERT_THAT(status_or_thunks, absl_testing::IsOk());
  const ThunkSequence& thunks = *status_or_thunks;
  ASSERT_EQ(thunks.size(), 1);
  EXPECT_EQ(thunks[0]->kind(), Thunk::kMemzero);
}

}  // namespace
}  // namespace xla::gpu
