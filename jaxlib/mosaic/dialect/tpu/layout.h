/* Copyright 2023 The JAX Authors.

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

#ifndef JAXLIB_MOSAIC_DIALECT_TPU_LAYOUT_H_
#define JAXLIB_MOSAIC_DIALECT_TPU_LAYOUT_H_

#include <array>
#include <cstdint>
#include <optional>
#include <ostream>
#include <tuple>

#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/OpImplementation.h"

namespace mlir::tpu {

// TODO(apaszke): Optimize this to encode the optional in the value itself
// and use a narrower type.
// An offset is nullopt when the value is replicated along sublanes or lanes.
using LayoutOffsets = std::array<std::optional<int64_t>, 2>;

class VectorLayout {
 public:
  enum class ImplicitDim {
    kNone = 0,  // To make if (implicit_dim) work.
    kMinor,
    kSecondMinor,
  };
  VectorLayout(int8_t bitwidth, LayoutOffsets offsets,
               std::array<int64_t, 2> tiling,
               ImplicitDim implicit_dim = ImplicitDim::kNone)
      : offsets_(offsets),
        tiling_(tiling),
        bitwidth_(bitwidth),
        implicit_dim_(implicit_dim) {}

  int8_t bitwidth() const { return bitwidth_; }
  const LayoutOffsets &offsets() const { return offsets_; }
  const std::array<int64_t, 2> &tiling() const { return tiling_; }
  ImplicitDim implicit_dim() const { return implicit_dim_; }

  bool operator==(const VectorLayout &other) const;

  template <typename Stream>
  void print(Stream &os) const;

  static std::optional<VectorLayout> join(const VectorLayout &l,
                                          const VectorLayout &r,
                                          ArrayRef<int64_t> shape);

  static std::optional<VectorLayout> parse(llvm::StringRef *data);

 private:
  std::tuple<std::optional<int64_t>, std::optional<int64_t>, int64_t, int64_t,
             int8_t, ImplicitDim>
  as_tuple() const;

  friend llvm::hash_code hash_value(const VectorLayout &layout);

  LayoutOffsets offsets_;
  std::array<int64_t, 2> tiling_;
  int8_t bitwidth_;
  ImplicitDim implicit_dim_;
};

using Layout = std::optional<VectorLayout>;
extern const Layout kNoLayout;

std::ostream &operator<<(std::ostream &os, const Layout &v);
llvm::raw_ostream &operator<<(llvm::raw_ostream &os, const Layout &v);
llvm::hash_code hash_value(const VectorLayout &layout);

std::optional<Layout> parseLayout(mlir::AsmParser &parser);

}  // namespace mlir::tpu

#endif  // JAXLIB_MOSAIC_DIALECT_TPU_LAYOUT_H_
