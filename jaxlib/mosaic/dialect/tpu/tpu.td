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

#ifndef TPU_BASE
#define TPU_BASE

include "mlir/IR/BuiltinAttributes.td"
include "mlir/IR/BuiltinTypeInterfaces.td"

def TPU_Dialect : Dialect {
  let name = "tpu";
  let cppNamespace = "::mlir::tpu";
  let useDefaultAttributePrinterParser = 1;
  let useDefaultTypePrinterParser = 1;
  let extraClassDeclaration = [{
    static StringRef GetCoreTypeKey() { return "tpu.core_type"; }

    static std::optional<CoreType> GetCoreTypeAttr(Operation *op);
  }];
  let hasConstantMaterializer = 1;
  let hasCanonicalizer = 1;
}

class TPU_Attr<string name, string mnemonic_, list<Trait> traits = []>
    : AttrDef<TPU_Dialect, name, traits> {
  let mnemonic = mnemonic_;
}

def TPU_DotDimensionNumbersAttr : TPU_Attr<"DotDimensionNumbers", "dot_dimension_numbers"> {
  let parameters = (ins
    ArrayRefParameter<"int64_t", "">:$lhs_contracting_dims,
    ArrayRefParameter<"int64_t", "">:$rhs_contracting_dims,
    ArrayRefParameter<"int64_t", "">:$lhs_non_contracting_dims,
    // Empty when rhs is a 1-D vector.
    OptionalArrayRefParameter<"int64_t", "">:$rhs_non_contracting_dims,
    // The contract is a flattened structure, wherein, each element is half of a
    // pair of indices. The first element is always 0 (lhs) or 1 (rhs) and the
    // second index is the index from the lhs or rhs.
    ArrayRefParameter<"int64_t", "">:$output_dim_order,
    OptionalArrayRefParameter<"int64_t", "">:$lhs_batch_dims,
    OptionalArrayRefParameter<"int64_t", "">:$rhs_batch_dims
    );
    let assemblyFormat = "`<` `[` $lhs_contracting_dims `]` `,` `[` $rhs_contracting_dims `]` `,` "
                     "`[` $lhs_non_contracting_dims `]` `,` `[` (`]`):($rhs_non_contracting_dims^ `]`)? `,` "
                     "`[` $output_dim_order `]` `,` "
                     "`[` (`]`):($lhs_batch_dims^ `]`)? `,` "
                     "`[` (`]`):($rhs_batch_dims^ `]`)? `>`";
    let constBuilderCall = "::mlir::tpu::DotDimensionNumbersAttr::get($_builder.getContext(), $0)";
}

class TPU_Type<string name, string mnemonic_, list<Trait> traits = [],
               string baseCppType = "::mlir::Type">
    : TypeDef<TPU_Dialect, name, traits, baseCppType> {
  let mnemonic = mnemonic_;
}

def TPU_Float8EXMYType : TPU_Type<"Float8EXMY", "float8_exmy",
                            [DeclareTypeInterfaceMethods<FloatTypeInterface, ["getFloatSemantics"]>]> {
    let summary = "EXMY type in a 8 bit container";
    let description = [{
      EXMY type in a 8 bit container. Meaningful bits are aligned to LSB, and
      bits higher than the underlying exmy type in the container are considered
      as ignored. See https://arxiv.org/abs/2405.13938 for more details.
    }];

    let parameters = (ins
      TypeParameter<"::mlir::FloatType", "Underlying EXMY type">:$underlying_type
    );

    let assemblyFormat = [{
      `<` $underlying_type `>`
    }];
}

#endif  // TPU_BASE
