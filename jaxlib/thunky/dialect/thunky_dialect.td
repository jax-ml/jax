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

#ifndef JAXLIB_THUNKY_DIALECT_THUNKY_DIALECT_TD_
#define JAXLIB_THUNKY_DIALECT_THUNKY_DIALECT_TD_

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/DialectBase.td"

def ThunkyDialect : Dialect {
  let name = "thunky";
  let description = [{
    MLIR dialect representing XLA:GPU thunk sequences and buffer allocations.
  }];
  let cppNamespace = "::mlir::thunky";
  let useDefaultTypePrinterParser = 1;
}

class Thunky_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<ThunkyDialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def Thunky_BufferType : Thunky_Type<"Buffer", "buffer"> {
  let summary = "Device buffer slice of a fixed byte size";
  let parameters = (ins "int64_t":$size);
  let assemblyFormat = "`<` $size `>`";
}

def Thunky_TokenType : Thunky_Type<"Token", "token"> {
  let summary = "Token representing an in-flight asynchronous execution scope";
}

#endif  // JAXLIB_THUNKY_DIALECT_THUNKY_DIALECT_TD_
