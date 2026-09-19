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

#ifndef JAXLIB_THUNKY_DIALECT_THUNKY_OPS_TD_
#define JAXLIB_THUNKY_DIALECT_THUNKY_OPS_TD_

include "mlir/IR/OpBase.td"
include "jaxlib/thunky/dialect/thunky_dialect.td"

class Thunky_Op<string mnemonic, list<Trait> traits = []>
    : Op<ThunkyDialect, mnemonic, traits>;

def Thunky_SliceBufferOp : Thunky_Op<"slice_buffer"> {
  let summary = "Creates a sub-slice view of an existing GPU buffer";
  let arguments = (ins
    Thunky_BufferType:$buffer,
    I64Attr:$offset
  );
  let results = (outs Thunky_BufferType:$result);
  let assemblyFormat = "$buffer `offset` `=` $offset attr-dict `:` type($buffer) `->` type($result)";
}

def Thunky_MemzeroOp : Thunky_Op<"memzero"> {
  let summary = "Zeroes out a device buffer";
  let arguments = (ins Thunky_BufferType:$buffer);
  let assemblyFormat = "$buffer attr-dict `:` type($buffer)";
}

def Thunky_Memset32Op : Thunky_Op<"memset32"> {
  let summary = "Fills a device buffer with a 32-bit pattern";
  let arguments = (ins
    Thunky_BufferType:$buffer,
    I32Attr:$value
  );
  let assemblyFormat = "$buffer `,` $value attr-dict `:` type($buffer)";
}

def Thunky_CopyOp : Thunky_Op<"copy"> {
  let summary = "Copies device-to-device memory from source to destination";
  let arguments = (ins
    Thunky_BufferType:$src,
    Thunky_BufferType:$dst
  );
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

def Thunky_CallThunkProtoOp : Thunky_Op<"call_thunk_proto"> {
  let summary = "Executes a serialized XLA ThunkProto on buffer slices";
  let arguments = (ins
    Variadic<Thunky_BufferType>:$buffers,
    StrAttr:$kernel_name,
    StrAttr:$thunk_proto,
    StrAttr:$asm_text,
    StrAttr:$binary,
    OptionalAttr<StrAttr>:$executable_metadata_proto
  );
  let assemblyFormat = "($buffers^ `:` type($buffers))? `name` `=` $kernel_name attr-dict";
}

def Thunky_PtxKernelOp : Thunky_Op<"ptx_kernel"> {
  let summary = "Executes a GPU kernel from PTX assembly on buffer slices";
  let arguments = (ins
    Variadic<Thunky_BufferType>:$buffers,
    StrAttr:$kernel_name,
    StrAttr:$ptx,
    DenseI64ArrayAttr:$grid_dim,
    DenseI64ArrayAttr:$block_dim,
    DenseBoolArrayAttr:$written,
    I64Attr:$shmem_bytes
  );
  let assemblyFormat = "($buffers^ `:` type($buffers))? `name` `=` $kernel_name attr-dict";
}

def Thunky_CustomCallOp : Thunky_Op<"custom_call"> {
  let summary = "Executes an XLA FFI CustomCallThunk";
  let arguments = (ins
    Variadic<Thunky_BufferType>:$buffers,
    I64Attr:$num_operands,
    I64Attr:$num_results,
    StrAttr:$target_name,
    DictionaryAttr:$backend_config,
    OptionalAttr<ArrayAttr>:$operand_shapes,
    OptionalAttr<ArrayAttr>:$result_shapes
  );
  let assemblyFormat = "($buffers^ `:` type($buffers))? `target` `=` $target_name attr-dict";
}

def Thunky_YieldOp : Thunky_Op<"yield", [Terminator]> {
  let summary = "Terminator for thunky control flow regions";
  let assemblyFormat = "attr-dict";
}

def Thunky_CondOp : Thunky_Op<"cond", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Executes branches[branch_index] (or branch 0 if true / branch 1 if false for bool predicate)";
  let arguments = (ins Thunky_BufferType:$branch_index);
  let regions = (region VariadicRegion<SizedRegion<1>>:$branches);
  let assemblyFormat = "$branch_index `:` type($branch_index) $branches attr-dict";
}

def Thunky_WhileOp : Thunky_Op<"while", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Executes cond_region, and while condition_buffer is true, executes body_region";
  let arguments = (ins Thunky_BufferType:$condition_buffer);
  let regions = (region SizedRegion<1>:$cond_region, SizedRegion<1>:$body_region);
  let assemblyFormat = "$condition_buffer `:` type($condition_buffer) `cond` $cond_region `body` $body_region attr-dict";
}

def Thunky_AsyncStartOp : Thunky_Op<"async_start", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Launches a nested thunk region asynchronously on a computation or communication stream";
  let arguments = (ins
    I64Attr:$stream_id,
    DefaultValuedAttr<BoolAttr, "false">:$is_communication
  );
  let results = (outs Thunky_TokenType:$token);
  let regions = (region SizedRegion<1>:$body_region);
  let assemblyFormat = "`stream` `=` $stream_id $body_region attr-dict `:` type($token)";
}

def Thunky_AsyncDoneOp : Thunky_Op<"async_done"> {
  let summary = "Waits for an asynchronous execution scope to complete on the current stream";
  let arguments = (ins Thunky_TokenType:$token);
  let assemblyFormat = "$token `:` type($token) attr-dict";
}

def Thunky_AllReduceOp : Thunky_Op<"all_reduce"> {
  let summary = "Executes an XLA:GPU AllReduceThunk across participating devices";
  let arguments = (ins
    Thunky_BufferType:$src,
    Thunky_BufferType:$dst,
    StrAttr:$reduction_kind,
    TypeAttr:$element_type,
    ArrayAttr:$replica_groups,
    StrAttr:$group_mode
  );
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

def Thunky_AllGatherOp : Thunky_Op<"all_gather"> {
  let summary = "Executes an XLA:GPU AllGatherThunk across participating devices";
  let arguments = (ins
    Thunky_BufferType:$src,
    Thunky_BufferType:$dst,
    TypeAttr:$element_type,
    ArrayAttr:$replica_groups,
    StrAttr:$group_mode
  );
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

def Thunky_ReduceScatterOp : Thunky_Op<"reduce_scatter"> {
  let summary = "Executes an XLA:GPU ReduceScatterThunk across participating devices";
  let arguments = (ins
    Thunky_BufferType:$src,
    Thunky_BufferType:$dst,
    StrAttr:$reduction_kind,
    TypeAttr:$element_type,
    ArrayAttr:$replica_groups,
    StrAttr:$group_mode
  );
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

def Thunky_AllToAllOp : Thunky_Op<"all_to_all", [SameVariadicOperandSize]> {
  let summary = "Executes an XLA:GPU AllToAllThunk across participating devices";
  let arguments = (ins
    Variadic<Thunky_BufferType>:$src,
    Variadic<Thunky_BufferType>:$dst,
    TypeAttr:$element_type,
    ArrayAttr:$replica_groups,
    StrAttr:$group_mode,
    BoolAttr:$has_split_dimension
  );
  let assemblyFormat = "`(` $src `)` `,` `(` $dst `)` attr-dict `:` `(` type($src) `)` `,` `(` type($dst) `)`";
}

def Thunky_CollectivePermuteOp : Thunky_Op<"collective_permute"> {
  let summary = "Executes an XLA:GPU CollectivePermuteThunk across participating devices";
  let arguments = (ins
    Thunky_BufferType:$src,
    Thunky_BufferType:$dst,
    TypeAttr:$element_type,
    ArrayAttr:$source_target_pairs,
    StrAttr:$group_mode,
    OptionalAttr<ArrayAttr>:$replica_groups
  );
  let assemblyFormat = "$src `,` $dst attr-dict `:` type($src) `,` type($dst)";
}

def Thunky_CollectiveGroupOp : Thunky_Op<"collective_group", [SingleBlockImplicitTerminator<"YieldOp">]> {
  let summary = "Fuses nested collective or copy thunks into a single NCCL group dispatch (CollectiveGroupThunk)";
  let regions = (region SizedRegion<1>:$body_region);
  let assemblyFormat = "$body_region attr-dict";
}

#endif  // JAXLIB_THUNKY_DIALECT_THUNKY_OPS_TD_
