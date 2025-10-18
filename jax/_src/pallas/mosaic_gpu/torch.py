# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PyTorch interop for Mosaic GPU."""

from __future__ import annotations

import functools
import itertools
import operator
from typing import TypeGuard

import jax
from jax._src import util
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func
from jax._src.lib.mlir.dialects import hlo
from jax.experimental.mosaic.gpu import core as mgpu_core


def as_torch_kernel(fn):
  """Makes a Mosaic GPU kernel callable with PyTorch tensors.

  Args:
    fn: A JAX function that invokes a Mosaic GPU kernel. Note that
      the implementation currently only supports functions that contain a
      single Mosaic GPU kernel invocation, without any other JAX API calls,
      e.g. from :mod:`jax.numpy`.

  Returns:
    A wrapper function that accepts PyTorch tensors as inputs and returns
    PyTorch tensors as outputs. The output tensors are allocated on the
    same device as the input tensors.

  Example::

      @functools.partial(
          pl.pallas_call, out_shape=jax.ShapeDtypeStruct([128], jnp.int32)
      )
      def add_kernel(x_ref, y_ref, o_ref):
        o_ref[...] = x_ref[...] + y_ref[...]

      x = torch.arange(128, dtype=torch.int32, device="cuda")
      y = x * x
      out = plgpu.as_torch_kernel(add_kernel)(x, y)
  """
  @functools.wraps(fn)
  def wrapper(*args):
    in_structs = jax.tree.map(
        lambda arg: jax.ShapeDtypeStruct(
            # Drop the "torch." prefix from the dtype string, if present.
            arg.shape,
            str(arg.dtype).split(".")[-1],
        ),
        args,
    )
    return _compile_fn(fn, in_structs)(*args)

  return wrapper


def _find_mgpu_call(module: ir.Module) -> hlo.CustomCallOp:
  mgpu_call: hlo.CustomCallOp | None = None
  allocs = set()
  for func_op in module.body.operations:
    if not isinstance(func_op, func.FuncOp):
      continue
    for block in func_op.body.blocks:
      try:
        idx = next(
            idx
            for idx, op in enumerate(block.operations)
            if _is_custom_call(op, "mosaic_gpu_v2")
        )
      except StopIteration:
        continue
      else:
        if mgpu_call is not None:
          raise RuntimeError("Multiple Mosaic GPU calls found in the module")
      # We only accept functions where the Mosaic GPU call is immediately
      # followed by a return op, and all preceding ops are outpput buffer
      # allocations, which must be passed into the Mosaic GPU call.
      has_illegal_op = False
      for op in itertools.islice(block.operations, idx):
        if not _is_custom_call(op, "AllocateBuffer"):
          has_illegal_op = True
          break
        allocs.add(op)
      terminator_op = block.operations[idx + 1]
      if has_illegal_op or not isinstance(terminator_op, func.ReturnOp):
        raise RuntimeError(
            "Mosaic GPU call must be the only operation in the function"
        )
      mgpu_call = block.operations[idx]
      if len(terminator_op.operands_) != len(mgpu_call.results):
        raise RuntimeError(
            "The function must return all Mosaic GPU call results and nothing"
            " else"
        )
      if any(map(operator.ne, mgpu_call.results, terminator_op.operands_)):
        raise RuntimeError(
            "Mosaic GPU call results are returned in the wrong order"
        )
  if mgpu_call is None:
    raise RuntimeError("No Mosaic GPU call found in the module")
  if allocs - {op.owner for op in mgpu_call.operands}:
    raise RuntimeError(
        "Not all buffer allocations are passed into the Mosaic GPU call"
    )
  return mgpu_call


def _is_custom_call(op: ir.Operation, name: str) -> TypeGuard[hlo.CustomCallOp]:
  return isinstance(op, hlo.CustomCallOp) and op.call_target_name.value == name


@util.weakref_lru_cache
def _compile_fn(fn, in_structs):
  traced = jax.jit(fn).trace(*in_structs)
  main_module = traced.lower().compiler_ir()
  mgpu_call = _find_mgpu_call(main_module)
  backend_config = mgpu_call.attributes["mhlo.backend_config"]
  if not isinstance(in_structs, tuple):
    in_structs = (in_structs,)
  unwrap_output_tuple = False
  if not isinstance(out_structs := traced.out_info, tuple):
    out_structs = (out_structs,)
    unwrap_output_tuple = True
  return mgpu_core._as_torch_gpu_kernel(
      backend_config["module"].value.encode(),
      in_structs,
      out_structs,
      unwrap_output_tuple=unwrap_output_tuple,
  )
