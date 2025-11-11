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

import ctypes
from collections import defaultdict
import functools
import itertools
from typing import Callable, TypeGuard, Mapping
import weakref

import jax
import jax.numpy as jnp
from jax._src import util
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import func
from jax._src.lib.mlir.dialects import hlo
from jax.experimental.mosaic.gpu import core as mgpu_core


def as_torch_kernel(fn, mesh=None):
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
    return _compile_fn(fn, in_structs, mesh)(*args)

  return wrapper


def _find_mgpu_call_in_module(module: ir.Module, collective: bool):
  main_funcs = [
      op
      for op in module.body.operations
      if isinstance(op, func.FuncOp) and op.name.value == "main"
  ]
  # TODO(apaszke): Add support for jax.jit, which will call another function
  # from main.
  if len(main_funcs) != 1:
    raise ValueError("Expected a single function in the kernel module")
  [func_body] = main_funcs[0].body.blocks
  if not collective:
    return _find_mgpu_call(func_body, list(func_body.arguments), False)
  if len(func_body.operations) != 2:
    raise ValueError(
        "Expected a manual sharding call and a terminator op in a collective"
        " kernel module"
    )
  manual_sharding_op, return_op = func_body.operations
  if manual_sharding_op.name != "sdy.manual_computation":
    raise ValueError("Expected a manual computation call")
  if return_op.name != "func.return":
    raise ValueError("Expected a return op")
  if list(manual_sharding_op.results) != list(return_op.operands):
    raise ValueError(
        "The manual sharding call must return all values and nothing else"
    )
  [body_block] = manual_sharding_op.body.blocks
  return _find_mgpu_call(body_block, list(func_body.arguments), True)


def _mlir_to_torch_dtype(torch, mlir_dtype: ir.Type):
  if mlir_dtype == ir.F32Type.get():
    return torch.float32
  if mlir_dtype == ir.F16Type.get():
    return torch.float16
  if mlir_dtype == ir.BF16Type.get():
    return torch.bfloat16
  if ir.IntegerType.isinstance(mlir_dtype):
    int_type = ir.IntegerType(mlir_dtype)
    if int_type.is_signed or int_type.is_signless:
      return getattr(torch, f"int{int_type.width}")
    else:
      return getattr(torch, f"uint{int_type.width}")
  raise NotImplementedError(f"Unsupported MLIR type: {mlir_dtype}")


def _find_mgpu_call(block: ir.Block, args: list[ir.Value]):
  import torch  # type: ignore[import-not-found]  # pytype: disable=import-error
  import torch.distributed._symmetric_memory as symm_mem
  import torch.distributed as dist
  mgpu_call: hlo.CustomCallOp | None = None
  get_outputs = None
  to_evaluate: list[Callable] = []
  init_env = {}
  value_names: Mapping[ir.Value, int] = defaultdict(int)
  for op in block.operations:
    if _is_custom_call(op, "AllocateBuffer"):
      def allocate_torch_buffer(
          env,
          device,
          _shape=op.result.type.shape,
          _dtype=_mlir_to_torch_dtype(torch, op.result.type.element_type),
          _result_name=value_names[op.result],
      ):
        env[_result_name] = torch.empty(_shape, dtype=_dtype, device=device)
        if collective:
          # TODO(apaszke): Not all args need to be in symmetric memory
          alloc = env[_result_name] = symm_mem.empty(_shape, dtype=_dtype, device=device)
          symm_mem.rendezvous(alloc, dist.group.WORLD)
        else:
          env[_result_name] = torch.empty(_shape, dtype=_dtype, device=device)
      to_evaluate.append(allocate_torch_buffer)
    elif _is_custom_call(op, "mosaic_gpu_v2"):
      if mgpu_call is not None:
        raise ValueError("Multiple Mosaic GPU kernels found in the module")
      operands = list(op.operands)
      if operands[:len(args)] != args:
        raise ValueError("The Mosaic GPU kernel operands must match the function arguments")
      mgpu_call = op
    elif op.name == "func.call":
      raise NotImplementedError("Pallas kernels calls wrapped in jax.jit are not supported")
    elif op.name == "func.return" or op.name == "sdy.return":
      if mgpu_call is None:
        raise ValueError("No Mosaic GPU call found in the module")
      if get_outputs is not None:
        raise ValueError("Multiple return ops found in the module")
      mgpu_results = list(mgpu_call.results)
      try:
        out_indices = [mgpu_results.index(o) for o in op.operands]
      except ValueError:
        raise ValueError("The function can only return kernel results") from None
      def get_outputs(*results, _out_indices=out_indices):
        return tuple(results[i] for i in _out_indices)
    elif op.name == "stablehlo.constant":
      result_type = ir.ShapedType(op.result.type)
      if result_type.shape:
        raise ValueError(f"Only scalar constants are supported, got {op}")
      if not op.value.is_splat:
        raise ValueError(f"Only splat constants are supported, got {op}")
      if result_type.element_type == ir.IntegerType.get_signless(32):
        init_env[value_names[op.result]] = ir.IntegerAttr(
            op.value.get_splat_value()
        ).value
      else:
        raise NotImplementedError(f"Only i32 constants are supported, got {op}")
    elif op.name == "stablehlo.broadcast_in_dim":
      if op.broadcast_dimensions:
        raise ValueError("Only scalar broadcasts are supported")
      target_shape = tuple(op.result.type.shape)
      result_name = value_names[op.result]
      operand_name = value_names[op.operand]
      dtype = torch.int32
      def run_broadcast(
          env,
          device,
          _target_shape=target_shape,
          _dtype=dtype,
          _operand_name=operand_name,
          _result_name=result_name,
      ):
        env[_result_name] = torch.broadcast_to(
            torch.as_tensor(env[_operand_name], dtype=_dtype, device=device),
            _target_shape,
        )

      to_evaluate.append(run_broadcast)
    else:
      raise ValueError(f"Unsupported operation found in the kernel module: {op}")
  if mgpu_call is None:
    raise ValueError("No Mosaic GPU call found in the module")
  if get_outputs is None:
    raise ValueError("No return op found in the module")

  arg_names = [value_names[arg] for arg in mgpu_call.operands]
  def prepare_args(*user_args, device):
    env = dict(init_env)
    # Only a prefix of operands are user args
    for name, arg in zip(arg_names, user_args, strict=False):
      env[name] = arg
    for thunk in to_evaluate:
      thunk(env, device)
    def _make_symmetric(arg):
      if not collective:
        return arg
      symm_arg = symm_mem.empty(arg.shape, dtype=arg.dtype, device=device)
      symm_mem.rendezvous(symm_arg, dist.group.WORLD)
      return symm_arg
    scratch_args = (_make_symmetric(env[name]) for name in arg_names[len(user_args):])
    return (*user_args, *scratch_args)
  output_input_aliases = [None] * len(mgpu_call.results)
  for alias in mgpu_call.output_operand_aliases:
    alias = hlo.OutputOperandAlias(alias)
    if alias.operand_tuple_indices:
      raise NotImplementedError("Tupled operand indices not supported")
    if len(alias.output_tuple_indices) > 1:
      raise NotImplementedError("Expected one element in output_tuple_indices")
    [output_index] = alias.output_tuple_indices or (0,)
    output_input_aliases[output_index] = alias.operand_index

  output_types = [
      (result.type.shape, _mlir_to_torch_dtype(torch, result.type.element_type))
      for result in mgpu_call.results
  ]
  def prepare_outputs(*all_args, device):
    outputs = []
    for ty, alias in zip(output_types, output_input_aliases, strict=True):
      if alias is not None:
        outputs.append(all_args[alias])
        continue
      if collective:
        # TODO(apaszke): Not all kernels need outputs in symmetric memory!
        out = symm_mem.empty(ty[0], dtype=ty[1], device=device)
        symm_mem.rendezvous(out, dist.group.WORLD)
      else:
        out = torch.empty(ty[0], dtype=ty[1], device=device)
      outputs.append(out)
    return outputs

  return mgpu_call, prepare_args, prepare_outputs, get_outputs


def _is_custom_call(op: ir.Operation, name: str) -> TypeGuard[hlo.CustomCallOp]:
  return isinstance(op, hlo.CustomCallOp) and op.call_target_name.value == name


@util.weakref_lru_cache
def _compile_fn(fn, in_structs, mesh):
  try:
    import torch  # type: ignore[import-not-found]  # pytype: disable=import-error
  except ImportError:
    raise RuntimeError("Can't compile for PyTorch: import torch failed") from None

  if collective := (mesh is not None):
    all_axes_spec = jax.sharding.PartitionSpec(*mesh.axis_names)
    # We use shard_map only to ensure that the kernel is traced with the right
    # mesh context.
    fn = jax.shard_map(
        fn,
        mesh=mesh,
        in_specs=all_axes_spec,
        out_specs=all_axes_spec,
        check_vma=False,
    )

  traced = jax.jit(fn).trace(*in_structs)
  main_module = traced.lower().compiler_ir()
  with main_module.context:
    mgpu_call, prepare_args, prepare_outputs, get_outputs = _find_mgpu_call_in_module(
        main_module, collective
    )

  if not isinstance(in_structs, tuple):
    in_structs = (in_structs,)
  unwrap_output_tuple = False
  if not isinstance(out_structs := traced.out_info, tuple):
    out_structs = (out_structs,)
    unwrap_output_tuple = True
  flat_arg_types, expected_arg_treedef = jax.tree.flatten(in_structs)
  _, out_treedef = jax.tree.flatten(out_structs)

  backend_config = mgpu_call.attributes["mhlo.backend_config"]
  module_asm = backend_config["module"].value_bytes
  launch, unload = mgpu_core._compile_as_torch_gpu_kernel(module_asm)

  def as_torch_dtype(dtype):
    # torch contains NumPy-compatible dtypes in its top namespace
    return getattr(torch, jnp.dtype(dtype).name)

  def apply(*user_args):
    flat_user_args, arg_treedef = jax.tree.flatten(user_args)
    if arg_treedef != expected_arg_treedef:
      raise ValueError(
          f"Invalid argument structure: expected {expected_arg_treedef}, got"
          f" {arg_treedef}, ({user_args=})"
      )
    for arg, expected_ty in zip(flat_user_args, flat_arg_types):
      if arg.shape != expected_ty.shape:
        raise ValueError(
            f"Argument shape mismatch: expected {expected_ty.shape}, got"
            f" {arg.shape}"
        )
      if arg.dtype != as_torch_dtype(expected_ty.dtype):
        raise ValueError(
            "Argument dtype mismatch: expected"
            f" {as_torch_dtype(expected_ty.dtype)}, got {arg.dtype}"
        )

    # We run all the ops that are necessary to prepare the arguments
    device = torch.device("cuda")
    flat_args = prepare_args(*flat_user_args, device=device)
    flat_outs = prepare_outputs(*flat_args, device=device)
    # Construct a device pointer list like in the XLA calling convention
    buffers = (ctypes.c_void_p * (len(flat_args) + len(flat_outs)))()
    for i, arg in enumerate(itertools.chain(flat_args, flat_outs)):
      buffers[i] = arg.data_ptr()
    launch(buffers, device)
    user_outs = get_outputs(*flat_outs)
    out = jax.tree.unflatten(out_treedef, user_outs)
    return out[0] if unwrap_output_tuple else out

  # Unload the compiled code when the Python function is destroyed.
  apply.destructor = weakref.ref(apply, lambda _weak_ref: unload)

  return apply
