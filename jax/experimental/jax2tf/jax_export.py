# Copyright 2023 The JAX Authors.
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
"""JAX APIs for exporting code for interoperation.

This module is used with jax2tf, but should have no TensorFlow dependencies.
"""
import dataclasses
from functools import partial
import re
from typing import  Callable, Dict, List, Optional, Sequence, Set, Union

from absl import logging

import jax
from jax import sharding

from jax._src import core
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.lib import xla_client
from jax._src.lib.mlir.dialects import stablehlo
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.lib.mlir.dialects import func as func_dialect


map = util.safe_map
zip = util.safe_zip

# These are the JAX custom call target names that are guaranteed to be stable.
# Their backwards compatibility is tested by back_compat_test.py.
_CUSTOM_CALL_TARGETS_GUARANTEED_STABLE = [
    "Sharding", "SPMDFullToShardShape", "SPMDShardToFullShape",
    "ducc_fft", "cu_threefry2x32",
    # eigh on CPU
    "lapack_ssyevd", "lapack_dsyevd", "lapack_cheevd", "lapack_zheevd",
    # eigh on GPU
    "cusolver_syevj", "cusolver_syevd",
    # eigh on TPU
    "Eigh",
    # qr on CPU
    "lapack_sgeqrf", "lapack_dgeqrf", "lapack_cgeqrf", "lapack_zgeqrf",
    "lapack_sorgqr", "lapack_dorgqr", "lapack_cungqr", "lapack_zungqr",
    # qr on GPU
    "cusolver_geqrf", "cublas_geqrf_batched",
    "cusolver_geqrf", "cusolver_orgqr",
    # qr and svd on TPU
    "Qr", "ProductOfElementaryHouseholderReflectors",
    # TODO(atondwal, necula): add back_compat tests for lu on CPU/GPU
    # # lu on CPU
    # "lapack_sgetrf" , "lapack_dgetrf" , "lapack_cgetrf" , "lapack_zgetrf",
    # # lu on GPU
    # "cublas_getrf_batched", "cusolver_getrf",
    # "hipblas_getrf_batched", "hipsolver_getrf",
    # lu on TPU
    "LuDecomposition",
]


@dataclasses.dataclass
class Exported:
  """Represents a lowered and serialized JAX module."""
  in_avals: Sequence[core.ShapedArray]
  out_avals: Sequence[core.ShapedArray]
  # The in_shardings reflect only the module_kept_var_idx
  in_shardings: Sequence[Union[sharding.XLACompatibleSharding, pxla.UnspecifiedValue]]
  out_shardings: Sequence[Union[sharding.XLACompatibleSharding, pxla.UnspecifiedValue]]
  lowering_platform: str  # One of "tpu", "cpu", "cuda", "rocm"

  mlir_module: mlir.ir.Module
  mlir_module_serialized: bytes  # VHLO bytecode format
  xla_call_module_version: int  # Follows the versions of XlaCallModule
  module_kept_var_idx: Sequence[int]  # Specifies if an argument is kept in the
                                      # lowering. Same length as `in_shardings`.


def default_jax_backend() -> str:
  # Canonicalize to turn into CUDA or ROCM
  return xb.canonicalize_platform(jax.default_backend())


def serialize_native(fun_jax: Callable,
                     args_avals: Sequence[core.ShapedArray], *,
                     lowering_platform: Optional[str],
                     strict_checks: bool) -> Exported:
  arg_specs_jax = [
    jax.ShapeDtypeStruct(aval.shape, aval.dtype, named_shape=aval.named_shape)
    for aval in args_avals
  ]

  if not hasattr(fun_jax, "lower"):
    # We support convert(pjit(f_jax)) and convert(jit(f_jax)) but also
    # convert(f_jax), in which case a "jit" is implied. In that case we raise
    # an error if the lowered function contains non-replicated sharding annotations.
    fun_jax_lower = jax.jit(fun_jax).lower
    allow_non_replicated_sharding = False
  else:
    # If we have a pjit or pmap already we do not wrap with another, and we
    # allow shardings.
    fun_jax_lower = fun_jax.lower
    allow_non_replicated_sharding = True

  lowered = fun_jax_lower(
      *arg_specs_jax,
      _experimental_lowering_platform=lowering_platform)._lowering  # type: ignore

  mlir_module = lowered.stablehlo()
  if "kept_var_idx" in lowered.compile_args:
    module_kept_var_idx = tuple(sorted(lowered.compile_args["kept_var_idx"]))
  else:
    # For pmap
    module_kept_var_idx = tuple(range(len(args_avals)))

  if not all(core.is_constant_shape(a.shape) for a in args_avals):
    # All arguments are kept if we have dimension variables.
    assert len(module_kept_var_idx) == len(args_avals)
    mlir_module = compute_dim_vars(mlir_module, args_avals)

  xla_call_module_version = 4
  mlir_str = mlir.module_to_bytecode(mlir_module)
  target_version = stablehlo.get_earliest_forward_compatible_version()
  mlir_module_serialized = xla_client._xla.mlir.serialize_portable_artifact(
      mlir_str, target_version)

  # Figure out the result types and shapes
  if "global_out_avals" in lowered.compile_args:
    # This is currently the case for pjit
    out_avals = lowered.compile_args["global_out_avals"]
  elif "shards" in lowered.compile_args:  # for PmapComputation
    out_avals = lowered.compile_args["shards"].out_sharded_avals
  else:
    out_avals = lowered.compile_args["out_avals"]
  if lowered.compile_args["host_callbacks"]:
    raise NotImplementedError("host_callbacks are not yet implemented for the jax2tf native lowering")

  # Log and then check the module.
  if logging.vlog_is_on(3):
    mlir_module_text = mlir.module_to_string(mlir_module)
    logmsg = f"version={xla_call_module_version} lowering_platform={lowering_platform}"
    logging.vlog(3, "Lowered JAX module: %s\n%s", logmsg, mlir_module_text)

  check_module(mlir_module,
               allow_non_replicated_sharding=allow_non_replicated_sharding,
               allow_all_custom_calls=not strict_checks)

  return Exported(
      in_avals=args_avals,
      out_avals=out_avals,
      in_shardings=lowered.compile_args["in_shardings"],
      out_shardings=lowered.compile_args["out_shardings"],
      lowering_platform=lowering_platform or default_jax_backend(),
      mlir_module=mlir_module,
      mlir_module_serialized=mlir_module_serialized,
      module_kept_var_idx=module_kept_var_idx,
      xla_call_module_version=xla_call_module_version)


def compute_dim_vars(module: mlir.ir.Module,
                     args_avals: Sequence[core.ShapedArray]) -> mlir.ir.Module:
  """Wraps the lowered module with a new "main" that computes the dim vars.

  JAX lowering in presence of shape polymorphism produces a `module` that
  takes one or more dimension arguments, specified using 0-dimensional tensors
  of type i32 or i64, followed by the regular array arguments.
  The dimension arguments correspond to the dimension variables appearing in
  the `args_avals`, in sorted order.

  Consider the lowering of a function with one array argument of type "f32[w, h]",
  where "w" and "h" are two dimension variables. The `module` will also
  contain two dimension arguments, corresponding to "h" and "w" respectively:

      func public main(arg_h: i32, arg_w: i32, arg: f32[?, ?]) {
        ...
      }

      we rename "main" to "_wrapped_jax_export_main" and add a new "main":

      func public main(arg: f32[?, ?]) {
         arg_h = hlo.get_dimension_size(arg, 1)
         arg_w = hlo.get_dimension_size(arg, 0)
         res = call _wrapped_jax_export_main(arg_h, arg_w, arg)
         return res
      }

  Args:
    module: the HLO module as obtained from lowering. May have a number of
      dimension arguments, followed by the kept array arguments.
    args_avals: the avals for all the arguments of the lowered function, which
      correspond to the array arguments of the `module`.

  Returns the wrapped module.
  """
  dim_args_builders = get_dim_arg_builders(args_avals)

  # Make a new module, do not mutate the "module" because it may be cached
  context = mlir.make_ir_context()
  with context, ir.Location.unknown(context):
    new_module = ir.Module.parse(mlir.module_to_bytecode(module))
    symbol_table = ir.SymbolTable(new_module.operation)
    orig_main = symbol_table["main"]
    orig_main.attributes["sym_visibility"] = ir.StringAttr.get("private")
    orig_main_name = "_wrapped_jax_export_main"
    symbol_table.set_symbol_name(orig_main, orig_main_name)

    orig_input_types = orig_main.type.inputs
    nr_array_args = len(orig_input_types) - len(dim_args_builders)
    assert nr_array_args >= 0

    new_main_input_types = orig_input_types[- nr_array_args:]
    orig_output_types = orig_main.type.results

    ftype = ir.FunctionType.get(new_main_input_types, orig_output_types)
    new_main_op = func_dialect.FuncOp(
        "main", ftype, ip=ir.InsertionPoint.at_block_begin(new_module.body))
    new_main_op.attributes["sym_visibility"] = ir.StringAttr.get("public")
    try:
      new_main_op.arg_attrs = list(orig_main.arg_attrs)[- nr_array_args:]
    except KeyError:
      pass  # TODO: better detection if orig_main.arg_attrs does not exist
    try:
      new_main_op.result_attrs = orig_main.result_attrs
    except KeyError:
      pass
    symbol_table.insert(new_main_op)
    entry_block = new_main_op.add_entry_block()
    with ir.InsertionPoint(entry_block):
      orig_main_args = []
      # The first arguments are the dimension variable
      for dim_arg_idx, dim_arg_builder in enumerate(dim_args_builders):
        orig_main_args.append(
            dim_arg_builder(new_main_op.arguments, orig_input_types[dim_arg_idx]))
      # Then the array arguments
      orig_main_args.extend(new_main_op.arguments)
      call = func_dialect.CallOp(orig_output_types,
                                 ir.FlatSymbolRefAttr.get(orig_main_name),
                                 orig_main_args)
      func_dialect.ReturnOp(call.results)
    symbol_table.set_symbol_name(new_main_op, "main")
    return new_module


# A dimension argument builder computes a dimension argument given
# the array arguments and the desired type of the dimension argument.
DimArgBuilder = Callable[[Sequence[mlir.ir.Value], mlir.ir.Type], mlir.ir.Value]

def get_dim_arg_builders(
    args_avals: Sequence[core.ShapedArray]) -> Sequence[DimArgBuilder]:
  """For each dimension variable, return a builder.

  Args:
    args_avals: the abstract values of the array arguments.

  Returns:
    a list of DimArgBuilder, for each dimension variable appearing in `args_avals`
    in the sorted order of dimension variable name.
  """
  def get_dim_arg(array_arg_idx: int, dim_idx: int,
                  array_args: Sequence[mlir.ir.Value],
                  dim_arg_type: mlir.ir.Type) -> mlir.ir.Value:
    dim_arg = hlo.GetDimensionSizeOp(array_args[array_arg_idx], dim_idx)
    if dim_arg.result.type != dim_arg_type:
      return hlo.ConvertOp(dim_arg_type, dim_arg).result
    else:
      return dim_arg.result

  dim_args_builder_dict: Dict[str, DimArgBuilder] = {}  # a builder for each dim var by name
  all_dim_vars: Set[str] = set()
  for arg_idx, aval in enumerate(args_avals):
    for axis_idx, d in enumerate(aval.shape):
      if not core.is_constant_dim(d):
        all_dim_vars = all_dim_vars.union(d.get_vars())
        d_var = d.to_var()
        # TODO(necula): compute dim vars from non-trivial expressions also
        if d_var is None: continue
        if not d_var in dim_args_builder_dict:
          dim_args_builder_dict[d_var] = partial(get_dim_arg, arg_idx, axis_idx)

  if all_dim_vars:
    dim_vars_with_builders_set = set(dim_args_builder_dict.keys())
    if dim_vars_with_builders_set != all_dim_vars:
      missing = all_dim_vars.difference(dim_vars_with_builders_set)
      args_list = [f"  Arg[{arg_idx}]: {aval}"
                   for arg_idx, aval in enumerate(args_avals)]
      raise ValueError(
          "The following dimension variables cannot be computed from the static "
          f"shapes of the array arguments: {missing}. The argument shapes are:\n" +
          "\n".join(args_list) +
          "\n"
          "Please see https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#dimension-variables-must-be-solvable-from-the-input-shapes for more details.")

    # In sorted order by name
    builders = [dim_args_builder_dict[d_var] for d_var in sorted(dim_args_builder_dict.keys())]
  else:
    builders = []
  return builders


def check_module(mod: mlir.ir.Module, *,
                 allow_non_replicated_sharding: bool,
                 allow_all_custom_calls: bool):
  """Run a number of checks on the module.

  Args:
    allow_non_replicated_sharding: whether the module is allowed to contain
      non_replicated sharding annotations.
    allow_all_custom_calls: whether we should allow all custom calls, or
      only those who we have explicitly marked as stable.
  """
  sharding_attr = mlir.ir.StringAttr.get("Sharding", mod.context)
  allowed_custom_call_targets_attrs = [
      mlir.ir.StringAttr.get(target, mod.context)
      for target in _CUSTOM_CALL_TARGETS_GUARANTEED_STABLE]
  disallowed_custom_call_ops: List[str] = []
  def check_sharding(op_str: str, loc: mlir.ir.Location):
    # Check the shardings in an operation or attribute (`op_str`)
    if not allow_non_replicated_sharding:
      m = re.search(r'mhlo.sharding\s*=\s*"([^"]+)"', op_str)
      if m and m.group(1) not in ["{replicated}", ""]:
        raise ValueError(
            "Lowered function does not have a top-level pjit but it has "
            f"non-replicated sharding annotations, e.g., {op_str} at {loc}.\n"
            "See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#support-for-partitioning for a discussion.")

  def check_op(op: mlir.ir.Operation):
    op_name = op.operation.name
    if op_name == "func.func":
      for a in op.operation.attributes:
        # TODO: figure out how to parse the attributes properly
        check_sharding(str(a), op.location)

    elif op_name == "stablehlo.custom_call":
      call_target_name_attr = op.operation.attributes["call_target_name"]
      if (not allow_all_custom_calls and
          call_target_name_attr not in allowed_custom_call_targets_attrs):
        disallowed_custom_call_ops.append(str(op))
      if call_target_name_attr == sharding_attr:
        check_sharding(str(op), op.location)

  def walk_operations(op):
    check_op(op)
    for region in op.operation.regions:
      for block in region:
        for op in block:
          walk_operations(op)

  walk_operations(mod)
  if disallowed_custom_call_ops:
    disallowed_custom_call_ops_str = "\n".join(disallowed_custom_call_ops)
    msg = ("Cannot serialize code with custom calls whose targets have no "
           "compatibility guarantees. Examples are:\n"
           f"{disallowed_custom_call_ops_str}.\n"
           "See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#native-lowering-supports-only-select-custom-calls")
    raise ValueError(msg)
