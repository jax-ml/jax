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
import re
from typing import  Callable, Dict, List, Optional, Sequence, Set, Union

from absl import logging

import jax
from jax import config
from jax import sharding

from jax._src import core
from jax._src import util
from jax._src import xla_bridge as xb
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.lib import xla_client
from jax._src.lib.mlir.dialects import stablehlo


map = util.safe_map
zip = util.safe_zip

# These are the JAX custom call target names that are guaranteed to be stable.
# They are tested by back_compat_test.py.
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
    "tf_embedded_graph",
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
                                      # lowering. As long as `out_avals`.
  dim_args_spec: Sequence[str]


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

  if xla_client.mlir_api_version >= 46:
    xla_call_module_version = 4
    mlir_str = mlir.module_to_bytecode(mlir_module)
    target_version = stablehlo.get_earliest_forward_compatible_version()
    mlir_module_serialized = xla_client._xla.mlir.serialize_portable_artifact(
        mlir_str, target_version)
  else:
    xla_call_module_version = 3
    mlir_module_serialized = mlir.module_to_bytecode(mlir_module)

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

  if "kept_var_idx" in lowered.compile_args:
    module_kept_var_idx = tuple(sorted(lowered.compile_args["kept_var_idx"]))
  else:
    # For pmap
    module_kept_var_idx = tuple(range(len(args_avals)))

  # We must compute the dim_args_spec: for each dimension variable, encode how
  # to compute its value from the shape of the explicit arguments. E.g., "2.1"
  # denotes args_tf[2].shape[1]. The order of the dimension variables must match
  # the order of the first N arguments of the lowered function.
  # If we use --jax_dynamic_shapes, the dimension variables are listed in the
  # order in which they are encountered by scanning the arguments and their
  # shapes in order. Otherwise, the dimension variables are passed in the
  # alphabetical order of their names.
  dim_args_spec_dict: Dict[str, str] = {}  # map dim var name to dim_args_spec
  dim_vars_order: List[str] = []
  all_dim_vars: Set[str] = set()
  current_kept_arg_idx = -1  # The index among the kept arguments
  for arg_idx, aval in enumerate(args_avals):
    is_kept = arg_idx in module_kept_var_idx
    if is_kept:
      current_kept_arg_idx += 1

    for axis_idx, d in enumerate(aval.shape):
      if not core.is_constant_dim(d):
        # We collect dimension variables even from dropped args
        all_dim_vars = all_dim_vars.union(d.get_vars())
        if not is_kept: continue
        d_var = d.to_var()
        # We can compute dim vars only from trivial polynomials
        if d_var is None: continue
        if d_var not in dim_args_spec_dict:
          dim_vars_order.append(d_var)
          dim_args_spec_dict[d_var] = f"{current_kept_arg_idx}.{axis_idx}"

  if all_dim_vars:
    dim_args_spec_set = set(dim_vars_order)
    if dim_args_spec_set != all_dim_vars:
      missing = all_dim_vars.difference(dim_args_spec_set)
      args_list = [f"  Arg[{arg_idx}] - {'KEPT   ' if arg_idx in module_kept_var_idx else 'DROPPED'}: {aval}"
                   for arg_idx, aval in enumerate(args_avals)]
      raise ValueError(
          "The following dimension variables cannot be computed from the static "
          f"shapes of the kept lowered arguments: {missing}. These are the "
          "argument shapes:\n" +
          "\n".join(args_list) +
          "\n"
          "Please see https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#dimension-variables-must-be-solvable-from-the-input-shapes for more details.")

    if config.jax_dynamic_shapes:
      # In the order we have seen them
      dim_args_spec = [dim_args_spec_dict[d_var] for d_var in dim_vars_order]
    else:
      # In sorted order by name
      dim_args_spec = [dim_args_spec_dict[d_var] for d_var in sorted(dim_vars_order)]
  else:
    dim_args_spec = []

  # Log and then check the module.
  if logging.vlog_is_on(3):
    mlir_module_text = mlir.module_to_string(mlir_module)
    logmsg = f"version={xla_call_module_version} lowering_platform={lowering_platform}, dim_args_spec=" + ", ".join(dim_args_spec)
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
      xla_call_module_version=xla_call_module_version,
      dim_args_spec=dim_args_spec
  )


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
