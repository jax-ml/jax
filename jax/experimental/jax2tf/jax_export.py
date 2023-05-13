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
"""JAX APIs for exporting JAX functions for interoperation.

This module is used with jax2tf, but has no TensorFlow dependencies.
"""
import dataclasses
import functools
import itertools
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from absl import logging

import jax
from jax import sharding

from jax._src import core
from jax._src import dispatch
from jax._src import pjit
from jax._src import sharding_impls
from jax._src import source_info_util
from jax._src.interpreters import mlir
from jax._src.interpreters import pxla
from jax._src.lib import xla_client
from jax._src.lib.mlir.dialects import stablehlo
from jax._src.lib.mlir import ir
from jax._src.lib.mlir.dialects import hlo
from jax._src.lib.mlir.dialects import func as func_dialect
from jax._src import tree_util
from jax._src import util
from jax._src import xla_bridge as xb

from jax.experimental.jax2tf import shape_poly

map = util.safe_map
zip = util.safe_zip

DType = Any

@dataclasses.dataclass(frozen=True)
class Exported:
  """A JAX function lowered to StableHLO.

  Attributes:
    fun_name: the name of the exported function, for error messages.
    in_tree: a PyTreeDef describing the tuple (args, kwargs) of the lowered JAX
        function. The actual lowering does not depend on the `in_tree`, but this
        can be used to invoke the exported function using the same argument
        structure.
    in_avals: the flat tuple of input abstract values. May contain dimension
        expressions in the shapes.
    out_tree: a PyTreeDef describing the result of the lowered JAX function.
    out_avals: the flat tuple of output abstract values. May contain dimension
        expressions in the shapes, with dimension variables among those in
        `in_avals.
    in_shardings: the flattened input shardings. Only for the inputs that are
        specified in `module_kept_var_idx`.
    out_shardings: the flattened output shardings, as long as `in_avals`.
    lowering_platform: one of 'tpu', 'cpu', 'cuda', 'rocm'

    mlir_module_serialized: the serialized lowered VHLO module.
    mlir_module_version: a version number for the serialized module.
        The following version numbers are valid:
           4 - mlir_module_serialized is a portable artifact.
    module_kept_var_idx: the sorted indices of the arguments among `in_avals` that
        must be passed to the module. The other arguments have been dropped
        because they are not used. Same length as `in_shardings`.
    strict_checks: whether the module was serialized with the following safety
        checking: (A) the lowered computation can only be executed on a platform
        for which it was lowered; (B) the serialized computation contains only
        custom calls with targets that are guaranteed to be stable, (more to come).
    _get_vjp: an optional function that takes the current exported function and
        returns the exported VJP function.
        The VJP function takes a flat list of arguments,
        starting with the primal arguments and followed by a cotangent argument
        for each primal output. It returns a tuple with the cotangents
        corresponding to the flattened primal inputs.
  """
  fun_name: str
  in_tree: tree_util.PyTreeDef
  in_avals: Tuple[core.AbstractValue, ...]
  out_tree: tree_util.PyTreeDef
  out_avals: Tuple[core.AbstractValue, ...]

  in_shardings: Tuple[Union[sharding.XLACompatibleSharding, pxla.UnspecifiedValue], ...]
  out_shardings: Tuple[Union[sharding.XLACompatibleSharding, pxla.UnspecifiedValue], ...]
  lowering_platform: str
  strict_checks: bool

  mlir_module_serialized: bytes
  xla_call_module_version: int
  module_kept_var_idx: Tuple[int, ...]

  _get_vjp: Optional[Callable[["Exported"], "Exported"]]

  @property
  def mlir_module(self) -> ir.Module:
    return xla_client._xla.mlir.deserialize_portable_artifact(self.mlir_module_serialized)

  def __str__(self):
    # This is called to make a MLIR source location when we call an Exported, and we
    # do not want the entire serialized module to end up in locations.
    return f"Exported(fun_name={self.fun_name}, ...)"

  def vjp(self) -> "Exported":
    """Gets the exported VJP.

    Returns None if not available, which can happen if the Exported has been
    loaded from an external format, without a VJP."""
    if self._get_vjp is None:
      raise ValueError("No VJP is available")
    return self._get_vjp(self)


def default_lowering_platform() -> str:
  # Canonicalize to turn 'gpu' into 'cuda' or 'rocm'
  return xb.canonicalize_platform(jax.default_backend())

def poly_spec(
    arg_shape: Sequence[Optional[int]],
    arg_dtype: DType,
    polymorphic_shape: Optional[str]) -> jax.ShapeDtypeStruct:
  """Constructs a jax.ShapeDtypeStruct with polymorphic shapes.

  Args:
    arg_shape: the shape, with possibly some unspecified dimensions.
    arg_dtype: the jax dtype.
    polymorphic_shape: a string specifying the polymorphic shape.

      .. warning:: The shape-polymorphic lowering is an experimental feature.
        It is meant to be sound, but it is known to reject some JAX programs
        that are shape polymorphic. The details of this feature can change.

      It should be either `None` (all dimensions are constant), or a string of
      specification for one axis, and can be either a constant, `_` denoting
      a constant dimension given by the `arg_shape`, or the name of a
      dimension variable assumed to range over dimension greater than 0. For
      convenience, zero or more trailing `_` can be abbreviated with `...`, and
      the surrounding parentheses may be missing.

      See [the README](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#shape-polymorphic-conversion)
      for more details.

  Returns: a jax.ShapeDTypeStruct with shapes that may contain symbolic
      expressions involving dimension variables.
  """
  aval_shape = shape_poly._parse_spec(polymorphic_shape, arg_shape)
  return jax.ShapeDtypeStruct(aval_shape, arg_dtype)

def shape_and_dtype_jax_array(a) -> Tuple[Sequence[Optional[int]], DType]:
  """Returns the shape and dtype of a jax.Array."""
  aval = core.raise_to_shaped(core.get_aval(a))
  return aval.shape, aval.dtype

def poly_specs(
    args,  # pytree of arguments
    polymorphic_shapes,  # prefix pytree of strings
    get_shape_and_dtype=shape_and_dtype_jax_array,
):
  """Constructs a pytree of jax.ShapeDtypeSpec.

  Args:
    args: a pytree of arguments
    polymorphic_shapes: should be `None` (all arguments are monomorphic),
      a single string (applies to all arguments), or a pytree matching a prefix
      of the `args`.
      See [how optional parameters are matched to
      arguments](https://jax.readthedocs.io/en/latest/pytrees.html#applying-optional-parameters-to-pytrees).

      See docstring of `poly_spec` and
      [the README](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#shape-polymorphic-conversion)
      for more details.

  Returns: a pytree of jax.ShapeDTypeStruct mathcing `args`.
  """
  args_flat, args_tree = tree_util.tree_flatten(args)

  shapes_and_dtypes = tuple(map(get_shape_and_dtype, args_flat))
  shapes, dtypes = util.unzip2(shapes_and_dtypes)

  if isinstance(args, tuple) and isinstance(polymorphic_shapes, list):
    # TODO: Remove backward-compatibility workaround
    polymorphic_shapes_ = tuple(polymorphic_shapes)
  else:
    polymorphic_shapes_ = polymorphic_shapes

  try:
    polymorphic_shapes_flat = tree_util.broadcast_prefix(
        polymorphic_shapes_, args,
        is_leaf=lambda x: x is None)
  except ValueError:
    e, *_ = tree_util.prefix_errors(
        polymorphic_shapes_, args,
        is_leaf=lambda x: x is None)
    raise e("jax_export polymorphic_shapes") from None

  # Now add in the polymorphic shapes
  args_specs_flat = tuple(
      map(poly_spec, shapes, dtypes, polymorphic_shapes_flat))

  return args_tree.unflatten(args_specs_flat)


def export(fun_jax: Callable,
           *,
           lowering_platform: Optional[str] = None,
           strict_checks: bool = True) -> Callable[..., Exported]:
  """Exports native serialization for a JAX function.

  Args:
    fun_jax: the function to lower and serialize.
    lowering_platform: one of 'tpu', 'cpu', 'cuda', 'rocm'. If None, then use
        the default JAX backend.
    strict_checks: whether to do strict safety checks. See Exported.strict_checks
        for more details.

  Returns: a function that takes args and kwargs pytrees of jax.ShapeDtypeStruct,
      or values with `.shape` and `.dtype` attributes, and returns an
      `Exported`.

  Usage:

      def f_jax(*args, **kwargs): ...
      exported = jax_export.export(f_jax)(*args, **kwargs)
  """
  fun_name = getattr(fun_jax, "__name__", "unknown")
  def do_export(*args_specs, **kwargs_specs) -> Exported:
    if not hasattr(fun_jax, "lower"):
      # We support convert(pjit(f_jax)) and convert(jit(f_jax)) but also
      # convert(f_jax), in which case a "jit" is implied. In that case we raise
      # an error if the lowered function contains non-replicated sharding annotations.
      wrapped_fun_jax = jax.jit(fun_jax)
      allow_non_replicated_sharding = False
    else:
      # If we have a pjit or pmap already we do not wrap with another, and we
      # allow shardings.
      wrapped_fun_jax = fun_jax  # type: ignore
      allow_non_replicated_sharding = True

    lowering_platform_str = lowering_platform or default_lowering_platform()
    lowered = wrapped_fun_jax.lower(
        *args_specs, **kwargs_specs,
        _experimental_lowering_platform=lowering_platform_str)
    lowering = lowered._lowering  # type: ignore

    _check_lowering(lowering)

    mlir_module = lowering.stablehlo()

    args_avals_flat, _ = tree_util.tree_flatten(lowered.in_avals)
    if "kept_var_idx" in lowering.compile_args:
      module_kept_var_idx = tuple(sorted(lowering.compile_args["kept_var_idx"]))
    else:
      # For pmap
      module_kept_var_idx = tuple(range(len(args_avals_flat)))

    if not all(
        core.is_constant_shape(a.shape) for a in args_avals_flat
    ) or lowering.compile_args.get("ordered_effects", []):
      # All arguments are kept if we have dimension variables.
      assert len(module_kept_var_idx) == len(args_avals_flat)
      mlir_module = _wrap_main_func(
          mlir_module, args_avals_flat, args_kwargs_tree=lowered.in_tree
      )

    xla_call_module_version = 5
    mlir_str = mlir.module_to_bytecode(mlir_module)
    if stablehlo.get_api_version() < 4:
      target_version = stablehlo.get_earliest_forward_compatible_version()
    else:
      # `target_version` is used to manage situations when a StableHLO producer
      # (in this case, jax2tf) and a StableHLO consumer were built using
      # different versions of StableHLO.
      #
      # Each StableHLO version `producer_version` has a compatibility window,
      # i.e. range of versions [`consumer_version_min`, `consumer_version_max`],
      # where StableHLO portable artifacts serialized by `producer_version`
      # can be deserialized by `consumer_version` within the window.
      # See https://github.com/openxla/stablehlo/blob/main/docs/compatibility.md
      # for the exact extent of these compatibility guarantees.
      #
      # `stablehlo.get_minimum_version()` returns `consumer_version_min`
      # for the current version of StableHLO. We are using it here to maximize
      # forward compatibility, i.e. to maximize how far into the past we can go
      # and still have the payloads produced by `serialize_portable_artifact`
      # compatible with potential consumers from the past.
      target_version = stablehlo.get_minimum_version()
    mlir_module_serialized = xla_client._xla.mlir.serialize_portable_artifact(
        mlir_str, target_version)

    # Figure out the result types and shapes
    if "global_out_avals" in lowering.compile_args:
      # This is currently the case for pjit
      out_avals_flat = lowering.compile_args["global_out_avals"]
    elif "shards" in lowering.compile_args:  # for PmapComputation
      out_avals_flat = lowering.compile_args["shards"].out_sharded_avals
    else:
      out_avals_flat = lowered.compile_args["out_avals"]

    # Log and then check the module.
    if logging.vlog_is_on(3):
      mlir_module_text = mlir.module_to_string(mlir_module)
      logmsg = (f"version={xla_call_module_version} "
                f"lowering_platform={lowering_platform_str} "
                f"strict_checks={strict_checks}")
      logging.info("Lowered JAX module: %s\n", logmsg)
      for l in mlir_module_text.splitlines():
        logging.info(l)

    _check_module(mlir_module,
                  allow_non_replicated_sharding=allow_non_replicated_sharding,
                  allow_all_custom_calls=not strict_checks)

    return Exported(
        fun_name=fun_name,
        in_tree=lowered.in_tree,
        out_tree=lowered.out_tree,
        in_avals=tuple(args_avals_flat),
        out_avals=tuple(out_avals_flat),
        in_shardings=lowering.compile_args["in_shardings"],
        out_shardings=lowering.compile_args["out_shardings"],
        lowering_platform=lowering_platform_str,
        strict_checks=strict_checks,
        mlir_module_serialized=mlir_module_serialized,
        module_kept_var_idx=module_kept_var_idx,
        xla_call_module_version=xla_call_module_version,
        _get_vjp=lambda exported: _export_native_vjp(fun_jax, exported))

  return do_export


def _wrap_main_func(
    module: ir.Module,
    args_avals_flat: Sequence[core.ShapedArray],
    *,
    args_kwargs_tree: tree_util.PyTreeDef,
) -> ir.Module:
  """Wraps the lowered module with a new "main".

  JAX lowering in presence of shape polymorphism produces a `module` that
  takes one or more dimension arguments, specified using 0-dimensional tensors
  of type i32 or i64, followed by the regular array arguments.
  The dimension arguments correspond to the dimension variables appearing in
  the `args_avals`, in sorted order.

  Consider the lowering of a function with one array argument of type "f32[w,
  h]",
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

  In addition, this function also removes token arguments/results from the main
  function by providing dummy values. This ensures that the main function's
  calling convention is as expected.

  Args:
    module: the HLO module as obtained from lowering. May have a number of
      dimension arguments, followed by the kept array arguments.
    args_avals_flat: the avals for all the arguments of the lowered function,
      which correspond to the array arguments of the `module`.
    args_kwargs_tree: the PyTreeDef corresponding to `(args, kwargs)`, for error
      messages.

  Returns the wrapped module.
  """
  dim_vars = shape_poly.all_dim_vars(args_avals_flat)

  # Make a new module, do not mutate the "module" because it may be cached
  context = mlir.make_ir_context()
  with context, ir.Location.unknown(context):
    new_module = ir.Module.parse(mlir.module_to_bytecode(module))
    symbol_table = ir.SymbolTable(new_module.operation)
    orig_main = symbol_table["main"]
    orig_main.attributes["sym_visibility"] = ir.StringAttr.get("private")
    symbol_table.set_symbol_name(orig_main, "_wrapped_jax_export_main")
    orig_main_name = ir.StringAttr(symbol_table.insert(orig_main)).value

    def is_token(attrs):
      try:
        return ir.BoolAttr(ir.DictAttr(attrs)["jax.token"]).value
      except KeyError:
        return False

    orig_input_types = orig_main.type.inputs
    arg_attrs = list(ir.ArrayAttr(orig_main.arg_attrs))
    nr_token_args = sum(1 for attrs in arg_attrs if is_token(attrs))
    nr_array_args = len(orig_input_types) - len(dim_vars) - nr_token_args
    assert nr_array_args >= 0
    assert not any(is_token(attrs) for attrs in arg_attrs[-nr_array_args:])
    new_main_input_types = orig_input_types[-nr_array_args:]

    orig_output_types = orig_main.type.results
    result_attrs = list(ir.ArrayAttr(orig_main.result_attrs))
    nr_token_results = sum(1 for attrs in result_attrs if is_token(attrs))
    nr_array_results = len(orig_output_types) - nr_token_results
    assert nr_array_results >= 0
    assert not any(
        is_token(attrs) for attrs in result_attrs[-nr_array_results:]
    )
    new_main_output_types = orig_output_types[-nr_array_results:]

    ftype = ir.FunctionType.get(new_main_input_types, new_main_output_types)
    new_main_op = func_dialect.FuncOp(
        "main", ftype, ip=ir.InsertionPoint.at_block_begin(new_module.body))
    new_main_op.attributes["sym_visibility"] = ir.StringAttr.get("public")
    try:
      new_main_op.arg_attrs = ir.ArrayAttr.get(arg_attrs[-nr_array_args:])
    except KeyError:
      pass  # TODO: better detection if orig_main.arg_attrs does not exist
    try:
      new_main_op.result_attrs = ir.ArrayAttr.get(
          result_attrs[-nr_array_results:]
      )
    except KeyError:
      pass
    symbol_table.insert(new_main_op)
    entry_block = new_main_op.add_entry_block()
    with ir.InsertionPoint(entry_block):
      orig_main_args: List[ir.Value] = []
      module_context = mlir.ModuleContext(
          "cpu", "cpu", sharding_impls.ShardingContext([]),
          source_info_util.new_name_stack(),
          [], itertools.count(1), [], module=new_module, context=context)
      ctx = mlir.LoweringRuleContext(module_context=module_context,
                                     primitive=None, avals_in=args_avals_flat, avals_out=None,
                                     tokens_in=mlir.TokenSet(), tokens_out=None)
      dim_args = _compute_dim_args(ctx, args_avals_flat, tuple(new_main_op.arguments),
                                   orig_input_types[:len(dim_vars)],
                                   args_kwargs_tree=args_kwargs_tree)
      # The first arguments are the dimension variable
      orig_main_args.extend(dim_args)
      # Then the token arguments
      orig_main_args.extend(list(mlir.dummy_token()) * nr_token_args)
      # Then the array arguments
      orig_main_args.extend(new_main_op.arguments)
      call = func_dialect.CallOp(orig_output_types,
                                 ir.FlatSymbolRefAttr.get(orig_main_name),
                                 orig_main_args)
      func_dialect.ReturnOp(call.results[-nr_array_results:])
    symbol_table.set_symbol_name(new_main_op, "main")
    return new_module


def _compute_dim_args(
    ctx: mlir.LoweringRuleContext,
    args_avals_flat: Sequence[core.ShapedArray],
    array_args: Sequence[ir.Value],
    dim_arg_types: Sequence[ir.Type], *,
    args_kwargs_tree: tree_util.PyTreeDef) -> Sequence[ir.Value]:
  """Compute the values of the dimension arguments.

  Args:
    args_avals_flat: the abstract values of the array arguments.
    array_args: the values of the array arguments.
    dim_arg_types: the desired types for the dimension arguments.
    args_kwargs_tree: the PyTreeDef corresponding to `(args, kwargs)`, for
      error messages.

  Returns:
    the values of the dimension variables, in the sorted order of the
    dimension variables.
  """
  dim_values = mlir.lower_fun(
      functools.partial(shape_poly.compute_dim_vars_from_arg_shapes,
                        args_avals_flat, args_kwargs_tree=args_kwargs_tree),
      multiple_results=True)(ctx, *array_args)

  res = []
  for dim_arg, dim_arg_type in zip(util.flatten(dim_values), dim_arg_types):
    if dim_arg.type != dim_arg_type:
      res.append(hlo.ConvertOp(dim_arg_type, dim_arg).result)
    else:
      res.append(dim_arg)
  return tuple(res)


def _check_lowering(lowering) -> None:
  if not isinstance(lowering, pxla.MeshComputation):
    raise NotImplementedError(f"serialization is supported only for pjit. {lowering}")

  if lowering.compile_args["host_callbacks"] or lowering.compile_args["keepalive"]:
    raise NotImplementedError("serialization of host_callbacks is not yet implemented")
  # Check that we do not see new compile_args. When we add a compile_args it is
  # safe to add it to the allowed_compile_args if it does not change the semantics
  # or the calling convention of the lowered module.
  allowed_compile_args = [
      "backend", "mesh", "global_in_avals",
      "global_out_avals", "in_shardings", "out_shardings", "kept_var_idx",
      "spmd_lowering", "auto_spmd_lowering",
      "tuple_args", "ordered_effects", "unordered_effects",
      "keepalive", "host_callbacks", "pmap_nreps", "committed",
      "device_assignment", "jaxpr_debug_info"]
  for compile_arg in lowering.compile_args.keys():
    if compile_arg not in allowed_compile_args:
      raise NotImplementedError(f"Unrecognized lowered.compile_args[{compile_arg}]")

  # We have not implemented support for some of the compile_args. Check here that
  # the compile_args have the values that have been implemented.
  not_implemented_msgs = []
  for compile_arg, check_value, err_msg in (
      ("spmd_lowering", lambda v: v, "True"),
      ("auto_spmd_lowering", lambda v: not v, "False"),
      # tuple_args is a compilation flag, does not affect lowering.
      ("tuple_args", lambda v: True, "N/A"),
      # unordered_effects do not change the calling convention. Those from
      # jax.debug will also result in keepalive being non-empty and unsupported
      # custom calls. The CallTfEffect is an exception, but we want to allow
      # that one.
      ("unordered_effects", lambda v: True, "N/A"),
      # ordered_effects are allowed and we ensure that the calling convention is
      # unmodified by passing dummy tokens in the main function wrapper.
      ("ordered_effects", lambda v: True, "N/A"),
      # used for TPU jax.debug, send/recv. Not supported yet.
      ("host_callbacks", lambda v: not v, "empty"),
      # used on all platforms for callbacks. Not supported yet.
      ("keepalive", lambda v: not v, "empty"),
      ("pmap_nreps", lambda v: v == 1, "1"),
  ):
    if compile_arg in lowering.compile_args:
      if not check_value(lowering.compile_args[compile_arg]):
        not_implemented_msgs.append(
            f"{compile_arg} must be {err_msg} and it is {lowering.compile_args[compile_arg]}")
  if not_implemented_msgs:
    raise NotImplementedError(
        "serialization error, unimplemented lowered.compile_args:\n" +
        "\n".join(not_implemented_msgs))

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
    # ApproxTopK on TPU
    "ApproxTopK",
]

def _check_module(mod: ir.Module, *,
                  allow_non_replicated_sharding: bool,
                  allow_all_custom_calls: bool):
  """Run a number of checks on the module.

  Args:
    allow_non_replicated_sharding: whether the module is allowed to contain
      non_replicated sharding annotations.
    allow_all_custom_calls: whether we should allow all custom calls, or
      only those who we have explicitly marked as stable.
  """
  sharding_attr = ir.StringAttr.get("Sharding", mod.context)
  allowed_custom_call_targets_attrs = [
      ir.StringAttr.get(target, mod.context)
      for target in _CUSTOM_CALL_TARGETS_GUARANTEED_STABLE]
  disallowed_custom_call_ops: List[str] = []
  def check_sharding(op: ir.Operation, loc: ir.Location):
    if not allow_non_replicated_sharding:
      try:
        sharding = op.attributes["mhlo.sharding"]
      except KeyError:
        pass
      else:
        if ir.StringAttr(sharding).value not in ["{replicated}", ""]:
          raise ValueError(
              "Lowered function does not have a top-level pjit but it has"
              f" non-replicated sharding annotations, e.g., {op} at {loc}.\nSee"
              " https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#support-for-partitioning"
              " for a discussion."
          )

  def check_op(op: ir.Operation):
    op_name = op.operation.name
    if op_name == "func.func":
      check_sharding(op.operation, op.location)

    elif op_name == "stablehlo.custom_call":
      call_target_name_attr = op.operation.attributes["call_target_name"]
      if (not allow_all_custom_calls and
          call_target_name_attr not in allowed_custom_call_targets_attrs):
        disallowed_custom_call_ops.append(str(op))
      if call_target_name_attr == sharding_attr:
        check_sharding(op, op.location)

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


def _export_native_vjp(primal_fun_jax, primal: Exported) -> Exported:
  # Export the VJP of `primal_fun_jax`. See documentation for Exported.vjp

  # Since jax.vjp does not handle kwargs, it is easier to do all the work
  # here with flattened functions.
  def fun_vjp_jax(*args_and_out_cts_flat_jax):
    # Takes a flat list of primals and output cotangents
    def flattened_primal_fun_jax(*args_flat):
      args, kwargs = primal.in_tree.unflatten(args_flat)
      res = primal_fun_jax(*args, **kwargs)
      res_flat, res_tree = tree_util.tree_flatten(res)
      assert res_tree == primal.out_tree
      return res_flat

    args_flat_jax, out_cts_flat_jax = util.split_list(args_and_out_cts_flat_jax,
                                                      [len(primal.in_avals)])
    _, pullback_jax = jax.vjp(flattened_primal_fun_jax, *args_flat_jax)
    return pullback_jax(out_cts_flat_jax)

  vjp_in_avals = list(
      itertools.chain(primal.in_avals,
                      map(lambda a: a.at_least_vspace(), primal.out_avals)))

  # Expand in_shardings to all in_avals even not kept ones.
  all_in_shardings = [sharding_impls.UNSPECIFIED] * len(primal.in_avals)
  for idx, in_s in zip(sorted(primal.module_kept_var_idx),
                       primal.in_shardings):
    all_in_shardings[idx] = in_s  # type: ignore
  all_shardings = all_in_shardings + list(primal.out_shardings)
  # Cannot mix unspecified and specified shardings. Make the unspecified
  # ones replicated.
  specified_shardings = [
      s for s in all_shardings if not sharding_impls.is_unspecified(s)]

  vjp_in_shardings: Any  # The primal inputs followed by output cotangents
  vjp_out_shardings: Any  # The primal output cotangents
  if 0 == len(specified_shardings):
    vjp_in_shardings = sharding_impls.UNSPECIFIED
    vjp_out_shardings = sharding_impls.UNSPECIFIED
  else:
    if len(specified_shardings) < len(all_shardings):
      # There are some specified, but not all; pjit front-end does not liwk
      in_s = specified_shardings[0]  # pjit will enforce that all have same devices
      assert isinstance(in_s, sharding.XLACompatibleSharding)
      replicated_s = sharding.GSPMDSharding.get_replicated(in_s._device_assignment)
      all_shardings = [
          s if not sharding_impls.is_unspecified(s) else replicated_s
          for s in all_shardings]

    vjp_in_shardings = tuple(all_shardings)
    vjp_out_shardings = tuple(all_shardings[:len(primal.in_avals)])
    if all(sharding_impls.is_unspecified(s) for s in vjp_out_shardings):
      vjp_out_shardings = sharding_impls.UNSPECIFIED

  fun_vjp_jax = pjit.pjit(fun_vjp_jax,
                          in_shardings=vjp_in_shardings,
                          out_shardings=vjp_out_shardings)

  return export(fun_vjp_jax,
                lowering_platform=primal.lowering_platform,
                strict_checks=primal.strict_checks)(*vjp_in_avals)

### Importing

def call_exported(exported: Exported) -> Callable[..., jax.Array]:
  @jax.custom_vjp
  def f_flat(*args_flat):
    return call_exported_p.bind(*args_flat, exported=exported)

  def f_flat_vjp_fwd(*args_flat):
    # Return the primal arguments as the residual
    # TODO: keep as residuals only the arguments that are needed
    return f_flat(*args_flat), args_flat

  def f_flat_vjp_bwd(residual, ct_res_flat):
    args_flat = residual  # residual is the primal argument flat tuple
    exp_vjp = exported.vjp()
    in_ct_flat = call_exported(exp_vjp)(*args_flat, *ct_res_flat)
    return in_ct_flat

  f_flat.defvjp(f_flat_vjp_fwd, f_flat_vjp_bwd)

  def f_imported(*args, **kwargs):
    # since custom_vjp does not support kwargs, flatten the function first.
    args_flat, in_tree = tree_util.tree_flatten((args, kwargs))
    if in_tree != exported.in_tree:
      # Give errors with the precise tree difference; use fake leaves so we can
      # use tree_util.equality_errors.
      in_args = in_tree.unflatten([0] * in_tree.num_leaves)
      exp_in_args = exported.in_tree.unflatten([0] * exported.in_tree.num_leaves)

      msg = (
          "The invocation args and kwargs must have the same pytree structure "
          f"as when the function '{exported.fun_name}' was exported, but they "
          "have the following structural differences:\n" +
          ("\n".join(
             f"   - {shape_poly.args_kwargs_path_to_str(path)} is a {thing1} in the invocation and a "
             f"{thing2} when exported, so {explanation}.\n"
             for path, thing1, thing2, explanation
             in tree_util.equality_errors(in_args, exp_in_args))))
      raise ValueError(msg)

    res_flat = f_flat(*args_flat)
    return exported.out_tree.unflatten(res_flat)
  return f_imported


# A JAX primitive for invoking a serialized JAX function.
call_exported_p = core.Primitive("call_exported")
call_exported_p.multiple_results = True

@util.cache()
def _call_exported_abstract_eval(*in_avals: core.AbstractValue,
                                 exported: Exported) -> Tuple[core.AbstractValue, ...]:
  exported_dim_vars = shape_poly.all_dim_vars(exported.in_avals)
  assert len(in_avals) == len(exported.in_avals)  # since the pytrees have the same structure
  # Check that the expected shapes match the actual ones
  for arg_idx, (exp_aval, actual_aval) in enumerate(zip(exported.in_avals, in_avals)):
    def pp_arg_dim(dim_idx: Optional[int]) -> str:
      return shape_poly.pretty_print_dimension_descriptor(exported.in_tree,
                                                          arg_idx, dim_idx)
    if len(exp_aval.shape) != len(actual_aval.shape):
      raise ValueError(
          f"Rank mismatch for {pp_arg_dim(None)}: expected {exp_aval.shape} "
          f"and called with {actual_aval.shape}")
    if exp_aval.dtype != actual_aval.dtype:
      raise ValueError(
          f"Dtype mismatch for {pp_arg_dim(None)}: expected {exp_aval.dtype} "
          f"and called with {actual_aval.dtype}")
    for dim_idx, aval_d in enumerate(exp_aval.shape):
      # If the exp_aval has a constant dimension then the actual argument must have
      # a matching constant dimension.
      if core.is_constant_dim(aval_d):
        if (not core.is_constant_dim(actual_aval.shape[dim_idx]) or
            aval_d != actual_aval.shape[dim_idx]):
          raise ValueError(
              f"Shape mismatch for {pp_arg_dim(dim_idx)} (expected constant): "
              f"expected {exp_aval.shape} and called with {actual_aval.shape}")

  # Must express the exported_dim_vars in terms of the shapes in in_avals.
  solution, shape_constraints, known_dim_vars = shape_poly.solve_dim_vars(
      exported.in_avals, args_kwargs_tree=exported.in_tree)
  known_env = {vname: in_avals[arg_idx].shape[dim_idx]
               for (vname, arg_idx, dim_idx) in known_dim_vars}
  shape_constraints.check(known_env)
  exported_dim_values = [solution[var].evaluate(known_env)
                         for var in exported_dim_vars]
  return tuple(
      core.ShapedArray(core.evaluate_shape(out_aval.shape, exported_dim_vars,
                                           *exported_dim_values),
                       dtype=out_aval.dtype, weak_type=out_aval.weak_type,
                       named_shape=out_aval.named_shape)
      for out_aval in exported.out_avals)


call_exported_p.def_abstract_eval(_call_exported_abstract_eval)

def _call_exported_impl(*args, exported: Exported):
  return dispatch.apply_primitive(call_exported_p, *args, exported=exported)

call_exported_p.def_impl(_call_exported_impl)

def _call_exported_lowering(ctx: mlir.LoweringRuleContext, *args,
                            platform: str,
                            exported: Exported):
  if platform != exported.lowering_platform:
    raise ValueError(
        f"The exported function '{exported.fun_name}' was lowered for "
        f"platform '{exported.lowering_platform}' but it is used "
        f"on '{platform}'.")
  submodule = ir.Module.parse(exported.mlir_module)
  symtab = ir.SymbolTable(submodule.operation)
  # The called function may have been exported with polymorphic shapes and called
  # now with more refined shapes. We insert hlo.ConvertOp to ensure the module
  # is valid.
  def convert_shape(x: ir.Value, x_aval: core.AbstractValue, new_aval: core.AbstractValue) -> ir.Value:
    new_ir_type = mlir.aval_to_ir_type(new_aval)
    if x.type != new_ir_type:
      return mlir.convert_hlo(ctx, x, x_aval, new_aval)
    else:
      return x

  callee_result_types = symtab["main"].type.results
  # TODO: maybe cache multiple calls
  fn = mlir.merge_mlir_modules(ctx.module_context.module,
                               f"call_exported_{exported.fun_name}",
                               submodule)
  kept_args = [
      convert_shape(a, a_aval, exported_in_aval)
      for i, (a, a_aval, exported_in_aval) in enumerate(zip(args, ctx.avals_in, exported.in_avals))
      if i in exported.module_kept_var_idx]
  call = func_dialect.CallOp(callee_result_types,
                             ir.FlatSymbolRefAttr.get(fn),
                             kept_args)
  # The ctx.avals_out already contain the abstract values refined by
  # _call_exported_abstract_eval.
  return tuple(convert_shape(out, out_aval, refined_out_aval)
               for out, out_aval, refined_out_aval in zip(call.results, exported.out_avals, ctx.avals_out))


for _p in ("cpu", "tpu", "cuda", "rocm"):
  mlir.register_lowering(call_exported_p,
                         functools.partial(_call_exported_lowering, platform=_p),
                         platform=_p)
