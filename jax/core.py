# Copyright 2022 The JAX Authors.
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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.core import (
  AbstractToken as AbstractToken,
  AbstractValue as AbstractValue,
  Atom as Atom,
  CallPrimitive as CallPrimitive,
  DebugInfo as DebugInfo,
  DShapedArray as DShapedArray,
  DropVar as DropVar,
  Effect as Effect,
  Effects as Effects,
  get_opaque_trace_state as get_opaque_trace_state,
  InconclusiveDimensionOperation as InconclusiveDimensionOperation,
  JaxprPpContext as JaxprPpContext,
  JaxprPpSettings as JaxprPpSettings,
  JaxprTypeError as JaxprTypeError,
  nonempty_axis_env as nonempty_axis_env_DO_NOT_USE,  # noqa: F401
  OutputType as OutputType,
  ParamDict as ParamDict,
  ShapedArray as ShapedArray,
  Trace as Trace,
  Tracer as Tracer,
  unsafe_am_i_under_a_jit as unsafe_am_i_under_a_jit_DO_NOT_USE,  # noqa: F401
  unsafe_am_i_under_a_vmap as unsafe_am_i_under_a_vmap_DO_NOT_USE,  # noqa: F401
  unsafe_get_axis_names as unsafe_get_axis_names_DO_NOT_USE,  # noqa: F401
  UnshapedArray as UnshapedArray,
  Value as Value,
  abstract_token as abstract_token,
  aval_mapping_handlers as aval_mapping_handlers,
  call as call,
  call_impl as call_impl,
  check_jaxpr as check_jaxpr,
  concrete_or_error as concrete_or_error,
  concretization_function_error as concretization_function_error,
  custom_typechecks as custom_typechecks,
  ensure_compile_time_eval as ensure_compile_time_eval,
  eval_context as eval_context,
  eval_jaxpr as eval_jaxpr,
  find_top_trace as find_top_trace,
  gensym as gensym,
  get_aval as get_aval,
  is_concrete as is_concrete,
  is_constant_dim as is_constant_dim,
  is_constant_shape as is_constant_shape,
  jaxprs_in_params as jaxprs_in_params,
  literalable_types as literalable_types,
  mapped_aval as mapped_aval,
  max_dim as max_dim,
  min_dim as min_dim,
  new_jaxpr_eqn as new_jaxpr_eqn,
  no_axis_name as no_axis_name,
  no_effects as no_effects,
  primal_dtype_to_tangent_dtype as primal_dtype_to_tangent_dtype,
  pytype_aval_mappings as pytype_aval_mappings,
  set_current_trace as set_current_trace,
  subjaxprs as subjaxprs,
  take_current_trace as take_current_trace,
  trace_ctx as trace_ctx,
  TraceTag as TraceTag,
  traverse_jaxpr_params as traverse_jaxpr_params,
  unmapped_aval as unmapped_aval,
  valid_jaxtype as valid_jaxtype,
)


from jax._src import core as _src_core
_deprecations = {
    # Added 2024-12-11
    "axis_frame": ("jax.core.axis_frame is deprecated.", _src_core.axis_frame),
    "AxisName": ("jax.core.AxisName is deprecated.", _src_core.AxisName),
    "ConcretizationTypeError": ("jax.core.ConcretizationTypeError is deprecated; "
                                "use jax.errors.ConcretizationTypeError.",
                                _src_core.ConcretizationTypeError),
    "call_p": ("jax.core.call_p is deprecated. Use jax.extend.core.primitives.call_p",
               _src_core.call_p),
    "closed_call_p": ("jax.core.closed_call_p is deprecated. Use jax.extend.core.primitives.closed_call_p",
                      _src_core.closed_call_p),
    "get_type": ("jax.core.get_type is deprecated.", _src_core.get_aval),
    "trace_state_clean": ("jax.core.trace_state_clean is deprecated.",
                          _src_core.trace_state_clean),
    "typecheck": ("jax.core.typecheck is deprecated.", _src_core.typecheck),
    "typematch": ("jax.core.typematch is deprecated.", _src_core.typematch),
    # Added 2024-12-10
    "full_lower": ("jax.core.full_lower is deprecated. It is a no-op as of JAX v0.4.36.", None),
    "jaxpr_as_fun": ("jax.core.jaxpr_as_fun was removed in JAX v0.6.0. Use jax.extend.core.jaxpr_as_fun instead, "
                     "and see https://docs.jax.dev/en/latest/jax.extend.html for details.",
                     None),
    "lattice_join": ("jax.core.lattice_join is deprecated. It is a no-op as of JAX v0.4.36.", None),
    # Finalized 2025-03-25 for JAX v0.6.0; remove after 2025-06-25
    "AxisSize": ("jax.core.AxisSize was removed in JAX v0.6.0.", None),
    "ClosedJaxpr": ("jax.core.ClosedJaxpr was removed in JAX v0.6.0. Use jax.extend.core.ClosedJaxpr instead, "
                    "and see https://docs.jax.dev/en/latest/jax.extend.html for details.", None),
    "EvalTrace": ("jax.core.EvalTrace was removed in JAX v0.6.0.", None),
    "InDBIdx": ("jax.core.InDBIdx was removed in JAX v0.6.0.", None),
    "InputType": ("jax.core.InputType was removed in JAX v0.6.0.", None),
    "Jaxpr": ("jax.core.Jaxpr was removed in JAX v0.6.0. Use jax.extend.core.Jaxpr instead, "
              "and see https://docs.jax.dev/en/latest/jax.extend.html for details.", None),
    "JaxprEqn": ("jax.core.JaxprEqn was removed in JAX v0.6.0. Use jax.extend.core.JaxprEqn instead, "
                 "and see https://docs.jax.dev/en/latest/jax.extend.html for details.", None),
    "Literal": ("jax.core.Literal was removed in JAX v0.6.0. Use jax.extend.core.Literal instead, "
                "and see https://docs.jax.dev/en/latest/jax.extend.html for details.", None),
    "MapPrimitive": ("jax.core.MapPrimitive was removed in JAX v0.6.0.", None),
    "OpaqueTraceState": ("jax.core.OpaqueTraceState was removed in JAX v0.6.0.", None),
    "OutDBIdx": ("jax.core.OutDBIdx was removed in JAX v0.6.0.", None),
    "Primitive": ("jax.core.Primitive was removed in JAX v0.6.0. Use jax.extend.core.Primitive instead, "
                  "and see https://docs.jax.dev/en/latest/jax.extend.html for details.", None),
    "Token": ("jax.core.Token was removed in JAX v0.6.0. Use jax.extend.core.Token instead, "
              "and see https://docs.jax.dev/en/latest/jax.extend.html for details.", None),
    "TRACER_LEAK_DEBUGGER_WARNING": ("jax.core.TRACER_LEAK_DEBUGGER_WARNING was removed in JAX v0.6.0.", None),
    "Var": ("jax.core.Var was removed in JAX v0.6.0. Use jax.extend.core.Var instead, "
            "and see https://docs.jax.dev/en/latest/jax.extend.html for details.", None),
    "concrete_aval": ("jax.core.concrete_aval was removed in JAX v0.6.0.", None),
    "dedup_referents": ("jax.core.dedup_referents was removed in JAX v0.6.0.", None),
    "escaped_tracer_error": ("jax.core.escaped_tracer_error was removed in JAX v0.6.0.", None),
    "extend_axis_env_nd": ("jax.core.extend_axis_env_nd was removed in JAX v0.6.0.", None),
    "get_referent": ("jax.core.get_referent was removed in JAX v0.6.0.", None),
    "join_effects": ("jax.core.join_effects was removed in JAX v0.6.0.", None),
    "leaked_tracer_error": ("jax.core.leaked_tracer_error was removed in JAX v0.6.0.", None),
    "maybe_find_leaked_tracers": ("jax.core.maybe_find_leaked_tracers was removed in JAX v0.6.0.", None),
    "raise_to_shaped": ("jax.core.raise_to_shaped was removed in JAX v0.6.0. It is a no-op as of JAX v0.4.36.", None),
    "raise_to_shaped_mappings": ("jax.core.raise_to_shaped_mappings was removed in JAX v0.6.0."
                                 " It is unused as of jax v0.4.36.", None),
    "reset_trace_state": ("jax.core.reset_trace_state was removed in JAX v0.6.0.", None),
    "str_eqn_compact": ("jax.core.str_eqn_compact was removed in JAX v0.6.0.", None),
    "substitute_vars_in_output_ty": ("jax.core.substitute_vars_in_output_ty was removed in JAX v0.6.0.", None),
    "typecompat": ("jax.core.typecompat was removed in JAX v0.6.0.", None),
    "used_axis_names_jaxpr": ("jax.core.used_axis_names_jaxpr was removed in JAX v0.6.0.", None),
}

import typing
if typing.TYPE_CHECKING:
  AxisName = _src_core.AxisName
  ConcretizationTypeError = _src_core.ConcretizationTypeError
  axis_frame = _src_core.axis_frame
  call_p = _src_core.call_p
  closed_call_p = _src_core.closed_call_p
  get_type = _src_core.get_aval
  trace_state_clean = _src_core.trace_state_clean
  typecheck = _src_core.typecheck
  typematch = _src_core.typematch
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
del _src_core
