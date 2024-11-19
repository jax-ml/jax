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
  axis_frame as axis_frame,
  AxisSize as AxisSize,
  AxisName as AxisName,
  CallPrimitive as CallPrimitive,
  ClosedJaxpr as ClosedJaxpr,
  ConcretizationTypeError as ConcretizationTypeError,
  DShapedArray as DShapedArray,
  DropVar as DropVar,
  Effect as Effect,
  Effects as Effects,
  EvalTrace as EvalTrace,
  get_opaque_trace_state as get_opaque_trace_state,
  InDBIdx as InDBIdx,
  InconclusiveDimensionOperation as InconclusiveDimensionOperation,
  InputType as InputType,
  Jaxpr as Jaxpr,
  JaxprDebugInfo as JaxprDebugInfo,
  JaxprEqn as JaxprEqn,
  JaxprPpContext as JaxprPpContext,
  JaxprPpSettings as JaxprPpSettings,
  JaxprTypeError as JaxprTypeError,
  Literal as Literal,
  MapPrimitive as MapPrimitive,
  nonempty_axis_env as nonempty_axis_env_DO_NOT_USE,  # noqa: F401
  OpaqueTraceState as OpaqueTraceState,
  OutDBIdx as OutDBIdx,
  OutputType as OutputType,
  ParamDict as ParamDict,
  Primitive as Primitive,
  ShapedArray as ShapedArray,
  TRACER_LEAK_DEBUGGER_WARNING as TRACER_LEAK_DEBUGGER_WARNING,
  Token as Token,
  Trace as Trace,
  Tracer as Tracer,
  unsafe_am_i_under_a_jit as unsafe_am_i_under_a_jit_DO_NOT_USE,  # noqa: F401
  unsafe_am_i_under_a_vmap as unsafe_am_i_under_a_vmap_DO_NOT_USE,  # noqa: F401
  unsafe_get_axis_names as unsafe_get_axis_names_DO_NOT_USE,  # noqa: F401
  unsafe_get_current_trace as unsafe_get_current_trace_DO_NOT_USE,  # noqa: F401
  UnshapedArray as UnshapedArray,
  Value as Value,
  Var as Var,
  abstract_token as abstract_token,
  aval_mapping_handlers as aval_mapping_handlers,
  call as call,
  call_impl as call_impl,
  call_p as call_p,
  check_jaxpr as check_jaxpr,
  closed_call_p as closed_call_p,
  concrete_aval as concrete_aval,
  concrete_or_error as concrete_or_error,
  concretization_function_error as concretization_function_error,
  custom_typechecks as custom_typechecks,
  dedup_referents as dedup_referents,
  ensure_compile_time_eval as ensure_compile_time_eval,
  escaped_tracer_error as escaped_tracer_error,
  eval_context as eval_context,
  eval_jaxpr as eval_jaxpr,
  extend_axis_env_nd as extend_axis_env_nd,
  find_top_trace as find_top_trace,
  full_lower as full_lower,
  gensym as gensym,
  get_aval as get_aval,
  get_type as get_type,
  get_referent as get_referent,
  is_concrete as is_concrete,
  is_constant_dim as is_constant_dim,
  is_constant_shape as is_constant_shape,
  jaxpr_as_fun as jaxpr_as_fun,
  jaxprs_in_params as jaxprs_in_params,
  join_effects as join_effects,
  lattice_join as lattice_join,
  leaked_tracer_error as leaked_tracer_error,
  literalable_types as literalable_types,
  mapped_aval as mapped_aval,
  maybe_find_leaked_tracers as maybe_find_leaked_tracers,
  max_dim as max_dim,
  min_dim as min_dim,
  new_jaxpr_eqn as new_jaxpr_eqn,
  no_axis_name as no_axis_name,
  no_effects as no_effects,
  primal_dtype_to_tangent_dtype as primal_dtype_to_tangent_dtype,
  pytype_aval_mappings as pytype_aval_mappings,
  raise_to_shaped as raise_to_shaped,
  raise_to_shaped_mappings as raise_to_shaped_mappings,
  reset_trace_state as reset_trace_state,
  set_current_trace as set_current_trace,
  str_eqn_compact as str_eqn_compact,
  subjaxprs as subjaxprs,
  substitute_vars_in_output_ty as substitute_vars_in_output_ty,
  take_current_trace as take_current_trace,
  trace_ctx as trace_ctx,
  trace_state_clean as trace_state_clean,
  TraceTag as TraceTag,
  traverse_jaxpr_params as traverse_jaxpr_params,
  typecheck as typecheck,
  typecompat as typecompat,
  typematch as typematch,
  unmapped_aval as unmapped_aval,
  used_axis_names_jaxpr as used_axis_names_jaxpr,
  valid_jaxtype as valid_jaxtype,
)

_deprecations = {
    # Finalized 2024-09-25; remove after 2024-12-25
    "pp_aval": ("jax.core.pp_aval was removed in JAX v0.4.34.", None),
    "pp_eqn": ("jax.core.pp_eqn was removed in JAX v0.4.34.", None),
    "pp_eqn_rules": ("jax.core.pp_eqn_rules was removed in JAX v0.4.34.", None),
    "pp_eqns": ("jax.core.pp_eqns was removed in JAX v0.4.34.", None),
    "pp_jaxpr": ("jax.core.pp_jaxpr was removed in JAX v0.4.34.", None),
    "pp_jaxpr_eqn_range": ("jax.core.pp_jaxpr_eqn_range was removed in JAX v0.4.34.", None),
    "pp_jaxpr_skeleton": ("jax.core.pp_jaxpr_skeleton was removed in JAX v0.4.34.", None),
    "pp_jaxprs": ("jax.core.pp_jaxprs was removed in JAX v0.4.34.", None),
    "pp_kv_pair": ("jax.core.pp_kv_pair was removed in JAX v0.4.34.", None),
    "pp_kv_pairs": ("jax.core.pp_kv_pairs was removed in JAX v0.4.34.", None),
    "pp_var": ("jax.core.pp_var was removed in JAX v0.4.34.", None),
    "pp_vars": ("jax.core.pp_vars was removed in JAX v0.4.34.", None),
}

import typing
if not typing.TYPE_CHECKING:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
