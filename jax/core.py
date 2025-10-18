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
