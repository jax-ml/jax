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

import jax._src.core as _src_core

from jax._src.core import (
  AbstractValue as AbstractValue,
  Atom as Atom,
  ParamDict as ParamDict,
  ShapedArray as ShapedArray,
  Trace as Trace,
  Tracer as Tracer,
  Value as Value,
  ensure_compile_time_eval as ensure_compile_time_eval,
  eval_context as eval_context,
  eval_jaxpr as eval_jaxpr,
  max_dim as max_dim,
  min_dim as min_dim,
)

_deprecations = {
  # Deprecated in v0.8.2; finalized in v0.10.0.
  # TODO(jakevdp) remove entries in v0.11.0.
  "get_aval": (
    "jax.core.get_aval was deprecated in JAX v0.8.2 and removed in JAX v0.10.0;"
    " use jax.typeof instead.",
    None,
  ),
  "mapped_aval": (
    "jax.core.mapped_aval was deprecated in JAX v0.8.2 and removed in JAX"
    " v0.10.0. Use jax.extend.core.mapped_aval.",
    None,
  ),
  "unmapped_aval": (
    "jax.core.unmapped_aval was deprecated in JAX v0.8.2 and removed in JAX"
    " v0.10.0. Use jax.extend.core.unmapped_aval.",
    None,
  ),
  "set_current_trace": (
    "jax.core.set_current_trace was deprecated in JAX v0.8.2 and removed in"
    " JAX v0.10.0. Use jax.extend.core.set_current_trace.",
    None,
  ),
  "take_current_trace": (
    "jax.core.take_current_trace was deprecated in JAX v0.8.2 and removed in"
    " JAX v0.10.0. Use jax.extend.core.take_current_trace.",
    None,
  ),
  "traverse_jaxpr_params": (
    "jax.core.traverse_jaxpr_params was deprecated in JAX v0.8.2 and removed in"
    " JAX v0.10.0.",
    None,
  ),
  "TraceTag": (
    "jax.core.TraceTag was deprecated in JAX v0.8.2 and removed in JAX v0.10.0."
    " Use jax.extend.core.TraceTag.",
    None,
  ),
  "call_impl": (
    "jax.core.call_impl was deprecated in JAX v0.8.2 and removed in JAX"
    " v0.10.0. Use jax.extend.core.call_impl.",
    None,
  ),
  "subjaxprs": (
    "jax.core.subjaxprs was deprecated in JAX v0.8.2 and removed in JAX"
    " v0.10.0. Use jax.extend.core.subjaxprs.",
    None,
  ),
  "AbstractToken": (
    "jax.core.AbstractToken was deprecated in JAX v0.8.2 and removed in JAX"
    " v0.10.0. Use jax.extend.core.AbstractToken.",
    None,
  ),
  # Deprecated in JAX v0.10.0; TODO(jakevdp) finalize in v0.11.0
  "CallPrimitive": (
    "jax.core.CallPrimitive is deprecated. Use jax.extend.core.CallPrimitive.",
    _src_core.CallPrimitive,
  ),
  "DebugInfo": (
    "jax.core.DebugInfo is deprecated. Use jax.extend.core.DebugInfo.",
    _src_core.DebugInfo,
  ),
  "DropVar": (
    "jax.core.DropVar is deprecated. Use jax.extend.core.DropVar.",
    _src_core.DropVar,
  ),
  "Effect": (
    "jax.core.Effect is deprecated. Use jax.extend.core.Effect.",
    _src_core.Effect,
  ),
  "Effects": (
    "jax.core.Effects is deprecated. Use jax.extend.core.Effects.",
    _src_core.Effects,
  ),
  "InconclusiveDimensionOperation": (
    "jax.core.InconclusiveDimensionOperation is deprecated. Use jax.extend.core.InconclusiveDimensionOperation.",
    _src_core.InconclusiveDimensionOperation,
  ),
  "JaxprTypeError": (
    "jax.core.JaxprTypeError is deprecated. Use jax.extend.core.JaxprTypeError.",
    _src_core.JaxprTypeError,
  ),
  "check_jaxpr": (
    "jax.core.check_jaxpr is deprecated. Use jax.extend.core.check_jaxpr.",
    _src_core.check_jaxpr,
  ),
  "concrete_or_error": (
    "jax.core.concrete_or_error is deprecated. Use jax.extend.core.concrete_or_error.",
    _src_core.concrete_or_error,
  ),
  "find_top_trace": (
    "jax.core.find_top_trace is deprecated. Use jax.extend.core.find_top_trace.",
    _src_core.find_top_trace,
  ),
  "gensym": (
    "jax.core.gensym is deprecated. Use jax.extend.core.gensym.",
    _src_core.gensym,
  ),
  "get_opaque_trace_state": (
    "jax.core.get_opaque_trace_state is deprecated. Use jax.extend.core.get_opaque_trace_state.",
    _src_core.get_opaque_trace_state,
  ),
  "jaxprs_in_params": (
    "jax.core.jaxprs_in_params is deprecated. Use jax.extend.core.jaxprs_in_params.",
    _src_core.jaxprs_in_params,
  ),
  "new_jaxpr_eqn": (
    "jax.core.new_jaxpr_eqn is deprecated. Use jax.extend.core.new_jaxpr_eqn.",
    _src_core.new_jaxpr_eqn,
  ),
  "no_effects": (
    "jax.core.no_effects is deprecated. Use jax.extend.core.no_effects.",
    _src_core.no_effects,
  ),
  "nonempty_axis_env_DO_NOT_USE": (
    "jax.core.nonempty_axis_env_DO_NOT_USE is deprecated.",
    _src_core.nonempty_axis_env,
  ),
  "primal_dtype_to_tangent_dtype": (
    "jax.core.primal_dtype_to_tangent_dtype is deprecated. Use jax.extend.core.primal_dtype_to_tangent_dtype.",
    _src_core.primal_dtype_to_tangent_dtype,
  ),
  "unsafe_am_i_under_a_jit_DO_NOT_USE": (
    "jax.core.unsafe_am_i_under_a_jit_DO_NOT_USE is deprecated.",
    _src_core.unsafe_am_i_under_a_jit,
  ),
  "unsafe_am_i_under_a_vmap_DO_NOT_USE": (
    "jax.core.unsafe_am_i_under_a_vmap_DO_NOT_USE is deprecated.",
    _src_core.unsafe_am_i_under_a_vmap,
  ),
  "unsafe_get_axis_names_DO_NOT_USE": (
    "jax.core.unsafe_get_axis_names_DO_NOT_USE is deprecated.",
    _src_core.unsafe_get_axis_names,
  ),
  "valid_jaxtype": (
    "jax.core.valid_jaxtype is deprecated. Use jax.extend.core.valid_jaxtype.",
    _src_core.valid_jaxtype,
  ),
  "JaxprPpContext": (
    "jax.core.JaxprPpContext is deprecated.",
    _src_core.JaxprPpContext,
  ),
  "JaxprPpSettings": (
    "jax.core.JaxprPpSettings is deprecated.",
    _src_core.JaxprPpSettings,
  ),
  "OutputType": (
    "jax.core.OutputType is deprecated.",
    _src_core.OutputType,
  ),
  "abstract_token": (
    "jax.core.abstract_token is deprecated.",
    _src_core.abstract_token,
  ),
  "aval_mapping_handlers": (
    "jax.core.aval_mapping_handlers is deprecated.",
    _src_core.aval_mapping_handlers,
  ),
  "call": (
    "jax.core.call is deprecated.",
    _src_core.call,
  ),
  "concretization_function_error": (
    "jax.core.concretization_function_error is deprecated.",
    _src_core.concretization_function_error,
  ),
  "custom_typechecks": (
    "jax.core.custom_typechecks is deprecated.",
    _src_core.custom_typechecks,
  ),
  "is_concrete": (
    "jax.core.is_concrete is deprecated.",
    _src_core.is_concrete,
  ),
  "is_constant_dim": (
    "jax.core.is_constant_dim is deprecated.",
    _src_core.is_constant_dim,
  ),
  "is_constant_shape": (
    "jax.core.is_constant_shape is deprecated.",
    _src_core.is_constant_shape,
  ),
  "literalable_types": (
    "jax.core.literalable_types is deprecated.",
    _src_core.literalable_types,
  ),
  "no_axis_name": (
    "jax.core.no_axis_name is deprecated.",
    _src_core.no_axis_name,
  ),
  "pytype_aval_mappings": (
    "jax.core.pytype_aval_mappings is deprecated.",
    _src_core.pytype_aval_mappings,
  ),
  "trace_ctx": (
    "jax.core.trace_ctx is deprecated.",
    _src_core.trace_ctx,
  ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  CallPrimitive = _src_core.CallPrimitive
  DebugInfo = _src_core.DebugInfo
  DropVar = _src_core.DropVar
  Effect = _src_core.Effect
  Effects = _src_core.Effects
  InconclusiveDimensionOperation = _src_core.InconclusiveDimensionOperation
  JaxprPpContext = _src_core.JaxprPpContext
  JaxprPpSettings = _src_core.JaxprPpSettings
  JaxprTypeError = _src_core.JaxprTypeError
  OutputType = _src_core.OutputType
  abstract_token = _src_core.abstract_token
  aval_mapping_handlers = _src_core.aval_mapping_handlers
  call = _src_core.call
  check_jaxpr = _src_core.check_jaxpr
  concrete_or_error = _src_core.concrete_or_error
  concretization_function_error = _src_core.concretization_function_error
  custom_typechecks = _src_core.custom_typechecks
  find_top_trace = _src_core.find_top_trace
  gensym = _src_core.gensym
  get_opaque_trace_state = _src_core.get_opaque_trace_state
  is_concrete = _src_core.is_concrete
  is_constant_dim = _src_core.is_constant_dim
  is_constant_shape = _src_core.is_constant_shape
  jaxprs_in_params = _src_core.jaxprs_in_params
  literalable_types = _src_core.literalable_types
  new_jaxpr_eqn = _src_core.new_jaxpr_eqn
  no_axis_name = _src_core.no_axis_name
  no_effects = _src_core.no_effects
  nonempty_axis_env_DO_NOT_USE = _src_core.nonempty_axis_env
  primal_dtype_to_tangent_dtype = _src_core.primal_dtype_to_tangent_dtype
  pytype_aval_mappings = _src_core.pytype_aval_mappings
  trace_ctx = _src_core.trace_ctx
  unsafe_am_i_under_a_jit_DO_NOT_USE = _src_core.unsafe_am_i_under_a_jit
  unsafe_am_i_under_a_vmap_DO_NOT_USE = _src_core.unsafe_am_i_under_a_vmap
  unsafe_get_axis_names_DO_NOT_USE = _src_core.unsafe_get_axis_names
  valid_jaxtype = _src_core.valid_jaxtype
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
del _src_core
