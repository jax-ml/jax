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
  CallPrimitive as CallPrimitive,
  DebugInfo as DebugInfo,
  DropVar as DropVar,
  Effect as Effect,
  Effects as Effects,
  InconclusiveDimensionOperation as InconclusiveDimensionOperation,
  JaxprPpContext as JaxprPpContext,
  JaxprPpSettings as JaxprPpSettings,
  JaxprTypeError as JaxprTypeError,
  OutputType as OutputType,
  ParamDict as ParamDict,
  ShapedArray as ShapedArray,
  Trace as Trace,
  Tracer as Tracer,
  Value as Value,
  abstract_token as abstract_token,
  aval_mapping_handlers as aval_mapping_handlers,
  call as call,
  check_jaxpr as check_jaxpr,
  concrete_or_error as concrete_or_error,
  concretization_function_error as concretization_function_error,
  custom_typechecks as custom_typechecks,
  ensure_compile_time_eval as ensure_compile_time_eval,
  eval_context as eval_context,
  eval_jaxpr as eval_jaxpr,
  find_top_trace as find_top_trace,
  gensym as gensym,
  get_opaque_trace_state as get_opaque_trace_state,
  is_concrete as is_concrete,
  is_constant_dim as is_constant_dim,
  is_constant_shape as is_constant_shape,
  jaxprs_in_params as jaxprs_in_params,
  literalable_types as literalable_types,
  max_dim as max_dim,
  min_dim as min_dim,
  new_jaxpr_eqn as new_jaxpr_eqn,
  no_axis_name as no_axis_name,
  no_effects as no_effects,
  nonempty_axis_env as nonempty_axis_env_DO_NOT_USE,  # noqa: F401
  primal_dtype_to_tangent_dtype as primal_dtype_to_tangent_dtype,
  pytype_aval_mappings as pytype_aval_mappings,
  trace_ctx as trace_ctx,
  unsafe_am_i_under_a_jit as unsafe_am_i_under_a_jit_DO_NOT_USE,  # noqa: F401
  unsafe_am_i_under_a_vmap as unsafe_am_i_under_a_vmap_DO_NOT_USE,  # noqa: F401
  unsafe_get_axis_names as unsafe_get_axis_names_DO_NOT_USE,  # noqa: F401
  valid_jaxtype as valid_jaxtype,
)

_deprecations = {
  # Added for v0.8.2
  "call_impl": (
    "jax.core.call_impl is deprecated.",
    _src_core.call_impl,
  ),
  "get_aval": (
    "jax.core.get_aval is deprecated; use jax.typeof instead.",
    _src_core.get_aval,
  ),
  "mapped_aval": (
    "jax.core.mapped_aval is deprecated. Use jax.extend.core.mapped_aval.",
    _src_core.mapped_aval,
  ),
  "set_current_trace": (
    "jax.core.set_current_trace is deprecated.",
    _src_core.set_current_trace,
  ),
  "subjaxprs": (
    "jax.core.subjaxprs is deprecated.",
    _src_core.subjaxprs,
  ),
  "take_current_trace": (
    "jax.core.take_current_trace is deprecated.",
    _src_core.take_current_trace,
  ),
  "traverse_jaxpr_params": (
    "jax.core.traverse_jaxpr_params is deprecated.",
    _src_core.traverse_jaxpr_params,
  ),
  "unmapped_aval": (
    "jax.core.unmapped_aval is deprecated. Use jax.extend.core.unmapped_aval.",
    _src_core.unmapped_aval,
  ),
  "AbstractToken": (
    "jax.core.AbstractToken is deprecated.",
    _src_core.AbstractToken,
  ),
  "TraceTag": (
    "jax.core.TraceTag is deprecated.",
    _src_core.TraceTag,
  ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  call_impl = _src_core.call_impl
  get_aval = _src_core.get_aval
  mapped_aval = _src_core.mapped_aval
  subjaxprs = _src_core.subjaxprs
  set_current_trace = _src_core.set_current_trace
  take_current_trace = _src_core.take_current_trace
  traverse_jaxpr_params = _src_core.traverse_jaxpr_params
  unmapped_aval = _src_core.unmapped_aval
  AbstractToken = _src_core.AbstractToken
  TraceTag = _src_core.TraceTag
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
del _src_core
