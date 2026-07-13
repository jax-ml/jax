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
  # Deprecated in JAX v0.10.0, finalized in v0.11.0.
  # TODO(jakevdp) remove in v0.12.0
  "CallPrimitive": (
    "jax.core.CallPrimitive was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0. Use jax.extend.core.CallPrimitive.",
    None,
  ),
  "DebugInfo": (
    "jax.core.DebugInfo  was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0. Use jax.extend.core.DebugInfo.",
    None,
  ),
  "DropVar": (
    "jax.core.DropVar  was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0. Use jax.extend.core.DropVar.",
    None,
  ),
  "Effect": (
    "jax.core.Effect  was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0. Use jax.extend.core.Effect.",
     None,
  ),
  "Effects": (
    "jax.core.Effects was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0.  Use jax.extend.core.Effects.",
    None,
  ),
  "InconclusiveDimensionOperation": (
    "jax.core.InconclusiveDimensionOperation  was deprecated in JAX v0.10.0 and"
    " removed in JAX v0.11.0.  Use jax.errors.InconclusiveDimensionOperation.",
    None,
  ),
  "JaxprTypeError": (
    "jax.core.JaxprTypeError  was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0.  Use jax.errors.JaxprTypeError.",
    None,
  ),
  "abstract_token": (
    "jax.core.abstract_token was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0. Use jax.ffi.abstract_token, available in JAX v0.11.0.",
    None,
  ),
  "check_jaxpr": (
    "jax.core.check_jaxpr  was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0.  Use jax.extend.core.check_jaxpr.",
    None,
  ),
  "concrete_or_error": (
    "jax.core.concrete_or_error was deprecated in JAX v0.10.0 and removed in"
    " JAX v0.11.0.  Use jax.extend.core.concrete_or_error.",
    None,
  ),
  "find_top_trace": (
    "jax.core.find_top_trace was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0.  Use jax.extend.core.find_top_trace.",
    None,
  ),
  "gensym": (
    "jax.core.gensym was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0.  Use jax.extend.core.gensym.",
    None,
  ),
  "get_opaque_trace_state": (
    "jax.core.get_opaque_trace_state was deprecated in JAX v0.10.0 and removed"
    " in JAX v0.11.0.  Use jax.extend.core.get_opaque_trace_state.",
    None,
  ),
  "is_concrete": (
    "jax.core.is_concrete was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0.",
    None,
  ),
  "is_constant_dim": (
    "jax.core.is_constant_dim was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0.",
    None,
  ),
  "is_constant_shape": (
    "jax.core.is_constant_shape was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0.",
    None,
  ),
  "jaxprs_in_params": (
    "jax.core.jaxprs_in_params was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0. Use jax.extend.core.jaxprs_in_params.",
    None,
  ),
  "new_jaxpr_eqn": (
    "jax.core.new_jaxpr_eqn was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0.  Use jax.extend.core.new_jaxpr_eqn.",
    None,
  ),
  "no_effects": (
    "jax.core.no_effects was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0.  Use jax.extend.core.no_effects.",
    None,
  ),
  "nonempty_axis_env_DO_NOT_USE": (
    "jax.core.nonempty_axis_env_DO_NOT_USE  was deprecated in JAX v0.10.0 and"
    " removed in JAX v0.11.0.",
    None,
  ),
  "primal_dtype_to_tangent_dtype": (
    "jax.core.primal_dtype_to_tangent_dtype was deprecated in JAX v0.10.0 and"
    " removed in JAX v0.11.0.  Use jax.extend.core.primal_dtype_to_tangent_dtype.",
    None,
  ),
  "unsafe_am_i_under_a_jit_DO_NOT_USE": (
    "jax.core.unsafe_am_i_under_a_jit_DO_NOT_USE was deprecated in JAX v0.10.0"
    " and removed in JAX v0.11.0.",
    None,
  ),
  "unsafe_am_i_under_a_vmap_DO_NOT_USE": (
    "jax.core.unsafe_am_i_under_a_vmap_DO_NOT_USE was deprecated in JAX v0.10.0"
    " and removed in JAX v0.11.0.",
    None,
  ),
  "unsafe_get_axis_names_DO_NOT_USE": (
    "jax.core.unsafe_get_axis_names_DO_NOT_USE was deprecated in JAX v0.10.0"
    " and removed in JAX v0.11.0.",
    None,
  ),
  "valid_jaxtype": (
    "jax.core.valid_jaxtype was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0.  Use jax.extend.core.valid_jaxtype.",
    None,
  ),
  "JaxprPpContext": (
    "jax.core.JaxprPpContext was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0.",
    None,
  ),
  "JaxprPpSettings": (
    "jax.core.JaxprPpSettings was deprecated in JAX v0.10.0 and removed in"
    " JAX v0.11.0.",
    None,
  ),
  "OutputType": (
    "jax.core.OutputType was deprecated in JAX v0.10.0 and removed in JAX"
    " v0.11.0.",
    None,
  ),
  "aval_mapping_handlers": (
    "jax.core.aval_mapping_handlers was deprecated in JAX v0.10.0"
    " and removed in JAX v0.11.0.",
    None,
  ),
  "call": (
    "jax.core.call was deprecated in JAX v0.10.0 and removed in JAX v0.11.0.",
    None,
  ),
  "concretization_function_error": (
    "jax.core.concretization_function_error was deprecated in JAX v0.10.0"
    " and removed in JAX v0.11.0.",
    None,
  ),
  "custom_typechecks": (
    "jax.core.custom_typechecks was deprecated in JAX v0.10.0"
    " and removed in JAX v0.11.0.",
    None,
  ),
  "literalable_types": (
    "jax.core.literalable_types was deprecated in JAX v0.10.0"
    " and removed in JAX v0.11.0.",
    None,
  ),
  "no_axis_name": (
    "jax.core.no_axis_name was deprecated in JAX v0.10.0"
    " and removed in JAX v0.11.0.",
    None,
  ),
  "trace_ctx": (
    "jax.core.trace_ctx was deprecated in JAX v0.10.0"
    " and removed in JAX v0.11.0.",
    None,
  ),
  # Deprecated in JAX v0.10.0, TODO(jakevdp) finalize after v0.11.0
  "pytype_aval_mappings": (
    "jax.core.pytype_aval_mappings is deprecated.",
    _src_core.pytype_aval_mappings,
  ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  pytype_aval_mappings = _src_core.pytype_aval_mappings
  trace_ctx = _src_core.trace_ctx
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
del _src_core
