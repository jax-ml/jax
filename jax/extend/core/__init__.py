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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.abstract_arrays import (
  array_types as array_types
)

from jax._src.core import (
  AbstractToken as AbstractToken,
  CallPrimitive as CallPrimitive,
  ClosedJaxpr as ClosedJaxpr,
  DebugInfo as DebugInfo,
  DropVar as DropVar,
  Effect as Effect,
  Effects as Effects,
  InconclusiveDimensionOperation as InconclusiveDimensionOperation,
  Jaxpr as Jaxpr,
  JaxprEqn as JaxprEqn,
  JaxprTypeError as JaxprTypeError,
  Literal as Literal,
  Primitive as Primitive,
  Token as Token,
  TraceTag as TraceTag,
  Var as Var,
  call_impl as call_impl,
  check_jaxpr as check_jaxpr,
  concrete_or_error as concrete_or_error,
  find_top_trace as find_top_trace,
  gensym as gensym,
  get_opaque_trace_state as get_opaque_trace_state,
  jaxpr_as_fun as jaxpr_as_fun,
  jaxprs_in_params as jaxprs_in_params,
  mapped_aval as mapped_aval,
  new_jaxpr_eqn as new_jaxpr_eqn,
  no_effects as no_effects,
  nonempty_axis_env as nonempty_axis_env_DO_NOT_USE,  # noqa: F401
  primal_dtype_to_tangent_dtype as primal_dtype_to_tangent_dtype,
  set_current_trace as set_current_trace,
  subjaxprs as subjaxprs,
  take_current_trace as take_current_trace,
  unmapped_aval as unmapped_aval,
  unsafe_am_i_under_a_jit as unsafe_am_i_under_a_jit_DO_NOT_USE,  # noqa: F401
  unsafe_am_i_under_a_vmap as unsafe_am_i_under_a_vmap_DO_NOT_USE,  # noqa: F401
  unsafe_get_axis_names as unsafe_get_axis_names_DO_NOT_USE,  # noqa: F401
  valid_jaxtype as valid_jaxtype,
)

from . import primitives as primitives
