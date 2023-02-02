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
# See PEP 484 & https://github.com/google/jax/issues/7570

from __future__ import annotations

from jax._src.interpreters.ad import (
  CustomJVPException as CustomJVPException,
  CustomVJPException as CustomVJPException,
  JVPTrace as JVPTrace,
  JVPTracer as JVPTracer,
  UndefinedPrimal as UndefinedPrimal,
  Zero as Zero,
  _closed_call_transpose as _closed_call_transpose,
  _custom_lin_transpose as _custom_lin_transpose,
  _interleave as _interleave,
  _jvp_jaxpr as _jvp_jaxpr,
  _perm as _perm,
  _primal_tangent_shapes_match as _primal_tangent_shapes_match,
  _raise_custom_vjp_error_on_jvp as _raise_custom_vjp_error_on_jvp,
  _update_annotation as _update_annotation,
  add_jaxvals as add_jaxvals,
  add_jaxvals_p as add_jaxvals_p,
  add_tangents as add_tangents,
  as_hashable_function as as_hashable_function,
  backward_pass as backward_pass,
  bilinear_transpose as bilinear_transpose,
  call_p as call_p,
  call_param_updaters as call_param_updaters,
  call_transpose as call_transpose,
  call_transpose_param_updaters as call_transpose_param_updaters,
  closed_backward_pass as closed_backward_pass,
  custom_lin_p as custom_lin_p,
  defbilinear as defbilinear,
  defjvp as defjvp,
  defjvp2 as defjvp2,
  defjvp_zero as defjvp_zero,
  deflinear as deflinear,
  deflinear2 as deflinear2,
  dtype as dtype,
  f_jvp_traceable as f_jvp_traceable,
  flatten_fun as flatten_fun,
  flatten_fun_nokwargs as flatten_fun_nokwargs,
  float0 as float0,
  get_aval as get_aval,
  get_primitive_transpose as get_primitive_transpose,
  identity as identity,
  instantiate_zeros as instantiate_zeros,
  instantiate_zeros_aval as instantiate_zeros_aval,
  is_undefined_primal as is_undefined_primal,
  jvp as jvp,
  jvp_jaxpr as jvp_jaxpr,
  jvp_subtrace as jvp_subtrace,
  jvp_subtrace_aux as jvp_subtrace_aux,
  jvpfun as jvpfun,
  linear_jvp as linear_jvp,
  linear_transpose as linear_transpose,
  linear_transpose2 as linear_transpose2,
  linearize as linearize,
  map_transpose as map_transpose,
  nonzero_outputs as nonzero_outputs,
  nonzero_tangent_outputs as nonzero_tangent_outputs,
  partition_list as partition_list,
  primitive_jvps as primitive_jvps,
  primitive_transposes as primitive_transposes,
  raise_to_shaped as raise_to_shaped,
  rearrange_binders as rearrange_binders,
  recast_to_float0 as recast_to_float0,
  reducing_transposes as reducing_transposes,
  replace_float0s as replace_float0s,
  standard_jvp as standard_jvp,
  standard_jvp2 as standard_jvp2,
  traceable as traceable,
  unpair_pval as unpair_pval,
  unzip2 as unzip2,
  vjp as vjp,
  weakref_lru_cache as weakref_lru_cache,
  wrap_name as wrap_name,
  zero_jvp as zero_jvp,
  zeros_like_aval as zeros_like_aval,
  zeros_like_jaxval as zeros_like_jaxval,
  zeros_like_p as zeros_like_p,
)

from jax.config import config
from jax._src import source_info_util
