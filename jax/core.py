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
# See PEP 484 & https://github.com/google/jax/issues/7570

from jax._src.core import (
  AbstractToken as AbstractToken,
  AbstractValue as AbstractValue,
  Atom as Atom,
  AxisSize as AxisSize,
  CallPrimitive as CallPrimitive,
  ClosedJaxpr as ClosedJaxpr,
  ConcreteArray as ConcreteArray,
  ConcretizationTypeError as ConcretizationTypeError,
  DShapedArray as DShapedArray,
  DropVar as DropVar,
  Effect as Effect,
  Effects as Effects,
  EvalTrace as EvalTrace,
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
  MainTrace as MainTrace,
  MapPrimitive as MapPrimitive,
  NameGatheringSubst as NameGatheringSubst,
  NamedShape as NamedShape,
  OutDBIdx as OutDBIdx,
  OutputType as OutputType,
  ParamDict as ParamDict,
  Primitive as Primitive,
  ShapedArray as ShapedArray,
  Sublevel as Sublevel,
  TRACER_LEAK_DEBUGGER_WARNING as TRACER_LEAK_DEBUGGER_WARNING,
  ThreadLocalState as ThreadLocalState,
  Token as Token,
  Trace as Trace,
  TraceStack as TraceStack,
  TraceState as TraceState,
  Tracer as Tracer,
  UnshapedArray as UnshapedArray,
  Value as Value,
  Var as Var,
  abstract_token as abstract_token,
  apply_todos as apply_todos,
  as_named_shape as as_named_shape,
  aval_mapping_handlers as aval_mapping_handlers,
  axis_frame as axis_frame,
  call as call,
  call_bind_with_continuation as call_bind_with_continuation,
  call_impl as call_impl,
  call_p as call_p,
  canonicalize_shape as canonicalize_shape,  # TODO(necula): remove this API
  check_eqn as check_eqn,
  check_jaxpr as check_jaxpr,
  check_type as check_type,
  check_valid_jaxtype as check_valid_jaxtype,
  closed_call_p as closed_call_p,
  concrete_aval as concrete_aval,
  concrete_or_error as concrete_or_error,
  concretization_function_error as concretization_function_error,
  cur_sublevel as cur_sublevel,
  custom_typechecks as custom_typechecks,
  dedup_referents as dedup_referents,
  definitely_equal as definitely_equal,  # TODO(necula): remove this API
  dimension_as_value as dimension_as_value,  # TODO(necula): remove this API
  do_subst_axis_names_jaxpr as do_subst_axis_names_jaxpr,
  ensure_compile_time_eval as ensure_compile_time_eval,
  escaped_tracer_error as escaped_tracer_error,
  eval_context as eval_context,
  eval_jaxpr as eval_jaxpr,
  extend_axis_env as extend_axis_env,
  extend_axis_env_nd as extend_axis_env_nd,
  find_top_trace as find_top_trace,
  full_lower as full_lower,
  gensym as gensym,
  get_aval as get_aval,
  get_referent as get_referent,
  is_constant_dim as is_constant_dim,
  is_constant_shape as is_constant_shape,
  jaxpr_as_fun as jaxpr_as_fun,
  jaxpr_uses_outfeed as jaxpr_uses_outfeed,
  jaxprs_in_params as jaxprs_in_params,
  join_effects as join_effects,
  join_named_shapes as join_named_shapes,
  lattice_join as lattice_join,
  leaked_tracer_error as leaked_tracer_error,
  literalable_types as literalable_types,
  map_bind as map_bind,
  map_bind_with_continuation as map_bind_with_continuation,
  mapped_aval as mapped_aval,
  maybe_find_leaked_tracers as maybe_find_leaked_tracers,
  max_dim as max_dim,
  min_dim as min_dim,
  new_base_main as new_base_main,
  new_jaxpr_eqn as new_jaxpr_eqn,
  new_main as new_main,
  new_sublevel as new_sublevel,
  no_axis_name as no_axis_name,
  no_effects as no_effects,
  non_negative_dim as non_negative_dim,
  outfeed_primitives as outfeed_primitives,
  pp_aval as pp_aval,
  pp_eqn as pp_eqn,
  pp_eqn_rules as pp_eqn_rules,
  pp_eqns as pp_eqns,
  pp_jaxpr as pp_jaxpr,
  pp_jaxpr_eqn_range as pp_jaxpr_eqn_range,
  pp_jaxpr_skeleton as pp_jaxpr_skeleton,
  pp_jaxprs as pp_jaxprs,
  pp_kv_pair as pp_kv_pair,
  pp_kv_pairs as pp_kv_pairs,
  pp_var as pp_var,
  pp_vars as pp_vars,
  primal_dtype_to_tangent_dtype as primal_dtype_to_tangent_dtype,
  primitive_uses_outfeed as primitive_uses_outfeed,
  process_env_traces_call as process_env_traces_call,
  process_env_traces_map as process_env_traces_map,
  pytype_aval_mappings as pytype_aval_mappings,
  raise_as_much_as_possible as raise_as_much_as_possible,
  raise_to_shaped as raise_to_shaped,
  raise_to_shaped_mappings as raise_to_shaped_mappings,
  reset_trace_state as reset_trace_state,
  stash_axis_env as stash_axis_env,
  str_eqn_compact as str_eqn_compact,
  subjaxprs as subjaxprs,
  subst_axis_names as subst_axis_names,
  subst_axis_names_eqn as subst_axis_names_eqn,
  subst_axis_names_jaxpr as subst_axis_names_jaxpr,
  subst_axis_names_var as subst_axis_names_var,
  substitute_vars_in_output_ty as substitute_vars_in_output_ty,
  thread_local_state as thread_local_state,
  token as token,
  trace_state_clean as trace_state_clean,
  traverse_jaxpr_params as traverse_jaxpr_params,
  typecheck as typecheck,
  typecompat as typecompat,
  typematch as typematch,
  unmapped_aval as unmapped_aval,
  used_axis_names as used_axis_names,
  used_axis_names_jaxpr as used_axis_names_jaxpr,
  valid_jaxtype as valid_jaxtype,
)

symbolic_equal_dim = definitely_equal  # TODO(necula): remove this API

from jax._src import core as _src_core
_deprecations = {
    # Added Oct 11, 2023:
    "DimSize": (
        "jax.core.DimSize is deprecated. Use DimSize = int | Any.",
        _src_core.DimSize,
    ),
    "Shape": (
        "jax.core.Shape is deprecated. Use Shape = Sequence[int | Any].",
        _src_core.Shape,
    ),
    "TracerArrayConversionError": (
        "jax.core.TracerArrayConversionError is deprecated. Use jax.errors.TracerArrayConversionError",
        _src_core.TracerArrayConversionError,
    ),
    "TracerIntegerConversionError": (
        "jax.core.TracerIntegerConversionError is deprecated. Use jax.errors.TracerIntegerConversionError",
        _src_core.TracerIntegerConversionError,
    ),
    "UnexpectedTracerError": (
        "jax.core.UnexpectedTracerError is deprecated. Use jax.errors.UnexpectedTracerError",
        _src_core.UnexpectedTracerError,
    ),
    "as_hashable_function": (
        "jax.core.as_hashable_function is deprecated. Use jax.util.as_hashable_function directly.",
        _src_core.as_hashable_function,
    ),
    "collections": (
        "jax.core.collections is deprecated. Use the collections module directly.",
        _src_core.collections,
    ),
    "dtypes": (
        "jax.core.dtypes is deprecated. Use jax.dtypes directly.",
        _src_core.dtypes,
    ),
    "lu": (
        "jax.core.lu is deprecated. Use lu = jax.extend.linear_util",
        _src_core.lu,
    ),
    "map": (
        "jax.core.map is deprecated. Use the built-in map function.",
        _src_core.map,
    ),
    "namedtuple": (
        "jax.core.namedtuple is deprecated. Use collections.namedtuple directly.",
        _src_core.namedtuple,
    ),
    "partial": (
        "jax.core.partial is deprecated. Use functools.partial directly.",
        _src_core.partial,
    ),
    "pp": (
        "jax.core.pp is deprecated. jax._src.pretty_printer is a non-public API.",
        _src_core.pp,
    ),
    "ref": (
        "jax.core.ref is deprecated. Use weakref.ref directly.",
        _src_core.ref,
    ),
    "safe_map": (
        "jax.core.safe_map is deprecated. Use jax.util.safe_map directly.",
        _src_core.safe_map,
    ),
    "safe_zip": (
        "jax.core.safe_zip is deprecated. Use jax.util.safe_zip directly.",
        _src_core.safe_zip,
    ),
    "source_info_util": (
        "jax.core.source_info_util is deprecated. Use jax.extend.source_info_util.",
        _src_core.source_info_util,
    ),
    "total_ordering": (
        "jax.core.total_ordering is deprecated. Use functools.total_ordering directly.",
        _src_core.total_ordering,
    ),
    "traceback_util": (
        "jax.core.traceback_util is deprecated. jax._src.traceback_util is a non-public API.",
        _src_core.traceback_util,
    ),
    "tuple_delete": (
        "jax.core.tuple_delete is deprecated. Use tuple_delete = lambda t, i: (*t[:i], *t[i+1:])",
        _src_core.tuple_delete,
    ),
    "tuple_insert": (
        "jax.core.tuple_insert is deprecated. Use tuple_insert = lambda t, v, i: (*t[:i], v, *t[i:])",
        _src_core.tuple_insert,
    ),
    "zip": (
        "jax.core.zip is deprecated. Use the built-in zip function.",
        _src_core.zip,
    ),
}

import typing
if typing.TYPE_CHECKING:
  DimSize = _src_core.DimSize
  Shape = _src_core.Shape
  TracerArrayConversionError = _src_core.TracerArrayConversionError
  TracerIntegerConversionError = _src_core.TracerIntegerConversionError
  UnexpectedTracerError = _src_core.UnexpectedTracerError
  as_hashable_function = _src_core.as_hashable_function
  collections = _src_core.collections
  dtypes = _src_core.dtypes
  lu = _src_core.lu
  map = _src_core.map
  namedtuple = _src_core.namedtuple
  partial = _src_core.partial
  pp = _src_core.pp
  ref = _src_core.ref
  safe_map = _src_core.safe_map
  safe_zip = _src_core.safe_zip
  source_info_util = _src_core.source_info_util
  total_ordering = _src_core.total_ordering
  traceback_util = _src_core.traceback_util
  tuple_delete = _src_core.tuple_delete
  tuple_insert = _src_core.tuple_insert
  zip = _src_core.zip
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
del _src_core
