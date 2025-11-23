# Copyright 2018 The JAX Authors.
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

from jax._src import core as _core_src
from jax._src.interpreters import partial_eval as _pe_src

from jax._src.interpreters.partial_eval import (
  DynamicJaxprTracer as DynamicJaxprTracer,
  JaxprTracer as JaxprTracer,
  PartialVal as PartialVal,
  Val as Val,
  custom_partial_eval_rules as custom_partial_eval_rules,
  dce_jaxpr as dce_jaxpr,
  dce_jaxpr_call_rule as dce_jaxpr_call_rule,
  dce_jaxpr_closed_call_rule as dce_jaxpr_closed_call_rule,
  dce_jaxpr_consts as dce_jaxpr_consts,
  dce_rules as dce_rules,
  partial_eval_jaxpr_custom_rules as partial_eval_jaxpr_custom_rules,
  trace_to_jaxpr_dynamic as trace_to_jaxpr_dynamic,
  trace_to_jaxpr_nounits as trace_to_jaxpr_nounits,
)


_deprecations = {
  # Deprecated for JAX v0.7.1; finalize in JAX v0.9.0.
  "Const": (
    "jax.interpreters.partial_eval.Const is deprecated.",
    _pe_src.Const,
  ),
  "ConstFoldRule": (
    "jax.interpreters.partial_eval.ConstFoldRule is deprecated.",
    _pe_src.ConstFoldRule,
  ),
  "ConstVar": (
    "jax.interpreters.partial_eval.ConstVar is deprecated.",
    _pe_src.ConstVar,
  ),
  "DCERule": (
    "jax.interpreters.partial_eval.DCERule is deprecated.",
    _pe_src.DCERule,
  ),
  "DynamicJaxprTrace": (
    "jax.interpreters.partial_eval.DynamicJaxprTrace is deprecated.",
    _pe_src.DynamicJaxprTrace,
  ),
  "ForwardingRule": (
    "jax.interpreters.partial_eval.ForwardingRule is deprecated.",
    _pe_src.ForwardingRule,
  ),
  "FreeVar": (
    "jax.interpreters.partial_eval.FreeVar is deprecated.",
    _pe_src.FreeVar,
  ),
  "Jaxpr": (
    (
        "jax.interpreters.partial_eval.Jaxpr is deprecated. Use"
        " jax.extend.core.Jaxpr, and please note that you must"
        " `import jax.extend` explicitly."
    ),
    _core_src.Jaxpr,
  ),
  "JaxprEqnRecipe": (
    "jax.interpreters.partial_eval.JaxprEqnRecipe is deprecated.",
    _pe_src.JaxprEqnRecipe,
  ),
  "JaxprStackFrame": (
    "jax.interpreters.partial_eval.JaxprStackFrame is deprecated.",
    _pe_src.JaxprStackFrame,
  ),
  "JaxprTrace": (
    "jax.interpreters.partial_eval.JaxprTrace is deprecated.",
    _pe_src.JaxprTrace,
  ),
  "JaxprTracerRecipe": (
    "jax.interpreters.partial_eval.JaxprTracerRecipe is deprecated.",
    _pe_src.JaxprTracerRecipe,
  ),
  "LambdaBinding": (
    "jax.interpreters.partial_eval.LambdaBinding is deprecated.",
    _pe_src.LambdaBinding,
  ),
  "ParamsUpdater": (
    "jax.interpreters.partial_eval.ParamsUpdater is deprecated.",
    _pe_src.ParamsUpdater,
  ),
  "PartialEvalCustomResult": (
    "jax.interpreters.partial_eval.PartialEvalCustomResult is deprecated.",
    _pe_src.PartialEvalCustomResult,
  ),
  "PartialEvalCustomRule": (
    "jax.interpreters.partial_eval.PartialEvalCustomRule is deprecated.",
    _pe_src.PartialEvalCustomRule,
  ),
  "ResAvalUpdater": (
    "jax.interpreters.partial_eval.ResAvalUpdater is deprecated.",
    _pe_src.ResAvalUpdater,
  ),
  "TracerAsName": (
    "jax.interpreters.partial_eval.TracerAsName is deprecated.",
    _pe_src.TracerAsName,
  ),
  "TracerId": (
    "jax.interpreters.partial_eval.TracerId is deprecated.",
    _pe_src.TracerId,
  ),
  "abstract_eval_fun": (
    "jax.interpreters.partial_eval.abstract_eval_fun is deprecated.",
    _pe_src.abstract_eval_fun,
  ),
  "call_param_updaters": (
    "jax.interpreters.partial_eval.call_param_updaters is deprecated.",
    _pe_src.call_param_updaters,
  ),
  "call_partial_eval_custom_rule": (
    "jax.interpreters.partial_eval.call_partial_eval_custom_rule is deprecated.",
    _pe_src.call_partial_eval_custom_rule,
  ),
  "call_partial_eval_rules": (
    "jax.interpreters.partial_eval.call_partial_eval_rules is deprecated.",
    _pe_src.call_partial_eval_rules,
  ),
  "close_jaxpr": (
    "jax.interpreters.partial_eval.close_jaxpr is deprecated.",
    _pe_src.close_jaxpr,
  ),
  "closed_call_partial_eval_custom_rule": (
    "jax.interpreters.partial_eval.closed_call_partial_eval_custom_rule is deprecated.",
    _pe_src.closed_call_partial_eval_custom_rule,
  ),
  "config": (
    "jax.interpreters.partial_eval.config is deprecated; use jax.config directly.",
    _pe_src.config,
  ),
  "const_fold_rules": (
    "jax.interpreters.partial_eval.const_fold_rules is deprecated.",
    _pe_src.const_fold_rules,
  ),
  "convert_constvars_jaxpr": (
    "jax.interpreters.partial_eval.convert_constvars_jaxpr is deprecated.",
    _pe_src.convert_constvars_jaxpr,
  ),
  "convert_envvars_to_constvars": (
    "jax.interpreters.partial_eval.convert_envvars_to_constvars is deprecated.",
    _pe_src.convert_envvars_to_constvars,
  ),
  "convert_invars_to_constvars": (
    "jax.interpreters.partial_eval.convert_invars_to_constvars is deprecated.",
    _pe_src.convert_invars_to_constvars,
  ),
  "custom_staging_rules": (
    "jax.interpreters.partial_eval.custom_staging_rules is deprecated.",
    _pe_src.custom_staging_rules,
  ),
  "forwarding_rules": (
    "jax.interpreters.partial_eval.forwarding_rules is deprecated.",
    _pe_src.forwarding_rules,
  ),
  "has_effects": (
    "jax.interpreters.partial_eval.has_effects is deprecated.",
    _pe_src.has_effects,
  ),
  "instantiate_const_at": (
    "jax.interpreters.partial_eval.instantiate_const_at is deprecated.",
    _pe_src.instantiate_const_at,
  ),
  "make_jaxpr_effects": (
    "jax.interpreters.partial_eval.make_jaxpr_effects is deprecated.",
    _pe_src.make_jaxpr_effects,
  ),
  "move_binders_to_back": (
    "jax.interpreters.partial_eval.move_binders_to_back is deprecated.",
    _pe_src.move_binders_to_back,
  ),
  "move_binders_to_front": (
    "jax.interpreters.partial_eval.move_binders_to_front is deprecated.",
    _pe_src.move_binders_to_front,
  ),
  "new_eqn_recipe": (
    "jax.interpreters.partial_eval.new_eqn_recipe is deprecated.",
    _pe_src.new_eqn_recipe,
  ),
  "partial_eval_jaxpr_custom": (
    "jax.interpreters.partial_eval.partial_eval_jaxpr_custom is deprecated.",
    _pe_src.partial_eval_jaxpr_custom,
  ),
  "partial_eval_jaxpr_custom_rule_not_implemented": (
    "jax.interpreters.partial_eval.partial_eval_jaxpr_custom_rule_not_implemented is deprecated.",
    _pe_src.partial_eval_jaxpr_custom_rule_not_implemented,
  ),
  "partial_eval_jaxpr_nounits": (
    "jax.interpreters.partial_eval.partial_eval_jaxpr_nounits is deprecated.",
    _pe_src.partial_eval_jaxpr_nounits,
  ),
  "partial_eval_wrapper_nounits": (
    "jax.interpreters.partial_eval.partial_eval_wrapper_nounits is deprecated.",
    _pe_src.partial_eval_wrapper_nounits,
  ),
  "partition_pvals": (
    "jax.interpreters.partial_eval.partition_pvals is deprecated.",
    _pe_src.partition_pvals,
  ),
  "recipe_to_eqn": (
    "jax.interpreters.partial_eval.recipe_to_eqn is deprecated.",
    _pe_src.recipe_to_eqn,
  ),
  "trace_to_subjaxpr_nounits": (
    "jax.interpreters.partial_eval.trace_to_subjaxpr_nounits is deprecated.",
    _pe_src.trace_to_subjaxpr_nounits,
  ),
  "trace_to_subjaxpr_nounits_fwd": (
    "jax.interpreters.partial_eval.trace_to_subjaxpr_nounits_fwd is deprecated.",
    _pe_src.trace_to_subjaxpr_nounits_fwd,
  ),
  "tracers_to_jaxpr": (
    "jax.interpreters.partial_eval.tracers_to_jaxpr is deprecated.",
    _pe_src.tracers_to_jaxpr,
  ),
}

import typing
if typing.TYPE_CHECKING:
  AbstractedAxesSpec = _pe_src.AbstractedAxesSpec
  AbstractedAxisName = _pe_src.AbstractedAxisName
  BoundedAxisSize = _pe_src.BoundedAxisSize
  Const = _pe_src.Const
  ConstFoldRule = _pe_src.ConstFoldRule
  ConstVar = _pe_src.ConstVar
  DCERule = _pe_src.DCERule
  DynamicJaxprTrace = _pe_src.DynamicJaxprTrace
  ForwardingRule = _pe_src.ForwardingRule
  FreeVar = _pe_src.FreeVar
  Jaxpr = _core_src.Jaxpr
  JaxprEqnRecipe = _pe_src.JaxprEqnRecipe
  JaxprStackFrame = _pe_src.JaxprStackFrame
  JaxprTrace = _pe_src.JaxprTrace
  JaxprTracerRecipe = _pe_src.JaxprTracerRecipe
  LambdaBinding = _pe_src.LambdaBinding
  ParamsUpdater = _pe_src.ParamsUpdater
  PartialEvalCustomResult = _pe_src.PartialEvalCustomResult
  PartialEvalCustomRule = _pe_src.PartialEvalCustomRule
  ResAvalUpdater = _pe_src.ResAvalUpdater
  TracerAsName = _pe_src.TracerAsName
  TracerId = _pe_src.TracerId
  abstract_eval_fun = _pe_src.abstract_eval_fun
  call_param_updaters = _pe_src.call_param_updaters
  call_partial_eval_custom_rule = _pe_src.call_partial_eval_custom_rule
  call_partial_eval_rules = _pe_src.call_partial_eval_rules
  close_jaxpr = _pe_src.close_jaxpr
  closed_call_partial_eval_custom_rule = _pe_src.closed_call_partial_eval_custom_rule
  config = _pe_src.config
  const_fold_rules = _pe_src.const_fold_rules
  convert_constvars_jaxpr = _pe_src.convert_constvars_jaxpr
  convert_envvars_to_constvars = _pe_src.convert_envvars_to_constvars
  convert_invars_to_constvars = _pe_src.convert_invars_to_constvars
  custom_staging_rules = _pe_src.custom_staging_rules
  forwarding_rules = _pe_src.forwarding_rules
  has_effects = _pe_src.has_effects
  infer_lambda_input_type = _pe_src.infer_lambda_input_type
  instantiate_const_at = _pe_src.instantiate_const_at
  make_jaxpr_effects = _pe_src.make_jaxpr_effects
  move_binders_to_back = _pe_src.move_binders_to_back
  move_binders_to_front = _pe_src.move_binders_to_front
  new_eqn_recipe = _pe_src.new_eqn_recipe
  partial_eval_jaxpr_custom = _pe_src.partial_eval_jaxpr_custom
  partial_eval_jaxpr_custom_rule_not_implemented = _pe_src.partial_eval_jaxpr_custom_rule_not_implemented
  partial_eval_jaxpr_nounits = _pe_src.partial_eval_jaxpr_nounits
  partial_eval_wrapper_nounits = _pe_src.partial_eval_wrapper_nounits
  partition_pvals = _pe_src.partition_pvals
  recipe_to_eqn = _pe_src.recipe_to_eqn
  trace_to_jaxpr_dynamic2 = _pe_src.trace_to_jaxpr_dynamic2
  trace_to_subjaxpr_nounits = _pe_src.trace_to_subjaxpr_nounits
  trace_to_subjaxpr_nounits_fwd = _pe_src.trace_to_subjaxpr_nounits_fwd
  tracers_to_jaxpr = _pe_src.tracers_to_jaxpr
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing, _pe_src, _core_src
