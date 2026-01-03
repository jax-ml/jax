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
  # Remove in v0.10.0
  "Const": (
    "jax.interpreters.partial_eval.Const is deprecated.",
    None,
  ),
  "ConstFoldRule": (
    "jax.interpreters.partial_eval.ConstFoldRule is deprecated.",
    None,
  ),
  "ConstVar": (
    "jax.interpreters.partial_eval.ConstVar is deprecated.",
    None,
  ),
  "DCERule": (
    "jax.interpreters.partial_eval.DCERule is deprecated.",
    None,
  ),
  "DynamicJaxprTrace": (
    "jax.interpreters.partial_eval.DynamicJaxprTrace is deprecated.",
    None,
  ),
  "ForwardingRule": (
    "jax.interpreters.partial_eval.ForwardingRule is deprecated.",
    None,
  ),
  "FreeVar": (
    "jax.interpreters.partial_eval.FreeVar is deprecated.",
    None,
  ),
  "Jaxpr": (
    (
        "jax.interpreters.partial_eval.Jaxpr is deprecated. Use"
        " jax.extend.core.Jaxpr, and please note that you must"
        " `import jax.extend` explicitly."
    ),
    None,
  ),
  "JaxprEqnRecipe": (
    "jax.interpreters.partial_eval.JaxprEqnRecipe is deprecated.",
    None,
  ),
  "JaxprStackFrame": (
    "jax.interpreters.partial_eval.JaxprStackFrame is deprecated.",
    None,
  ),
  "JaxprTrace": (
    "jax.interpreters.partial_eval.JaxprTrace is deprecated.",
    None,
  ),
  "JaxprTracerRecipe": (
    "jax.interpreters.partial_eval.JaxprTracerRecipe is deprecated.",
    None,
  ),
  "LambdaBinding": (
    "jax.interpreters.partial_eval.LambdaBinding is deprecated.",
    None,
  ),
  "ParamsUpdater": (
    "jax.interpreters.partial_eval.ParamsUpdater is deprecated.",
    None,
  ),
  "PartialEvalCustomResult": (
    "jax.interpreters.partial_eval.PartialEvalCustomResult is deprecated.",
    None,
  ),
  "PartialEvalCustomRule": (
    "jax.interpreters.partial_eval.PartialEvalCustomRule is deprecated.",
    None,
  ),
  "ResAvalUpdater": (
    "jax.interpreters.partial_eval.ResAvalUpdater is deprecated.",
    None,
  ),
  "TracerAsName": (
    "jax.interpreters.partial_eval.TracerAsName is deprecated.",
    None,
  ),
  "TracerId": (
    "jax.interpreters.partial_eval.TracerId is deprecated.",
    None,
  ),
  "abstract_eval_fun": (
    "jax.interpreters.partial_eval.abstract_eval_fun is deprecated.",
    None,
  ),
  "call_param_updaters": (
    "jax.interpreters.partial_eval.call_param_updaters is deprecated.",
    None,
  ),
  "call_partial_eval_custom_rule": (
    "jax.interpreters.partial_eval.call_partial_eval_custom_rule is deprecated.",
    None,
  ),
  "call_partial_eval_rules": (
    "jax.interpreters.partial_eval.call_partial_eval_rules is deprecated.",
    None,
  ),
  "close_jaxpr": (
    "jax.interpreters.partial_eval.close_jaxpr is deprecated.",
    None,
  ),
  "closed_call_partial_eval_custom_rule": (
    "jax.interpreters.partial_eval.closed_call_partial_eval_custom_rule is deprecated.",
    None,
  ),
  "config": (
    "jax.interpreters.partial_eval.config is deprecated; use jax.config directly.",
    None,
  ),
  "const_fold_rules": (
    "jax.interpreters.partial_eval.const_fold_rules is deprecated.",
    None,
  ),
  "convert_constvars_jaxpr": (
    "jax.interpreters.partial_eval.convert_constvars_jaxpr is deprecated.",
    None,
  ),
  "convert_envvars_to_constvars": (
    "jax.interpreters.partial_eval.convert_envvars_to_constvars is deprecated.",
    None,
  ),
  "convert_invars_to_constvars": (
    "jax.interpreters.partial_eval.convert_invars_to_constvars is deprecated.",
    None,
  ),
  "custom_staging_rules": (
    "jax.interpreters.partial_eval.custom_staging_rules is deprecated.",
    None,
  ),
  "forwarding_rules": (
    "jax.interpreters.partial_eval.forwarding_rules is deprecated.",
    None,
  ),
  "has_effects": (
    "jax.interpreters.partial_eval.has_effects is deprecated.",
    None,
  ),
  "instantiate_const_at": (
    "jax.interpreters.partial_eval.instantiate_const_at is deprecated.",
    None,
  ),
  "make_jaxpr_effects": (
    "jax.interpreters.partial_eval.make_jaxpr_effects is deprecated.",
    None,
  ),
  "move_binders_to_back": (
    "jax.interpreters.partial_eval.move_binders_to_back is deprecated.",
    None,
  ),
  "move_binders_to_front": (
    "jax.interpreters.partial_eval.move_binders_to_front is deprecated.",
    None,
  ),
  "new_eqn_recipe": (
    "jax.interpreters.partial_eval.new_eqn_recipe is deprecated.",
    None,
  ),
  "partial_eval_jaxpr_custom": (
    "jax.interpreters.partial_eval.partial_eval_jaxpr_custom is deprecated.",
    None,
  ),
  "partial_eval_jaxpr_custom_rule_not_implemented": (
    "jax.interpreters.partial_eval.partial_eval_jaxpr_custom_rule_not_implemented is deprecated.",
    None,
  ),
  "partial_eval_jaxpr_nounits": (
    "jax.interpreters.partial_eval.partial_eval_jaxpr_nounits is deprecated.",
    None,
  ),
  "partial_eval_wrapper_nounits": (
    "jax.interpreters.partial_eval.partial_eval_wrapper_nounits is deprecated.",
    None,
  ),
  "partition_pvals": (
    "jax.interpreters.partial_eval.partition_pvals is deprecated.",
    None,
  ),
  "recipe_to_eqn": (
    "jax.interpreters.partial_eval.recipe_to_eqn is deprecated.",
    None,
  ),
  "trace_to_subjaxpr_nounits": (
    "jax.interpreters.partial_eval.trace_to_subjaxpr_nounits is deprecated.",
    None,
  ),
  "trace_to_subjaxpr_nounits_fwd": (
    "jax.interpreters.partial_eval.trace_to_subjaxpr_nounits_fwd is deprecated.",
    None,
  ),
  "tracers_to_jaxpr": (
    "jax.interpreters.partial_eval.tracers_to_jaxpr is deprecated.",
    None,
  ),
}

from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
