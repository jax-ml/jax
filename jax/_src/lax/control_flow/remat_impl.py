# Copyright 2022 Google LLC
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
"""Module for the remat implementation."""
from functools import partial

from typing import Optional

import jax
from jax import core
from jax import lax
from jax.config import config
from jax.interpreters import mlir
from jax.interpreters import partial_eval as pe
from jax.interpreters import xla
from jax.tree_util import tree_flatten, tree_unflatten
from jax._src import ad_checkpoint
from jax._src import util
from jax._src.util import safe_map, wrap_name
from jax._src.lax.control_flow.conditionals import cond
from jax._src.lib.mlir.dialects import mhlo
from jax._src.lax.control_flow.loops import while_loop
import numpy as np

_map = safe_map

def _dummy_remat_result(aval: core.AbstractValue):
  """A result that will be discarded"""
  if aval is core.abstract_token:
    return lax.create_token()
  else:
    return lax.broadcast(np.array(0, dtype=aval.dtype), aval.shape)  # type: ignore

def _remat_translation_using_cond(*args,
                                  jaxpr: core.Jaxpr):
  # Implements:
  #  if(rng(0, 1) < 2)
  #    return eval_jaxpr(*args)
  #  else:
  #    return 0
  avals_out = tuple(ov.aval for ov in jaxpr.outvars)

  def remat_comp(*args):
    return tuple(core.eval_jaxpr(jaxpr, (), *args))
  def dummy_comp(*args):
    return tuple(_map(_dummy_remat_result, avals_out))

  cond_pred = (lax.rng_uniform(np.float32(0), np.float32(1), shape=()) < np.float32(2))
  return cond(cond_pred, remat_comp, dummy_comp, *args)

def _remat_translation_using_while(*args,
                                   jaxpr: core.Jaxpr):
  # Implements:
  #  for(counter=0, result=0; counter < rng(1, 2); counter ++) {
  #     result = eval_jaxpr(*args)
  #  }
  # The loop carry is a tuple: (counter, result, args)
  avals_out = tuple(ov.aval for ov in jaxpr.outvars)
  dummies_like_result = tuple(_map(_dummy_remat_result, avals_out))
  carry_init = (np.int32(0), dummies_like_result, args)
  def cond(carry):
    counter, _, _ = carry
    return counter < lax.rng_uniform(np.int32(1), np.int32(2), shape=())

  def body(carry):
    counter, _, args = carry
    results = core.eval_jaxpr(jaxpr, (), *args)
    return (counter + 1, tuple(results), args)

  carry_res = while_loop(cond, body, carry_init)
  return carry_res[1]


def _remat_translation_using_opt_barrier(*args, jaxpr: core.Jaxpr):
  args = _optimization_barrier(args)
  return core.eval_jaxpr(jaxpr, (), *args)


def remat_impl(*args,
               call_jaxpr: Optional[core.Jaxpr] = None,
               jaxpr: Optional[core.Jaxpr] = None,
               prevent_cse: bool, differentiated: bool,
               policy,
               is_gpu_platform: bool = False,
               concrete: bool = False,
               name: str = "checkpoint"):
  # Support either "jaxpr" (for remat2) and "call_jaxpr" (for remat)
  # name is not passed for remat2, defaults to "checkpoint"
  # TODO: remove call_jaxpr once we drop the remat call primitive
  if jaxpr is None:
    jaxpr = call_jaxpr
  assert jaxpr is not None
  assert not jaxpr.constvars

  del concrete, policy  # Unused.
  if differentiated and prevent_cse:
    if config.jax_remat_opt_barrier:
      translation_rule = _remat_translation_using_opt_barrier
    elif is_gpu_platform:
      translation_rule = _remat_translation_using_while
    else:
      translation_rule = _remat_translation_using_cond
  else:
    translation_rule = lambda *args, jaxpr: core.eval_jaxpr(jaxpr, (), *args)

  return jax.named_call(translation_rule, name=wrap_name(name, "remat"))(*args, jaxpr=jaxpr)

for remat_primitive in (pe.remat_call_p, ad_checkpoint.remat_p):  # type: ignore
  mlir.register_lowering(remat_primitive,
                         mlir.lower_fun(remat_impl, multiple_results=True))
  mlir.register_lowering(remat_primitive,
                         mlir.lower_fun(partial(remat_impl,
                                                is_gpu_platform=True),
                                        multiple_results=True),
                         platform="gpu")


def _optimization_barrier_abstract_eval(*args):
  return args


def _optimization_barrier_lowering_rule(ctx, *args):
  barrier_types = _map(mlir.aval_to_ir_types, ctx.avals_in)
  flat_barrier_types = util.flatten(barrier_types)

  flat_args = mlir.flatten_lowering_ir_args(args)
  barrier_op = mhlo.OptimizationBarrierOp(flat_barrier_types, flat_args)
  return util.unflatten(barrier_op.results, _map(len, barrier_types))


def _optimization_barrier(arg):
  flat_args, treedef = tree_flatten(arg)
  return tree_unflatten(treedef, optimization_barrier_p.bind(*flat_args))


optimization_barrier_p = core.Primitive('optimization_barrier')
optimization_barrier_p.multiple_results = True
optimization_barrier_p.def_impl(
    partial(xla.apply_primitive, optimization_barrier_p))
optimization_barrier_p.def_abstract_eval(_optimization_barrier_abstract_eval)
mlir.register_lowering(optimization_barrier_p,
                       _optimization_barrier_lowering_rule)
