# Copyright 2025 The JAX Authors.
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
"""Alternative API definitions for repros.

The functions in this module are alternative JAX APIs that help the repro
machinery to handle caches.

For example, instead of:

     f_jit = jax.jit(f)
     ra = f_jit(a)
     rb = f_jit(b)

we prefer:

     ra = jax_jit_call(f, a)
     rb = jax_jit_call(f, b)

because the latter can make sure to create a fresh wrapper for the argument
`f`, which will result in foiling the JAX tracing caches, and ensuring that
each wrapper has at most one invocation from which we generate the repro.

The regular JAX APIs are rewritten in terms of these APIs by a system of
trampolines (see `repro.boundary_trampolines`).

Additionally, some of the wrappers do quite a bit more, e.g., `jax_fuser_fusible`.

In order to avoid infinite recursion, the definitions below use
`core.bypass_repro_wrapper` to ensure that when we call into the real JAX
APIs we do not use the repro machinery.

This file contains most `import` statements locally, to avoid circular imports.
"""

from functools import partial
from typing import Any, Callable

from jax._src import core
from jax._src import tree_util

def _jax_jit_call_map_user_funcs(to_apply, *args, **_):
  # jit sometimes takes static args that may be callables, e.g., jnp.float32,
  # so we specify precisely that only the first arg is a user function.
  return tuple((to_apply(args[0]), *args[1:]))

@partial(core.repro_boundary, api_name="jax_jit_call",
         map_user_funcs=_jax_jit_call_map_user_funcs)
def jax_jit_call(f: Callable, trans_args: tuple[Any,...],
                 trans_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api
  return core.bypass_repro_wrapper(api.jit)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(core.repro_boundary, api_name="jax_jit_aot_trace_call",
         map_user_funcs=_jax_jit_call_map_user_funcs)
def jax_jit_aot_trace_call(fun, jit_args: tuple[Any], jit_kwargs: dict[str, Any],
                           *args, **kwargs):
  from jax._src import api
  jit_new = core.bypass_repro_wrapper(api.jit)(fun, *jit_args, **jit_kwargs)
  return jit_new.trace(*args, **kwargs)


@partial(core.repro_boundary, api_name="jax_jit_aot_lower_call",
         map_user_funcs=_jax_jit_call_map_user_funcs)
def jax_jit_aot_lower_call(fun, jit_args: tuple[Any], jit_kwargs: dict[str, Any],
                           *args, **kwargs):
  from jax._src import api
  jit_new = core.bypass_repro_wrapper(api.jit)(fun, *jit_args, **jit_kwargs)
  return jit_new.lower(*args, **kwargs)


@partial(core.repro_boundary, api_name="jax_grad_call")
def jax_grad_call(f: Callable, trans_args: tuple[Any], trans_kwargs: dict[str, Any],
                  *args, **kwargs):
  from jax._src import api
  return core.bypass_repro_wrapper(api.grad)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(core.repro_boundary, api_name="jax_saved_input_vjp")
def jax_saved_input_vjp(f: Callable, *args, **kwargs):
  from jax._src import api
  return core.bypass_repro_wrapper(api.saved_input_vjp)(f, *args, **kwargs)


@partial(core.repro_boundary, api_name="jax_linear_transpose_call")
def jax_linear_transpose_call(f: Callable, trans_args: tuple[Any, ...], trans_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api
  return core.bypass_repro_wrapper(api.linear_transpose)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(core.repro_boundary, api_name="jax_jacfwd_call")
def jax_jacfwd_call(f: Callable, trans_args: tuple[Any, ...], trans_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api
  return core.bypass_repro_wrapper(api.jacfwd)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(core.repro_boundary, api_name="jax_jacrev_call")
def jax_jacrev_call(f: Callable, trans_args: tuple[Any, ...], trans_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api
  return core.bypass_repro_wrapper(api.jacrev)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(core.repro_boundary, api_name="jax_checkpoint_call")
def jax_checkpoint_call(f: Callable, trans_args: tuple[Any, ...], trans_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import ad_checkpoint
  return core.bypass_repro_wrapper(ad_checkpoint.remat)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(core.repro_boundary, api_name="jax_value_and_grad_call")
def jax_value_and_grad_call(f: Callable, value_and_grad_args: tuple[Any], value_and_grad_kwargs: dict[str, Any],
                  *args, **kwargs):
  from jax._src import api
  return core.bypass_repro_wrapper(api.value_and_grad)(f, *value_and_grad_args, **value_and_grad_kwargs)(*args, **kwargs)

@partial(core.repro_boundary, api_name="jax_vmap_call")
def jax_vmap_call(f: Callable, trans_args: tuple[Any, ...], trans_kwargs: dict[str, Any],
                  *args, **kwargs):
  from jax._src import api
  return core.bypass_repro_wrapper(api.vmap)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(core.repro_boundary, api_name="jax_shard_map_call")
def jax_shard_map_call(f: Callable, trans_args: tuple[Any, ...], trans_kwargs: dict[str, Any],
                      *args, **kwargs):
  from jax._src import shard_map
  return core.bypass_repro_wrapper(shard_map._shard_map)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(core.repro_boundary, api_name="jax_pmap_call")
def jax_pmap_call(f: Callable, trans_args: tuple[Any, ...], trans_kwargs: dict[str, Any],
                  *args, **kwargs):
  from jax._src import api
  return core.bypass_repro_wrapper(api.pmap)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(core.repro_boundary, api_name="jax_custom_jvp_call")
def jax_custom_jvp_call(fun, cjvp_kwargs: dict[str, Any], defjvp_kwargs: dict[str, Any],
                        *fun_jvps_and_args, uses_defjvps: bool, jvps_count: int, **kwargs):
  from jax._src import custom_derivatives
  cjvp_new = custom_derivatives.custom_jvp(fun, **cjvp_kwargs)
  if uses_defjvps:
    cjvp_new.defjvps(*fun_jvps_and_args[:jvps_count])
  else:
    assert jvps_count == 1
    cjvp_new.defjvp(fun_jvps_and_args[0], **defjvp_kwargs)
  return core.bypass_repro_wrapper(cjvp_new.__call__)(cjvp_new, *fun_jvps_and_args[jvps_count:], **kwargs)


@partial(core.repro_boundary, api_name="jax_custom_vjp_call")
def jax_custom_vjp_call(fun: Callable, fwd: Callable, bwd: Callable,
                        custom_vjp_kwargs, *args, **kwargs):
  from jax._src import custom_derivatives
  cvjp = custom_derivatives.custom_vjp(fun, **custom_vjp_kwargs)
  cvjp.defvjp(fwd, bwd)
  return core.bypass_repro_wrapper(cvjp.__call__)(cvjp, *args, **kwargs)


@partial(core.repro_boundary, api_name="jax_pallas_call")
def jax_pallas_call(kernel: Callable, out_shape, pl_call_kwargs, *args):
  from jax._src.pallas import pallas_call
  return core.bypass_repro_wrapper(pallas_call._pallas_call)(kernel, out_shape, **pl_call_kwargs)(*args)


@partial(core.repro_boundary, api_name="jax_fuser_fuse")
def jax_fuser_fuse(fun: Callable, trans_kwargs: dict[str, Any],
                  *args):
  from jax._src.pallas import fuser
  return core.bypass_repro_wrapper(fuser.fuse)(fun, **trans_kwargs)(*args)


@partial(core.repro_boundary, api_name="jax_fuser_fusible")
def jax_fuser_fusible(fun_1: Callable, fun_2: Callable,
                      trans_kwargs: dict[str, Any],
                      *args):
  # Since the original `fuser.fusible` traces the function twice, once
  # with the "output" fusion being `None` and once with actual functions,
  # we create two separate functions `fun_1` and `fun_2` that we then
  # merge into a `joint_fun`.
  from jax._src.pallas import fuser
  def wrap_fusion(fn: fuser.Fusion | Any) -> fuser.Fusion:
    if not isinstance(fn, fuser.Fusion): return fn
    res = core.wrap_repro_boundary_callable(fn, is_jax=True)
    if res is not fn:
      res.shape = fn.shape
    return res
  def joint_fun(*args, **kwargs):
    wrapped_args = tree_util.tree_map(wrap_fusion, args)
    if args[-1] is None:
      return fun_1(*wrapped_args, **kwargs)
    else:
      return fun_2(*wrapped_args, **kwargs)
  return core.bypass_repro_wrapper(fuser.fusible)(joint_fun, **trans_kwargs)(*args)


@partial(core.repro_boundary, api_name="pallas_custom_fusion_call")
def pallas_custom_fusion_call(fun: Callable,
                              eval_rule: Callable,
                              pull_block_spec_rule: Callable,
                              push_block_spec_rule: Callable | None,
                              impl_rule: Callable | None,
                              *args, **kwargs):
    from jax._src.pallas import fuser
    cfus = fuser.custom_fusion(fun)
    cfus.def_eval_rule(eval_rule)
    cfus.def_push_block_spec(push_block_spec_rule)
    cfus.def_pull_block_spec(pull_block_spec_rule)
    cfus.def_pallas_impl(impl_rule)
    return core.bypass_repro_wrapper(cfus.__call__)(cfus, *args, **kwargs)
