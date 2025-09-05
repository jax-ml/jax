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

See https://docs.jax.dev/en/latest/debugging/repro.html#handling-caches.


In order to avoid infinite recursion, the definitions below use
`repro.bypass_wrapper` to ensure that when we call into the real JAX
APIs we do not use the repro machinery.

This file contains most `import` statements locally, to avoid circular imports.
"""

import contextlib
from functools import partial
from typing import Any, Callable

from jax._src import traceback_util
from jax._src import tree_util


def repro_bypass_wrapper(f: Callable) -> Callable:
  from jax._src.repro import tracker
  return tracker.bypass_wrapper(f)


def _jit_context(is_pjit: bool, ctx_mesh):
  from jax._src.sharding_impls import set_mesh  # type: ignore
  from jax._src import mesh as mesh_lib  # type: ignore
  if ctx_mesh is not None:
    if is_pjit:
      # We'd like to transition to set_mesh, but we get an error if this
      # is done under a jit.
      return ctx_mesh
    prev_ctx = mesh_lib.get_concrete_mesh()
    if prev_ctx == ctx_mesh:
      return contextlib.nullcontext()  # Already set, cannot set again
    return set_mesh(ctx_mesh)
  else:
    return contextlib.nullcontext()


@partial(traceback_util.repro.boundary, api_name="jax_repro_collect")
def jax_repro_collect(f: Callable):
  return f()


@partial(traceback_util.repro.boundary, api_name="jax_jit_call")
def jax_jit_call(f: Callable, ctx_mesh,
                 jit_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api  # type: ignore
  with _jit_context(False, ctx_mesh):
    return repro_bypass_wrapper(api.jit)(f, **jit_kwargs)(*args, **kwargs)


@partial(traceback_util.repro.boundary, api_name="jax_pjit_call")
def jax_pjit_call(f: Callable, ctx_mesh,
                  jit_kwargs: dict[str, Any], *args, **kwargs):
  from jax.experimental import pjit  # type: ignore
  with _jit_context(True, ctx_mesh):
    return repro_bypass_wrapper(pjit.pjit)(f, **jit_kwargs)(*args, **kwargs)


@partial(traceback_util.repro.boundary, api_name="jax_jit_aot_trace_call")
def jax_jit_aot_trace_call(fun, ctx_mesh, jit_kwargs: dict[str, Any],
                           *args, **kwargs):
  from jax._src import api  # type: ignore
  with _jit_context(False, ctx_mesh):
    jit_new = repro_bypass_wrapper(api.jit)(fun, **jit_kwargs)
    return jit_new.trace(*args, **kwargs)


@partial(traceback_util.repro.boundary, api_name="jax_pjit_aot_trace_call")
def jax_pjit_aot_trace_call(fun, ctx_mesh, jit_kwargs: dict[str, Any],
                            *args, **kwargs):
  from jax.experimental import pjit  # type: ignore
  with _jit_context(True, ctx_mesh):
    jit_new = repro_bypass_wrapper(pjit.pjit)(fun, **jit_kwargs)
    return jit_new.trace(*args, **kwargs)


@partial(traceback_util.repro.boundary, api_name="jax_jit_aot_lower_call")
def jax_jit_aot_lower_call(fun, ctx_mesh, jit_kwargs: dict[str, Any],
                           *args, **kwargs):
  from jax._src import api  # type: ignore
  with _jit_context(False, ctx_mesh):
    jit_new = repro_bypass_wrapper(api.jit)(fun, **jit_kwargs)
    return jit_new.lower(*args, **kwargs)


@partial(traceback_util.repro.boundary, api_name="jax_pjit_aot_lower_call")
def jax_pjit_aot_lower_call(fun, ctx_mesh, jit_kwargs: dict[str, Any],
                            *args, **kwargs):
  from jax.experimental import pjit  # type: ignore
  with _jit_context(True, ctx_mesh):
    jit_new = repro_bypass_wrapper(pjit.pjit)(fun, **jit_kwargs)
    return jit_new.lower(*args, **kwargs)


@partial(traceback_util.repro.boundary, api_name="jax_grad_call")
def jax_grad_call(f: Callable, trans_args: tuple[Any], trans_kwargs: dict[str, Any],
                  *args, **kwargs):
  from jax._src import api  # type: ignore
  return repro_bypass_wrapper(api.grad)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(traceback_util.repro.boundary, api_name="jax_vjp")
def jax_vjp(f: Callable, *primals, has_aux=False, reduce_axes=()):
  from jax._src import api  # type: ignore
  if has_aux:
    out, f_vjp, aux = repro_bypass_wrapper(api.vjp)(f, *primals, has_aux=has_aux, reduce_axes=reduce_axes)
    return out, traceback_util.repro.boundary(f_vjp, is_user=False), aux
  else:
    out, f_vjp = repro_bypass_wrapper(api.vjp)(f, *primals, has_aux=has_aux, reduce_axes=reduce_axes)
    return out, traceback_util.repro.boundary(f_vjp, is_user=False)


@partial(traceback_util.repro.boundary, api_name="jax_saved_input_vjp")
def jax_saved_input_vjp(f: Callable, which, *primals, allow_unused=True, allow_opaque=True):
  from jax._src import api  # type: ignore
  out, f_vjp = repro_bypass_wrapper(api.saved_input_vjp)(f, which, *primals, allow_unused=allow_unused, allow_opaque=allow_opaque)
  return out, traceback_util.repro.boundary(f_vjp, is_user=False)


@partial(traceback_util.repro.boundary, api_name="jax_linear_transpose_call")
def jax_linear_transpose_call(f: Callable, trans_args: tuple[Any, ...], trans_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api  # type: ignore
  return repro_bypass_wrapper(api.linear_transpose)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(traceback_util.repro.boundary, api_name="jax_linearize")
def jax_linearize(f: Callable, *args, **kwargs):
  from jax._src import api  # type: ignore
  out, f_vjp = repro_bypass_wrapper(api.linearize)(f, *args, **kwargs)
  return out, traceback_util.repro.boundary(f_vjp, is_user=False)


@partial(traceback_util.repro.boundary, api_name="jax_jacfwd_call")
def jax_jacfwd_call(f: Callable, trans_args: tuple[Any, ...], trans_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api  # type: ignore
  return repro_bypass_wrapper(api.jacfwd)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(traceback_util.repro.boundary, api_name="jax_jacrev_call")
def jax_jacrev_call(f: Callable, trans_args: tuple[Any, ...], trans_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api  # type: ignore
  return repro_bypass_wrapper(api.jacrev)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(traceback_util.repro.boundary, api_name="jax_hessian_call")
def jax_hessian_call(f: Callable, trans_args: tuple[Any, ...], trans_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api  # type: ignore
  return repro_bypass_wrapper(api.hessian)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(traceback_util.repro.boundary, api_name="jax_checkpoint_call")
def jax_checkpoint_call(f: Callable, trans_args: tuple[Any, ...], trans_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import ad_checkpoint  # type: ignore
  return repro_bypass_wrapper(ad_checkpoint.checkpoint)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(traceback_util.repro.boundary, api_name="jax_value_and_grad_call")
def jax_value_and_grad_call(f: Callable, value_and_grad_args: tuple[Any], value_and_grad_kwargs: dict[str, Any],
                  *args, **kwargs):
  from jax._src import api  # type: ignore
  return repro_bypass_wrapper(api.value_and_grad)(f, *value_and_grad_args, **value_and_grad_kwargs)(*args, **kwargs)


@partial(traceback_util.repro.boundary, api_name="jax_vmap_call")
def jax_vmap_call(f: Callable, trans_args: tuple[Any, ...], trans_kwargs: dict[str, Any],
                  *args, **kwargs):
  from jax._src import api  # type: ignore
  return repro_bypass_wrapper(api.vmap)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(traceback_util.repro.boundary, api_name="jax_shard_map_call")
def jax_shard_map_call(f: Callable, trans_args: tuple[Any, ...], trans_kwargs: dict[str, Any],
                      *args, **kwargs):
  from jax._src import shard_map  # type: ignore
  return repro_bypass_wrapper(shard_map._shard_map)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(traceback_util.repro.boundary, api_name="jax_pmap_call")
def jax_pmap_call(f: Callable, trans_args: tuple[Any, ...], trans_kwargs: dict[str, Any],
                  *args, **kwargs):
  from jax._src import api  # type: ignore
  return repro_bypass_wrapper(api.pmap)(f, *trans_args, **trans_kwargs)(*args, **kwargs)


@partial(traceback_util.repro.boundary, api_name="jax_custom_jvp_call",
         map_user_func_args=(
    lambda toapply, fun, cjvp_kwargs, defjvp_kwargs, *fun_jvps_and_args, uses_defjvps, jvps_count: (
          toapply(fun), cjvp_kwargs, defjvp_kwargs,
          *(map(toapply, fun_jvps_and_args[:jvps_count])),
          *fun_jvps_and_args[jvps_count:])))
def jax_custom_jvp_call(fun, cjvp_kwargs: dict[str, Any], defjvp_kwargs: dict[str, Any],
                        *fun_jvps_and_args, uses_defjvps: bool, jvps_count: int):
  from jax._src import custom_derivatives  # type: ignore
  cjvp_new = custom_derivatives.custom_jvp(fun, **cjvp_kwargs)
  if uses_defjvps:
    cjvp_new.defjvps(*fun_jvps_and_args[:jvps_count])
  else:
    assert jvps_count == 1
    cjvp_new.defjvp(fun_jvps_and_args[0], **defjvp_kwargs)
  return repro_bypass_wrapper(cjvp_new.__call__)(cjvp_new, *fun_jvps_and_args[jvps_count:])


@partial(traceback_util.repro.boundary, api_name="jax_custom_vjp_call",
         map_user_func_args=(
             lambda toapply, fun, fwd, bwd, custom_vjp_kwargs, *args: (
                   toapply(fun), toapply(fwd), toapply(bwd), custom_vjp_kwargs, *args)))
def jax_custom_vjp_call(fun: Callable, fwd: Callable, bwd: Callable,
                        custom_vjp_kwargs, *args):
  from jax._src import custom_derivatives  # type: ignore
  cvjp = custom_derivatives.custom_vjp(fun, **custom_vjp_kwargs)
  cvjp.defvjp(fwd, bwd)
  return repro_bypass_wrapper(cvjp.__call__)(cvjp, *args)


@partial(traceback_util.repro.boundary, api_name="jax_pallas_call")
def jax_pallas_call(kernel: Callable, out_shape, pl_call_kwargs, *args):
  from jax._src.pallas import pallas_call  # type: ignore
  return repro_bypass_wrapper(pallas_call._pallas_call)(kernel, out_shape, **pl_call_kwargs)(*args)


@partial(traceback_util.repro.boundary, api_name="jax_fuser_fuse")
def jax_fuser_fuse(fun: Callable, trans_kwargs: dict[str, Any],
                  *args):
  from jax.experimental.pallas import fuser  # type: ignore
  return repro_bypass_wrapper(fuser.fuse)(fun, **trans_kwargs)(*args)


@partial(traceback_util.repro.boundary, api_name="jax_fuser_fusible",
         map_user_func_args=lambda toapply, fun_1, fun_2, trans_kwargs, *args:(
          toapply(fun_1), toapply(fun_2), trans_kwargs, *args))
def jax_fuser_fusible(fun_1: Callable, fun_2: Callable,
                      trans_kwargs: dict[str, Any],
                      *args):
  # Since the original `fuser.fusible` traces the function twice, once
  # with the "output" fusion being `None` and once with actual functions,
  # we create two separate functions `fun_1` and `fun_2` that we then
  # merge into a `joint_fun`.
  from jax._src.repro import tracker  # type: ignore
  from jax.experimental.pallas import fuser  # type: ignore
  def wrap_fusion(fn: fuser.Fusion | Any) -> fuser.Fusion:
    if not isinstance(fn, fuser.Fusion): return fn
    res = tracker.wrap_callable(fn, is_user=False)
    if res is not fn:
      res.shape = fn.shape
    return res
  def joint_fun(*args, **kwargs):
    wrapped_args = tree_util.tree_map(wrap_fusion, args)
    if args[-1] is None:
      return fun_1(*wrapped_args, **kwargs)
    else:
      return fun_2(*wrapped_args, **kwargs)
  return repro_bypass_wrapper(fuser.fusible)(joint_fun, **trans_kwargs)(*args)


@partial(traceback_util.repro.boundary, api_name="pallas_custom_fusion_call",
         map_user_func_args=lambda toapply, fun, eval_rule, pull_block_spec_rule, push_block_spec_rule, impl_rule, *args:(
             toapply(fun), toapply(eval_rule), toapply(pull_block_spec_rule),
             toapply(push_block_spec_rule), toapply(impl_rule),
             *args))
def pallas_custom_fusion_call(fun: Callable,
                              eval_rule: Callable,
                              pull_block_spec_rule: Callable,
                              push_block_spec_rule: Callable | None,
                              impl_rule: Callable | None,
                              *args):
  from jax.experimental.pallas import fuser  # type: ignore
  cfus = fuser.custom_fusion(fun)
  cfus.def_eval_rule(eval_rule)
  cfus.def_push_block_spec(push_block_spec_rule)  # type: ignore
  cfus.def_pull_block_spec(pull_block_spec_rule)
  cfus.def_pallas_impl(impl_rule)
  return repro_bypass_wrapper(cfus.__call__)(cfus, *args)


@partial(traceback_util.repro.boundary, api_name="jax_export_call")
def jax_export_call(fun: Callable, jit_kwargs: dict[str, Any], exp_kwargs: dict[str, Any],
                    *args, **kwargs):
  from jax._src import api  # type: ignore
  from jax.export import export  # type: ignore
  fun_jit = repro_bypass_wrapper(api.jit)(fun, **jit_kwargs)
  return repro_bypass_wrapper(export)(fun_jit, **exp_kwargs)(*args, **kwargs)


# TODO: move this out, it is about Flax
@partial(traceback_util.repro.boundary, api_name="jax_flax_axes_scan_call")
def jax_flax_axes_scan_call(body_fun: Callable,
                            scan_args: tuple[Any, ...], scan_kwargs: dict[str, Any],
                            *args, **kwargs):
  try:
    import flax  # type: ignore  # noqa: F401
    from flax.core import axes_scan  # type: ignore
    return repro_bypass_wrapper(axes_scan.scan)(body_fun, *scan_args, **scan_kwargs)(*args, **kwargs)
  except ImportError:
    raise NotImplementedError("flax.core.axes_scan.scan is not available.")
