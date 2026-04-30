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

See https://docs.jax.dev/en/latest/debugging/repro.html#trampolines.


In order to avoid infinite recursion, the definitions below use
`repro.bypass_wrapper` to ensure that when we call into the real JAX
APIs we do not use the repro machinery.

This file contains most `import` statements locally, to avoid circular imports.
"""

import contextlib
import copy
import dataclasses
from functools import partial
from typing import Any, Callable, Sequence

from jax._src import traceback_util
from jax._src import tree_util

if traceback_util.repro is not None:
  # For the cases when repros are run without repro installed
  repro_boundary = traceback_util.repro.boundary
else:
  def repro_boundary(f: Callable, **_): return f


def repro_bypass_wrapper(f: Callable) -> Callable:
  if traceback_util.repro is not None:
    from jax._src.repro import tracker  # type: ignore
    return tracker.bypass_wrapper(f)
  else:
    return f


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


@partial(repro_boundary, repro_api_name="jax_repro_collect")
def jax_repro_collect(f: Callable):
  return f()


@partial(repro_boundary, repro_api_name="jit_call")
def jit_call(f: Callable, ctx_mesh,
             jit_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api  # type: ignore
  with _jit_context(False, ctx_mesh):
    return repro_bypass_wrapper(api.jit)(f, **jit_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="pjit_call")
def pjit_call(f: Callable, ctx_mesh,
              jit_kwargs: dict[str, Any], *args, **kwargs):
  from jax.experimental import pjit  # type: ignore
  with _jit_context(True, ctx_mesh):
    return repro_bypass_wrapper(pjit.pjit)(f, **jit_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="jit_aot_trace_call")
def jit_aot_trace_call(fun, ctx_mesh, jit_kwargs: dict[str, Any],
                       *args, **kwargs):
  from jax._src import api  # type: ignore
  with _jit_context(False, ctx_mesh):
    jit_new = repro_bypass_wrapper(api.jit)(fun, **jit_kwargs)
    return jit_new.trace(*args, **kwargs)


@partial(repro_boundary, repro_api_name="pjit_aot_trace_call")
def pjit_aot_trace_call(fun, ctx_mesh, jit_kwargs: dict[str, Any],
                        *args, **kwargs):
  from jax.experimental import pjit  # type: ignore
  with _jit_context(True, ctx_mesh):
    jit_new = repro_bypass_wrapper(pjit.pjit)(fun, **jit_kwargs)
    return jit_new.trace(*args, **kwargs)


@partial(repro_boundary, repro_api_name="jit_aot_lower_call")
def jit_aot_lower_call(fun, ctx_mesh, jit_kwargs: dict[str, Any],
                       *args, **kwargs):
  from jax._src import api  # type: ignore
  with _jit_context(False, ctx_mesh):
    jit_new = repro_bypass_wrapper(api.jit)(fun, **jit_kwargs)
    return jit_new.lower(*args, **kwargs)


@partial(repro_boundary, repro_api_name="pjit_aot_lower_call")
def pjit_aot_lower_call(fun, ctx_mesh, jit_kwargs: dict[str, Any],
                        *args, **kwargs):
  from jax.experimental import pjit  # type: ignore
  with _jit_context(True, ctx_mesh):
    jit_new = repro_bypass_wrapper(pjit.pjit)(fun, **jit_kwargs)
    return jit_new.lower(*args, **kwargs)


@partial(repro_boundary, repro_api_name="grad_call")
def grad_call(f: Callable, api_args: tuple[Any],
                  api_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api  # type: ignore
  return repro_bypass_wrapper(api.grad)(
    f, *api_args, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="vjp")
def vjp(f: Callable, *primals, has_aux=False, reduce_axes=()):
  from jax._src import api  # type: ignore
  if has_aux:
    out, f_vjp, aux = repro_bypass_wrapper(api.vjp)(
      f, *primals, has_aux=has_aux, reduce_axes=reduce_axes)
    return out, repro_boundary(f_vjp, is_user=False), aux
  else:
    out, f_vjp = repro_bypass_wrapper(api.vjp)(
      f, *primals, has_aux=has_aux, reduce_axes=reduce_axes)
    return out, repro_boundary(f_vjp, is_user=False)


@partial(repro_boundary, repro_api_name="custom_gradient_call")
def custom_gradient_call(f: Callable, api_args: tuple[Any, ...],
                             api_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import custom_derivatives  # type: ignore
  return repro_bypass_wrapper(custom_derivatives.custom_gradient)(
    f, *api_args, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="linear_transpose_call")
def linear_transpose_call(f: Callable, api_args: tuple[Any, ...],
                              api_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api  # type: ignore
  return repro_bypass_wrapper(api.linear_transpose)(
    f, *api_args, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="linearize")
def linearize(f: Callable, *args, **kwargs):
  from jax._src import api  # type: ignore
  out, f_vjp = repro_bypass_wrapper(api.linearize)(f, *args, **kwargs)
  return out, repro_boundary(f_vjp, is_user=False)


@partial(repro_boundary, repro_api_name="jacfwd_call")
def jacfwd_call(f: Callable, api_args: tuple[Any, ...],
                    api_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api  # type: ignore
  return repro_bypass_wrapper(api.jacfwd)(
    f, *api_args, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="jacrev_call")
def jacrev_call(f: Callable, api_args: tuple[Any, ...],
                    api_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api  # type: ignore
  return repro_bypass_wrapper(api.jacrev)(
    f, *api_args, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="hessian_call")
def hessian_call(f: Callable, api_args: tuple[Any, ...],
                     api_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api  # type: ignore
  return repro_bypass_wrapper(api.hessian)(
    f, *api_args, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="checkpoint_call")
def checkpoint_call(f: Callable, api_args: tuple[Any, ...],
                        api_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import ad_checkpoint  # type: ignore
  return repro_bypass_wrapper(ad_checkpoint.checkpoint)(
    f, *api_args, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="value_and_grad_call")
def value_and_grad_call(f: Callable, api_args: tuple[Any],
                          api_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api  # type: ignore
  return repro_bypass_wrapper(api.value_and_grad)(
    f, *api_args, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="vmap_call")
def vmap_call(f: Callable, api_args: tuple[Any, ...],
                  api_kwargs : dict[str, Any], *args, **kwargs):
  from jax._src import api  # type: ignore
  return repro_bypass_wrapper(api.vmap)(
    f, *api_args, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="shard_map_call")
def shard_map_call(f: Callable, api_args: tuple[Any, ...],
                       api_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import shard_map  # type: ignore
  return repro_bypass_wrapper(shard_map._shard_map)(
    f, *api_args, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="custom_jvp_call",
         map_user_func_args=(
    lambda toapply, fun, cjvp_kwargs, defjvp_kwargs, *fun_jvps_and_args, uses_defjvps, jvps_count: (
          (toapply(fun), cjvp_kwargs, defjvp_kwargs,
           *(map(toapply, fun_jvps_and_args[:jvps_count])),
           *fun_jvps_and_args[jvps_count:]),
          {"uses_defjvps": uses_defjvps, "jvps_count": jvps_count})))
def custom_jvp_call(fun, cjvp_kwargs: dict[str, Any], defjvp_kwargs: dict[str, Any],
                    *fun_jvps_and_args, uses_defjvps: bool, jvps_count: int):
  from jax._src import custom_derivatives  # type: ignore
  cjvp_new = custom_derivatives.custom_jvp(fun, **cjvp_kwargs)
  if uses_defjvps:
    cjvp_new.defjvps(*fun_jvps_and_args[:jvps_count])
  else:
    assert jvps_count == 1
    cjvp_new.defjvp(fun_jvps_and_args[0], **defjvp_kwargs)
  return repro_bypass_wrapper(cjvp_new.__call__)(cjvp_new, *fun_jvps_and_args[jvps_count:])


@partial(repro_boundary, repro_api_name="custom_vjp_call",
         map_user_func_args=(
             lambda toapply, fun, fwd, bwd, custom_vjp_kwargs, *args, **kwargs: (
                   (toapply(fun), toapply(fwd), toapply(bwd),
                    custom_vjp_kwargs, *args),
                   kwargs)))
def custom_vjp_call(fun: Callable, fwd: Callable, bwd: Callable,
                    custom_vjp_kwargs, *args):
  from jax._src import custom_derivatives  # type: ignore
  cvjp = custom_derivatives.custom_vjp(fun, **custom_vjp_kwargs)
  cvjp.defvjp(fwd, bwd)
  return repro_bypass_wrapper(cvjp.__call__)(cvjp, *args)


def _jax_cond_map_user_func_args(toapply, pred, true_fun: Callable,
                                 false_fun: Callable, *operands, **kwargs):
  if not (callable(true_fun) and callable(false_fun)):
    # try falling back to the old, deprecated version of `cond`
    if callable(false_fun) and len(operands) == 2 and callable(operands[1]):
      x_true, f_true, x_false, f_false = true_fun, false_fun, *operands
      return (pred, x_true, toapply(f_true), x_false, toapply(f_false)), kwargs
    else:
      raise NotImplementedError("Unrecognized old-style (?) lax.cond")
  else:
    return (pred, toapply(true_fun), toapply(false_fun), *operands), kwargs


@partial(repro_boundary, repro_api_name="cond", map_user_func_args=_jax_cond_map_user_func_args)
def cond(pred, true_fun: Callable, false_fun: Callable, *operands, **kwargs):
  from jax._src.lax.control_flow.conditionals import cond  # type: ignore
  return repro_bypass_wrapper(cond)(pred, true_fun, false_fun, *operands, **kwargs)


@partial(repro_boundary, repro_api_name="switch",
         map_user_func_args=lambda toapply, index, branches, operands, **kwargs: (
             (index, tuple(map(toapply, branches)), operands), kwargs))
def switch(index, branches: Sequence[Callable], operands: Sequence[Any], **kwargs):
  from jax._src.lax.control_flow.conditionals import _switch_internal  # type: ignore
  return repro_bypass_wrapper(_switch_internal)(index, branches, operands, **kwargs)


@partial(repro_boundary, repro_api_name="fori_loop",
         map_user_func_args=lambda toapply, lower, upper, body_fun, *args, **kwargs: (
             (lower, upper, toapply(body_fun), *args), kwargs))
def fori_loop(lower: int, upper: int,
              body_fun: Callable[[int, Any], Any], init_val: Any, **kwargs):
  from jax._src.lax.control_flow.loops import fori_loop  # type: ignore
  return repro_bypass_wrapper(fori_loop)(lower, upper, body_fun, init_val, **kwargs)


@partial(repro_boundary, repro_api_name="while_loop",
         map_user_func_args=lambda toapply, cond_fun, body_fun, *args, **kwargs: (
             (toapply(cond_fun), toapply(body_fun), *args), kwargs))
def while_loop(cond_fun: Callable[[Any], Any],
               body_fun: Callable[[Any], Any],
               init_val: Any, **kwargs):
  from jax._src.lax.control_flow.loops import while_loop  # type: ignore
  return repro_bypass_wrapper(while_loop)(cond_fun, body_fun, init_val, **kwargs)

# TODO: move this out, it is about Flax
@partial(repro_boundary, repro_api_name="flax_axes_scan_call")
def flax_axes_scan_call(body_fun: Callable,
                        scan_args: tuple[Any, ...], scan_kwargs: dict[str, Any],
                        *args, **kwargs):
  try:
    import flax  # type: ignore  # noqa: F401
    from flax.core import axes_scan  # type: ignore
    return repro_bypass_wrapper(axes_scan.scan)(body_fun, *scan_args, **scan_kwargs)(*args, **kwargs)
  except ImportError:
    raise NotImplementedError("flax.core.axes_scan.scan is not available.")


@partial(repro_boundary, repro_api_name="export_call")
def export_call(fun: Callable, jit_kwargs: dict[str, Any],
                exp_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api  # type: ignore
  from jax._src.export import _export  # type: ignore
  fun_jit = repro_bypass_wrapper(api.jit)(fun, **jit_kwargs)
  return repro_bypass_wrapper(_export._export_internal)(fun_jit, **exp_kwargs)(*args, **kwargs)


# TODO(necula): this is needed only for jax2tf
@partial(repro_boundary, repro_api_name="export_internal_call")
def export_internal_call(fun: Callable, jit_kwargs: dict[str, Any],
                         exp_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api  # type: ignore
  from jax._src.export import _export  # type: ignore
  fun_jit = repro_bypass_wrapper(api.jit)(fun, **jit_kwargs)
  return repro_bypass_wrapper(_export._export_internal)(fun_jit, **exp_kwargs)(*args, **kwargs)
