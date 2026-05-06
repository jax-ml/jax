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


def _pallas_map_blockspec(toapply: Callable, bs):
  from jax.experimental.pallas import BlockSpec  # type: ignore
  from jax._src.pallas.core import _IndexMapFunc  # type: ignore

  if isinstance(bs, BlockSpec):
    if bs.index_map is not None:
      orig_map = bs.index_map.index_map if isinstance(bs.index_map, _IndexMapFunc) else bs.index_map
      return dataclasses.replace(bs, index_map=toapply(orig_map))
  return bs


def _pallas_map_gridspec(toapply: Callable, gs):
  from jax._src.pallas.core import GridSpec  # type: ignore

  if isinstance(gs, GridSpec):
    new_fields = {}
    new_fields["in_specs"] = \
      tree_util.tree_map(partial(_pallas_map_blockspec, toapply), gs.in_specs)
    new_fields["out_specs"] = \
      tree_util.tree_map(partial(_pallas_map_blockspec, toapply), gs.out_specs)
    new_spec = copy.copy(gs)
    for k, v in new_fields.items():
      object.__setattr__(new_spec, k, v)
    return new_spec
  return gs


def _pallas_call_call_map_user_func_args(toapply, kernel: Callable,
                                         api_args: tuple[Any, ...],
                                         api_kwargs: dict[str, Any],
                                         *args, **kwargs):
  new_api_kwargs = dict(api_kwargs)
  if (in_specs := new_api_kwargs.get("in_specs")) is not None:
    new_api_kwargs["in_specs"] = \
      tree_util.tree_map(partial(_pallas_map_blockspec, toapply), in_specs)
  if (out_specs := new_api_kwargs.get("out_specs")) is not None:
    new_api_kwargs["out_specs"] = \
        tree_util.tree_map(partial(_pallas_map_blockspec, toapply), out_specs)
  if (grid_spec := new_api_kwargs.get("grid_spec")) is not None:
    new_api_kwargs["grid_spec"] = _pallas_map_gridspec(toapply, grid_spec)

  return (toapply(kernel), api_args, new_api_kwargs, *args), kwargs


@partial(repro_boundary, repro_api_name="pallas_call_call",
         map_user_func_args=_pallas_call_call_map_user_func_args)
def pallas_call_call(kernel: Callable, api_args: tuple[Any, ...],
                     api_kwargs: dict[str, Any], *args, **kwargs):

  from jax._src.pallas import pallas_call  # type: ignore
  return repro_bypass_wrapper(pallas_call.pallas_call)(
    kernel, *api_args, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="pallas_core_map")
def pallas_core_map(f: Callable, mesh, core_map_kwargs):
  from jax._src.pallas import core as pallas_core  # type: ignore
  return repro_bypass_wrapper(pallas_core.core_map)(mesh, **core_map_kwargs)(f)


@partial(repro_boundary, repro_api_name="pallas_run_state_call")
def pallas_run_state_call(f: Callable, api_args: tuple[Any, ...],
                          api_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src.state import discharge  # type: ignore
  return repro_bypass_wrapper(discharge.run_state)(
    f, *api_args, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="pallas_gpu_kernel_call")
def pallas_gpu_kernel_call(body: Callable, api_args: tuple[Any, ...],
                           api_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src.pallas.mosaic_gpu import core as gpu_core  # type: ignore
  return repro_bypass_wrapper(gpu_core.kernel)(
    body, *api_args, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="pallas_kernel_call")
def pallas_kernel_call(body: Callable, kernel_kwargs: dict[str, Any],
                       *operands):
  from jax._src.pallas import helpers as pallas_helpers  # type: ignore
  return repro_bypass_wrapper(pallas_helpers.kernel)(
    body, **kernel_kwargs)(*operands)


@partial(repro_boundary, repro_api_name="pallas_parallel_loop_call")
def pallas_parallel_loop_call(body: Callable, api_args: tuple[Any, ...],
                              api_kwargs: dict[str, Any]):
  from jax._src.pallas.mosaic import sc_primitives  # type: ignore
  return repro_bypass_wrapper(sc_primitives.parallel_loop)(
    *api_args, **api_kwargs)(body)


@partial(repro_boundary, repro_api_name="pallas_gpu_emit_pipeline_call",
         map_user_func_args=_pallas_call_call_map_user_func_args)
def pallas_gpu_emit_pipeline_call(f: Callable, api_args: tuple[Any, ...],
                                 api_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src.pallas.mosaic_gpu import pipeline as gpu_pipeline  # type: ignore
  return repro_bypass_wrapper(gpu_pipeline.emit_pipeline)(
    f, *api_args, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="pallas_tpu_emit_pipeline_call",
         map_user_func_args=_pallas_call_call_map_user_func_args)
def pallas_tpu_emit_pipeline_call(f: Callable, api_args: tuple[Any, ...],
                                  api_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src.pallas.mosaic import pipeline as tpu_pipeline  # type: ignore
  return repro_bypass_wrapper(tpu_pipeline.emit_pipeline)(
    f, *api_args, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="pallas_tpu_emit_pipeline_with_allocations",
         map_user_func_args=_pallas_call_call_map_user_func_args)
def pallas_tpu_emit_pipeline_with_allocations(body: Callable, kwargs):
  from jax._src.pallas.mosaic import pipeline as tpu_pipeline  # type: ignore
  return repro_bypass_wrapper(tpu_pipeline.emit_pipeline_with_allocations)(body, **kwargs)


@partial(repro_boundary, repro_api_name="fuser_fuse")
def fuser_fuse(fun: Callable, api_kwargs: dict[str, Any], *args, **kwargs):
  from jax.experimental.pallas import fuser  # type: ignore
  return repro_bypass_wrapper(fuser.fuse)(fun, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="fuser_evaluate")
def fuser_evaluate(fun: Callable, trans_kwargs: dict[str, Any],
                   *args):
  from jax.experimental.pallas import fuser  # type: ignore
  return repro_bypass_wrapper(fuser.evaluate)(fun, **trans_kwargs)(*args)


@partial(repro_boundary, repro_api_name="fuser_fusible",
         map_user_func_args=lambda toapply, fun_1, fun_2, api_kwargs, *args, **kwargs:((
          toapply(fun_1), toapply(fun_2), api_kwargs, *args), kwargs))
def fuser_fusible(fun_1: Callable, fun_2: Callable,
                  api_kwargs: dict[str, Any],
                  *args, **kwargs):
  # Since the original `fuser.fusible` traces the function twice, once
  # with the "output" fusion being `None` and once with actual functions,
  # we create two separate functions `fun_1` and `fun_2` that we then
  # merge into a `joint_fun`.
  from jax._src.repro import tracker  # type: ignore
  from jax.experimental.pallas import fuser  # type: ignore
  def wrap_fusion(fn: fuser.Fusion | Any) -> fuser.Fusion:
    if not isinstance(fn, fuser.Fusion): return fn
    res = tracker.wrap_callable(fn, is_user=False)
    # TODO: maybe we can forward all properties?
    if res is not fn:
      res.dtype = fn.dtype
      res.shape = fn.shape
      res.type = fn.type
      res.out_type = fn.out_type
      res.in_dtype = fn.in_dtype
      res.in_shape = fn.in_shape
      res.in_type = fn.in_type
    return res
  def joint_fun(*args, **kwargs):
    wrapped_args = tree_util.tree_map(wrap_fusion, args)
    if not tree_util.tree_leaves(args[-1]):  # all None
      return fun_1(*wrapped_args, **kwargs)
    else:
      return fun_2(*wrapped_args, **kwargs)
  return repro_bypass_wrapper(fuser.fusible)(joint_fun, **api_kwargs)(*args)


@partial(repro_boundary, repro_api_name="pallas_custom_fusion_call",
         map_user_func_args=lambda toapply, fun, eval_rule, pull_block_spec_rule, push_block_spec_rule, impl_rule, *args, **kwargs:(
             (toapply(fun), toapply(eval_rule), toapply(pull_block_spec_rule),
              toapply(push_block_spec_rule), toapply(impl_rule),
              *args), kwargs))
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


@partial(repro_boundary, repro_api_name="fuser_pull_block_spec_call",
         map_user_func_args=lambda toapply, fun, api_args, api_kwargs, *args, **kwargs: (
             (toapply(fun),
              tree_util.tree_map(partial(_pallas_map_blockspec, toapply), api_args),
              tree_util.tree_map(partial(_pallas_map_blockspec, toapply), api_kwargs),
              *args), kwargs))
def fuser_pull_block_spec_call(fun: Callable, api_args, api_kwargs,
                              *args, **kwargs):
  from jax._src.pallas.fuser import block_spec  # type: ignore

  res = repro_bypass_wrapper(block_spec.pull_block_spec)(
      fun, *api_args, **api_kwargs)(*args, **kwargs)
  kernel_fn, in_block_arg_specs, in_block_kwarg_specs = res

  wrapped_in_args = tree_util.tree_map(
      lambda x: _pallas_map_blockspec(lambda f: repro_boundary(f, is_user=False), x),
      in_block_arg_specs
  )
  wrapped_in_kwargs = tree_util.tree_map(
      lambda x: _pallas_map_blockspec(lambda f: repro_boundary(f, is_user=False), x),
      in_block_kwarg_specs
  )
  return repro_boundary(kernel_fn, is_user=False), wrapped_in_args, wrapped_in_kwargs

@partial(repro_boundary, repro_api_name="fuser_push_block_spec_call",
         map_user_func_args=lambda toapply, fun, api_args, api_kwargs, *args, **kwargs: (
             (toapply(fun),
              tree_util.tree_map(partial(_pallas_map_blockspec, toapply), api_args),
              tree_util.tree_map(partial(_pallas_map_blockspec, toapply), api_kwargs),
              *args), kwargs))
def fuser_push_block_spec_call(fun: Callable, api_args, api_kwargs,
                                *args, **kwargs):
  from jax._src.pallas.fuser import block_spec  # type: ignore

  res = repro_bypass_wrapper(block_spec.push_block_spec)(
      fun, *api_args, **api_kwargs)(*args, **kwargs)

  wrapped_res = tree_util.tree_map(
      lambda x: _pallas_map_blockspec(lambda f: repro_boundary(f, is_user=False), x),
      res
  )
  return wrapped_res


@partial(repro_boundary, repro_api_name="fuser_get_fusion_values")
def fuser_get_fusion_values(fun: Callable, *api_args, **api_kwargs):
  from jax._src.pallas.fuser import block_spec  # type: ignore
  kernel_fn, *rest_res = repro_bypass_wrapper(block_spec.get_fusion_values)(
      fun, *api_args, **api_kwargs)

  return repro_boundary(kernel_fn, is_user=False), *rest_res


@partial(repro_boundary, repro_api_name="fuser_make_scalar_prefetch_handler",
         # No user functions in the args
         map_user_func_args=lambda toapply, *args, **kwargs: (args, kwargs))
def fuser_make_scalar_prefetch_handler(*args, **kwargs):
  from jax._src.pallas.fuser import block_spec  # type: ignore
  handler_fn = repro_bypass_wrapper(block_spec.make_scalar_prefetch_handler)(*args, **kwargs)
  return repro_boundary(handler_fn, is_user=False)


@partial(repro_boundary, repro_api_name="jax_vjphiprimitive_call",
         map_user_func_args=(
             lambda toapply, expand_fun, jvp_fun, batch_fun, fwd_fun, bwd_retval_fun, *args, **kwargs: (
              (toapply(expand_fun), toapply(jvp_fun), toapply(batch_fun),
              toapply(fwd_fun), toapply(bwd_retval_fun), *args), kwargs)))
def jax_vjphiprimitive_call(expand_fun: Callable,
                            jvp_fun: Callable, batch_fun: Callable,
                            fwd_fun: Callable, bwd_retval_fun: Callable,
                            *args, in_avals, out_aval, params, prim=None):
  # prim will be present when called from tracker, but is dropped from repros
  # (like a static argument), in tracker.py.
  # TODO: find a cleaner solution
  from jax.experimental import hijax  # type: ignore  # noqa: F401
  from jax._src import core  # type: ignore  # noqa: F401

  # class ReproHiType(hijax.HiType):
  #   def __init__(self, typ):
  #     self._typ = typ

  #   def to_tangent_aval(self):
  #     return self._typ

  #   def __hash__(self):
  #     return hash(self._typ)
  #   def __eq__(self, other):
  #     return self._typ == other._typ

  #   # Keep a memo table for all types we have seen
  #   typ_memo: dict[hijax.HiType, "ReproHiType"] = {}
  #   @classmethod
  #   def wrap_aval(cls, typ: core.AbstractValue) -> "ReproHiType":
  #     if not isinstance(typ, hijax.HiType):
  #       return typ
  #     if (res := ReproHiType.typ_memo.get(typ)) is not None:
  #       return res
  #     res = ReproHiType(typ)
  #     ReproHiType.memo[typ] = res
  #     return res

  # class ReproHiValue:
  #   def __init__(self, typ):
  #     self._typ = typ

  class ReproVJPHiPrimitive(hijax.VJPHiPrimitive):
    def __init__(self):
      self.in_avals = in_avals
      self.out_aval = out_aval
      self.params = params
      self._prim = prim
      self._expand_fun = expand_fun
      self._jvp_fun = jvp_fun
      self._batch_fun = batch_fun
      self._fwd_fun = fwd_fun
      self._bwd_retval_fun = bwd_retval_fun
      super().__init__()

    def expand(self, *args):
      res = self._expand_fun(self, *args)
      return res

    def jvp(self, primals, tangents):
      return self._jvp_fun(self, primals, tangents)

    def batch(self, axis_data, args, dims):
      return self._batch_fun(self, axis_data, args, dims)

    def vjp_fwd(self, *args):
      res, residuals = self._fwd_fun(self, *args)
      return res, residuals

    def vjp_bwd_retval(self, *args):
      return self._bwd_retval_fun(self, *args)

    def __getattr__(self, name):
      """Forward attributes to the original primitive.
      This is only needed when called from tracker. When called from the repros
      we don't need to access any attributes of the original primitive.
      """
      return getattr(self._prim, name)

  hi_prim = ReproVJPHiPrimitive()
  # Now call hijax.VJPHiPrimitive.__call__
  return repro_bypass_wrapper(hi_prim.__call__)(hi_prim, *args)

# def hitype(tangent_aval):
#   from jax.experimental import hijax  # type: ignore  # noqa: F401
#   class ReproHiType(hijax.HiType):
#     def __init__(self):
#       self.in_avals = in_avals
#       self.out_aval = out_aval  # TODO: wrap the hitypes
#       self.params = params
#       self._prim = prim
#       self._expand_fun = expand_fun
#       self._jvp_fun = jvp_fun
#       self._batch_fun = batch_fun
#       self._fwd_fun = fwd_fun
#       self._bwd_retval_fun = bwd_retval_fun
#       super().__init__()

#     def to_tangent_aval(self):
#       return tangent_aval
