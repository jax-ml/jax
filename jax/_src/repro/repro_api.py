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

This file, along with repro_runtime.py, contains the definitions needed to
run the repros. We split these in two files to mitigate circular imports:
we keep in repro_api.py very few top-level imports and we define here all
the functions needed to be imported by the rest of the repro infrastructure.
We put in repro_runtime.py the top-level imports for the modules referenced
in the repros.

In order to avoid infinite recursion, the definitions below use
`repro.bypass_wrapper` to ensure that when we call into the real JAX
APIs we do not use the repro machinery.

This file contains most `import` statements locally, to avoid circular imports.
"""

import contextlib
import copy
import dataclasses
from functools import partial, wraps
import types
import threading
from typing import Any, Callable, Sequence

import numpy as np

from jax._src import config
from jax._src import traceback_util
from jax._src import tree_util
from jax._src import xla_bridge

from jax._src.repro import repro_primitives

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


def jax_primitive_bind(p_name: str) -> Callable:
  return repro_primitives.primitives_by_name[p_name].bind


def jax_get_device(platform: str, id: int):
  devs = [d for d in xla_bridge.devices(platform) if d.id == id]
  if not devs:
    raise NotImplementedError(f"Cannot find device id={id} for platform {platform}")
  return devs[0]


def request_cpu_devices(nr_devices: int):
  if xla_bridge.num_cpu_devices.value < nr_devices:
    xla_bridge.get_backend.cache_clear()
    # Don't raise an error for `request_cpu_devices` because we initialize the
    # backend in OSS during collecting tests in pytest via `device_under_test`.
    try:
      config.update("jax_num_cpu_devices", nr_devices)
    except RuntimeError:
      pass


class _ReproRuntimeState(threading.local):
  def __init__(self):
    """This is used in repros in case of errors, and specify the original
    platform. This makes it sometimes possible to reproduce a tracing or
    lowering error on a machine with a different set of accelerators.

    TODO: this make sense only for tracing and lowering errors, not for
    compilation or runtime errors.
    """
    self.default_lowering_platform: str | None = None

_local_state = _ReproRuntimeState()


from jax._src.interpreters import pxla  # type: ignore
old_lower_sharding_computation = pxla.lower_sharding_computation
def new_lower_sharding_computation(*args, lowering_platforms, **kwargs):
  new_platforms = lowering_platforms
  if (lowering_platforms is None and
      (platform := _local_state.default_lowering_platform) is not None):
    if platform == "gpu": platform = "cuda"  # TODO: clean
    new_platforms = (platform,)

  return old_lower_sharding_computation(*args, lowering_platforms=new_platforms,
                                        **kwargs)
pxla.lower_sharding_computation = new_lower_sharding_computation

@contextlib.contextmanager
def default_lowering_platform(platform: str, has_error: bool):
  # Used in repros to override the lowering platform to what it was originally
  if not has_error and platform != xla_bridge.local_devices()[0].platform:
    raise ValueError(
      f"This repro was generated on platform={platform} and you are trying "
      f"to run it on platform={xla_bridge.local_devices()[0].platform}. "
      "This is not allowed because the repro does not expect an error.")
  prev = _local_state.default_lowering_platform
  try:
    _local_state.default_lowering_platform = platform
    yield
  finally:
    _local_state.default_lowering_platform = prev


def state_context(**kwargs) -> contextlib.AbstractContextManager[Any]:
  """Sets thread local configuration variables."""
  class _LocalContextManager:
    # Make a context manager for non-State configuration variables (those
    # defined as config_ext.Config)
    def __init__(self, c: config.Config, new_val: Any):
      self.c = c
      if c.name == "abstract_mesh_context_manager" and new_val is None:  # pyrefly: ignore [missing-attribute]
        new_val = config.config_ext.unset
      self.prev_val = self.c.swap_local(new_val)  # pyrefly: ignore [missing-attribute]

    def __enter__(self):
      pass

    def __exit__(self, exc_type, exc_val, exc_tb):
      self.c.set_local(self.prev_val)  # pyrefly: ignore [missing-attribute]

  def get_state_context_manager(k: str, new_val: Any) -> contextlib.AbstractContextManager[Any]:
    if (s := config.config_states.get(k)) is not None:
      assert isinstance(s, config.State)
      return s(new_val)
    if k == "default_lowering_platform":
      return default_lowering_platform(new_val, True)

    # A few tracing context keys are not in config_states, but are defined
    # in the config module as context_ext.Config, luckily with the same name
    return _LocalContextManager(getattr(config, k), new_val)

  ctx = contextlib.ExitStack()
  for k, v in kwargs.items():
    ctx.enter_context(get_state_context_manager(k, v))
  return ctx


def fake_prng_key(impl, shape):
  from jax._src.random import prng  # type: ignore
  # This is only for fake keys
  # TODO: implement properly the shape, based on impl
  return prng.PRNGKeyArray(impl, np.zeros(shape + (2,), dtype=np.uint32))


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
def jax_repro_collect(f: Callable, *args,
                      collect_static_argnums=(),
                      collect_static_argnames=(),
                      **kwargs):
  """A fake JAX API to wrap a set of JAX calls to be collected in a reproducer"""
  return f(*args, **kwargs)


@partial(repro_boundary, repro_api_name="jit_call")
def jit_call(f: Callable, ctx_mesh,
             jit_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src import api  # type: ignore
  with _jit_context(False, ctx_mesh):
    jitted = repro_bypass_wrapper(api.jit)(f, **jit_kwargs)
    return jitted(*args, **kwargs)


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


def _wrap_custom_gradient_f(wrap_user, f: Callable):
  # The function passed to custom_gradient is expected to return both
  # the primal output and a gradient function. We wrap that result as
  # a user function.
  @wraps(f)
  def wrapped_f(*args, **kwargs):
    res, grad_f = f(*args, **kwargs)
    return res, wrap_user(grad_f)
  return wrap_user(wrapped_f)

@partial(repro_boundary, repro_api_name="custom_gradient_call",
         map_user_func_args=lambda wrap_user, f, *args, **kwargs: (
          (_wrap_custom_gradient_f(wrap_user, f), *args), kwargs))
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
    lambda wrap_user, fun, cjvp_kwargs, defjvp_kwargs, *fun_jvps_and_args, uses_defjvps, jvps_count: (
          (wrap_user(fun), cjvp_kwargs, defjvp_kwargs,
           *(map(wrap_user, fun_jvps_and_args[:jvps_count])),
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
             lambda wrap_user, fun, fwd, bwd, custom_vjp_kwargs, *args, **kwargs: (
                   (wrap_user(fun), wrap_user(fwd), wrap_user(bwd),
                    custom_vjp_kwargs, *args),
                   kwargs)))
def custom_vjp_call(fun: Callable, fwd: Callable, bwd: Callable,
                    custom_vjp_kwargs, *args):
  from jax._src import custom_derivatives  # type: ignore
  cvjp = custom_derivatives.custom_vjp(fun, **custom_vjp_kwargs)
  cvjp.defvjp(fwd, bwd)
  return repro_bypass_wrapper(cvjp.__call__)(cvjp, *args)


def _jax_cond_map_user_func_args(wrap_user, pred, true_fun: Callable,
                                 false_fun: Callable, *operands, **kwargs):
  if not (callable(true_fun) and callable(false_fun)):
    # try falling back to the old, deprecated version of `cond`
    if callable(false_fun) and len(operands) == 2 and callable(operands[1]):
      x_true, f_true, x_false, f_false = true_fun, false_fun, *operands
      return (pred, x_true, wrap_user(f_true), x_false, wrap_user(f_false)), kwargs
    else:
      raise NotImplementedError("Unrecognized old-style (?) lax.cond")
  else:
    return (pred, wrap_user(true_fun), wrap_user(false_fun), *operands), kwargs


@partial(repro_boundary, repro_api_name="cond", map_user_func_args=_jax_cond_map_user_func_args)
def cond(pred, true_fun: Callable, false_fun: Callable, *operands, **kwargs):
  from jax._src.lax.control_flow.conditionals import cond  # type: ignore
  return repro_bypass_wrapper(cond)(pred, true_fun, false_fun, *operands, **kwargs)


@partial(repro_boundary, repro_api_name="switch",
         map_user_func_args=lambda wrap_user, index, branches, operands, **kwargs: (
             (index, tuple(map(wrap_user, branches)), operands), kwargs))
def switch(index, branches: Sequence[Callable], operands: Sequence[Any], **kwargs):
  from jax._src.lax.control_flow.conditionals import _switch_internal  # type: ignore
  return repro_bypass_wrapper(_switch_internal)(index, branches, operands, **kwargs)


@partial(repro_boundary, repro_api_name="fori_loop",
         map_user_func_args=lambda wrap_user, lower, upper, body_fun, *args, **kwargs: (
             (lower, upper, wrap_user(body_fun), *args), kwargs))
def fori_loop(lower: int, upper: int,
              body_fun: Callable[[int, Any], Any], init_val: Any, **kwargs):
  from jax._src.lax.control_flow.loops import fori_loop  # type: ignore
  return repro_bypass_wrapper(fori_loop)(lower, upper, body_fun, init_val, **kwargs)


@partial(repro_boundary, repro_api_name="while_loop",
         map_user_func_args=lambda wrap_user, cond_fun, body_fun, *args, **kwargs: (
             (wrap_user(cond_fun), wrap_user(body_fun), *args), kwargs))
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


def _pallas_map_blockspec(wrap_user: Callable, bs):
  from jax.experimental.pallas import BlockSpec  # type: ignore
  from jax._src.pallas.core import _IndexMapFunc  # type: ignore

  if isinstance(bs, BlockSpec):
    if bs.index_map is not None:
      assert isinstance(bs.index_map, _IndexMapFunc), bs.index_map
      orig_map = bs.index_map.index_map
      return dataclasses.replace(bs, index_map=wrap_user(orig_map))
  return bs


def _pallas_map_gridspec(wrap_user: Callable, gs):
  from jax._src.pallas.core import GridSpec  # type: ignore

  if isinstance(gs, GridSpec):
    new_fields = {}
    new_fields["in_specs"] = \
      tree_util.tree_map(partial(_pallas_map_blockspec, wrap_user), gs.in_specs)
    new_fields["out_specs"] = \
      tree_util.tree_map(partial(_pallas_map_blockspec, wrap_user), gs.out_specs)
    new_spec = copy.copy(gs)
    for k, v in new_fields.items():
      object.__setattr__(new_spec, k, v)
    return new_spec
  return gs


def _map_user_blockspec(wrap_user, api_kwargs):
  new_api_kwargs = dict(api_kwargs)
  if (in_specs := new_api_kwargs.get("in_specs")) is not None:
    new_api_kwargs["in_specs"] = \
      tree_util.tree_map(partial(_pallas_map_blockspec, wrap_user), in_specs)
  if (out_specs := new_api_kwargs.get("out_specs")) is not None:
    new_api_kwargs["out_specs"] = \
        tree_util.tree_map(partial(_pallas_map_blockspec, wrap_user), out_specs)
  if (grid_spec := new_api_kwargs.get("grid_spec")) is not None:
    new_api_kwargs["grid_spec"] = _pallas_map_gridspec(wrap_user, grid_spec)
  return new_api_kwargs


def _pallas_call_call_map_user_func_args(wrap_user, kernel, api_args,
                                         api_kwargs, *args, **kwargs):
  new_api_kwargs = _map_user_blockspec(wrap_user, api_kwargs)
  return (wrap_user(kernel), api_args, new_api_kwargs, *args), kwargs


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


@partial(repro_boundary, repro_api_name="pallas_gpu_kernel_call",
         map_user_func_args=_pallas_call_call_map_user_func_args)
def pallas_gpu_kernel_call(body: Callable, api_args: tuple[Any, ...],
                           api_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src.pallas.mosaic_gpu import core as plgpu_core  # type: ignore
  return repro_bypass_wrapper(plgpu_core.kernel)(
    body, *api_args, **api_kwargs)(*args, **kwargs)


def _pallas_kernel_call_map(wrap_user, body, kernel_kwargs, *operands):
  if isinstance(body, Sequence):
    body = tuple(wrap_user(b) for b in body)
  else:
    body = wrap_user(body)
  return (body, kernel_kwargs, *operands), {}


@partial(repro_boundary, repro_api_name="pallas_kernel_call",
         map_user_func_args=_pallas_kernel_call_map)
def pallas_kernel_call(body: Callable | Sequence[Callable],
                       kernel_kwargs: dict[str, Any],
                       *operands):
  from jax._src.pallas import helpers as pallas_helpers  # type: ignore
  return repro_bypass_wrapper(pallas_helpers.kernel)(
    body, **kernel_kwargs)(*operands)


@partial(repro_boundary, repro_api_name="pallas_mpmd_map_call",
         map_user_func_args=lambda wrap_user, mesh_and_fns, kernel_kwargs, *operands: (
            (tuple((m, wrap_user(f)) for m, f in mesh_and_fns),
             kernel_kwargs, *operands), {}))
def pallas_mpmd_map_call(mesh_and_fns: Sequence, api_args: tuple[Any, ...],
                         api_kwargs: dict[str, Any], *args):
  from jax._src.pallas import mpmd  # type: ignore
  return repro_bypass_wrapper(mpmd.mpmd_map)(
      mesh_and_fns, *api_args, **api_kwargs)(*args)


@partial(repro_boundary, repro_api_name="pallas_parallel_loop")
def pallas_parallel_loop(fun: Callable, *api_args, **api_kwargs):
  from jax._src.pallas.mosaic import sc_primitives  # type: ignore
  return repro_bypass_wrapper(sc_primitives.parallel_loop)(
    *api_args, **api_kwargs)(fun)


# The only reason we need a trampolines for async_copy and friends is because
# the sync_copy, async_copy, dma_start and dma_wait methods take pytreedef
# as parameters and we don't want to use that in the repro, because they
# may have user-nodes in them.
@partial(repro_boundary, repro_api_name="pallas_wait_send",
         map_user_func_args=lambda wrap_user, *args, **kwargs: (args, kwargs))
def pallas_wait_send(desc):
  return desc.wait_send()


@partial(repro_boundary, repro_api_name="pallas_wait_recv",
         map_user_func_args=lambda wrap_user, *args, **kwargs: (args, kwargs))
def pallas_wait_recv(desc):
  return desc.wait_recv()


@partial(repro_boundary, repro_api_name="pallas_dma_start",
         map_user_func_args=lambda wrap_user, *args, **kwargs: (args, kwargs))
def pallas_dma_start(desc, *args, **kwargs):
  return desc.start(*args, **kwargs)


def _wrap_pallas_AsyncCopyDescriptor(desc):  # : AsyncCopyDescriptor
  from jax._src.pallas.mosaic.primitives import AsyncCopyDescriptor  # type: ignore
  desc.start = types.MethodType(
      repro_boundary(AsyncCopyDescriptor.start, is_user=False,
                     repro_api_name="pallas_dma_start"),
      desc)
  desc.wait_send = types.MethodType(
      repro_boundary(AsyncCopyDescriptor.wait_send, is_user=False,
                     repro_api_name="pallas_wait_send",),
      desc)
  desc.wait_recv = types.MethodType(
      repro_boundary(AsyncCopyDescriptor.wait_recv, is_user=False,
                     repro_api_name="pallas_wait_recv"),
      desc)
  return desc


@partial(repro_boundary, repro_api_name="pallas_async_copy",
         map_user_func_args=lambda wrap_user, *args, **kwargs: (args, kwargs))
def pallas_async_copy(*args, **kwargs):
  from jax.experimental.pallas import tpu as pltpu  # type: ignore
  desc = repro_bypass_wrapper(pltpu.async_copy)(*args, **kwargs)
  return _wrap_pallas_AsyncCopyDescriptor(desc)


@partial(repro_boundary, repro_api_name="pallas_make_async_copy",
         map_user_func_args=lambda wrap_user, *args, **kwargs: (args, kwargs))
def pallas_make_async_copy(*args, **kwargs):
  from jax.experimental.pallas import tpu as pltpu  # type: ignore
  desc = repro_bypass_wrapper(pltpu.make_async_copy)(*args, **kwargs)
  return _wrap_pallas_AsyncCopyDescriptor(desc)


@partial(repro_boundary, repro_api_name="pallas_make_async_remote_copy",
         map_user_func_args=lambda wrap_user, *args, **kwargs: (args, kwargs))
def pallas_make_async_remote_copy(*args, **kwargs):
  from jax.experimental.pallas import tpu as pltpu  # type: ignore
  desc = repro_bypass_wrapper(pltpu.make_async_remote_copy)(*args, **kwargs)
  return _wrap_pallas_AsyncCopyDescriptor(desc)


def _pallas_gpu_pipeline_map_user_func_args(wrap_user, kernel, api_args,
                                            api_kwargs, *args, **kwargs):
  new_api_kwargs = _map_user_blockspec(wrap_user, api_kwargs)
  if (compute_context := new_api_kwargs.get("compute_context")) is not None:
    new_api_kwargs["compute_context"] = wrap_user(compute_context)
  return (wrap_user(kernel), api_args, new_api_kwargs, *args), kwargs


@partial(repro_boundary, repro_api_name="pallas_gpu_emit_pipeline_call",
         map_user_func_args=_pallas_call_call_map_user_func_args)
def pallas_gpu_emit_pipeline_call(f: Callable, api_args: tuple[Any, ...],
                                  api_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src.pallas.mosaic_gpu import pipeline as gpu_pipeline  # type: ignore
  return repro_bypass_wrapper(gpu_pipeline.emit_pipeline)(
    f, *api_args, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="pallas_gpu_emit_pipeline_warp_specialized_call",
         map_user_func_args=_pallas_gpu_pipeline_map_user_func_args)
def pallas_gpu_emit_pipeline_warp_specialized_call(f: Callable, api_args: tuple[Any, ...],
                                                   api_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src.pallas.mosaic_gpu import pipeline as gpu_pipeline  # type: ignore
  return repro_bypass_wrapper(gpu_pipeline.emit_pipeline_warp_specialized)(
    f, *api_args, **api_kwargs)(*args, **kwargs)



@partial(repro_boundary, repro_api_name="pallas_tpu_emit_pipeline_call",
         map_user_func_args=_pallas_call_call_map_user_func_args)
def pallas_tpu_emit_pipeline_call(f: Callable, api_args: tuple[Any, ...],
                                  api_kwargs: dict[str, Any], *args, **kwargs):
  from jax._src.pallas.mosaic import pipeline as tpu_pipeline  # type: ignore
  return repro_bypass_wrapper(tpu_pipeline.emit_pipeline)(
    f, *api_args, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="pallas_tpu_emit_pipeline_with_allocations",
         map_user_func_args=lambda wrap_user, body, *args, **kwargs: (
          (wrap_user(body),),
          _map_user_blockspec(wrap_user, kwargs)))
def pallas_tpu_emit_pipeline_with_allocations(body: Callable, **kwargs):
  from jax._src.pallas.mosaic import pipeline as tpu_pipeline  # type: ignore
  res, make_allocations = \
    repro_bypass_wrapper(tpu_pipeline.emit_pipeline_with_allocations)(body, **kwargs)
  return res, repro_boundary(make_allocations, is_user=False)


@partial(repro_boundary, repro_api_name="fuser_fuse_call")
def fuser_fuse_call(fun: Callable, api_kwargs: dict[str, Any], *args, **kwargs):
  from jax.experimental.pallas import fuser  # type: ignore
  return repro_bypass_wrapper(fuser.fuse)(fun, **api_kwargs)(*args, **kwargs)


@partial(repro_boundary, repro_api_name="fuser_evaluate_call")
def fuser_evaluate_call(fun: Callable, trans_kwargs: dict[str, Any],
                        *args):
  from jax.experimental.pallas import fuser  # type: ignore
  return repro_bypass_wrapper(fuser.evaluate)(fun, **trans_kwargs)(*args)


@partial(repro_boundary, repro_api_name="fuser_fusible_call",
         map_user_func_args=lambda wrap_user, fun_1, fun_2, api_kwargs, *args, **kwargs:((
          wrap_user(fun_1), wrap_user(fun_2), api_kwargs, *args), kwargs))
def fuser_fusible_call(fun_1: Callable, fun_2: Callable,
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
         map_user_func_args=lambda wrap_user, fun, eval_rule, pull_block_spec_rule, push_block_spec_rule, impl_rule, *args, **kwargs:(
             (wrap_user(fun), wrap_user(eval_rule), wrap_user(pull_block_spec_rule),
              wrap_user(push_block_spec_rule), wrap_user(impl_rule),
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
         map_user_func_args=lambda wrap_user, fun, api_args, api_kwargs, *args, **kwargs: (
             (wrap_user(fun),
              tree_util.tree_map(partial(_pallas_map_blockspec, wrap_user), api_args),
              tree_util.tree_map(partial(_pallas_map_blockspec, wrap_user), api_kwargs),
              *args), kwargs))
def fuser_pull_block_spec_call(fun: Callable, api_args, api_kwargs,
                              *args, **kwargs):
  from jax._src.pallas.fuser import block_spec  # type: ignore

  res = repro_bypass_wrapper(block_spec.pull_block_spec)(
      fun, *api_args, **api_kwargs)(*args, **kwargs)
  kernel_fn, in_block_arg_specs, in_block_kwarg_specs = res

  wrapped_in_block_arg_specs = tree_util.tree_map(
      lambda x: _pallas_map_blockspec(lambda f: repro_boundary(f, is_user=False), x),
      in_block_arg_specs
  )
  wrapped_in_block_kwarg_specs = tree_util.tree_map(
      lambda x: _pallas_map_blockspec(lambda f: repro_boundary(f, is_user=False), x),
      in_block_kwarg_specs
  )
  return (repro_boundary(kernel_fn, is_user=False),
          wrapped_in_block_arg_specs,
          wrapped_in_block_kwarg_specs)


@partial(repro_boundary, repro_api_name="fuser_push_block_spec_call",
         map_user_func_args=lambda wrap_user, fun, api_args, api_kwargs, *args, **kwargs: (
             (wrap_user(fun),
              tree_util.tree_map(partial(_pallas_map_blockspec, wrap_user), api_args),
              tree_util.tree_map(partial(_pallas_map_blockspec, wrap_user), api_kwargs),
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
         map_user_func_args=lambda wrap_user, *args, **kwargs: (args, kwargs))
def fuser_make_scalar_prefetch_handler(*args, **kwargs):
  from jax._src.pallas.fuser import block_spec  # type: ignore
  handler_fn = repro_bypass_wrapper(block_spec.make_scalar_prefetch_handler)(*args, **kwargs)
  return repro_boundary(handler_fn, is_user=False)


## HIJAX
try:
  from jax._src import hijax  # type: ignore

  class ReproHiType(hijax.HiType):
    def __init__(self, name: str, unique_id: int, *,
                 tangent_aval = None):
      self.name = name
      self.unique_id = unique_id
      self._tangent_aval = tangent_aval
      super().__init__()

    def __hash__(self):
      return hash(self.unique_id)

    def __eq__(self, other):
      return isinstance(other, type(self)) and self.unique_id == other.unique_id

    def lower_val(self, hi_val: "ReproHiVal"):
      assert isinstance(hi_val, ReproHiVal), hi_val
      return hi_val.lo_vals

    def raise_val(self, *lo_vals) -> "ReproHiVal":
      return ReproHiVal(self, lo_vals)

    def to_tangent_aval(self):
      if self._tangent_aval is not None:
        if self._tangent_aval == "self":
          return self
        return self._tangent_aval
      raise NotImplementedError(f"to_tangent_aval: {self}")

    def dec_rank(self, size: int | None, spec: hijax.MappingSpec):
      # TODO: This may be Ok?
      return self

  class ReproHiVal:
    def __init__(self, typ: ReproHiType, lo_vals: tuple[Any, ...]):
      self.typ = typ
      self.lo_vals = lo_vals

  hijax.register_hitype(ReproHiVal, lambda v: v.typ)

  class ReproMappingSpec(hijax.MappingSpec):
    def __init__(self, name: str):
      self.name = name

    def __eq__(self, other):
      return isinstance(other, ReproMappingSpec) and self.name == other.name

    def __hash__(self):
      return hash(self.name)


except ImportError:
  hijax = None
  class ReproHiType:
    def __init__(self, *args, **kwargs):
      self.name = ""
      self.unique_id = 0
      self._tangent_aval = None

  class ReproHiVal:
    def __init__(self, *args, **kwargs):
      self.typ = ReproHiType("", 0)
      self.lo_vals = ()

  class ReproMappingSpec:
    def __init__(self, name: str):
      self.name = name


_hitype_memo_lock = threading.Lock()
# Map HiType to unique integers
hitype_memo: dict[Any, int] = {}


def register_repro_hitype(hityp):
  with _hitype_memo_lock:
    if hityp in hitype_memo:
      return
    type_id = len(hitype_memo)
    hitype_memo[hityp] = type_id


@partial(repro_boundary, repro_api_name="jax_vjphiprimitive_call",
         map_user_func_args=(
             lambda wrap_user, *args, expand, jvp, batch, fwd, bwd_retval, **kwargs: (
              (args, dict(expand=wrap_user(expand), jvp=wrap_user(jvp),
                          batch=wrap_user(batch), fwd=wrap_user(fwd),
                          bwd_retval=wrap_user(bwd_retval), **kwargs)))))
def jax_vjphiprimitive_call(hi_prim_name: str, *args, expand: Callable,
                            jvp: Callable, batch: Callable,
                            fwd: Callable, bwd_retval: Callable,
                            in_avals, out_aval, params, prim=None):
  # prim will be present when called from tracker, but is dropped from repros
  # (like a static argument), in tracker.py.
  # TODO: find a cleaner solution
  from jax._src import hijax  # type: ignore  # noqa: F401
  from jax._src import core  # type: ignore  # noqa: F401

  class ReproVJPHiPrimitive(hijax.VJPHiPrimitive):
    def __init__(self):
      self.name = hi_prim_name
      self.in_avals = in_avals
      self.out_aval = out_aval
      self.params = params
      self._prim = prim
      self._expand = expand
      self._jvp = jvp
      self._batch = batch
      self._fwd = fwd
      self._bwd_retval = bwd_retval
      super().__init__()

    def __repr__(self):
      return f"{self.__class__.__name__}[{self.name}:{self.params}]"

    def expand(self, *args):
      return self._expand(self, *args)

    def jvp(self, primals, tangents):
      return self._jvp(self, primals, tangents)

    def batch(self, axis_data, args, dims):
      return self._batch(self, axis_data, args, dims)

    def vjp_fwd(self, *args):
      res, residuals = self._fwd(self, *args)
      return res, residuals

    def vjp_bwd_retval(self, *args):
      return self._bwd_retval(self, *args)

    def __getattr__(self, name):
      """Forward attributes to the original primitive.
      This is only needed when called from tracker. When called from the repros
      we don't need to access any attributes of the original primitive.
      """
      return getattr(self._prim, name)

  hi_prim = ReproVJPHiPrimitive()
  # Now call hijax.VJPHiPrimitive.__call__
  return repro_bypass_wrapper(hi_prim.__call__)(hi_prim, *args)
