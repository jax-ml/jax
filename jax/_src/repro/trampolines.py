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
"""
See https://docs.jax.dev/en/latest/debugging/repro.html#trampolines
"""

from __future__ import annotations
from typing import Any, Callable
from functools import partial

Args = tuple[Any, ...]  # positional args
KWArgs = dict[str, Any]

# An api_trampoline takes the real_api_fun as an argument, and returns
# a function that should have the same in-out semantics, while being friendly
# to repro tracking.
# Dict indexed by repro_api_name.
api_trampolines: dict[str, Callable[..., Any]] = {}
def api_trampoline(repro_api_name: str):
  def decorator(f: Callable):
    api_trampolines[repro_api_name] = f
    return f
  return decorator

def uncurry_trampoline(api_name: str, api_fun: Callable) -> Callable:
  """
  A trampoline to use for an API function that returns a single
  first-order function. This trampoline turns it into the uncurried function.
  If `api_fun` is used like:

     api_fun(fun, *api_args, **api_kwargs)(*args, **kwargs)

  then `uncurry_trampoline("api_name", api_fun)` when used in lieu
  of `api_fun` will defer all processing until `args` and `kwargs` are
  available and then call

      repro_api.<api_name>_call(fun, api_args, api_kwargs, *args, **kwargs)

  """
  api_call_name = f"{api_name}_call"
  from jax._src.repro import repro_api

  def trampoline(arg0, *api_args, **api_kwargs):
    def call_trampoline(*args, **kwargs):
      return getattr(repro_api, api_call_name)(
        arg0, api_args, api_kwargs, *args, **kwargs)
    call_trampoline.__name__ = f"{api_name}_call_trampoline"
    call_trampoline.__qualname__ = call_trampoline.__name__
    return call_trampoline
  trampoline.__name__ = f"{api_name}_trampoline"
  trampoline.__qualname__ = trampoline.__name__
  trampoline.real_api_fun = api_fun  # pyrefly: ignore[missing-attribute]
  return trampoline


for repro_api_name, transform_name in [
  ("jax.shard_map", "shard_map"),
  ("jax.vmap", "vmap"),
  ("jax.grad", "grad"),
  ("jax.linear_transpose", "linear_transpose"),
  ("jax.jacfwd", "jacfwd"),
  ("jax.jacrev", "jacrev"),
  ("jax.hessian", "hessian"),
  ("jax.value_and_grad", "value_and_grad"),
  ("jax.checkpoint", "checkpoint"),
  ("jax.experimental.pallas.pallas_call", "pallas_call"),
  ("jax.experimental.pallas.run_state", "pallas_run_state"),
  ("jax.experimental.pallas.mosaic_gpu.kernel", "pallas_gpu_kernel"),
  ("jax.custom_gradient", "custom_gradient"),
  ("flax.core.axes_scan.scan", "flax_axes_scan"),
  ("jax.experimental.pallas.mosaic_gpu.emit_pipeline", "pallas_gpu_emit_pipeline"),
  ("jax.experimental.pallas.tpu.emit_pipeline", "pallas_tpu_emit_pipeline"),
  ("fuser.pull_block_spec", "fuser_pull_block_spec"),
  ("fuser.push_block_spec", "fuser_push_block_spec"),
]:
  api_trampolines[repro_api_name] = partial(uncurry_trampoline, transform_name)


def uncurry_decorator_trampoline(api_name: str, api_fun: Callable) -> Callable:
  """
  A trampoline to use for an API function that acts as a decorator factory.
  The API is called first with configuration args/kwargs, and returns a
  decorator that takes a single function (body). If `api_fun` is used like:

     api_fun(*api_args, **api_kwargs)(body)

  then `uncurrying_decorator_trampoline("api_name", api_fun)` when used
  in lieu of `api_fun` will defer all processing until `body` is available
  and then call

      repro_api.<api_name>_call(body, api_args, api_kwargs)

  """
  api_call_name = f"{api_name}_call"
  from jax._src.repro import repro_api

  def trampoline(*api_args, **api_kwargs):
    def decorator(body):
      return getattr(repro_api, api_call_name)(
        body, api_args, api_kwargs)
    return decorator
  trampoline.__name__ = f"{api_name}_trampoline"
  trampoline.__qualname__ = trampoline.__name__
  trampoline.real_api_fun = api_fun  # pyrefly: ignore[missing-attribute]
  return trampoline


def redirect_trampoline(api_name: str, api_fun: Callable) -> Callable:
  """
  Redirects an API call `api_fun` to a function in
  repro_api.`api_name`.
  Use when you need to specify map_user_func_args, or we cannot
  use directly the JAX implementation.

  See https://docs.jax.dev/en/latest/debugging/repro.html#trampolines.

  Instead of:

     api_fun(*trans_args, **trans_kwargs)

  then `redirect_trampoline("api_name", api_fun)` will call

      repro_api.<api_name>(*trans_args, **trans_kwargs)

  """
  from jax._src.repro import repro_api

  def trampoline(*args, **kwargs):
    return getattr(repro_api, api_name)(*args, **kwargs)

  trampoline.__name__ = f"{api_name}_trampoline"
  trampoline.__qualname__ = trampoline.__name__
  trampoline.real_api_fun = api_fun  # pyrefly: ignore[missing-attribute]
  return trampoline

for repro_api_name, transform_name in [
  ("jax.lax.cond", "cond"),
  ("jax.lax.switch", "switch"),
  ("jax.lax.fori_loop", "fori_loop"),
  ("jax.lax.while_loop", "while_loop"),
  ("jax.linearize", "linearize"),
  ("jax.vjp", "vjp"),
  ("jax.experimental.pallas.tpu.emit_pipeline_with_allocations", "pallas_tpu_emit_pipeline_with_allocations"),
  ("fuser.make_scalar_prefetch_handler", "fuser_make_scalar_prefetch_handler"),
]:
  api_trampolines[repro_api_name] = partial(redirect_trampoline, transform_name)


def jit_trampoline(is_pjit: bool, real_jit: Callable) -> Callable:
  # A trampoline for both jit and pjit
  from jax._src import mesh as mesh_lib  # type: ignore

  def get_ctx_mesh() -> mesh_lib.Mesh | None:  # type: ignore
    ctx_mesh = mesh_lib.get_concrete_mesh()
    if ctx_mesh.empty and is_pjit:
      ctx_mesh = mesh_lib.thread_resources.env.physical_mesh
    return ctx_mesh if not ctx_mesh.empty else None

  def jit_trampoline_wrapper(fun: Callable | None = None, /, **jit_kwargs):
    from jax._src.repro.repro_api import jit_call, jit_aot_trace_call, jit_aot_lower_call
    from jax._src.repro.repro_api import pjit_call, pjit_aot_trace_call, pjit_aot_lower_call
    if fun is None:  # Starting with JAX v0.8.1, jax.jit(**kwargs) can be a decorator
      return lambda fun: jit_trampoline_wrapper(fun, **jit_kwargs)

    # Ignore calls from xla_primitive_callable which use jit over an internal
    # function that just binds the primitive
    if getattr(fun, "_apply_primitive", None):
      return real_jit(fun, **jit_kwargs)

    def jit_call_trampoline(*args, **kwargs):
      to_call = pjit_call if is_pjit else jit_call
      return to_call(fun, get_ctx_mesh(), jit_kwargs, *args, **kwargs)

    def jit_aot_trace_trampoline(*args, **kwargs):
      to_call = pjit_aot_trace_call if is_pjit else jit_aot_trace_call
      return to_call(fun, get_ctx_mesh(), jit_kwargs, *args, **kwargs)

    jit_call_trampoline.trace = jit_aot_trace_trampoline  # pyrefly: ignore[missing-attribute]

    def jit_aot_lower_trampoline(*args, **kwargs):
      to_call = pjit_aot_lower_call if is_pjit else jit_aot_lower_call
      return to_call(fun, get_ctx_mesh(), jit_kwargs, *args, **kwargs)

    if hasattr(fun, "__name__"):
      jit_call_trampoline.__name__ = fun.__name__
    if hasattr(fun, "__qualname__"):
      jit_call_trampoline.__qualname__ = fun.__qualname__
    # Set __wrapped__ so that inspect.signature sees through to the original
    # function's signature. This is needed for resolve_kwargs to work correctly
    # when a jit-wrapped function is used with custom_vjp/custom_jvp.
    jit_call_trampoline._is_repro_trampoline = True  # pyrefly: ignore[missing-attribute]
    jit_call_trampoline.__wrapped__ = fun  # pyrefly: ignore[missing-attribute]
    jit_call_trampoline.lower = jit_aot_lower_trampoline  # pyrefly: ignore[missing-attribute]
    jit_call_trampoline.clear_cache = lambda: None  # pyrefly: ignore[missing-attribute]
    # Save some stuff for jax_export_trampoline
    jit_call_trampoline.fun = fun  # pyrefly: ignore[missing-attribute]
    jit_call_trampoline.jit_kwargs = jit_kwargs  # pyrefly: ignore[missing-attribute]
    jit_call_trampoline.is_pjit = is_pjit  # pyrefly: ignore[missing-attribute]
    return jit_call_trampoline

  jit_trampoline_wrapper.real_api_fun = real_jit  # pyrefly: ignore[missing-attribute]
  return jit_trampoline_wrapper

api_trampolines["jax.jit"] = partial(jit_trampoline, False)
api_trampolines["pjit.pjit"] = partial(jit_trampoline, True)


@api_trampoline("jax.export.export")
def jax_export_trampoline(real_export: Callable) -> Callable:
  def export_trampoline(fun_jit, **exp_kwargs):
    def exported_call(*args, **kwargs):
      from jax._src.repro import repro_api
      # We must be using a jitted function, expect jit_trampoline to have added
      # the necessary attributes.
      if not (hasattr(fun_jit, "fun") and hasattr(fun_jit, "jit_kwargs")):
        raise NotImplementedError("jax.export.export called on a non-jitted function")
      return repro_api.export_call(
          fun_jit.fun, fun_jit.jit_kwargs, exp_kwargs, *args, **kwargs)

    return exported_call

  export_trampoline.real_api_fun = real_export  # pyrefly: ignore[missing-attribute]
  return export_trampoline


@api_trampoline("jax.custom_jvp.__call__")
def custom_jvp_trampoline(real_api_fun: Callable):
  from jax._src import api_util  # type: ignore
  from jax._src.repro.repro_api import custom_jvp_call

  def custom_jvp_call_trampoline(*args, **kwargs):
    cjvp_orig, *rest_args = args
    resolved_args = api_util.resolve_kwargs(cjvp_orig.fun, rest_args, kwargs)
    if cjvp_orig.jvps is None:
      jvps_count = 1
      uses_defjvps = False
      new_args = (cjvp_orig.jvp, *resolved_args)
    else:
      jvps_count = len(cjvp_orig.jvps)
      uses_defjvps = True
      new_args = (*cjvp_orig.jvps, *resolved_args)

    return custom_jvp_call(cjvp_orig.fun,
                           dict(nondiff_argnums=cjvp_orig.nondiff_argnums),
                           dict(symbolic_zeros=cjvp_orig.symbolic_zeros),
                           *new_args, uses_defjvps=uses_defjvps, jvps_count=jvps_count)

  custom_jvp_call_trampoline.real_api_fun = real_api_fun  # pyrefly: ignore[missing-attribute]
  return custom_jvp_call_trampoline


@api_trampoline("jax.custom_vjp.__call__")
def custom_vjp_trampoline(real_api_fun: Callable):
  def custom_vjp_call_trampoline(*args, **kwargs):
    from jax._src import api_util  # type: ignore
    from jax._src.repro.repro_api import custom_vjp_call
    cvjp_orig, *rest_args = args
    resolved_args = api_util.resolve_kwargs(cvjp_orig.fun, rest_args, kwargs)
    return custom_vjp_call(cvjp_orig.fun, cvjp_orig.fwd, cvjp_orig.bwd,
                           dict(nondiff_argnums=cvjp_orig.nondiff_argnums),
                           *resolved_args)
  custom_vjp_call_trampoline.real_api_fun = real_api_fun  # pyrefly: ignore[missing-attribute]
  return custom_vjp_call_trampoline


@api_trampoline("jax.named_call")
def named_call_trampoline(real_api_fun: Callable):
  # TODO: handle named_call. The problem that a named_call can wrap a jit
  # with statics and the statics are then not handle properly
  return (lambda fun, *args, **kwargs: fun)


@api_trampoline("jax.experimental.pallas.core_map")
def pallas_core_map_trampoline(real_api_fun: Callable):
  from jax._src.repro.repro_api import pallas_core_map

  def pallas_core_map_trampoline(mesh, **core_map_kwargs):
    def pallas_core_map_decorator(f):
      return pallas_core_map(f, mesh, core_map_kwargs)
    return pallas_core_map_decorator
  pallas_core_map_trampoline.real_api_fun = real_api_fun  # pyrefly: ignore[missing-attribute]
  return pallas_core_map_trampoline


@api_trampoline("fuser.fuse")
def fuser_fuse_trampoline(real_api_fun: Callable) -> Callable:
  from jax._src.repro.repro_api import fuser_fuse

  def jax_fuser_fuse_trampoline(fun: Callable | None = None, **trans_kwargs):
    def actual_decorator(fun: Callable):
      return partial(fuser_fuse, fun, trans_kwargs)
    if fun is None:
      return actual_decorator
    return actual_decorator(fun)

  jax_fuser_fuse_trampoline.real_api_fun = real_api_fun  # pyrefly: ignore[missing-attribute]
  return jax_fuser_fuse_trampoline


@api_trampoline("fuser.evaluate")
def fuser_evaluate_trampoline(real_api_fun: Callable) -> Callable:
  from jax._src.repro.repro_api import fuser_evaluate

  def fuser_evaluate_trampoline(fun: Callable | None = None, **trans_kwargs):
    def actual_decorator(fun: Callable):
      return partial(fuser_evaluate, fun, trans_kwargs)
    if fun is None:
      return actual_decorator
    return actual_decorator(fun)

  fuser_evaluate_trampoline.real_api_fun = real_api_fun  # pyrefly: ignore[missing-attribute]
  return fuser_evaluate_trampoline


@api_trampoline("fuser.fusible")
def fuser_fusible_trampoline(real_api_fun: Callable) -> Callable:
  from jax._src.repro.repro_api import fuser_fusible

  def jax_fuser_fusible_trampoline(fun: Callable | None = None, **trans_kwargs):
    def actual_decorator(fun: Callable):
      return partial(fuser_fusible, fun, fun, trans_kwargs)
    if fun is None:
      return actual_decorator
    return actual_decorator(fun)

  jax_fuser_fusible_trampoline.real_api_fun = real_api_fun  # pyrefly: ignore[missing-attribute]
  return jax_fuser_fusible_trampoline


@api_trampoline("jax.pallas.custom_fusion.__call__")
def pallas_custom_fusion_trampoline(real_api_fun: Callable):
  from jax._src import api_util  # type: ignore
  from jax._src.repro.repro_api import pallas_custom_fusion_call

  def custom_fusion_trampoline(*args, **kwargs):
    cfus, *rest_args = args
    resolved_args = api_util.resolve_kwargs(cfus.fun, rest_args, kwargs)
    return pallas_custom_fusion_call(
        cfus.fun, cfus.eval_rule, cfus.pull_block_spec_rule,
        cfus.push_block_spec_rule, cfus.pallas_impl, *resolved_args)

  custom_fusion_trampoline.real_api_fun = real_api_fun  # pyrefly: ignore[missing-attribute]
  return custom_fusion_trampoline
