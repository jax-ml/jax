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
  ("jax.custom_gradient", "custom_gradient"),
  ("flax.core.axes_scan.scan", "flax_axes_scan"),
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
    jit_call_trampoline.clear_cache = lambda: None  # Caches are foiled  # pyrefly: ignore[missing-attribute]
    return jit_call_trampoline

  jit_trampoline_wrapper.real_api_fun = real_jit  # pyrefly: ignore[missing-attribute]
  return jit_trampoline_wrapper

api_trampolines["jax.jit"] = partial(jit_trampoline, False)
api_trampolines["pjit.pjit"] = partial(jit_trampoline, True)


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
