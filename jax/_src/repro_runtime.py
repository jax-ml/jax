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
"""Runtime definitions for repros.

HIGHLY EXPERIMENTAL.
"""

from typing import Any, Callable

import numpy as np

from jax import lax  # type: ignore  # noqa: F401
from jax.lax import Precision  # type: ignore  # noqa: F401
from jax import random
from jax.extend.core import primitives
from jax import sharding  # type: ignore  # noqa: F401

from jax._src import api
from jax._src import core
from jax._src import custom_derivatives
from jax._src import dtypes  # type: ignore  # noqa: F401
from jax._src import literals  # type: ignore  # noqa: F401
from jax._src import shard_map
from jax._src import xla_bridge

# TODO: this is how we print numpy arrays for now, using the built-in
# repr
array = np.array
int8 = np.int8
uint8 = np.uint8
int16 = np.int16
uint16 = np.uint16
int32 = np.int32
uint32 = np.uint32
int64 = np.int64
uint64 = np.uint64

float16 = np.float16
float32 = np.float32
float64 = np.float64

_jax_primitives: dict[str, core.Primitive] = {}

# Initialize the table of primitives
def _add_primitives_from_module(m):
  from jax._src import core
  for p_var_name, p in vars(m).items():
    if type(p) is core.Primitive:
      _jax_primitives[p.name] = p

_add_primitives_from_module(primitives)
del _add_primitives_from_module
del core, random, primitives

def jax_primitive_bind(p_name: str) -> Callable:
  return _jax_primitives[p_name].bind

from jax._src.random import resolve_prng_impl  # type: ignore  # noqa: F401
from jax._src.lax.control_flow.conditionals import _cond as jax_cond  # type: ignore  # noqa: F401

def jax_jit_call(f: Callable, jit_args: tuple[Any,...], jit_kwargs: dict[str, Any], *args, **kwargs):
  return api.jit(f, *jit_args, **jit_kwargs)(*args, **kwargs)

def jax_vmap_call(f: Callable, vmap_args: tuple[Any, ...], vmap_kwargs: dict[str, Any],
                  *args, **kwargs):
  return api.vmap(f, *vmap_args, **vmap_kwargs)(*args, **kwargs)

def jax_grad_call(f: Callable, grad_args: tuple[Any], grad_kwargs: dict[str, Any],
                  *args, **kwargs):
  return api.grad(f, *grad_args, **grad_kwargs)(*args, **kwargs)

def jax_value_and_grad_call(f: Callable, value_and_grad_args: tuple[Any], value_and_grad_kwargs: dict[str, Any],
                  *args, **kwargs):
  return api.value_and_grad(f, *value_and_grad_args, **value_and_grad_kwargs)(*args, **kwargs)

def jax_custom_vjp_call(fun: Callable, fwd: Callable, bwd: Callable,
                        custom_vjp_kwargs, *args, **kwargs):
  cvjp = custom_derivatives.custom_vjp(fun, **custom_vjp_kwargs)
  cvjp.defvjp(fwd, bwd)
  return cvjp(*args, **kwargs)

def jax_custom_jvp_call(fun: Callable, cjvp_kwargs: dict[str, Any], defjvp_kwargs: dict[str, Any],
                        *fun_jvps_and_args, uses_defjvps: bool, jvps_count: int, **kwargs):
  from jax._src import custom_derivatives
  cjvp_new = custom_derivatives.custom_jvp(fun, **cjvp_kwargs)
  if uses_defjvps:
    cjvp_new.defjvps(*fun_jvps_and_args[:jvps_count])
  else:
    assert jvps_count == 1
    cjvp_new.defjvp(fun_jvps_and_args[0], **defjvp_kwargs)
  return cjvp_new(*fun_jvps_and_args[jvps_count:], **kwargs)

def jax_named_call_call(f: Callable, t_args: tuple[Any, ...], t_kwargs: dict[str, Any], *args, **kwargs):
  assert False  # We bypass named_call for now
  return api.named_call(f, *t_args, **t_kwargs)(*args, **kwargs)

def jax_shard_map_call(f: Callable, shard_map_kwargs: dict[str, Any], *args, **kwargs):
  return shard_map._shard_map(f, **shard_map_kwargs)(*args, **kwargs)

def jax_get_device(platform: str, id: int):
  devs = [d for d in xla_bridge.devices(platform) if d.id == id]
  if not devs:
    raise NotImplementedError(f"Cannot find device id={id} for platform {platform}")
  return devs[0]
