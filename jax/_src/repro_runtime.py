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

from jax._src import core
from jax._src import dtypes  # type: ignore  # noqa: F401
from jax.lax import Precision  # type: ignore  # noqa: F401
from jax import random
from jax.extend.core import primitives
from jax._src import literals  # type: ignore  # noqa: F401
from jax._src import sharding  # type: ignore  # noqa: F401
from typing import Callable

import numpy as np
# TODO: this is how we print numpy arrays for now
array = np.array
int32 = np.int32
float32 = np.float32

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

def jax_defjvp(jvp_obj, *args, **kwargs):
  return jvp_obj.defjvp(*args, **kwargs)

def jax_defjvps(jvp_obj, *args, **kwargs):
  return jvp_obj.defjvps(*args, **kwargs)

def jax_call_custom_jvp(jvp_obj, *args, **kwargs):
  return jvp_obj(*args, **kwargs)

def jax_defvjp(vjp_obj, *args, **kwargs):
  return vjp_obj.defvjp(*args, **kwargs)

def jax_call_custom_vjp(vjp_obj, *args, **kwargs):
  return vjp_obj(*args, **kwargs)

def jax_aot_trace(jit_f, *args, **kwargs):
  return jit_f.trace(*args, **kwargs)

def jax_aot_lower(jit_f, *args, **kwargs):
  return jit_f.fun.lower(*args, **kwargs)

def jax_aot_eval_shape(jit_f, *args, **kwargs):
  return jit_f.fun.eval_shape(*args, **kwargs)

from jax._src.random import resolve_prng_impl as jax_random_prngs  # type: ignore  # noqa: F401
from jax._src.lax.control_flow.conditionals import _cond as jax_cond  # type: ignore  # noqa: F401
