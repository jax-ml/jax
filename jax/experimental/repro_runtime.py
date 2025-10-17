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

import numpy as np
from typing import Callable

# Most of the imports below are for the purpose of populating the
# lowering tables, from which we collect the set of JAX primitives.

from jax import lax  # type: ignore  # noqa: F401
from jax.lax import Precision  # type: ignore  # noqa: F401
from jax import sharding  # type: ignore  # noqa: F401
from jax.experimental.pallas import tpu as pltpu  # type: ignore  # noqa: F401

from jax._src import ad_checkpoint  # type: ignore  # noqa: F401
from jax._src import core  # type: ignore  # noqa: F401
from jax._src import custom_derivatives   # type: ignore  # noqa: F401
from jax._src import dtypes  # type: ignore  # noqa: F401
from jax._src import literals  # type: ignore  # noqa: F401
from jax._src.pallas import core as pallas_core  # type: ignore  # noqa: F401
from jax.experimental.pallas import fuser  # type: ignore  # noqa: F401
from jax._src.pallas import primitives  # type: ignore  # noqa: F401
from jax._src import prng  # type: ignore  # noqa: F401
from jax._src import shard_map  # type: ignore  # noqa: F401
from jax._src.state import indexing  # type: ignore  # noqa: F401
from jax._src import xla_bridge  # type: ignore  # noqa: F401
from jax._src.lax.control_flow.conditionals import _cond as jax_cond  # type: ignore  # noqa: F401
from jax._src.interpreters import mlir  # type: ignore  # noqa: F401
from jax._src.lib import xla_client  # type: ignore  # noqa: F401
from jax._src.random import resolve_prng_impl  # type: ignore  # noqa: F401

from jax._src.repro.repro_api import *  # type: ignore  # noqa: F401,F403

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

inf = np.inf
nan = np.nan

_jax_primitives: dict[str, core.Primitive] = {}

# Initialize the table of primitives from the lowering rule tables
for prim in mlir._lowerings:
  _jax_primitives[prim.name] = prim
for platform_lowerings in mlir._platform_specific_lowerings.values():
  for prim in platform_lowerings:
    _jax_primitives[prim.name] = prim
_jax_primitives[primitives.program_id_p.name] = primitives.program_id_p
_jax_primitives[primitives.num_programs_p.name] = primitives.num_programs_p
del primitives


def jax_primitive_bind(p_name: str) -> Callable:
  return _jax_primitives[p_name].bind


def jax_get_device(platform: str, id: int):
  devs = [d for d in xla_bridge.devices(platform) if d.id == id]
  if not devs:
    raise NotImplementedError(f"Cannot find device id={id} for platform {platform}")
  return devs[0]

def fake_prng_key(impl, shape):
  # This is only for fake keys
  # TODO: implement properly the shape, based on impl
  return prng.PRNGKeyArray(impl, np.zeros(shape + (2,), dtype=np.uint32))
