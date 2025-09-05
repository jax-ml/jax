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

# Allow the use of the repro_runtime even if Pallas is not available
try:
  from jax._src.pallas import core as pallas_core  # type: ignore  # noqa: F401
except ImportError:
  pallas_core = None  # type: ignore
try:
  from jax.experimental.pallas import tpu as pltpu  # type: ignore  # noqa: F401
except ImportError:
  pltpu = None  # type: ignore
try:
  from jax.experimental.pallas import fuser  # type: ignore  # noqa: F401
except ImportError:
  fuser = None  # type: ignore
try:
  from jax._src.pallas import primitives as pallas_primitives # type: ignore  # noqa: F401
except ImportError:
  pallas_primitives = None  # type: ignore

# The following imports are references by the emitter rules
from jax._src import ad_checkpoint  # type: ignore  # noqa: F401
from jax._src import config
from jax._src import core  # type: ignore  # noqa: F401
from jax._src import custom_derivatives   # type: ignore  # noqa: F401
from jax._src import dtypes  # type: ignore  # noqa: F401
from jax._src.frozen_dict import FrozenDict  # type: ignore  # noqa: F401
from jax._src import lax  # type: ignore  # noqa: F401
from jax._src import literals  # type: ignore  # noqa: F401
from jax._src.literals import TypedFloat, TypedInt  # type: ignore  # noqa: F401
from jax._src.mesh import AbstractDevice, AbstractMesh, AxisType, Mesh  # type: ignore  # noqa: F401
from jax._src.named_sharding import NamedSharding  # type: ignore  # noqa: F401
from jax._src.partition_spec import PartitionSpec  # type: ignore  # noqa: F401
from jax._src import prng  # type: ignore  # noqa: F401
from jax._src import shard_map  # type: ignore  # noqa: F401
from jax._src import sharding  # type: ignore  # noqa: F401
from jax._src.typing import DeprecatedArg  # type: ignore  # noqa: F401
from jax._src.state import indexing  # type: ignore  # noqa: F401
from jax._src import xla_bridge  # type: ignore  # noqa: F401
from jax._src.lax.control_flow.conditionals import cond as jax_cond  # type: ignore  # noqa: F401
from jax._src.lax.control_flow.conditionals import _switch_internal as jax_switch  # type: ignore  # noqa: F401
from jax._src.interpreters import mlir  # type: ignore  # noqa: F401
from jax._src.lib import xla_client  # type: ignore  # noqa: F401
from jax._src.random import resolve_prng_impl  # type: ignore  # noqa: F401


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

bfloat16 = dtypes.bfloat16
float16 = np.float16
float32 = np.float32
float64 = np.float64

inf = np.inf
nan = np.nan

# Initialize the table of primitives from the lowering rule tables
_jax_primitives: dict[str, core.Primitive] = {}
for prim in mlir._lowerings:
  _jax_primitives[prim.name] = prim
for platform_lowerings in mlir._platform_specific_lowerings.values():
  for prim in platform_lowerings:
    _jax_primitives[prim.name] = prim
# Some primitives do not have lowering rules
from jax._src.interpreters import ad
for prim in ad.primitive_jvps:
  _jax_primitives[prim.name] = prim
for prim in ad.primitive_linearizations:
  _jax_primitives[prim.name] = prim
del ad

if pallas_primitives:
  _jax_primitives[pallas_primitives.program_id_p.name] = pallas_primitives.program_id_p
  _jax_primitives[pallas_primitives.num_programs_p.name] = pallas_primitives.num_programs_p
del pallas_primitives


def jax_primitive_bind(p_name: str) -> Callable:
  return _jax_primitives[p_name].bind


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

def fake_prng_key(impl, shape):
  # This is only for fake keys
  # TODO: implement properly the shape, based on impl
  return prng.PRNGKeyArray(impl, np.zeros(shape + (2,), dtype=np.uint32))
