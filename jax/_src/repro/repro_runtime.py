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

This file, along with repro_api.py, contains the definitions needed to
run the repros. We split these in two files to mitigate circular imports:
we keep in repro_api.py very few top-level imports and we define here all
the functions needed to be imported by the rest of the repro infrastructure.
We put in repro_runtime.py the top-level imports for the modules referenced
in the repros.

HIGHLY EXPERIMENTAL.
"""

import numpy as np

from jax._src.repro import repro_primitives

# The imports below are referenced in repros. Some of them are for the purpose
# of populating the lowering tables,
# from which we collect the set of JAX primitives.

# Allow the use of the repro_runtime even if Pallas is not available
try:
  from jax._src.pallas import core as pallas_core  # type: ignore  # noqa: F401
except ImportError:
  pallas_core = None  # type: ignore
try:
  from jax.experimental import pallas  # type: ignore  # noqa: F401
except ImportError:
  pallas = None  # type: ignore
try:
  from jax.experimental.pallas import tpu as pltpu  # type: ignore  # noqa: F401
except ImportError:
  pltpu = None  # type: ignore
try:
  from jax.experimental.pallas import tpu_sc as plsc  # type: ignore  # noqa: F401
except ImportError:
  plsc = None  # type: ignore
try:
  from jax.experimental.pallas import fuser  # type: ignore  # noqa: F401
except ImportError:
  fuser = None  # type: ignore
try:
  from jax._src.pallas import primitives as pallas_primitives # type: ignore  # noqa: F401
except ImportError:
  pallas_primitives = None  # type: ignore
try:
  from jax._src.pallas.mosaic import primitives as mosaic_primitives  # type: ignore  # noqa: F401
except ImportError:
  mosaic_primitives = None  # type: ignore
try:
  from jax._src.pallas.mosaic_gpu import core as plgpu_core  # type: ignore  # noqa: F401
except ImportError:
  plgpu_core = None  # type: ignore

try:
  from jax._src.pallas.mosaic_gpu import primitives as mosaic_gpu_primitives  # type: ignore  # noqa: F401
except ImportError:
  mosaic_gpu_primitives = None  # type: ignore

try:
  import jax.experimental.pallas.mosaic_gpu as plgpu  # type: ignore  # noqa: F401
except ImportError:
  plgpu = None  # type: ignore


from jax import export  # type: ignore  # noqa: F401
from jax._src import layout  # type: ignore  # noqa: F401

# The following imports are references by the emitter rules
from jax._src import ad_checkpoint  # type: ignore  # noqa: F401
from jax._src import core  # type: ignore  # noqa: F401
from jax._src import custom_derivatives   # type: ignore  # noqa: F401
from jax._src import dtypes  # type: ignore  # noqa: F401
from jax._src.frozen_dict import FrozenDict  # type: ignore  # noqa: F401
from jax._src import lax  # type: ignore  # noqa: F401
from jax._src.lax.lax import RaggedDotDimensionNumbers  # type: ignore  # noqa: F401
from jax._src import literals  # type: ignore  # noqa: F401
from jax._src.literals import TypedFloat, TypedInt  # type: ignore  # noqa: F401
from jax._src.mesh import AbstractDevice, AbstractMesh, AxisType, Mesh  # type: ignore  # noqa: F401
from jax._src.named_sharding import NamedSharding  # type: ignore  # noqa: F401
from jax._src.partition_spec import PartitionSpec  # type: ignore  # noqa: F401
from jax._src.sharding_impls import make_single_device_sharding  # type: ignore  # noqa: F401
from jax._src import shard_map  # type: ignore  # noqa: F401
from jax._src import sharding  # type: ignore  # noqa: F401
from jax._src.typing import DeprecatedArg  # type: ignore  # noqa: F401
from jax._src import traceback_util  # type: ignore  # noqa: F401
from jax._src.state import indexing  # type: ignore  # noqa: F401
from jax._src.state import types as state_types  # type: ignore  # noqa: F401
from jax._src import xla_bridge  # type: ignore  # noqa: F401
from jax._src.lax.control_flow.conditionals import cond as jax_cond  # type: ignore  # noqa: F401
from jax._src.lax.control_flow.conditionals import _switch_internal as jax_switch  # type: ignore  # noqa: F401
from jax._src.interpreters import mlir  # type: ignore  # noqa: F401
from jax._src.lib import xla_client  # type: ignore  # noqa: F401
from jax._src.random import prng  # type: ignore  # noqa: F401
from jax._src.random import resolve_prng_impl  # type: ignore  # noqa: F401
from jax._src.xla_metadata_lib import XlaMetadata  # type: ignore  # noqa: F401

ignored = np.float32(0)

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

repro_primitives.populate()
