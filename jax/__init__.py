# Copyright 2018 Google LLC
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

# Set default logging level before any logging happens.
import os as _os
_os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
del _os

# Set Cloud TPU env vars if necessary before transitively loading C++ backend
from .cloud_tpu_init import cloud_tpu_init as _cloud_tpu_init
try:
  _cloud_tpu_init()
except Exception as exc:
  # Defensively swallow any exceptions to avoid making jax unimportable
  from warnings import warn as _warn
  _warn(f"cloud_tpu_init failed: {repr(exc)}\n This a JAX bug; please report "
        f"an issue at https://github.com/google/jax/issues")
  del _warn
del _cloud_tpu_init

# flake8: noqa: F401

# Confusingly there are two things named "config": the module and the class.
# We want the exported object to be the class, so we first import the module
# to make sure a later import doesn't overwrite the class.
from . import config as _config_module
del _config_module

from ._src.config import (
  config, enable_checks, check_tracer_leaks, checking_leaks,
  debug_nans, debug_infs, log_compiles, default_matmul_precision,
  numpy_rank_promotion
)
from ._src.api import (
  ad,  # TODO(phawkins): update users to avoid this.
  checkpoint,
  closure_convert,
  curry,  # TODO(phawkins): update users to avoid this.
  custom_ivjp,
  custom_gradient,
  custom_jvp,
  custom_vjp,
  default_backend,
  device_count,
  device_get,
  device_put,
  device_put_sharded,
  device_put_replicated,
  devices,
  disable_jit,
  eval_shape,
  flatten_fun_nokwargs,  # TODO(phawkins): update users to avoid this.
  float0,
  grad,
  hessian,
  host_count,
  host_id,
  host_ids,
  invertible,
  jacobian,
  jacfwd,
  jacrev,
  jit,
  jvp,
  local_device_count,
  local_devices,
  linearize,
  linear_transpose,
  make_jaxpr,
  mask,
  named_call,
  partial,  # TODO(phawkins): update callers to use functools.partial.
  pmap,
  process_count,
  process_index,
  pxla,  # TODO(phawkins): update users to avoid this.
  remat,
  shapecheck,
  ShapedArray,
  ShapeDtypeStruct,
  # TODO(phawkins): hide tree* functions from jax, update callers to use
  # jax.tree_util.
  treedef_is_leaf,
  tree_flatten,
  tree_leaves,
  tree_map,
  tree_multimap,
  tree_structure,
  tree_transpose,
  tree_unflatten,
  value_and_grad,
  vjp,
  vmap,
  xla,  # TODO(phawkins): update users to avoid this.
  xla_computation,
)
from .experimental.maps import soft_pmap
from .version import __version__

# These submodules are separate because they are in an import cycle with
# jax and rely on the names imported above.
from . import api
from . import dtypes
from . import errors
from . import image
from . import lax
from . import nn
from . import profiler
from . import random
from . import util

def _init():
  from . import numpy # side-effecting import sets up operator overloads

_init()
del _init
