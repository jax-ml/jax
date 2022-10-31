# Copyright 2018 The JAX Authors.
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
from jax._src.cloud_tpu_init import cloud_tpu_init as _cloud_tpu_init
try:
  _cloud_tpu_init()
except Exception as exc:
  # Defensively swallow any exceptions to avoid making jax unimportable
  from warnings import warn as _warn
  _warn(f"cloud_tpu_init failed: {repr(exc)}\n This a JAX bug; please report "
        f"an issue at https://github.com/google/jax/issues")
  del _warn
del _cloud_tpu_init

# Confusingly there are two things named "config": the module and the class.
# We want the exported object to be the class, so we first import the module
# to make sure a later import doesn't overwrite the class.
from jax import config as _config_module
del _config_module

from jax._src.basearray import Array as Array

from jax._src.config import (
  config as config,
  enable_checks as enable_checks,
  check_tracer_leaks as check_tracer_leaks,
  checking_leaks as checking_leaks,
  enable_custom_prng as enable_custom_prng,
  enable_custom_vjp_by_custom_transpose as enable_custom_vjp_by_custom_transpose,
  debug_nans as debug_nans,
  debug_infs as debug_infs,
  log_compiles as log_compiles,
  default_device as default_device,
  default_matmul_precision as default_matmul_precision,
  default_prng_impl as default_prng_impl,
  numpy_dtype_promotion as numpy_dtype_promotion,
  numpy_rank_promotion as numpy_rank_promotion,
  jax2tf_associative_scan_reductions as jax2tf_associative_scan_reductions,
  transfer_guard as transfer_guard,
  transfer_guard_host_to_device as transfer_guard_host_to_device,
  transfer_guard_device_to_device as transfer_guard_device_to_device,
  transfer_guard_device_to_host as transfer_guard_device_to_host,
  spmd_mode as spmd_mode,
)
from .core import eval_context as ensure_compile_time_eval
from jax._src.environment_info import print_environment_info as print_environment_info
from jax._src.api import (
  ad,  # TODO(phawkins): update users to avoid this.
  effects_barrier,
  block_until_ready as block_until_ready,
  checkpoint as checkpoint,
  checkpoint_policies as checkpoint_policies,
  clear_backends as clear_backends,
  closure_convert as closure_convert,
  curry,  # TODO(phawkins): update users to avoid this.
  custom_gradient as custom_gradient,
  custom_jvp as custom_jvp,
  custom_vjp as custom_vjp,
  default_backend as default_backend,
  device_count as device_count,
  device_get as device_get,
  device_put as device_put,
  device_put_sharded as device_put_sharded,
  device_put_replicated as device_put_replicated,
  devices as devices,
  disable_jit as disable_jit,
  eval_shape as eval_shape,
  flatten_fun_nokwargs,  # TODO(phawkins): update users to avoid this.
  float0 as float0,
  grad as grad,
  hessian as hessian,
  host_count as host_count,
  host_id as host_id,
  host_ids as host_ids,
  jacobian as jacobian,
  jacfwd as jacfwd,
  jacrev as jacrev,
  jit as jit,
  jvp as jvp,
  local_device_count as local_device_count,
  local_devices as local_devices,
  linearize as linearize,
  linear_transpose as linear_transpose,
  live_arrays as live_arrays,
  make_jaxpr as make_jaxpr,
  named_call as named_call,
  named_scope as named_scope,
  pmap as pmap,
  process_count as process_count,
  process_index as process_index,
  pure_callback as pure_callback,
  pxla,  # TODO(phawkins): update users to avoid this.
  remat as remat,
  ShapedArray as ShapedArray,
  ShapeDtypeStruct as ShapeDtypeStruct,
  value_and_grad as value_and_grad,
  vjp as vjp,
  vmap as vmap,
  xla,  # TODO(phawkins): update users to avoid this.
  xla_computation as xla_computation,
)

from jax._src.array import (
    make_array_from_single_device_arrays as make_array_from_single_device_arrays,
    make_array_from_callback as make_array_from_callback,
)

from jax.version import __version__ as __version__
from jax.version import __version_info__ as __version_info__

from jax._src.tree_util import (
  tree_map as tree_map,
  # TODO(jakevdp): remove these deprecated routines after October 2022
  _deprecated_treedef_is_leaf as treedef_is_leaf,
  _deprecated_tree_flatten as tree_flatten,
  _deprecated_tree_leaves as tree_leaves,
  _deprecated_tree_structure as tree_structure,
  _deprecated_tree_transpose as tree_transpose,
  _deprecated_tree_unflatten as tree_unflatten,
)

# These submodules are separate because they are in an import cycle with
# jax and rely on the names imported above.
from jax import abstract_arrays as abstract_arrays
from jax import api_util as api_util
from jax import distributed as distributed
from jax import debug as debug
from jax import dtypes as dtypes
from jax import errors as errors
from jax import image as image
from jax import lax as lax
from jax import nn as nn
from jax import numpy as numpy
from jax import ops as ops
from jax import profiler as profiler
from jax import random as random
from jax import scipy as scipy
from jax import sharding as sharding
from jax import stages as stages
from jax import tree_util as tree_util
from jax import util as util

import jax.lib  # TODO(phawkins): remove this export.

del jax._src
