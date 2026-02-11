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

# Set default C++ logging level before any logging happens.
import os as _os
_os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')
del _os

# Import version first, because other submodules may reference it.
from jax.version import __version__ as __version__
from jax.version import __version_info__ as __version_info__

# Set Cloud TPU env vars if necessary before transitively loading C++ backend
from jax._src.cloud_tpu_init import cloud_tpu_init as _cloud_tpu_init
try:
  _cloud_tpu_init()
except Exception as exc:
  # Defensively swallow any exceptions to avoid making jax unimportable
  from warnings import warn as _warn
  _warn(f"cloud_tpu_init failed: {exc!r}\n This a JAX bug; please report "
        f"an issue at https://github.com/jax-ml/jax/issues")
  del _warn
del _cloud_tpu_init

# Force early import, allowing use of `jax.core` after importing `jax`.
import jax.core as _core
del _core

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

from jax._src.basearray import Array as Array
from jax import tree as tree
from jax import typing as typing

from jax._src.config import (
  config as config,
  enable_checks as enable_checks,
  enable_x64 as enable_x64,
  debug_key_reuse as debug_key_reuse,
  check_tracer_leaks as check_tracer_leaks,
  checking_leaks as checking_leaks,
  enable_custom_prng as enable_custom_prng,
  softmax_custom_jvp as softmax_custom_jvp,
  enable_custom_vjp_by_custom_transpose as enable_custom_vjp_by_custom_transpose,
  debug_nans as debug_nans,
  debug_infs as debug_infs,
  log_compiles as log_compiles,
  no_tracing as no_tracing,
  no_execution as no_execution,
  explain_cache_misses as explain_cache_misses,
  default_device as default_device,
  default_matmul_precision as default_matmul_precision,
  default_prng_impl as default_prng_impl,
  numpy_dtype_promotion as numpy_dtype_promotion,
  numpy_rank_promotion as numpy_rank_promotion,
  allow_f16_reductions as allow_f16_reductions,
  jax2tf_associative_scan_reductions as jax2tf_associative_scan_reductions,
  legacy_prng_key as legacy_prng_key,
  threefry_partitionable as threefry_partitionable,
  array_garbage_collection_guard as array_garbage_collection_guard,
  transfer_guard as transfer_guard,
  transfer_guard_host_to_device as transfer_guard_host_to_device,
  transfer_guard_device_to_device as transfer_guard_device_to_device,
  transfer_guard_device_to_host as transfer_guard_device_to_host,
  make_user_context as make_user_context,
  remove_size_one_mesh_axis_from_type as remove_size_one_mesh_axis_from_type,
  thread_guard as thread_guard
)

from jax._src.core import ensure_compile_time_eval as ensure_compile_time_eval
from jax._src.environment_info import print_environment_info as print_environment_info

from jax._src.lib import xla_client as _xc
Device = _xc.Device
del _xc

from jax._src.core import typeof as typeof
from jax._src.api import effects_barrier as effects_barrier
from jax._src.api import block_until_ready as block_until_ready
from jax._src.ad_checkpoint import checkpoint as checkpoint
from jax._src.ad_checkpoint import checkpoint_policies as checkpoint_policies
from jax._src.ad_checkpoint import remat as remat
from jax._src.api import clear_caches as clear_caches
from jax._src.api import copy_to_host_async as copy_to_host_async
from jax._src.custom_derivatives import closure_convert as closure_convert
from jax._src.custom_derivatives import custom_gradient as custom_gradient
from jax._src.custom_derivatives import custom_jvp as custom_jvp
from jax._src.custom_derivatives import custom_vjp as custom_vjp
from jax._src.xla_bridge import default_backend as default_backend
from jax._src.xla_bridge import device_count as device_count
from jax._src.api import device_get as device_get
from jax._src.api import device_put as device_put
from jax._src.api import device_put_sharded as _deprecated_device_put_sharded
from jax._src.api import device_put_replicated as _deprecated_device_put_replicated
from jax._src.xla_bridge import devices as devices
from jax._src.api import disable_jit as disable_jit
from jax._src.api import eval_shape as eval_shape
from jax._src.dtypes import float0 as float0
from jax._src.api import fwd_and_bwd as fwd_and_bwd
from jax._src.api import grad as grad
from jax._src.api import hessian as hessian
from jax._src.xla_bridge import host_count as host_count
from jax._src.xla_bridge import host_id as host_id
from jax._src.xla_bridge import host_ids as host_ids
from jax._src.api import jacobian as jacobian
from jax._src.api import jacfwd as jacfwd
from jax._src.api import jacrev as jacrev
from jax._src.api import jit as jit
from jax._src.api import jvp as jvp
from jax._src.xla_bridge import local_device_count as local_device_count
from jax._src.xla_bridge import local_devices as local_devices
from jax._src.api import linearize as linearize
from jax._src.api import linear_transpose as linear_transpose
from jax._src.api import live_arrays as live_arrays
from jax._src.api import make_jaxpr as make_jaxpr
from jax._src.api import named_call as named_call
from jax._src.api import named_scope as named_scope
from jax._src.api import pmap as pmap
from jax._src.xla_bridge import process_count as process_count
from jax._src.xla_bridge import process_index as process_index
from jax._src.xla_bridge import process_indices as process_indices
from jax._src.callback import pure_callback as pure_callback
from jax._src.core import ShapeDtypeStruct as ShapeDtypeStruct
from jax._src.api import value_and_grad as value_and_grad
from jax._src.api import vjp as vjp
from jax._src.api import vmap as vmap
from jax._src.indexing import ds as ds
from jax._src.sharding_impls import NamedSharding as NamedSharding
from jax._src.sharding_impls import make_mesh as make_mesh
from jax._src.sharding_impls import set_mesh as set_mesh
from jax._src.partition_spec import P as P
from jax._src.pjit import reshard as reshard

from jax._src.shard_map import shard_map as shard_map
from jax._src.shard_map import smap as smap

from jax.ref import new_ref as new_ref
from jax.ref import empty_ref as empty_ref
from jax.ref import free_ref as free_ref
from jax.ref import freeze as freeze
from jax.ref import Ref as Ref

# Force import, allowing jax.interpreters.* to be used after import jax.
from jax.interpreters import ad, batching, mlir, partial_eval, pxla, xla
del ad, batching, mlir, partial_eval, pxla, xla

from jax._src.array import (
    make_array_from_single_device_arrays as make_array_from_single_device_arrays,
    make_array_from_callback as make_array_from_callback,
    make_array_from_process_local_data as make_array_from_process_local_data,
)

# These submodules are separate because they are in an import cycle with
# jax and rely on the names imported above.
from jax import custom_derivatives as custom_derivatives
from jax import custom_batching as custom_batching
from jax import custom_transpose as custom_transpose
from jax import api_util as api_util
from jax import distributed as distributed
from jax import debug as debug
from jax import dlpack as dlpack
from jax import dtypes as dtypes
from jax import errors as errors
from jax import export as export
from jax import ffi as ffi
from jax import image as image
from jax import lax as lax
from jax import monitoring as monitoring
from jax import nn as nn
from jax import numpy as numpy
from jax import ops as ops
from jax import profiler as profiler
from jax import random as random
from jax import scipy as scipy
from jax import sharding as sharding
from jax import memory as memory
from jax import stages as stages
from jax import tree_util as tree_util

# Also circular dependency.
from jax._src.array import Shard as Shard

import jax.experimental.compilation_cache.compilation_cache as _ccache
del _ccache

_deprecations = {
  # Remove in v0.10.0
  "array_ref": (
    "jax.array_ref was removed in JAX v0.9.0; use jax.new_ref instead.",
    None,
  ),
  "ArrayRef": (
    "jax.ArrayRef was removed in JAX v0.9.0; use jax.Ref instead.",
    None
  ),
  # Added for v0.8.1
  "device_put_replicated": (
    "jax.device_put_replicated is deprecated; use jax.device_put instead.",
    _deprecated_device_put_replicated
  ),
  # Added for v0.8.1
  "device_put_sharded": (
    "jax.device_put_sharded is deprecated; use jax.device_put instead.",
    _deprecated_device_put_sharded
  ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  device_put_replicated = _deprecated_device_put_replicated
  device_put_sharded = _deprecated_device_put_sharded
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing

import jax.lib  # TODO(phawkins): remove this export.  # noqa: F401

# trailer
del _deprecated_device_put_sharded
del _deprecated_device_put_replicated
