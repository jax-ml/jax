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

# flake8: noqa: F401
from .config import config
from .api import (
  ad,  # TODO(phawkins): update users to avoid this.
  argnums_partial,  # TODO(phawkins): update Haiku to not use this.
  checkpoint,
  curry,  # TODO(phawkins): update users to avoid this.
  custom_ivjp,
  custom_gradient,
  custom_jvp,
  custom_vjp,
  custom_transforms,
  defjvp,
  defjvp_all,
  defvjp,
  defvjp_all,
  device_count,
  device_get,
  device_put,
  devices,
  disable_jit,
  eval_shape,
  flatten_fun_nokwargs,  # TODO(phawkins): update users to avoid this.
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
  make_jaxpr,
  mask,
  partial,  # TODO(phawkins): update callers to use functools.partial.
  pmap,
  pxla,  # TODO(phawkins): update users to avoid this.
  remat,
  shapecheck,
  ShapedArray,
  ShapeDtypeStruct,
  soft_pmap,
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
from .version import __version__

# These submodules are separate because they are in an import cycle with
# jax and rely on the names imported above.
from . import image
from . import lax
from . import nn
from . import random

def _init():
  import os
  os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '1')

  from . import numpy # side-effecting import sets up operator overloads

_init()
del _init
