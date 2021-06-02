# coding=utf-8
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

# TODO(phawkins): fix users of these aliases and delete this file.

# flake8: noqa: F401
from jax._src.config import (
  FLAGS,
)
from jax._src.api import (
  AxisName,
  ShapedArray,
  ShapeDtypeStruct,
  custom_jvp,
  device_count,
  device_get,
  device_put,
  device_put_replicated,
  device_put_sharded,
  devices,
  eval_shape,
  flatten_fun_nokwargs,
  grad,
  jacfwd,
  jacobian,
  jacrev,
  jit,
  jvp,
  linear_transpose,
  local_device_count,
  make_jaxpr,
  pmap,
  raise_to_shaped,
  tree_flatten,
  tree_map,
  tree_multimap,
  vjp,
  vmap,
  wraps,
  xla_computation,
  _check_callable,
  _std_basis,
  _unravel_array_into_pytree,
)
