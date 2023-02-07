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

from jax._src.interpreters.xla import (
  AxisEnv as AxisEnv,
  Backend as Backend,
  Buffer as Buffer,
  ConcreteArray as ConcreteArray,
  Device as Device,
  DeviceArray as DeviceArray,
  Shape as Shape,
  ShapedArray as ShapedArray,
  SpatialSharding as SpatialSharding,
  TranslationContext as TranslationContext,
  TranslationRule as TranslationRule,
  XlaBuilder as XlaBuilder,
  XlaLoadedExecutable as XlaLoadedExecutable,
  XlaOp as XlaOp,
  XlaShape as XlaShape,
  _CppDeviceArray as _CppDeviceArray,
  _DeviceArray as _DeviceArray,
  abstractify as abstractify,
  aval_to_xla_shapes as aval_to_xla_shapes,
  axis_groups as axis_groups,
  axis_read as axis_read,
  backend_specific_translations as backend_specific_translations,
  canonicalize_dtype as canonicalize_dtype,
  canonicalize_dtype_handlers as canonicalize_dtype_handlers,
  check_backend_matches as check_backend_matches,
  dtype_to_primitive_type as dtype_to_primitive_type,
  extend_axis_env as extend_axis_env,
  extend_name_stack as extend_name_stack,
  jaxpr_collectives as jaxpr_collectives,
  lower_fun as lower_fun,
  make_device_array as make_device_array,
  make_op_metadata as make_op_metadata,
  new_name_stack as new_name_stack,
  parameter as parameter,
  partition_list as partition_list,
  primitive_subcomputation as primitive_subcomputation,
  pytype_aval_mappings as pytype_aval_mappings,
  register_collective_primitive as register_collective_primitive,
  register_initial_style_primitive as register_initial_style_primitive,
  register_translation as register_translation,
  sharding_to_proto as sharding_to_proto,
  translations as translations,
  xb as xb,
  xc as xc,
  xe as xe,
  xla_call as xla_call,
  xla_call_p as xla_call_p,
  xla_destructure as xla_destructure,
  xla_shape_handlers as xla_shape_handlers,
)

# TODO(phawkins): update users.
from jax._src.dispatch import (
  apply_primitive as apply_primitive,
  backend_compile as backend_compile,
  device_put as device_put,
)
