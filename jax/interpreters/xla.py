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
  TranslationContext as TranslationContext,
  TranslationRule as TranslationRule,
  abstractify as abstractify,
  axis_groups as axis_groups,
  backend_specific_translations as backend_specific_translations,
  canonicalize_dtype as canonicalize_dtype,
  canonicalize_dtype_handlers as canonicalize_dtype_handlers,
  check_backend_matches as check_backend_matches,
  parameter as parameter,
  pytype_aval_mappings as pytype_aval_mappings,
  register_collective_primitive as register_collective_primitive,
  register_initial_style_primitive as register_initial_style_primitive,
  register_translation as register_translation,
  sharding_to_proto as sharding_to_proto,
  translations as translations,
  xla_call as xla_call,
  xla_call_p as xla_call_p,
  xla_destructure as xla_destructure,
  xla_shape_handlers as xla_shape_handlers,
)

from jax._src.core import (
  ShapedArray as ShapedArray,
  ConcreteArray as ConcreteArray,
)

# TODO(phawkins): update users.
from jax._src.dispatch import (
  apply_primitive as apply_primitive,
  backend_compile as backend_compile,
  device_put as device_put,
)

from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc  # type: ignore

_deprecated_Device = xc.Device
XlaOp = xc.XlaOp
xe = xc._xla
Backend = xe.Client
Buffer = xc.Buffer
_CppDeviceArray = xe.Buffer

from jax._src.device_array import (
  make_device_array as make_device_array,
  _DeviceArray as _DeviceArray,
  DeviceArray as _deprecated_DeviceArray,
)

_deprecations = {
  # Added Feb 9, 2023:
  "Device": (
    "jax.interpreters.xla.Device is deprecated. Use jax.Device instead.",
    _deprecated_Device,
  ),
  "DeviceArray": (
    "jax.interpreters.xla.DeviceArray is deprecated. Use jax.Array instead.",
    _deprecated_DeviceArray,
  ),
}

from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr

import typing
if typing.TYPE_CHECKING:
  Device = xc.Device
  from jax._src.device_array import (
    DeviceArray as DeviceArray,
  )
del typing
