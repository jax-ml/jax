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
)

from jax._src.sharding_impls import (
  AxisEnv as AxisEnv,
)

from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc  # type: ignore

XlaOp = xc.XlaOp
xe = xc._xla
Backend = xe.Client
