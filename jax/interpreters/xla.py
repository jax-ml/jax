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
  abstractify as abstractify,
  canonicalize_dtype as canonicalize_dtype,
  canonicalize_dtype_handlers as canonicalize_dtype_handlers,
  pytype_aval_mappings as pytype_aval_mappings,

  # Deprecations
  backend_specific_translations as _deprecated_backend_specific_translations,
  register_translation as _deprecated_register_translation,
  translations as _deprecated_translations,
  xla_destructure as _deprecated_xla_destructure,
  TranslationContext as _deprecated_TranslationContext,
  TranslationRule as _deprecated_TranslationRule,
)
from jax._src.interpreters.pxla import (
  axis_groups as _deprecated_axis_groups,
)

from jax._src.core import (
  ShapedArray as _deprecated_ShapedArray,
  ConcreteArray as _deprecated_ConcreteArray,
)

from jax._src.compiler import (
  backend_compile as _deprecated_backend_compile,
)

from jax._src.dispatch import (
  apply_primitive as apply_primitive,
)

from jax._src.sharding_impls import (
  AxisEnv as _deprecated_AxisEnv,
)

from jax._src import xla_bridge as xb
from jax._src.lib import xla_client as xc  # type: ignore

xe = xc._xla
Backend = xe.Client

# Deprecations
_deprecations = {
    # Added Aug 29, 2023:
    "backend_specific_translations": (
        "jax.interpreters.xla.backend_specific_translations is deprecated. "
        "Register custom primitives via jax.interpreters.mlir instead.",
        _deprecated_backend_specific_translations,
    ),
    "translations": (
        "jax.interpreters.xla.translations is deprecated. "
        "Register custom primitives via jax.interpreters.mlir instead.",
        _deprecated_translations,
    ),
    "register_translation": (
        "jax.interpreters.xla.register_translation is deprecated. "
        "Register custom primitives via jax.interpreters.mlir instead.",
        _deprecated_register_translation,
    ),
    "xla_destructure": (
        "jax.interpreters.xla.xla_destructure is deprecated. "
        "Register custom primitives via jax.interpreters.mlir instead.",
        _deprecated_xla_destructure,
    ),
    "TranslationRule": (
        "jax.interpreters.xla.TranslationRule is deprecated. "
        "Register custom primitives via jax.interpreters.mlir instead.",
        _deprecated_TranslationRule,
    ),
    "TranslationContext": (
        "jax.interpreters.xla.TranslationContext is deprecated. "
        "Register custom primitives via jax.interpreters.mlir instead.",
        _deprecated_TranslationContext,
    ),
    "axis_groups": (
        "jax.interpreters.xla.axis_groups is deprecated.",
        _deprecated_axis_groups,
    ),
    "ShapedArray": (
        "jax.interpreters.xla.ShapedArray is deprecated. "
        "Use jax.core.ShapedArray instead.",
        _deprecated_ShapedArray,
    ),
    "ConcreteArray": (
        "jax.interpreters.xla.ConcreteArray is deprecated. "
        "Use jax.core.ConcreteArray instead.",
        _deprecated_ConcreteArray,
    ),
    "AxisEnv": (
        "jax.interpreters.xla.AxisEnv is deprecated.",
        _deprecated_AxisEnv,
    ),
    "backend_compile": (
        "jax.interpreters.xla.backend_compile is deprecated.",
        _deprecated_backend_compile,
    ),
    "XlaOp": (
        "jax.interpreters.xla.XlaOp is deprecated. "
        "Register custom primitives via jax.interpreters.mlir instead.",
        xc.XlaOp,
    ),
}

import typing
if typing.TYPE_CHECKING:
  backend_specific_translations = _deprecated_backend_specific_translations
  translations = _deprecated_translations
  register_translation = _deprecated_register_translation
  xla_destructure = _deprecated_xla_destructure
  TranslationRule = _deprecated_TranslationRule
  TranslationContext = _deprecated_TranslationContext
  axis_groups = _deprecated_axis_groups
  ShapedArray = _deprecated_ShapedArray
  ConcreteArray = _deprecated_ConcreteArray
  AxisEnv = _deprecated_AxisEnv
  backend_compile = _deprecated_backend_compile
  XlaOp = xc.XlaOp
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
