# Copyright 2022 The JAX Authors.
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

# Note: import <name> as <name> is required for names to be exported.
# See PEP 484 & https://github.com/jax-ml/jax/issues/7570

import jax._src.core as _src_core

from jax._src.core import (
  AbstractValue as AbstractValue,
  Atom as Atom,
  ParamDict as ParamDict,
  ShapedArray as ShapedArray,
  Trace as Trace,
  Tracer as Tracer,
  Value as Value,
  ensure_compile_time_eval as ensure_compile_time_eval,
  eval_context as eval_context,
  eval_jaxpr as eval_jaxpr,
  max_dim as max_dim,
  min_dim as min_dim,
)

_deprecations = {
  # Deprecated in JAX v0.10.0, TODO(jakevdp) finalize after v0.11.0
  "pytype_aval_mappings": (
    "jax.core.pytype_aval_mappings is deprecated.",
    _src_core.pytype_aval_mappings,
  ),
}

import typing as _typing
if _typing.TYPE_CHECKING:
  pytype_aval_mappings = _src_core.pytype_aval_mappings
  trace_ctx = _src_core.trace_ctx
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
del _src_core
