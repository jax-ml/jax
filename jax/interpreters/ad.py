# Copyright 2023 The JAX Authors.
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

from __future__ import annotations

from jax._src.interpreters.ad import (
  JVPTrace as JVPTrace,
  JVPTracer as JVPTracer,
  UndefinedPrimal as UndefinedPrimal,
  Zero as Zero,
  add_jaxvals as add_jaxvals,
  add_jaxvals_p as add_jaxvals_p,
  add_tangents as add_tangents,
  defbilinear as defbilinear,
  defjvp as defjvp,
  defjvp2 as defjvp2,
  deflinear as deflinear,
  deflinear2 as deflinear2,
  get_primitive_transpose as get_primitive_transpose,
  instantiate_zeros as instantiate_zeros,
  is_undefined_primal as is_undefined_primal,
  jvp as jvp,
  linearize as linearize,
  primitive_jvps as primitive_jvps,
  primitive_transposes as primitive_transposes,
  zeros_like_aval as zeros_like_aval,
)


_deprecations = {
    # Deprecated in v0.9.0; finalized in v0.10.0.
    # TODO(jakevdp) remove entry in v0.11.0.
    "reducing_transposes": (
        (
            "jax.interpreters.ad.reducing_transposes was deprecated in v0.9.0."
            " and removed in v0.10.0. It has been unused since JAX v0.4.38."
        ),
        None,
    ),
}

import typing
if not typing.TYPE_CHECKING:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
