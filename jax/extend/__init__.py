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

"""Modules for JAX extensions.

The :mod:`jax.extend` module provides modules for access to JAX
internal machinery. See
`JEP #15856 <https://docs.jax.dev/en/latest/jep/15856-jex.html>`_.

This module is not the only means by which JAX aims to be
extensible. For example, the main JAX API offers mechanisms for
`customizing derivatives
<https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>`_,
`registering custom pytree definitions
<https://docs.jax.dev/en/latest/pytrees.html#extending-pytrees>`_,
and more.

API policy
----------

Unlike the
`public API <https://docs.jax.dev/en/latest/api_compatibility.html>`_,
this module offers **no compatibility guarantee** across releases.
Breaking changes will be announced via the
`JAX project changelog <https://docs.jax.dev/en/latest/changelog.html>`_.
"""

from jax.extend import (
    backend as backend,
    core as core,
    ffi as _ffi,
    linear_util as linear_util,
    mlir as mlir,
    random as random,
    sharding as sharding,
    source_info_util as source_info_util,
)

_deprecations = {
    # Added 2025-7-7
    "ffi": (
        "The jax.extend.ffi module was deprecated in JAX v0.5.0, use jax.ffi instead.",
        _ffi,
    ),
}

import typing
if typing.TYPE_CHECKING:
  ffi = _ffi
else:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del typing
del _ffi
