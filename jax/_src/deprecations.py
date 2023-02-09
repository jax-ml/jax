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

import warnings

# Module __getattr__ factory that warns if deprecated names are used.
#
# Example usage:
# from jax._src.interpreters.pxla import (
#   Mesh as _deprecated_Mesh,
# )
#
# _deprecations = {
#   # Added Feb 8, 2023:
#   "Mesh": (
#     "jax.interpreters.pxla.Mesh is deprecated. Use jax.sharding.Mesh.",
#     _deprecated_Mesh,
#   ),
# }
#
# from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
# __getattr__ = _deprecation_getattr(__name__, _deprecations)
# del _deprecation_getattr, _deprecations

# Note that type checkers such as Pytype will not know about the deprecated
# names. If it is desirable that a deprecated name is known to the type checker,
# add:
# import typing
# if typing.TYPE_CHECKING:
#   from jax._src.interpreters.pxla import (
#     Mesh as Mesh,
#   )
# del typing
def deprecation_getattr(module, deprecations):
  def getattr(name):
    if name in deprecations:
      message, fn = deprecations[name]
      if fn is None:
        raise AttributeError(message)
      warnings.warn(message, DeprecationWarning)
      return fn
    raise AttributeError(f"module {module!r} has no attribute {name!r}")

  return getattr
