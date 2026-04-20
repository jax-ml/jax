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

from jax._src.interpreters.batching import (
  axis_primitive_batchers as axis_primitive_batchers,
  bdim_at_front as bdim_at_front,
  broadcast as broadcast,
  defbroadcasting as defbroadcasting,
  defreducer as defreducer,
  defvectorized as defvectorized,
  fancy_primitive_batchers as fancy_primitive_batchers,
  not_mapped as not_mapped,
  primitive_batchers as primitive_batchers,
  register_vmappable as register_vmappable,
  unregister_vmappable as unregister_vmappable,
)


_deprecations = {
  # Deprecated in JAX v0.7.1; removed in JAX v0.10.0.
  # TODO(jakevdp):remove this for JAX v0.11.0
  "NotMapped": (
    "jax.interpreters.batching.NotMapped is deprecated.",
    None,
  ),
}


import typing as _typing
if not _typing.TYPE_CHECKING:
  from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
  __getattr__ = _deprecation_getattr(__name__, _deprecations)
  del _deprecation_getattr
del _typing
