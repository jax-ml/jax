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

from jax._src.lib import (
    _jax as _jax
)

deserialize_portable_artifact = _jax.mlir.deserialize_portable_artifact
serialize_portable_artifact = _jax.mlir.serialize_portable_artifact
refine_polymorphic_shapes = _jax.mlir.refine_polymorphic_shapes
hlo_to_stablehlo = _jax.mlir.hlo_to_stablehlo

del _jax
