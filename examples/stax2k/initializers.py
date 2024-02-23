# ---
# Copyright 2024 The JAX Authors.
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

import jax
from util import Any, JaxType, JaxVal, tree_map_with_index

Key = Any
State = Any
Params = Any
Values = Any
OptState = Any
TangentType = Any

class Initializer:
  param_type : JaxType
  def new_params(self, k:Key) -> JaxVal:
    raise NotImplementedError(type(self))

class IIDNormalInitializer(Initializer):
  def __init__(self, param_type:JaxType, offset=1.0, scale=1.0):
    # TODO: check that the jaxtype is a pytree of floats
    self.param_type = param_type
    self.offset = offset
    self.scale = scale

  def new_params(self, k:Key) -> JaxVal:
    def new_leaf(i, ty):
      ki = jax.random.fold_in(k, i)
      return jax.random.normal(ki, shape=ty.shape, dtype=ty.dtype)
    return tree_map_with_index(self.param_type, new_leaf)
