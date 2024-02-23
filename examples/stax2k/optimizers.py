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

from typing import Any

import jax
from util import tree_map, JaxType, PRNGKey, jax_type

Key = Any
State = Any
Params = Any
Values = Any
OptState = Any
TangentType = Any

def is_pytree_of_floats(ty:JaxType):
  return True  # todo

class Optimizer:
  param_type    : JaxType
  opt_state_type : JaxType

  def initial_state(self, key):
    assert False

  def opt_step(self, key: PRNGKey, opt_state:OptState, params:Params, grads:TangentType) -> tuple[Params, State]:
    raise NotImplementedError(type(self))

  def check_types(self):
    assert False

class SGDOptimizer(Optimizer):
  def __init__(self, param_type: JaxType, scale:float=0.1, lr:float=0.001):
    assert is_pytree_of_floats(param_type)
    self.param_type = param_type
    self.opt_state_type = ()
    self.scale = scale
    self.lr    = lr

  def initial_state(self, _):
    return ()

  def opt_step(self, _, __, params, grads):
    def leaf_step(param, grad):
      return param - self.lr * grad
    new_params = tree_map(leaf_step, params, grads)
    return (new_params, ())

class SerialOptimizer(Optimizer):
  def __init__(self, optimizers:list[Optimizer]):
    self.optimizers = optimizers

  def initial_state(self, key):
    assert False

  def opt_step(self, _, __, params, grads):
    assert False

# Could factor as a particular "pure optimizer". But this is less indirection
class ClipNorms(Optimizer):
  def __init__(self):
    assert False

  def initial_state(self, key):
    assert False

  def opt_step(self, _, __, params, grads):
    assert False

