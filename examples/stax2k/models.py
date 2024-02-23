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
from jax.nn import relu
from jax.tree_util import tree_map
from jax.scipy.special import logsumexp
from typing import Any, Callable, Generic, TypeVar
import jax.numpy as jnp
from dataclasses import dataclass

from util import JaxType, ArrayType, float_type, JaxVal, PRNGKey

State = Any
Params = Any
Values = Any
Intermediates = Any
TangentType = Any

S = TypeVar("S")  # state
P = TypeVar("P")  # params
I = TypeVar("I")  # inputs
O = TypeVar("O")  # outputs

T = TypeVar("O")  # targets, or general-purpose

@dataclass
class Model(Generic[S,P,I,O]):
  state_type  : JaxType
  param_type  : JaxType
  input_type  : JaxType  # no batch dimension
  output_type : JaxType  # no batch dimension

  def initial_state(self, prng:PRNGKey) -> State:
    raise NotImplementedError(type(self)) # subclass should implement

  def forward(self, state: State, params: Params, inputs: Values) -> tuple[Values, State]:
    raise NotImplementedError(type(self)) # subclass should implement

  # Default implementation only saves inputs. Subclasseses are free to override it.
  def forward_with_intermediates(self, state: State, params: Params, inputs: Values
      ) -> tuple[Values, State, Intermediates]:
    outputs, new_state = self.forward(state, params, inputs)
    return outputs, new_state, (params, inputs)

  # Default implementation. Subclasseses are free to override it. Gets
  # intermediates from `forward_with_intermediates` so they need to be in cahoots.
  def backward(self, state: State, intermediates:Intermediates, output_cts: Values) -> tuple[Params, Values, State]:
    params, inputs = intermediates
    def forward_fun(params_inputs, state):
      params, inputs = params_inputs
      return self.forward(state, params, inputs)

    _, vjpfun, new_state = jax.vjp(forward_fun, (params, inputs), state, has_aux=True)
    (params_cts, inputs_cts), _ = vjpfun(output_cts)
    return params_cts, inputs_cts, new_state

  def grad(self, state: State, params: Params, inputs: Values) -> (Params, State):
    assert self.output_type == float_type
    _, new_state, intermediates = self.forward_with_intermediates(state, params, inputs)
    grads, _, newer_state = self.backward(new_state, intermediates, 1.0)
    return grads, newer_state

class DenseModel(Model):
  def __init__(self, n_in, n_out, init_scale=1.0):
    self.n_in  = n_in
    self.n_out = n_out
    self.init_scale = init_scale
    param_type = dict(
      w = ArrayType((n_out, n_in), float),
      b = ArrayType((n_out,), float))
    input_type = ArrayType((n_in,), float)
    output_type = ArrayType((n_out,), float)
    super().__init__((), param_type, input_type, output_type)

  def initial_state(self, prng): return ()

  def forward(self, state, params, inputs):
    w, b = params['w'], params['b']
    outputs = jnp.dot(inputs, w.T) + b
    return outputs, state

# === second-order combinators (models parameterized by models) ===

class Sequential(Model):
  def __init__(self, modules, input_type=None):
    self.modules = modules

    if modules:
      output_type = modules[-1].output_type
      assert input_type is None or modules[0].input_type == input_type
    else:
      assert input_type is not None
      output_type = input_type

    for m1, m2 in zip(modules[:-1], modules[1:]):
      assert m1.output_type == m2.input_type

    state_type = [m.state_type for m in modules]
    param_type = [m.param_type for m in modules]
    super().__init__(state_type, param_type, input_type, output_type)

  def initial_state(self, prng):
    prngs = jax.random.split(prng, len(self.modules))
    return [m.initial_state(p) for p, m in zip(prngs, self.modules)]

  def forward(self, state, params, inputs):
    x = inputs
    states = []
    for m, s, p in zip(self.modules, state, params):
      x, s_new = m.forward(s, p, x)
      states.append(s_new)
    return x, states

# === pure functions ===

# Models that implement pure functions without any state, prng or trainable
# parameters. E.g. normalization layers, max-pooling layers, etc.
class FunctionModel(Model):
  def __init__(self, input_type, output_type):
    super().__init__((), (), input_type, output_type)

  def apply_function(self, inputs):
    raise NotImplementedError(type(self)) # subclass should implement

  def initial_state(self , prng:PRNGKey) -> State:
    return ()

  def forward(self, _, params: Params, inputs: Values) -> tuple[Values, State]:
    return self.apply_function(inputs), ()

class NormalizeLogits(FunctionModel):
  def __init__(self, num_classes):
    super().__init__(
      input_type  = ArrayType((num_classes,), float),
      output_type = ArrayType((num_classes,), float))

  def apply_function(self, logits):
    return logits - logsumexp(logits, axis=1, keepdims=True)

# === nonlinearities ===

class Relu(FunctionModel):
  def __init__(self, num_features):
    super().__init__(
      input_type  = ArrayType((num_features,), float),
      output_type = ArrayType((num_features,), float))

  def apply_function(self, activations):
    return relu(activations)

# === loss functions ===

# Wraps e.g. an `image -> logits` model to make an
# `(image, label) -> scalar_loss` model.
class Loss(Model):
  def __init__(self, model:Model, label_type:JaxType):
    state_type = model.state_type
    param_type = model.param_type
    input_type = (model.input_type, label_type)
    output_type = float_type
    self.model = model
    super().__init__(state_type, param_type, input_type, output_type)

  def compute_loss(self, outputs, labels) -> float:
    raise NotImplementedError(type(self)) # subclass should implement

  def forward(self, state: State, params: Params, inputs_labels: Values) -> tuple[Values, State]:
    inputs, labels = inputs_labels
    outputs, state = self.model.forward(state, params, inputs)
    loss = self.compute_loss(outputs, labels)
    return loss, state

class CategoricalLoss(Loss):
  def __init__(self, model:Model[S,P,I,O], target_type:JaxType):
    super().__init__(model, target_type)

  def compute_loss(self, preds: O, targets: T) -> float:
    return -jnp.mean(jnp.sum(preds * targets, axis=1))
