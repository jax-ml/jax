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

from functools import partial
import jax
from jax.nn import relu
from jax.tree_util import tree_map
from jax.scipy.special import logsumexp
from typing import Any, Callable, Generic, TypeVar
import jax.numpy as jnp
from dataclasses import dataclass

from util import JaxType, ArrayType, float_type, JaxVal, PRNGKey

State = Any
Cotangents = Any
ParamCotangents = Any
Params = Any
Values = Any
Residuals = Any
TangentType = Any

S = TypeVar("S")  # state
P = TypeVar("P")  # params
I = TypeVar("I")  # inputs
O = TypeVar("O")  # outputs
T = TypeVar("T")  # targets, or general-purpose

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

  # Default implementation only saves inputs and params. Subclasseses may override it.
  def forward_with_residuals(self, state: State, params: Params, inputs: Values
      ) -> tuple[Values, State, Residuals]:
    outputs, new_state = self.forward(state, params, inputs)
    return outputs, new_state, (params, inputs)

  def forward_general(self, state, params, inputs, save_residuals):
    if save_residuals:
      return self.forward_with_residuals(state, params, inputs)
    else:
      x, s_new = self.forward(state, params, inputs)
      return x, s_new, None

  # Default implementation using JAX's AD. Subclasseses are free to override it. Gets
  # residuals from `forward_with_residuals` so they need to be in cahoots.
  def backward(
      self, state: State, residuals:Residuals, output_cts: Cotangents
      ) -> tuple[ParamCotangents, Cotangents, State]:
    params, inputs = residuals
    _, vjpfun, new_state = jax.vjp(partial(self.forward, state), params, inputs, has_aux=True)
    params_cts, inputs_cts = vjpfun(output_cts)
    return params_cts, inputs_cts, new_state

  def grad(self, state: State, params: Params, inputs: Values) -> (Params, Values, State):
    assert self.output_type == float_type
    output, new_state, residuals = self.forward_with_residuals(state, params, inputs)
    grads, _, newer_state = self.backward(new_state, residuals, 1.0)
    return grads, output, newer_state

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

class CombinatorModel(Model):
  children : list[Model]
  def __init__(self, component_models, state_type, param_type, input_type, output_type):
    self.children = component_models
    super().__init__(state_type, param_type, input_type, output_type)

  def forward(self, state, params, inputs):
    output, state, _ = self.forward_general(state, params, inputs, False)
    return output, state

  def forward_with_residuals(self, state, params, inputs):
    return self.forward_general(state, params, inputs, True)

  # Combinators should implement forward_general instead of `forward` or `forward_with_residuals`
  def forward_general(self, state, params, inputs, save_residuals):
    raise NotImplementedError(type(self)) # subclass should implement

  # Combinators need to implement `backward` themselves. Using JAX's AD on `forward` will
  # ignore component models' custom `backward` implementations.
  def backward(self, state, residuals, output_cts: Values):
    raise NotImplementedError(type(self)) # subclass should implement

class Sequential(CombinatorModel):
  def __init__(self, models, input_type=None):
    if models:
      output_type = models[-1].output_type
      assert input_type is None or models[0].input_type == input_type
    else:
      assert input_type is not None
      output_type = input_type

    for m1, m2 in zip(models[:-1], models[1:]):
      assert m1.output_type == m2.input_type

    state_type = [m.state_type for m in models]
    param_type = [m.param_type for m in models]
    super().__init__(models, state_type, param_type, input_type, output_type)

  def initial_state(self, prng):
    prngs = jax.random.split(prng, len(self.children))
    return [m.initial_state(p) for p, m in zip(prngs, self.children)]

  def forward_general(self, state, params, inputs, save_residuals):
    x = inputs
    states = []
    residuals = []
    for m, s, p in zip(self.children, state, params):
      x, s_new, r = m.forward_general(s, p, x, save_residuals)
      states.append(s_new)
      residuals.append(r)
    return x, states, residuals

  def backward(self, state, residuals, ct):
    param_cts_rev = []
    states_rev = []
    for m, s, i in list(zip(self.children, state, residuals))[::-1]:
      p_ct, ct, s_new = m.backward(s, i, ct)
      states_rev.append(s_new)
      param_cts_rev.append(p_ct)
    return param_cts_rev[::-1], ct, states_rev[::-1]

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
class ModelWithLoss(CombinatorModel):
  def __init__(self, model:Model, label_type:JaxType):
    state_type = model.state_type
    param_type = model.param_type
    input_type = (model.input_type, label_type)
    output_type = float_type
    self.model = model
    super().__init__([model], state_type, param_type, input_type, output_type)

  def compute_loss(self, preds, labels) -> float:
    raise NotImplementedError(type(self)) # subclass should implement

  def forward_general(self, state, params, inputs_labels, save_residuals):
    inputs, labels = inputs_labels
    preds, state, model_residuals = self.model.forward_general(state, params, inputs, save_residuals)
    loss = self.compute_loss(preds, labels)
    return loss, state, (model_residuals, preds, labels)

  def backward(self, state: State, residuals:Residuals, loss_ct: Values) -> tuple[Params, Values, State]:
    model_residuals, preds, labels = residuals
    _, vjpfun = jax.vjp(self.compute_loss, preds, labels)
    preds_ct, labels_ct = vjpfun(loss_ct)
    params_ct, model_inputs_ct, new_state  = self.model.backward(state, model_residuals, preds_ct)
    return params_ct, (model_inputs_ct, labels_ct), new_state

class WithCategoricalLoss(ModelWithLoss):
  def __init__(self, model:Model[S,P,I,O]):
    assert isinstance(model.output_type, ArrayType)
    assert len(model.output_type.shape) == 1
    num_categories = model.output_type.shape[0]
    label_type = ArrayType((num_categories,), float)
    super().__init__(model, label_type)

  def compute_loss(self, preds: O, targets: T) -> float:
    return -jnp.mean(jnp.sum(preds * targets, axis=1))
