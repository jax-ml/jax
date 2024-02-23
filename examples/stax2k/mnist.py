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

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp

import datasets
from models import Model, Sequential, DenseModel, NormalizeLogits, CategoricalLoss, Relu
from initializers import Initializer, IIDNormalInitializer
from optimizers import Optimizer, SGDOptimizer
from util import JaxType, JaxVal, PRNGKey

# === generic training step (could be a library) ===

# Full state of training loop. Serialize this to resume training after a crash.
@dataclass
class TrainingState:
  train_iter   : int
  opt_state    : JaxVal
  model_state  : JaxVal
  model_params : JaxVal

def initial_train_state(model: Model, initializer: Initializer, optimizer: Optimizer, key: PRNGKey):
  k1, k2, k3 = jax.random.split(key, num=3)
  return TrainingState(
    train_iter = 0,
    opt_state = optimizer.initial_state(k1),
    model_state = model.initial_state(k2),
    model_params = initializer.new_params(k3))

def train_state_type(model: Model, initializer: Initializer, optimizer: Optimizer, key: PRNGKey) -> JaxType:
  assert False

def train_step(
    model: Model, opt:Optimizer, key: PRNGKey, batch, ts:TrainingState
    ) -> TrainingState:
  grads, new_model_state = model.grad(ts.model_state, ts.model_params, batch)
  new_params, new_opt_state = opt.opt_step(key, ts.opt_state, ts.model_params, grads)
  return TrainingState(
    train_iter = ts.train_iter + 1,
    opt_state = new_opt_state,
    model_state = new_model_state,
    model_params = new_params)

# === checkpointing (could be a library) ===

CheckpointName = Any  # string file name
DirPath = Any

def load_checkpoint(checkpoint_type: JaxType, checkpoint_dir: DirPath, checkpoint_name: CheckpointName):
  assert False

def save_checkpoint(checkpoint_data: JaxVal, checkpoint_dir: DirPath) -> CheckpointName:
  pass # todo!

# === data loading (could be a library) ===

Batch = Any
@dataclass
class BatchedData:
  batches : list[Batch]
  def __getitem__(self, i):
    return self.batches[i % len(self.batches)]

def mnist_data(batch_size):
  train_images, train_labels, test_images, test_labels = datasets.mnist()
  num_train = train_images.shape[0]
  num_batches = num_train % batch_size
  batches = []
  for i in range(num_batches):
    l = i * batch_size
    r = l + batch_size
    batches.append((train_images[l:r], train_labels[l:r]))
  eval_data = (train_images, train_labels, test_images, test_labels)
  return BatchedData(batches), eval_data

# === training config ===

@dataclass
class MLPConfig:
 input_size   : int
 output_size  : int
 hidden_sizes : list[int]

@dataclass
class TrainingConfig:
  model_cfg            : MLPConfig
  batch_size           : int
  param_scale          : float
  learning_rate        : float
  max_iters            : int
  init_key             : int
  eval_period          : int
  checkpoint_period    : int
  checkpoint_dir       : str
  checkpoint_to_load   : str | None

# === training loop implementation ===

def eval_accuracy(model:Model, ts: TrainingState, features, labels):
  logits, _ = model.forward(ts.model_state, ts.model_params, features)
  pred_classes = jnp.argmax(logits, axis=1)
  label_classes = jnp.argmax(labels, axis=1)
  return jnp.mean(pred_classes == label_classes)

def run_evals(model: Model, eval_data, ts: TrainingState):
  train_images, train_labels, test_images, test_labels = eval_data
  print(f"Iteration: {ts.train_iter}")
  print(f"Train accuracy: {eval_accuracy(model, ts, train_images, train_labels)}")
  print(f"Test  accuracy: {eval_accuracy(model, ts, test_images, test_labels)}")

def build_mlp(cfg: MLPConfig) -> Model:
  input_sizes = [cfg.input_size] + cfg.hidden_sizes
  output_sizes = cfg.hidden_sizes + [cfg.output_size]
  nonlinearities = [Relu] * len(cfg.hidden_sizes) + [NormalizeLogits]
  layers = []
  for (n_in, n_out, nonlinearity) in zip(input_sizes, output_sizes, nonlinearities):
    layers.append(DenseModel(n_in, n_out))
    layers.append(nonlinearity(n_out))
  return Sequential(layers)

def run_training_loop(cfg:TrainingConfig):
  k1, k2 = jax.random.split(cfg.init_key)
  model = build_mlp(cfg.model_cfg)
  model_with_loss = CategoricalLoss(model, 555)
  initializer = IIDNormalInitializer(model.param_type, scale=cfg.param_scale)
  optimizer = SGDOptimizer(param_type=model.param_type, scale=cfg.param_scale)
  train_batches, eval_data = mnist_data(cfg.batch_size)
  assert model.param_type == optimizer.param_type
  assert model.param_type == initializer.param_type

  if cfg.checkpoint_to_load:
    ts_type = train_state_type(model, initializer, optimizer)
    ts = load_checkpoint(ts_type, cfg.checkpoint_to_load)
  else:
    ts = initial_train_state(model, initializer, optimizer, k1)

  while ts.train_iter < cfg.max_iters:
    if ts.train_iter % cfg.eval_period == 0:
      run_evals(model, eval_data, ts)

    if ts.train_iter % cfg.checkpoint_period == 0:
      checkpoint_name = save_checkpoint(ts, cfg.checkpoint_dir)
      print(f"Checkpoint saved: {checkpoint_name}")

    batch = train_batches[ts.train_iter]
    key = jax.random.fold_in(k2, ts.train_iter)
    ts = train_step(model_with_loss, optimizer, key, batch, ts)

if __name__ == "__main__":

  model_config = MLPConfig(
    input_size   = 784,
    output_size  = 10,
    hidden_sizes = [1024, 1024])

  training_config = TrainingConfig(
    model_cfg     = model_config,
    init_key      = jax.random.key(0),
    batch_size    = 128,
    learning_rate = 0.01,
    param_scale   = 0.1,
    max_iters         = 1000,
    eval_period       = 20,
    checkpoint_period = 20,
    checkpoint_dir     = None,
    checkpoint_to_load = None)

  run_training_loop(training_config)
