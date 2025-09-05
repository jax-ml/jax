# Copyright 2025 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Default Hyperparameter configuration.

`get_config`
https://github.com/google/flax/blob/a1451a0e55dfc55b4bc9ca42f88bf28744357d8c/examples/gemma/configs/default.py
"""
import dataclasses
from typing import Any

from flax import nnx
from flax.examples.gemma import train
from flax.examples.gemma import transformer as transformer_lib
from flax.examples.gemma import utils
from flax.examples.gemma.train import MeshRules, TrainConfig
import jax
import jax.numpy as jnp
import optax


def get_fake_batch(batch_size: int) -> tuple[jnp.ndarray, ...]:
  """Returns fake data for the given batch size."""
  rng = jax.random.PRNGKey(0)
  batch = {}
  for k in (
      'inputs',
      'inputs_position',
      'inputs_segmentation',
      'targets',
      'targets_position',
      'targets_segmentation',
  ):
    batch[k] = jax.random.randint(rng, (batch_size, 128), 0, 9999999, jnp.int32)
  return batch


def get_apply_fn_and_args() -> tuple[jax.stages.Traced, tuple[Any, ...]]:
  """Returns the apply function and args for the model."""
  config = get_config()

  devices_array = utils.create_device_mesh(config)
  mesh = jax.sharding.Mesh(devices_array, config.mesh_axes)

  rng = jax.random.PRNGKey(config.seed)
  _, init_rng = jax.random.split(rng)

  model_config = transformer_lib.TransformerConfig.from_version_name(
      config.transformer_name,
      num_embed=config.vocab_size,
      dtype=jnp.bfloat16 if config.use_bfloat16 else jnp.float32,
      axis_rules=config.axis_rules,
  )

  def constructor(config: transformer_lib.TransformerConfig, key: jax.Array):
    return transformer_lib.Transformer(config, rngs=nnx.Rngs(params=key))

  learning_rate_fn = train.create_learning_rate_schedule(
      learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
  )

  optimizer = optax.adamw(
      learning_rate_fn,
      b1=0.9,
      b2=0.98,
      eps=1e-9,
      weight_decay=config.weight_decay,
  )

  train_state, state_sharding = utils.setup_initial_state(
      constructor, optimizer, model_config, init_rng, mesh
  )
  data_sharding = jax.sharding.NamedSharding(mesh, jax.P(config.data_sharding))

  batch = get_fake_batch(config.per_device_batch_size)
  batch = jax.tree.map(lambda x: jnp.asarray(x, device=data_sharding), batch)

  apply_fn = jax.jit(
      train.train_step,
      in_shardings=(
          state_sharding,
          data_sharding,
      ),  # type: ignore
      out_shardings=(state_sharding, None),  # type: ignore
      static_argnames=('learning_rate_fn', 'label_smoothing'),
      donate_argnums=0,
  )
  return apply_fn, (train_state, batch, learning_rate_fn, 0.0)


@dataclasses.dataclass(unsafe_hash=True)
class Config:
  # Path to load or store sentencepiece vocab file.
  vocab_path: str | None = None
  # Vocabulary size if `vocab_path` is not given.
  vocab_size: int = (
      35_000  # lm1b dataset vocab size: 35913  (Gemma expected vocab size: 262_144)
  )
  # Maximum number of characters to use for training.
  max_corpus_chars: int = 10**7
  # Name of TFDS translation dataset to use.
  dataset_name: str = 'lm1b'
  # Optional name of TFDS translation dataset to use for evaluation.
  eval_dataset_name: str = 'lm1b'
  # Optional name of TFDS split to use for evaluation.
  eval_split: str = 'test'
  # Per device batch size for training.
  per_device_batch_size: int = 32
  # Per device batch size for training.
  eval_per_device_batch_size: int = 32

  # Prompt for language model sampling
  prompts: tuple[str, ...] = (
      'Paris is a the capital',
      'Flax is a',
      # From train set:
      'The shutdown was aimed at creating efficiencies as',
      # -> the plant was already operating at its maximum capacity of 3,000 tonnes of cellulose paste per day
      'A big theme of this hire is that there are parts of',
      # -> our operations that to use a pretty trite phrase , need to be taken to the next level ...
      # From test set:
      'Because of Bear Stearns , many analysts are',
      # -> raising the odds that a 2008 recession could be worse than expected
      'Next month , the Brazilian bourse',
      # -> opens a London office',
  )
  # Temperature for top_p sampling.
  sampling_temperature: float = 0.0
  # Top-p sampling threshold.
  sampling_top_p: float = 0.95

  # Number of steps to take during training.
  num_train_steps: int = 500_000
  # Number of steps to take during evaluation.
  # Large enough to evaluate all samples: 306_688 / (32 * 8) = 1198
  num_eval_steps: int = 2_000
  # Number of steps to generate predictions.
  # -1 will use the whole eval dataset.
  num_predict_steps: int = 50
  # Base learning rate.
  learning_rate: float = 0.0016
  # Linear learning rate warmup.
  warmup_steps: int = 1000
  # Cross entropy loss label smoothing.
  label_smoothing: float = 0.0
  # Decay factor for AdamW style weight decay.
  weight_decay: float = 0.1
  # Maximum length cutoff for training examples.
  max_target_length: int = 128
  # Maximum length cutoff for eval examples.
  max_eval_target_length: int = 512

  # Gemma transformer name.
  # Possible values defined in transformer.TransformerConfig:
  # (gemma_2b, gemma_7b, gemma2_2b, gemma2_9b, gemma2_27b, gemma3_1b, gemma3_4b, ...)
  transformer_name: str | None = 'gemma3_1b'
  # or alternatively define the model using the dict of parameters
  transformer_params: dict[Any, Any] | None = None

  # Whether to save model checkpoints.
  save_checkpoints: bool = True
  # Whether to restore from existing model checkpoints.
  restore_checkpoints: bool = True
  # Save a checkpoint every these number of steps.
  checkpoint_every_steps: int = 10_000
  # Frequency of eval during training, e.g. every 1_000 steps.
  eval_every_steps: int = 5_000
  # Use bfloat16 mixed precision training instead of float32.
  use_bfloat16: bool = True
  # Integer for PRNG random seed.
  seed: int = 0

  # Parallelism
  mesh_axes: tuple[str, ...] = ('data', 'fsdp', 'tensor')
  axis_rules: MeshRules = MeshRules(
      embed='fsdp',
      mlp='tensor',
      kv='tensor',
      vocab='tensor',
  )
  data_sharding: tuple[str, ...] = ('data', 'fsdp')

  # One axis for each parallelism type may hold a placeholder (-1)
  # value to auto-shard based on available slices and devices.
  # By default, product of the DCN axes should equal number of slices
  # and product of the ICI axes should equal number of devices per slice.
  # ICI (Inter-Chip Interconnection): A high-speed connection between
  # sets of TPU chips, which form the TPU network.
  # DCN (Data Center Network): A connection between the TPU networks;
  # not as fast as ICI.
  # ICI has around 100x the bandwidth of DCN, but it is not a general
  # purpose connection, which is why DCN is necessary for scaling to
  # extremely large ML models.
  dcn_data_parallelism: int = -1
  dcn_fsdp_parallelism: int = 1
  dcn_tensor_parallelism: int = 1
  ici_data_parallelism: int = 1
  ici_fsdp_parallelism: int = -1
  ici_tensor_parallelism: int = 1


def get_config() -> TrainConfig:
  """Get the default hyperparameter configuration."""
  config = Config()
  return TrainConfig(**dataclasses.asdict(config))
