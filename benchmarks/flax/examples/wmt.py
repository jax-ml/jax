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
https://github.com/google/flax/blob/7eba0111e4f13eaaa9f8b79182676f1744c1fca2/examples/wmt/configs/default.py
"""
from typing import Any

from flax import jax_utils
from flax import linen as nn
from flax.examples.wmt import models
from flax.examples.wmt import train
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib
import jax
import jax.numpy as jnp
import ml_collections
import numpy as np
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
    batch[k] = jax.random.randint(
        rng, (batch_size, 256), 0, 9999999, dtype=jnp.int32
    )
  return batch


def get_apply_fn_and_args() -> tuple[jax.stages.Traced, tuple[Any, ...]]:
  """Returns the apply function and args for the model."""
  config = get_config()
  dtype = train.preferred_dtype(config)
  train_config = models.TransformerConfig(
      vocab_size=config.vocab_size,
      output_vocab_size=config.vocab_size,
      share_embeddings=config.share_embeddings,
      logits_via_embedding=config.logits_via_embedding,
      dtype=dtype,
      emb_dim=config.emb_dim,
      num_heads=config.num_heads,
      num_layers=config.num_layers,
      qkv_dim=config.qkv_dim,
      mlp_dim=config.mlp_dim,
      max_len=max(config.max_target_length, config.max_eval_target_length),
      dropout_rate=config.dropout_rate,
      attention_dropout_rate=config.attention_dropout_rate,
      deterministic=False,
      decode=False,
      kernel_init=nn.initializers.xavier_uniform(),
      bias_init=nn.initializers.normal(stddev=1e-6),
  )
  learning_rate_fn = train.create_learning_rate_schedule(
      learning_rate=config.learning_rate, warmup_steps=config.warmup_steps
  )
  rng = jax.random.key(config.seed)
  _, init_rng = jax.random.split(rng)
  input_shape = (config.per_device_batch_size, config.max_target_length)
  target_shape = (config.per_device_batch_size, config.max_target_length)
  m = models.Transformer(train_config.replace(deterministic=True))  # pytype: disable=attribute-error
  initial_variables = jax.jit(m.init)(
      init_rng,
      jnp.ones(input_shape, jnp.float32),
      jnp.ones(target_shape, jnp.float32),
  )
  dynamic_scale = None
  if dtype == jnp.float16:
    dynamic_scale = dynamic_scale_lib.DynamicScale()
  state = train.TrainState.create(
      apply_fn=m.apply,
      params=initial_variables['params'],
      tx=optax.adamw(
          learning_rate=learning_rate_fn,
          b1=0.9,
          b2=0.98,
          eps=1e-9,
          weight_decay=config.weight_decay,
      ),
      dynamic_scale=dynamic_scale,
  )
  # TODO(dsuo): We should be using pmap's in_axes.
  state = jax_utils.replicate(state)
  batch = get_fake_batch(config.per_device_batch_size)
  batch = common_utils.shard(jax.tree_util.tree_map(np.asarray, batch))
  dropout_rngs = jax.random.split(rng, jax.local_device_count())
  def apply_fn(state, batch, dropout_rngs):
    return train.train_step(
        state,
        batch,
        config=train_config,
        learning_rate_fn=learning_rate_fn,
        label_smoothing=config.label_smoothing,
        dropout_rng=dropout_rngs,
    )
  p_apply_fn = jax.pmap(
      apply_fn,
      axis_name='batch',
      donate_argnums=(0,),
  )  # pytype: disable=wrong-arg-types

  return p_apply_fn, (state, batch, dropout_rngs)


def get_config():
  """Get the default hyperparameter configuration."""
  config = ml_collections.ConfigDict()

  # Path to load or store sentencepiece vocab file.
  config.vocab_path = None

  # Vocabulary size if `vocab_path` is not given.
  config.vocab_size = 32_000

  config.max_corpus_chars = 10**7

  # Name of TFDS translation dataset to use.
  config.dataset_name = 'wmt17_translate/de-en'

  # Optional name of TFDS translation dataset to use for evaluation.
  config.eval_dataset_name = 'wmt14_translate/de-en'
  config.eval_split = 'test'

  # Reverse the direction of translation.
  config.reverse_translation = False

  # Per device batch size for training.
  config.per_device_batch_size = 32

  # Beam size for inference.
  config.beam_size = 4

  config.num_train_steps = 100_000

  # Number of steps to take during evaluation.
  config.num_eval_steps = 20
  # Number of steps to generate predictions (used for BLEU score).
  # -1 will use the whole eval dataset.
  config.num_predict_steps = -1

  # Base learning rate.
  config.learning_rate = 0.002

  # Linear learning rate warmup.
  config.warmup_steps = 1000

  # Cross entropy loss label smoothing.
  config.label_smoothing = 0.1

  # Decay factor for AdamW style weight decay.
  config.weight_decay = 0.0

  # Maximum length cutoff for training examples.
  config.max_target_length = 256
  # Maximum length cutoff for eval examples.
  config.max_eval_target_length = 256
  # Maximum length cutoff for predicted tokens.
  config.max_predict_length = 256

  # Inputs and targets share embedding.
  config.share_embeddings = True

  # Final logit transform uses embedding matrix transpose.
  config.logits_via_embedding = True

  # Number of transformer layers.
  config.num_layers = 6

  # Size of query/key/value for attention.
  config.qkv_dim = 1024
  # Size of embeddings.
  config.emb_dim = 1024
  # Size of the MLP.
  config.mlp_dim = 4096

  # Number of attention heads.
  config.num_heads = 16

  # Dropout rate.
  config.dropout_rate = 0.1

  # Attention dropout rate.
  config.attention_dropout_rate = 0.1

  # Whether to save model checkpoints.
  config.save_checkpoints = True
  # Whether to restore from existing model checkpoints.
  config.restore_checkpoints = True

  # Save a checkpoint every these number of steps.
  config.checkpoint_every_steps = 10_000
  # Frequency of eval during training, e.g. every 1000 steps.
  config.eval_every_steps = 1_000

  # Use float16/bfloat16 (GPU/TPU) mixed precision training instead of float32.
  config.use_mixed_precision = True

  # Integer for PRNG random seed.
  config.seed = 0

  return config


def metrics():
  return [
      'train_loss',
      'eval_loss',
      'bleu',
      'eval_accuracy',
      'train_accuracy',
      'uptime',
      'steps_per_sec',
      'train_learning_rate',
  ]
