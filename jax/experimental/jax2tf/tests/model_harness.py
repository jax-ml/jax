# Copyright 2022 The JAX Authors.
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
"""All the models to convert."""
import dataclasses
import functools
from typing import Any, Callable, Dict, List, Sequence

import numpy as np
import jraph

from jax.experimental.jax2tf.tests.flax_models import actor_critic
from jax.experimental.jax2tf.tests.flax_models import bilstm_classifier
from jax.experimental.jax2tf.tests.flax_models import cnn
from jax.experimental.jax2tf.tests.flax_models import gnn
from jax.experimental.jax2tf.tests.flax_models import resnet
from jax.experimental.jax2tf.tests.flax_models import seq2seq_lstm
from jax.experimental.jax2tf.tests.flax_models import transformer_lm1b as lm1b
from jax.experimental.jax2tf.tests.flax_models import transformer_nlp_seq as nlp_seq
from jax.experimental.jax2tf.tests.flax_models import transformer_wmt as wmt
from jax.experimental.jax2tf.tests.flax_models import vae

from jax import random

import tensorflow as tf


@dataclasses.dataclass
class ModelHarness:
  name: str
  apply: Callable[..., Any]
  variables: Dict[str, Any]
  inputs: Sequence[np.ndarray]
  rtol: float = 1e-4

  @property
  def tf_input_signature(self):
    return [
        tf.TensorSpec(x.shape, tf.dtypes.as_dtype(x.dtype)) for x in self.inputs
    ]

  def apply_with_vars(self, *args, **kwargs):
    return self.apply(self.variables, *args, **kwargs)


def _actor_critic_harness():
  model = actor_critic.ActorCritic(num_outputs=8)
  x = np.zeros((1, 84, 84, 4), np.float32)
  variables = model.init(random.PRNGKey(0), x)
  return ModelHarness('flax/actor_critic', model.apply, variables, [x])


def _bilstm_harness():
  model = bilstm_classifier.TextClassifier(
      # TODO(marcvanzee): This fails when
      # `embedding_size != hidden_size`. I suppose some arrays are
      # concatenated with incompatible shapes, which could mean
      # something is going wrong in the translation.
      embedding_size=3,
      hidden_size=1,
      vocab_size=13,
      output_size=1,
      dropout_rate=0.,
      word_dropout_rate=0.)
  x = np.array([[2, 4, 3], [2, 6, 3]], np.int32)
  lengths = np.array([2, 3], np.int32)
  variables = model.init(random.PRNGKey(0), x, lengths, deterministic=True)
  apply = functools.partial(model.apply, deterministic=True)
  return ModelHarness('flax/bilstm', apply, variables, [x, lengths])


def _cnn_harness():
  model = cnn.CNN()
  x = np.zeros((1, 28, 28, 1), np.float32)
  variables = model.init(random.PRNGKey(0), x)
  return ModelHarness('flax/cnn', model.apply, variables, [x])


def _get_gnn_graphs():
  n_node = np.arange(3, 11)
  n_edge = np.arange(4, 12)
  total_n_node = np.sum(n_node)
  total_n_edge = np.sum(n_edge)
  n_graph = n_node.shape[0]
  feature_dim = 10
  graphs = jraph.GraphsTuple(
      n_node=n_node,
      n_edge=n_edge,
      senders=np.zeros(total_n_edge, dtype=np.int32),
      receivers=np.ones(total_n_edge, dtype=np.int32),
      nodes=np.ones((total_n_node, feature_dim)),
      edges=np.zeros((total_n_edge, feature_dim)),
      globals=np.zeros((n_graph, feature_dim)),
  )
  return graphs


def _gnn_harness():
  # Setting taken from flax/examples/ogbg_molpcba/models_test.py.
  rngs = {
      'params': random.PRNGKey(0),
      'dropout': random.PRNGKey(1),
  }
  graphs = _get_gnn_graphs()
  model = gnn.GraphNet(
      latent_size=5,
      num_mlp_layers=2,
      message_passing_steps=2,
      output_globals_size=15,
      use_edge_model=True)
  variables = lambda: model.init(rngs, graphs)
  return ModelHarness('flax/gnn', model.apply, variables, [graphs])


def _gcn_harness():
  # Setting taken from flax/examples/ogbg_molpcba/models_test.py.
  rngs = {
      'params': random.PRNGKey(0),
      'dropout': random.PRNGKey(1),
  }
  graphs = _get_gnn_graphs()
  model = gnn.GraphConvNet(
      latent_size=5,
      num_mlp_layers=2,
      message_passing_steps=2,
      output_globals_size=5)
  variables = model.init(rngs, graphs)
  return ModelHarness('flax/gnn_conv', model.apply, variables, [graphs])


def _resnet50_harness():
  model = resnet.ResNet50(num_classes=2, dtype=np.float32)
  x = np.zeros((8, 244, 244, 3), np.float32)
  variables = model.init(random.PRNGKey(0), x)
  apply = functools.partial(model.apply, train=False, mutable=False)
  return ModelHarness('flax/resnet50', apply, variables, [x])


def _seq2seq_lstm_harness():
  model = seq2seq_lstm.Seq2seq(teacher_force=True, hidden_size=2, vocab_size=4)
  x = np.zeros((1, 2, 4), np.float32)
  rngs = {
      'params': random.PRNGKey(0),
      'lstm': random.PRNGKey(1),
  }
  variables = model.init(rngs, x, x)
  apply = functools.partial(model.apply, rngs={'lstm': random.PRNGKey(2)})
  return ModelHarness('flax/seq2seq', apply, variables, [x, x])


def _min_transformer_kwargs():
  return dict(
      vocab_size=8,
      output_vocab_size=8,
      emb_dim = 4,
      num_heads= 1,
      num_layers = 1,
      qkv_dim= 2,
      mlp_dim = 2,
      max_len = 2,
      dropout_rate = 0.,
      attention_dropout_rate = 0.)


def _full_transformer_kwargs():
  kwargs = dict(
      decode = True,
      deterministic = True,
      logits_via_embedding=False,
      share_embeddings=False)
  return {**kwargs, **_min_transformer_kwargs()}


def _transformer_lm1b_harness():
  config = lm1b.TransformerConfig(**_full_transformer_kwargs())
  model = lm1b.TransformerLM(config=config)
  x = np.zeros((2, 1), np.float32)
  rng1, rng2 = random.split(random.PRNGKey(0))
  variables = model.init(rng1, x)

  def apply(*args):
    # Don't return the new state (containing the cache).
    output, _ = model.apply(*args, rngs={'cache': rng2}, mutable=['cache'])
    return output

  return ModelHarness('flax/lm1b', apply, variables, [x])


def _transformer_nlp_seq_harness():
  config = nlp_seq.TransformerConfig(**_min_transformer_kwargs())
  model = nlp_seq.Transformer(config=config)
  x = np.zeros((2, 1), np.float32)
  variables = model.init(random.PRNGKey(0), x, train=False)
  apply = functools.partial(model.apply, train=False)
  return ModelHarness('flax/nlp_seq', apply, variables, [x])


def _transformer_wmt_harness():
  config = wmt.TransformerConfig(**_full_transformer_kwargs())
  model = wmt.Transformer(config=config)
  x = np.zeros((2, 1), np.float32)
  variables = model.init(random.PRNGKey(0), x, x)

  def apply(*args):
    # Don't return the new state (containing the cache).
    output, _ = model.apply(*args, mutable=['cache'])
    return output

  return ModelHarness('flax/wmt', apply, variables, [x, x])


def _vae_harness():
  model = vae.VAE(latents=3)
  x = np.zeros((1, 8, 8, 3), np.float32)
  rng1, rng2 = random.split(random.PRNGKey(0))
  variables = model.init(rng1, x, rng2)
  generate = lambda v, x: model.apply(v, x, method=model.generate)
  return ModelHarness('flax/vae', generate, variables, [x])


##### All harnesses in this file.
# Note: we store the functions and not their instantiation to avoid creating all
# parameters of all models at once.
ALL_HARNESS_FNS: List[Callable[[], ModelHarness]] = [
    _actor_critic_harness,
    _bilstm_harness,
    # TODO(marcvanzee): GNNs currently do not work since their __call__
    # function takes a jraph.GraphsTuple as an argument, which we don't support.
    # 'flax/gnn': _gnn_harness,
    # 'flax/gnn_conv': _gnn_conv_harness,
    _resnet50_harness,
    _seq2seq_lstm_harness,
    _transformer_lm1b_harness,
    _transformer_nlp_seq_harness,
    _transformer_wmt_harness,
    _vae_harness
]
