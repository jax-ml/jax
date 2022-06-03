# Copyright 2022 Google LLC
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
from typing import Any, Callable, Sequence

import jax.numpy as jnp
import jraph

from jax.experimental.jax2tf.converters_eval.test_models.flax import actor_critic
from jax.experimental.jax2tf.converters_eval.test_models.flax import bilstm_classifier
from jax.experimental.jax2tf.converters_eval.test_models.flax import cnn
from jax.experimental.jax2tf.converters_eval.test_models.flax import gnn
from jax.experimental.jax2tf.converters_eval.test_models.flax import resnet
from jax.experimental.jax2tf.converters_eval.test_models.flax import seq2seq_lstm
from jax.experimental.jax2tf.converters_eval.test_models.flax import transformer_lm1b as lm1b
from jax.experimental.jax2tf.converters_eval.test_models.flax import transformer_nlp_seq as nlp_seq
from jax.experimental.jax2tf.converters_eval.test_models.flax import transformer_wmt as wmt
from jax.experimental.jax2tf.converters_eval.test_models.flax import vae

from jax import random


@dataclasses.dataclass
class ModelTestCase:
  apply: Callable[..., Any]
  variables: Any
  input_specs: Sequence[jnp.ndarray]
  rtol: float = 1e-4


def _actor_critic():
  model = actor_critic.ActorCritic(num_outputs=8)
  x = jnp.zeros((1, 8, 8, 4))
  variables = model.init(random.PRNGKey(0), x)
  return ModelTestCase(model.apply, variables, [x])


def _bilstm():
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
  x = jnp.array([[2, 4, 3], [2, 6, 3]], jnp.int32)
  lengths = jnp.array([2, 3], jnp.int32)
  variables = model.init(random.PRNGKey(0), x, lengths, deterministic=True)
  apply = functools.partial(model.apply, deterministic=True)
  return ModelTestCase(apply, variables, [x, lengths])


def _cnn():
  model = cnn.CNN()
  x = jnp.zeros((1, 28, 28, 1))
  variables = model.init(random.PRNGKey(0), x)
  return ModelTestCase(model.apply, variables, [x])


def _get_gnn_graphs():
  n_node = jnp.arange(3, 11)
  n_edge = jnp.arange(4, 12)
  total_n_node = jnp.sum(n_node)
  total_n_edge = jnp.sum(n_edge)
  n_graph = n_node.shape[0]
  feature_dim = 10
  graphs = jraph.GraphsTuple(
      n_node=n_node,
      n_edge=n_edge,
      senders=jnp.zeros(total_n_edge, dtype=jnp.int32),
      receivers=jnp.ones(total_n_edge, dtype=jnp.int32),
      nodes=jnp.ones((total_n_node, feature_dim)),
      edges=jnp.zeros((total_n_edge, feature_dim)),
      globals=jnp.zeros((n_graph, feature_dim)),
  )
  return graphs


def _gnn():
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
  variables = model.init(rngs, graphs)
  return ModelTestCase(model.apply, variables, [graphs])


def _gcn():
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
  return ModelTestCase(model.apply, variables, [graphs])


def _resnet50():
  model = resnet.ResNet50(num_classes=2, dtype=jnp.float32)
  x = jnp.zeros((8, 244, 244, 3))
  variables = model.init(random.PRNGKey(0), x)
  apply = functools.partial(model.apply, train=False, mutable=False)
  return ModelTestCase(apply, variables, [x])


def _seq2seq_lstm():
  model = seq2seq_lstm.Seq2seq(teacher_force=True, hidden_size=2, vocab_size=4)
  x = jnp.zeros((1, 2, 4))
  rngs = {
      'params': random.PRNGKey(0),
      'lstm': random.PRNGKey(1),
  }
  variables = model.init(rngs, x, x)
  apply = functools.partial(model.apply, rngs={'lstm': random.PRNGKey(2)})
  return ModelTestCase(apply, variables, [x, x])


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


def _transformer_lm1b():
  config = lm1b.TransformerConfig(**_full_transformer_kwargs())
  model = lm1b.TransformerLM(config=config)
  x = jnp.zeros((2, 1))
  rng1, rng2 = random.split(random.PRNGKey(0))
  variables = model.init(rng1, x)
  apply = functools.partial(model.apply, rngs={'cache': rng2}, mutable=['cache'])
  return ModelTestCase(apply, variables, [x])


def _transformer_nlp_seq():
  config = nlp_seq.TransformerConfig(**_min_transformer_kwargs())
  model = nlp_seq.Transformer(config=config)
  x = jnp.zeros((2, 1))
  variables = model.init(random.PRNGKey(0), x, train=False)
  apply = functools.partial(model.apply, train=False)
  return ModelTestCase(apply, variables, [x])


def _transformer_wmt():
  config = wmt.TransformerConfig(**_full_transformer_kwargs())
  model = wmt.Transformer(config=config)
  x = jnp.zeros((2, 1))
  variables = model.init(random.PRNGKey(0), x, x)
  apply = functools.partial(model.apply, mutable=['cache'])
  return ModelTestCase(apply, variables, [x, x])


def _vae():
  model = vae.VAE(latents=3)
  x = jnp.zeros((1, 8, 8, 3))
  rng1, rng2 = random.split(random.PRNGKey(0))
  variables = model.init(rng1, x, rng2)
  generate = lambda v, x: model.apply(v, x, method=model.generate)
  return ModelTestCase(generate, variables, [x])


def get_test_cases():
  return {
      'flax/actor_critic': _actor_critic,
      'flax/bilstm': _bilstm,
      'flax/cnn': _cnn,
      # TODO(marcvanzee): GNNs currently do not work since their __call__
      # function takes a jraph.GraphsTuple as an argument. We should extend our
      # input logic so we can handle any pytree.
      # 'flax/gnn': _gnn,
      # 'flax/gnn_conv': _gnn_conv,
      'flax/resnet50': _resnet50,
      'flax/seq2seq_lstm': _seq2seq_lstm,
      'flax/transformer_lm1b': _transformer_lm1b,
      'flax/transformer_nlp_seq': _transformer_nlp_seq,
      'flax/transformer_wmt': _transformer_wmt,
      'flax/vae': _vae,
  }
