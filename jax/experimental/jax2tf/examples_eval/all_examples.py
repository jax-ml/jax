# Copyright 2020 Google LLC
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
"""All the examples to convert to TFLite or TFjs."""
import dataclasses
import enum
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp


class Arg(enum.Enum):
  """This enum is used to automatically generate args dependent on a Module's internal state."""
  RNG = '_RNG'
  ONES = '_ONES'
  VARS = '_VARS'
  INPUT = '_INPUTS'


@dataclasses.dataclass
class ModuleSpec:
  """Specification of a Module."""
  module_path: str
  input_shape: Tuple[int, ...] = ()
  module_args: Sequence[Any] = ()
  module_kwargs: Optional[Dict[str, Any]] = None
  init_args: Sequence[Any] = (Arg.RNG, Arg.ONES)
  init_kwargs: Optional[Dict[str, Any]] = None
  apply_args: Sequence[Any] = (Arg.VARS, Arg.INPUT)
  apply_kwargs: Optional[Dict[str, Any]] = None
  dtype: jnp.dtype = jnp.float32
  rng_key: int = 0
  apply_method_fn: str = '__call__'

  def __post_init__(self):
    self.module_kwargs = self.module_kwargs or {}
    self.init_kwargs = self.init_kwargs or {}
    self.apply_kwargs = self.apply_kwargs or {}


@dataclasses.dataclass
class ExampleSuite:
  """A suite of examples."""
  name: str
  description: str
  url: str
  examples: Dict[str, ModuleSpec]


@dataclasses.dataclass
class TransformerConfig:
  """Transformer config."""
  vocab_size: int = 8
  output_vocab_size: int = 8
  share_embeddings: bool = False
  logits_via_embedding: bool = False
  dtype: jnp.dtype = jnp.float32
  emb_dim: int = 4
  num_heads: int = 1
  num_layers: int = 1
  qkv_dim: int = 2
  mlp_dim: int = 2
  max_len: int = 2
  dropout_rate: float = 0.
  attention_dropout_rate: float = 0.
  kernel_init: Callable[..., Any] = nn.initializers.xavier_uniform()
  bias_init: Callable[..., Any] = nn.initializers.normal(stddev=1e-6)
  posemb_init: Optional[Callable[..., Any]] = None
  deterministic: bool = True
  decode: bool = True


def _flax_examples():
  return {
      'imagenet':
          ModuleSpec(
              module_path='imagenet.models.ResNet50',
              input_shape=(8, 224, 224, 3),
              module_kwargs=dict(num_classes=2, dtype=jnp.float32),
              apply_kwargs=dict(train=False, mutable=False)),
      'lm1b':
          ModuleSpec(
              module_path='lm1b.models.TransformerLM',
              input_shape=(2, 1),
              module_kwargs=dict(config=TransformerConfig()),
              apply_kwargs=dict(rngs={'cache': Arg.RNG}, mutable=['cache'])),
      'mnist':
          ModuleSpec(module_path='mnist.train.CNN', input_shape=(1, 28, 28, 1)),
      'nlp_seq':
          ModuleSpec(
              module_path='nlp_seq.models.Transformer',
              input_shape=(2, 1),
              init_args=(Arg.RNG,),
              init_kwargs=dict(inputs=Arg.ONES, train=False),
              module_kwargs=dict(config=TransformerConfig()),
              apply_args=(Arg.VARS,),
              apply_kwargs=dict(inputs=Arg.INPUT, train=False)),
      'pixelcnn++':
          ModuleSpec(
              module_path='pixelcnn.pixelcnn.PixelCNNPP',
              input_shape=(1, 32, 32, 3),
              init_kwargs=dict(train=False),
              module_kwargs=dict(
                  depth=1, features=2, logistic_components=2, dropout_p=0.),
              apply_kwargs=dict(train=False)),
      'ppo':
          ModuleSpec(
              module_path='ppo.models.ActorCritic',
              # TODO(marcvanzee): We get numerical differences if we run this
              # for input shapes (1, 8, 8, 4). conv_general_dilated then returns
              # only zeros for TFLite, but not for JAX. We should investigate
              # and fix this.
              input_shape=(1, 84, 84, 4),
              module_kwargs=dict(num_outputs=8)),
      'seq2seq':
          ModuleSpec(
              module_path='seq2seq.train.Seq2seq',
              input_shape=(1, 2, 15),
              module_kwargs=dict(teacher_force=True, hidden_size=2),
              init_args=({
                  'params': Arg.RNG,
                  'lstm': Arg.RNG
              }, Arg.ONES, Arg.ONES),
              apply_args=(Arg.VARS, Arg.INPUT, Arg.INPUT),
              apply_kwargs=dict(rngs={'lstm': Arg.RNG})),
      'sst2':
          ModuleSpec(
              module_path='sst2.models.TextClassifier',
              input_shape=(2, 3),
              module_kwargs=dict(
                  # TODO(marcvanzee): TFLite throws a concatenation error when
                  # `embedding_size != hidden_size`. I suppose some arrays are
                  # concatenated with incompatible shapes, which could mean
                  # something is going wrong in the translation.
                  embedding_size=3,
                  hidden_size=3,
                  vocab_size=13,
                  output_size=1,
                  dropout_rate=0.,
                  word_dropout_rate=0.),
              init_args=(Arg.RNG, Arg.ONES, jnp.array([2, 3], dtype=jnp.int32)),
              init_kwargs=dict(deterministic=True),
              apply_args=(Arg.VARS, Arg.INPUT, jnp.array([2, 3],
                                                         dtype=jnp.int32)),
              apply_kwargs=dict(deterministic=True),
              dtype=jnp.int32),
      'vae':
          ModuleSpec(
              module_path='vae.train.VAE',
              input_shape=(1, 8, 8, 3),
              module_args=(3,),
              init_args=(Arg.RNG, Arg.ONES, Arg.RNG),
              apply_method_fn='generate'),
      'wmt':
          ModuleSpec(
              module_path='wmt.models.Transformer',
              input_shape=(2, 1),
              module_kwargs=dict(config=TransformerConfig()),
              init_args=(Arg.RNG, Arg.ONES, Arg.ONES),
              apply_args=(Arg.VARS, Arg.INPUT, Arg.INPUT),
              apply_kwargs=dict(mutable=['cache'])),
  }

def get_suite(suite_name: str) -> Optional[ExampleSuite]:
  """Returns all examples in `suite_name`."""
  if suite_name == 'flax':
    return ExampleSuite(
        name='The Flax Examples',
        description="""List of examples maintained by the Flax team.
These exampls are representative for what the average ML researcher is interested in.""",
        url='https://github.com/google/flax/tree/main/examples',
        examples=_flax_examples()
    )
  else:
    return None
