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

"""Tests for refs module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

from absl.testing import absltest
from absl.testing import parameterized

import numpy as onp

from jax import test_util as jtu
from jax.test_util import check_grads
from jax import random
from jax import nn
from jax import grad, jit, vmap
from jax.refs import Ref, tree_load, tree_store, collect, inject, no_value
import jax
import jax.numpy as jnp

from jax.config import config
config.parse_flags_with_absl()

class RefsTest(jtu.JaxTestCase):

  def test_collect(self):
    ref = Ref()
    def foo(x):
      y = x ** 2
      ref.store(y)
      return y + 1
    pure_foo = collect(foo, ref)
    assert pure_foo(3) == (10, 9)

  def test_jit_collect(self):
    ref = Ref()
    def foo(x):
      y = x ** 2
      ref.store(y)
      return y + 1
    pure_foo = jit(collect(foo, ref))
    assert pure_foo(3) == (10, 9)

  def test_collect_jit(self):
    ref = Ref()
    def foo(x):
      y = x ** 2
      ref.store(y)
      return y + 1
    pure_foo = collect(jit(foo, refs=ref), ref)
    assert pure_foo(3) == (10, 9)

  def test_collect_list(self):
    refs = [Ref(), Ref()]

    # foo :: [Writer(refs)] float -> [Writer(refs)] float
    def foo(x):
      y = x ** 2
      refs[0].store(y)
      refs[1].store(y - 1)
      return y + 1

    # pure_foo :: float -> float, List[float]
    pure_foo = collect(foo, refs)
    assert pure_foo(3) == (10, [9, 8])

  def test_inject(self):
    ref = Ref()
    def foo(x):
      y = x ** 2
      y = ref.load()
      return y + 1
    pure_foo = inject(foo, ref)
    assert pure_foo(4, 3) == 5

  def test_tagging(self):

    _tag_refs = Ref(dict())
    def tag(name, value=no_value):
      _tag_vals = _tag_refs.load()
      if name not in _tag_vals:
        assert value is not no_value
        _tag_vals[name] = value
      else:
        value, _tag_vals[name] = _tag_vals[name], value
      _tag_refs.store(_tag_vals)
      return value


    def tagged_fn_with_aux(x):
      y = x ** 2
      tag('y', y)
      z = y + 1
      return z

    assert collect(tagged_fn_with_aux, _tag_refs)(2) == (5, {'y': 4})

    def tagged_fn_with_intermediate(x):
      y = x ** 2
      y = tag('y', y)
      z = y + 1
      return z

    assert inject(tagged_fn_with_intermediate, _tag_refs)({'y': 3}, 2) == 4

  def test_stateful_PRNG(self):

    _global_PRNG_key = Ref()
    def next_key():
      key1, key2 = random.split(_global_PRNG_key.load())
      _global_PRNG_key.store(key1)
      return key2

    def stateful_normal():
      return random.normal(next_key())

    pure_normal = inject(stateful_normal, _global_PRNG_key)
    pure_normal(random.PRNGKey(2))

  def test_stateful_NNs(self):

    _global_PRNG_key = Ref()
    def next_key():
      key1, key2 = random.split(_global_PRNG_key.load())
      _global_PRNG_key.store(key1)
      return key2

    class Parameter(Ref):
      """A trainable parameter."""

    class Buffer(Ref):
      """A container for non-trainable state."""


    class _ModuleMeta(type):
      def __init__(cls, name, bases, attrs):
        super(_ModuleMeta, cls).__init__(name, bases, attrs)
        def from_kv(keys, values):
          module = cls.__new__(cls)
          module.__dict__.update(**dict(zip(keys, values)))
          return module
        jax.tree_util.register_pytree_node(
            cls,
            lambda m: (list(m.__dict__.values()), list(m.__dict__.keys())),
            from_kv)

    class Module(metaclass=_ModuleMeta):
      def __repr__(self):
        s = ', '.join(k + '=' + repr(v) for k, v in self.__dict__.items())
        return self.__class__.__name__ + '(' + s + ')'


    class Linear(Module):
      def __init__(self, nI, nO, bias=True,
                  weight_init=nn.initializers.lecun_normal(),
                  bias_init=nn.initializers.zeros):
        self.W = Parameter(weight_init(next_key(), (nI, nO)))
        if bias:
          self.b = Parameter(bias_init(next_key(), (nO,)))
      def __call__(self, x):
        return x @ self.W.load() + self.b.load()

    class Sequential(Module):
      def __init__(self, layers):
        self.layers = layers
      def __call__(self, x):
        for layer in self.layers:
          x = layer(x)
        return x

    def arch1():
      layer1 = Linear(2, 2)
      layer2 = Linear(2, 2)
      layer1.W = layer2.W
      return Sequential([layer1, layer2])

    def arch2():
      layer1 = Linear(2, 2)
      layer2 = layer1
      return Sequential([layer1, layer2])

    model1 = inject(arch1, _global_PRNG_key)(random.PRNGKey(1))
    model2 = inject(arch2, _global_PRNG_key)(random.PRNGKey(1))

    def loss1(x):
      return jnp.sum(model1(x))

    def loss2(x):
      return jnp.sum(model2(x))

    params1 = tree_load(model1, Parameter)
    params2 = tree_load(model2, Parameter)

    def train_step1(params, x):
      grads = jax.grad(inject(loss1, model1))(params, x)
      return jax.tree_multimap(lambda p, g: p - g, params, grads)

    def train_step2(params, x):
      grads = jax.grad(inject(loss2, model2))(params, x)
      return jax.tree_multimap(lambda p, g: p - g, params, grads)

    x = random.normal(random.PRNGKey(0), (2,))
    params1 = train_step1(params1, x)
    tree_store(model1, params1)
    assert model1.layers[0].W is model1.layers[1].W
    assert model1.layers[0].b is not model1.layers[1].b
    params2 = train_step2(params2, x)
    tree_store(model2, params2)
    assert model2.layers[0].W is model2.layers[1].W
    assert model2.layers[0].b is model2.layers[1].b
