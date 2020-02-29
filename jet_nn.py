"""
Create some pared down examples to see if we can take jets through them.
"""
import time

import haiku as hk
import jax
import jax.numpy as jnp
import numpy.random as npr

from jax.flatten_util import ravel_pytree

import tensorflow_datasets as tfds
from functools import reduce

from scipy.special import factorial as fact


def sigmoid(z):
  """
  Defined using only numpy primitives (but probably less numerically stable).
  """
  return 1./(1. + jnp.exp(-z))

def repeated(f, n):
  def rfun(p):
    return reduce(lambda x, _: f(x), range(n), p)
  return rfun


def jvp_taylor(f, primals, series):
  def expansion(eps):
    tayterms = [
        sum([eps**(i + 1) * terms[i] / fact(i + 1) for i in range(len(terms))])
        for terms in series
    ]
    return f(*map(sum, zip(primals, tayterms)))

  n_derivs = []
  N = len(series[0]) + 1
  for i in range(1, N):
    d = repeated(jax.jacobian, i)(expansion)(0.)
    n_derivs.append(d)
  return f(*primals), n_derivs


def jvp_test_jet(f, primals, series, atol=1e-5):
  tic = time.time()
  y, terms = jax.jet(f, primals, series)
  print("jet done in {} sec".format(time.time() - tic))

  tic = time.time()
  y_jvp, terms_jvp = jvp_taylor(f, primals, series)
  print("jvp done in {} sec".format(time.time() - tic))

  assert jnp.allclose(y, y_jvp)
  assert jnp.allclose(terms, terms_jvp, atol=atol)


def softmax_cross_entropy(logits, labels):
  one_hot = hk.one_hot(labels, logits.shape[-1])
  return -jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1)
  # return -logits[labels]


rng = jax.random.PRNGKey(42)

order = 4

batch_size = 2
train_ds = tfds.load('mnist', split=tfds.Split.TRAIN)
train_ds = train_ds.cache().shuffle(1000).batch(batch_size)
batch = next(tfds.as_numpy(train_ds))


def test_mlp_x():
  """
  Test jet through a linear layer built with Haiku wrt x.
  """

  def loss_fn(images, labels):
    model = hk.Sequential([
        lambda x: x.astype(jnp.float32) / 255.,
        hk.Linear(10)
    ])
    logits = model(images)
    return jnp.mean(softmax_cross_entropy(logits, labels))

  loss_obj = hk.transform(loss_fn)

  # flatten for MLP
  batch['image'] = jnp.reshape(batch['image'], (batch_size, -1))
  images, labels = jnp.array(batch['image'], dtype=jnp.float32), jnp.array(batch['label'])

  params = loss_obj.init(rng, images, labels)

  flat_params, unravel = ravel_pytree(params)

  loss = loss_obj.apply(unravel(flat_params), images, labels)
  print("forward pass works")

  f = lambda images: loss_obj.apply(unravel(flat_params), images, labels)
  _ = jax.grad(f)(images)
  terms_in = [npr.randn(*images.shape) for _ in range(order)]
  jvp_test_jet(f, (images,), (terms_in,))


def test_mlp1():
  """
  Test jet through a linear layer built with Haiku.
  """

  def loss_fn(images, labels):
    model = hk.Sequential([
        lambda x: x.astype(jnp.float32) / 255.,
        hk.Linear(10)
    ])
    logits = model(images)
    return jnp.mean(softmax_cross_entropy(logits, labels))

  loss_obj = hk.transform(loss_fn)

  # flatten for MLP
  batch['image'] = jnp.reshape(batch['image'], (batch_size, -1))
  images, labels = jnp.array(batch['image'], dtype=jnp.float32), jnp.array(batch['label'])

  params = loss_obj.init(rng, images, labels)

  flat_params, unravel = ravel_pytree(params)

  loss = loss_obj.apply(unravel(flat_params), images, labels)
  print("forward pass works")

  f = lambda flat_params: loss_obj.apply(unravel(flat_params), images, labels)
  terms_in = [npr.randn(*flat_params.shape) for _ in range(order)]
  jvp_test_jet(f, (flat_params,), (terms_in,))


def test_mlp2():
  """
  Test jet through a MLP with sigmoid activations.
  """

  def loss_fn(images, labels):
    model = hk.Sequential([
        lambda x: x.astype(jnp.float32) / 255.,
        hk.Linear(100),
        sigmoid,
        hk.Linear(10)
    ])
    logits = model(images)
    return jnp.mean(softmax_cross_entropy(logits, labels))

  loss_obj = hk.transform(loss_fn)

  # flatten for MLP
  batch['image'] = jnp.reshape(batch['image'], (batch_size, -1))
  images, labels = jnp.array(batch['image'], dtype=jnp.float32), jnp.array(batch['label'])

  params = loss_obj.init(rng, images, labels)

  flat_params, unravel = ravel_pytree(params)

  loss = loss_obj.apply(unravel(flat_params), images, labels)
  print("forward pass works")

  f = lambda flat_params: loss_obj.apply(unravel(flat_params), images, labels)
  terms_in = [npr.randn(*flat_params.shape) for _ in range(order)]
  jvp_test_jet(f, (flat_params,), (terms_in,), atol=1e-4)


def test_res1():
  """
  Test jet through a simple convnet with sigmoid activations.
  """

  def loss_fn(images, labels):
    model = hk.Sequential([
        lambda x: x.astype(jnp.float32) / 255.,
        hk.Conv2D(output_channels=10,
                  kernel_shape=3,
                  stride=2,
                  padding="VALID"),
        sigmoid,
        hk.Reshape(output_shape=(batch_size, -1)),
        hk.Linear(10)
    ])
    logits = model(images)
    return jnp.mean(softmax_cross_entropy(logits, labels))

  loss_obj = hk.transform(loss_fn)

  images, labels = batch['image'], batch['label']

  params = loss_obj.init(rng, images, labels)

  flat_params, unravel = ravel_pytree(params)

  loss = loss_obj.apply(unravel(flat_params), images, labels)
  print("forward pass works")

  f = lambda flat_params: loss_obj.apply(unravel(flat_params), images, labels)
  terms_in = [npr.randn(*flat_params.shape) for _ in range(order)]
  jvp_test_jet(f, (flat_params,), (terms_in,))


def test_div():
  x = 1.
  y = 5.
  primals = (x, y)
  order = 4
  series_in = ([npr.randn() for _ in range(order)], [npr.randn() for _ in range(order)])
  jvp_test_jet(lambda a, b: a / b, primals, series_in)


def test_gather():
  npr.seed(0)
  D = 3  # dimensionality
  N = 6  # differentiation order
  x = npr.randn(D)
  terms_in = list(npr.randn(N, D))
  jvp_test_jet(lambda x: x[1:], (x,), (terms_in,))

def test_reduce_max():
  npr.seed(0)
  D1, D2 = 3, 5  # dimensionality
  N = 6  # differentiation order
  x = npr.randn(D1, D2)
  terms_in = [npr.randn(D1, D2) for _ in range(N)]
  jvp_test_jet(lambda x: x.max(axis=1), (x,), (terms_in,))

def test_sub():
  x = 1.
  y = 5.
  primals = (x, y)
  order = 4
  series_in = ([npr.randn() for _ in range(order)], [npr.randn() for _ in range(order)])
  jvp_test_jet(lambda a, b: a - b, primals, series_in)

def test_exp():
  npr.seed(0)
  D1, D2 = 3, 5  # dimensionality
  N = 6  # differentiation order
  x = npr.randn(D1, D2)
  terms_in = [npr.randn(D1, D2) for _ in range(N)]
  jvp_test_jet(lambda x: jnp.exp(x), (x,), (terms_in,))

def test_log():
  npr.seed(0)
  D1, D2 = 3, 5  # dimensionality
  N = 4  # differentiation order
  x = jnp.exp(npr.randn(D1, D2))
  terms_in = [jnp.exp(npr.randn(D1, D2)) for _ in range(N)]
  jvp_test_jet(lambda x: jnp.log(x), (x,), (terms_in,))


# test_div()
# test_gather()
# test_reduce_max()
# test_sub()
# test_exp()
# test_log()

# test_mlp_x()
# test_mlp1()
# test_mlp2()
# test_res1()
