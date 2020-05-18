# Copyright 2019 Google LLC
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

"""JAX efficiently trains a differentially private conv net on MNIST.

This script contains a JAX implementation of Differentially Private Stochastic
Gradient Descent (https://arxiv.org/abs/1607.00133). DPSGD requires clipping
the per-example parameter gradients, which is non-trivial to implement
efficiently for convolutional neural networks.  The JAX XLA compiler shines in
this setting by optimizing the minibatch-vectorized computation for 
convolutional architectures. Train time takes a few seconds per epoch on a
commodity GPU.

This code depends on tensorflow_privacy (https://github.com/tensorflow/privacy)
  Install instructions:
    $ pip install tensorflow
    $ git clone https://github.com/tensorflow/privacy
    $ cd privacy
    $ pip install .

The results match those in the reference TensorFlow baseline implementation:
  https://github.com/tensorflow/privacy/tree/master/tutorials

Example invocations:
  # this non-private baseline should get ~99% acc
  python -m examples.differentially_private_sgd \
    --dpsgd=False \
    --learning_rate=.1 \
    --epochs=20 \

   this private baseline should get ~95% acc
  python -m examples.differentially_private_sgd \
   --dpsgd=True \
   --noise_multiplier=1.3 \
   --l2_norm_clip=1.5 \
   --epochs=15 \
   --learning_rate=.25 \

  # this private baseline should get ~96.6% acc
  python -m examples.differentially_private_sgd \
   --dpsgd=True \
   --noise_multiplier=1.1 \
   --l2_norm_clip=1.0 \
   --epochs=60 \
   --learning_rate=.15 \

  # this private baseline should get ~97% acc
  python -m examples.differentially_private_sgd \
   --dpsgd=True \
   --noise_multiplier=0.7 \
   --l2_norm_clip=1.5 \
   --epochs=45 \
   --learning_rate=.25 \
"""

from functools import partial
import itertools
import time
import warnings

from absl import app
from absl import flags

from jax import grad
from jax import jit
from jax import random
from jax import tree_util
from jax import vmap
from jax.experimental import optimizers
from jax.experimental import stax
from jax.lax import stop_gradient
import jax.numpy as jnp
from examples import datasets
import numpy.random as npr

# https://github.com/tensorflow/privacy
from privacy.analysis.rdp_accountant import compute_rdp
from privacy.analysis.rdp_accountant import get_privacy_spent

FLAGS = flags.FLAGS

flags.DEFINE_boolean(
    'dpsgd', True, 'If True, train with DP-SGD. If False, '
    'train with vanilla SGD.')
flags.DEFINE_float('learning_rate', .15, 'Learning rate for training')
flags.DEFINE_float('noise_multiplier', 1.1,
                   'Ratio of the standard deviation to the clipping norm')
flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
flags.DEFINE_integer('batch_size', 256, 'Batch size')
flags.DEFINE_integer('epochs', 60, 'Number of epochs')
flags.DEFINE_integer('seed', 0, 'Seed for jax PRNG')
flags.DEFINE_integer(
    'microbatches', None, 'Number of microbatches '
    '(must evenly divide batch_size)')
flags.DEFINE_string('model_dir', None, 'Model directory')


init_random_params, predict = stax.serial(
    stax.Conv(16, (8, 8), padding='SAME', strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Conv(32, (4, 4), padding='VALID', strides=(2, 2)),
    stax.Relu,
    stax.MaxPool((2, 2), (1, 1)),
    stax.Flatten,
    stax.Dense(32),
    stax.Relu,
    stax.Dense(10),
)


def loss(params, batch):
  inputs, targets = batch
  logits = predict(params, inputs)
  logits = stax.logsoftmax(logits)  # log normalize
  return -jnp.mean(jnp.sum(logits * targets, axis=1))  # cross entropy loss


def accuracy(params, batch):
  inputs, targets = batch
  target_class = jnp.argmax(targets, axis=1)
  predicted_class = jnp.argmax(predict(params, inputs), axis=1)
  return jnp.mean(predicted_class == target_class)


def private_grad(params, batch, rng, l2_norm_clip, noise_multiplier,
                 batch_size):
  """Return differentially private gradients for params, evaluated on batch."""

  def _clipped_grad(params, single_example_batch):
    """Evaluate gradient for a single-example batch and clip its grad norm."""
    grads = grad(loss)(params, single_example_batch)

    nonempty_grads, tree_def = tree_util.tree_flatten(grads)
    total_grad_norm = jnp.linalg.norm(
        [jnp.linalg.norm(neg.ravel()) for neg in nonempty_grads])
    divisor = stop_gradient(jnp.amax((total_grad_norm / l2_norm_clip, 1.)))
    normalized_nonempty_grads = [g / divisor for g in nonempty_grads]
    return tree_util.tree_unflatten(tree_def, normalized_nonempty_grads)

  px_clipped_grad_fn = vmap(partial(_clipped_grad, params))
  std_dev = l2_norm_clip * noise_multiplier
  noise_ = lambda n: n + std_dev * random.normal(rng, n.shape)
  normalize_ = lambda n: n / float(batch_size)
  tree_map = tree_util.tree_map
  sum_ = lambda n: jnp.sum(n, 0)  # aggregate
  aggregated_clipped_grads = tree_map(sum_, px_clipped_grad_fn(batch))
  noised_aggregated_clipped_grads = tree_map(noise_, aggregated_clipped_grads)
  normalized_noised_aggregated_clipped_grads = (
      tree_map(normalize_, noised_aggregated_clipped_grads)
  )
  return normalized_noised_aggregated_clipped_grads


def shape_as_image(images, labels, dummy_dim=False):
  target_shape = (-1, 1, 28, 28, 1) if dummy_dim else (-1, 28, 28, 1)
  return jnp.reshape(images, target_shape), labels


def compute_epsilon(steps, num_examples=60000, target_delta=1e-5):
  if num_examples * target_delta > 1.:
    warnings.warn('Your delta might be too high.')
  q = FLAGS.batch_size / float(num_examples)
  orders = list(jnp.linspace(1.1, 10.9, 99)) + range(11, 64)
  rdp_const = compute_rdp(q, FLAGS.noise_multiplier, steps, orders)
  eps, _, _ = get_privacy_spent(orders, rdp_const, target_delta=target_delta)
  return eps


def main(_):

  if FLAGS.microbatches:
    raise NotImplementedError(
        'Microbatches < batch size not currently supported'
    )

  train_images, train_labels, test_images, test_labels = datasets.mnist()
  num_train = train_images.shape[0]
  num_complete_batches, leftover = divmod(num_train, FLAGS.batch_size)
  num_batches = num_complete_batches + bool(leftover)
  key = random.PRNGKey(FLAGS.seed)

  def data_stream():
    rng = npr.RandomState(FLAGS.seed)
    while True:
      perm = rng.permutation(num_train)
      for i in range(num_batches):
        batch_idx = perm[i * FLAGS.batch_size:(i + 1) * FLAGS.batch_size]
        yield train_images[batch_idx], train_labels[batch_idx]

  batches = data_stream()

  opt_init, opt_update, get_params = optimizers.sgd(FLAGS.learning_rate)

  @jit
  def update(_, i, opt_state, batch):
    params = get_params(opt_state)
    return opt_update(i, grad(loss)(params, batch), opt_state)

  @jit
  def private_update(rng, i, opt_state, batch):
    params = get_params(opt_state)
    rng = random.fold_in(rng, i)  # get new key for new random numbers
    return opt_update(
        i,
        private_grad(params, batch, rng, FLAGS.l2_norm_clip,
                     FLAGS.noise_multiplier, FLAGS.batch_size), opt_state)

  _, init_params = init_random_params(key, (-1, 28, 28, 1))
  opt_state = opt_init(init_params)
  itercount = itertools.count()

  steps_per_epoch = 60000 // FLAGS.batch_size
  print('\nStarting training...')
  for epoch in range(1, FLAGS.epochs + 1):
    start_time = time.time()
    # pylint: disable=no-value-for-parameter
    for _ in range(num_batches):
      if FLAGS.dpsgd:
        opt_state = \
            private_update(
                key, next(itercount), opt_state,
                shape_as_image(*next(batches), dummy_dim=True))
      else:
        opt_state = update(
            key, next(itercount), opt_state, shape_as_image(*next(batches)))
    # pylint: enable=no-value-for-parameter
    epoch_time = time.time() - start_time
    print('Epoch {} in {:0.2f} sec'.format(epoch, epoch_time))

    # evaluate test accuracy
    params = get_params(opt_state)
    test_acc = accuracy(params, shape_as_image(test_images, test_labels))
    test_loss = loss(params, shape_as_image(test_images, test_labels))
    print('Test set loss, accuracy (%): ({:.2f}, {:.2f})'.format(
        test_loss, 100 * test_acc))

    # determine privacy loss so far
    if FLAGS.dpsgd:
      delta = 1e-5
      num_examples = 60000
      eps = compute_epsilon(epoch * steps_per_epoch, num_examples, delta)
      print(
          'For delta={:.0e}, the current epsilon is: {:.2f}'.format(delta, eps))
    else:
      print('Trained with vanilla non-private SGD optimizer')


if __name__ == '__main__':
  app.run(main)
