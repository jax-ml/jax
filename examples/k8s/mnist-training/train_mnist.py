"""Distributed MNIST training with JAX on Kubernetes.

This example trains a simple ConvNet on MNIST using data-parallel training
across multiple JAX processes, orchestrated by a Kubernetes JobSet.

Each process trains on a shard of the data and gradients are averaged across
all processes via jax.lax.pmean.

Usage (local single-process):
    python train_mnist.py

Usage (Kubernetes):
    See the JobSet manifest in jobset.yaml and the README for instructions.
"""

import argparse
import time

import jax
import jax.numpy as jnp
from jax import random
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import optax
import numpy as np


# ---------------------------------------------------------------------------
# Model: simple ConvNet
# ---------------------------------------------------------------------------

def init_params(key):
    """Initialize parameters for a simple ConvNet."""
    keys = random.split(key, 5)
    params = {
        'conv1': {
            'w': random.normal(keys[0], (3, 3, 1, 32)) * 0.1,
            'b': jnp.zeros(32),
        },
        'conv2': {
            'w': random.normal(keys[1], (3, 3, 32, 64)) * 0.1,
            'b': jnp.zeros(64),
        },
        'fc1': {
            'w': random.normal(keys[2], (3136, 128)) * 0.05,
            'b': jnp.zeros(128),
        },
        'fc2': {
            'w': random.normal(keys[3], (128, 10)) * 0.05,
            'b': jnp.zeros(10),
        },
    }
    return params


def conv2d(x, w, b):
    """Simple 2D convolution with SAME-ish padding via JAX."""
    x = jax.lax.conv_general_dilated(
        x, w,
        window_strides=(1, 1),
        padding='VALID',
        dimension_numbers=('NHWC', 'HWIO', 'NHWC'),
    )
    return x + b


def forward(params, x):
    """Forward pass: Conv -> ReLU -> Conv -> ReLU -> Pool -> FC -> FC."""
    x = conv2d(x, params['conv1']['w'], params['conv1']['b'])
    x = jax.nn.relu(x)
    x = conv2d(x, params['conv2']['w'], params['conv2']['b'])
    x = jax.nn.relu(x)
    # Max pool 2x2
    x = jax.lax.reduce_window(
        x, -jnp.inf, jax.lax.max, (1, 2, 2, 1), (1, 2, 2, 1), 'VALID'
    )
    x = x.reshape((x.shape[0], -1))  # Flatten
    x = jax.nn.relu(x @ params['fc1']['w'] + params['fc1']['b'])
    x = x @ params['fc2']['w'] + params['fc2']['b']
    return jax.nn.log_softmax(x, axis=-1)


# ---------------------------------------------------------------------------
# Loss and training step
# ---------------------------------------------------------------------------

def cross_entropy_loss(params, images, labels):
    logits = forward(params, images)
    one_hot = jax.nn.one_hot(labels, 10)
    return -jnp.mean(jnp.sum(one_hot * logits, axis=-1))


@jax.jit
def train_step(params, opt_state, images, labels):
    """Single training step with gradient averaging across devices."""
    loss, grads = jax.value_and_grad(cross_entropy_loss)(params, images, labels)
    # Average gradients across all devices (data parallelism).
    grads = jax.lax.pmean(grads, axis_name='batch')
    loss = jax.lax.pmean(loss, axis_name='batch')
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@jax.jit
def eval_step(params, images, labels):
    logits = forward(params, images)
    predictions = jnp.argmax(logits, axis=-1)
    return jnp.sum(predictions == labels)


# ---------------------------------------------------------------------------
# Data loading (downloads MNIST via tensorflow_datasets or falls back to a
# simple HTTP fetch)
# ---------------------------------------------------------------------------

def load_mnist():
    """Load MNIST data as numpy arrays. Uses keras if available."""
    try:
        import tensorflow_datasets as tfds
        ds = tfds.load('mnist', split=['train', 'test'], as_supervised=True,
                       shuffle_files=False)
        train_images, train_labels = [], []
        for img, lbl in ds[0]:
            train_images.append(img.numpy())
            train_labels.append(lbl.numpy())
        test_images, test_labels = [], []
        for img, lbl in ds[1]:
            test_images.append(img.numpy())
            test_labels.append(lbl.numpy())
        train_images = np.stack(train_images).astype(np.float32) / 255.0
        train_labels = np.array(train_labels, dtype=np.int32)
        test_images = np.stack(test_images).astype(np.float32) / 255.0
        test_labels = np.array(test_labels, dtype=np.int32)
    except ImportError:
        # Fallback: use keras
        from keras.datasets import mnist as keras_mnist  # type: ignore
        (train_images, train_labels), (test_images, test_labels) = (
            keras_mnist.load_data()
        )
        train_images = train_images[..., np.newaxis].astype(np.float32) / 255.0
        test_images = test_images[..., np.newaxis].astype(np.float32) / 255.0
        train_labels = train_labels.astype(np.int32)
        test_labels = test_labels.astype(np.int32)

    return train_images, train_labels, test_images, test_labels


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Module-level optimizer so train_step can close over it.
optimizer = optax.adam(1e-3)


def main():
    parser = argparse.ArgumentParser(description='JAX MNIST on Kubernetes')
    parser.add_argument('--epochs', type=int, default=5,
                        help='number of training epochs (default: 5)')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='per-device batch size (default: 128)')
    parser.add_argument('--dry-run', action='store_true',
                        help='run one batch only, for testing')
    args = parser.parse_args()

    # ----- Distributed initialization -----
    # On Kubernetes (via JobSet) and Cloud TPU, jax.distributed.initialize()
    # auto-detects coordinator address, process count, and process ID from
    # the environment. No arguments needed.
    jax.distributed.initialize()

    process_idx = jax.process_index()
    num_processes = jax.process_count()
    local_devices = jax.local_devices()
    global_devices = jax.devices()

    if process_idx == 0:
        print(f"JAX distributed initialized: {num_processes} process(es)")
        print(f"Global devices: {len(global_devices)} — {global_devices}")
    print(f"[Process {process_idx}] Local devices: {local_devices}")

    # ----- Data loading and sharding -----
    train_images, train_labels, test_images, test_labels = load_mnist()

    # Each process trains on a different shard of the data.
    shard_size = len(train_images) // num_processes
    start = process_idx * shard_size
    end = start + shard_size
    train_images = train_images[start:end]
    train_labels = train_labels[start:end]

    if process_idx == 0:
        print(f"Training on {shard_size} samples per process, "
              f"{shard_size * num_processes} total")

    # ----- Initialize model and optimizer -----
    key = random.PRNGKey(42)
    params = init_params(key)
    opt_state = optimizer.init(params)

    # Create a mesh for data parallelism across all local devices.
    mesh = Mesh(np.array(local_devices), axis_names=('batch',))

    per_device_batch = args.batch_size
    num_local_devices = len(local_devices)
    local_batch_size = per_device_batch * num_local_devices

    # ----- Training loop -----
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        # Shuffle data each epoch
        perm = np.random.permutation(len(train_images))
        train_images = train_images[perm]
        train_labels = train_labels[perm]

        num_batches = len(train_images) // local_batch_size
        epoch_loss = 0.0

        for batch_idx in range(num_batches):
            i = batch_idx * local_batch_size
            batch_images = train_images[i:i + local_batch_size]
            batch_labels = train_labels[i:i + local_batch_size]

            # Reshape for per-device sharding: (num_devices, per_device_batch, ...)
            batch_images = batch_images.reshape(
                num_local_devices, per_device_batch, 28, 28, 1
            )
            batch_labels = batch_labels.reshape(
                num_local_devices, per_device_batch
            )

            batch_images = jnp.array(batch_images)
            batch_labels = jnp.array(batch_labels)

            with mesh:
                params, opt_state, loss = train_step(
                    params, opt_state, batch_images, batch_labels
                )

            epoch_loss += float(loss)

            if args.dry_run:
                print(f"[Process {process_idx}] Dry run batch loss: {loss:.4f}")
                jax.distributed.shutdown()
                return

        elapsed = time.time() - epoch_start
        avg_loss = epoch_loss / max(num_batches, 1)

        if process_idx == 0:
            print(f"Epoch {epoch}/{args.epochs} — "
                  f"loss: {avg_loss:.4f}, "
                  f"time: {elapsed:.1f}s")

    # ----- Evaluation -----
    correct = 0
    eval_batch = 1000
    for i in range(0, len(test_images), eval_batch):
        batch_images = jnp.array(test_images[i:i + eval_batch])
        batch_labels = jnp.array(test_labels[i:i + eval_batch])
        correct += int(eval_step(params, batch_images, batch_labels))

    if process_idx == 0:
        accuracy = correct / len(test_images) * 100
        print(f"\nTest accuracy: {correct}/{len(test_images)} ({accuracy:.1f}%)")

    jax.distributed.shutdown()


if __name__ == '__main__':
    main()
