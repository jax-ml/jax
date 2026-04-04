# Claude Code Prompt — JAX Kubernetes MNIST Training Example PR

Paste everything below this line into Claude Code:

---

I'm an Engineering Director at Google working on GKE, a maintainer of Kubernetes OSS, and chair of SIG API-Machinery. I'm contributing a distributed JAX training example to the `jax-ml/jax` repo under `examples/k8s/`.

## Context

The `jax-ml/jax` repo has an `examples/k8s/` directory with minimal scaffolding (just a `svc-acct.yaml` and a basic JobSet snippet in the docs). I'm adding a real end-to-end training example: distributed MNIST training on TPU via GKE, using JobSet for orchestration.

## Task

1. Fork and clone `jax-ml/jax`
2. Create a branch called `examples-k8s-mnist-training`
3. Create the directory `examples/k8s/mnist-training/`
4. Write the files listed below into that directory
5. Run linting (`ruff check` or `pyright`) on train_mnist.py and fix any issues
6. Verify the training script at least parses: `python -c "import ast; ast.parse(open('examples/k8s/mnist-training/train_mnist.py').read())"`
7. Prepare the commit but show me the final diff before pushing

## Files to create

### `examples/k8s/mnist-training/train_mnist.py`

```python
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
```

### `examples/k8s/mnist-training/Dockerfile`

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install JAX with TPU support.
# For GPU, replace jax[tpu] with jax[cuda12] (or the appropriate CUDA version).
RUN pip install --no-cache-dir \
    "jax[tpu]" \
    "jax[k8s]" \
    optax \
    keras \
    tensorflow-datasets

COPY train_mnist.py .

ENTRYPOINT ["python", "train_mnist.py"]
```

### `examples/k8s/mnist-training/svc-acct.yaml`

```yaml
# Service account and RBAC for JAX distributed peer discovery.
#
# JAX processes need to list pods in the same Job to auto-detect
# coordinator_address, num_processes, and process_id.
#
# Apply once per namespace:
#   kubectl apply -f svc-acct.yaml
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: jax-job-sa
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: jax-job-role
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: jax-job-rolebinding
subjects:
  - kind: ServiceAccount
    name: jax-job-sa
roleRef:
  kind: Role
  name: jax-job-role
  apiGroup: rbac.authorization.k8s.io
```

### `examples/k8s/mnist-training/jobset.yaml`

```yaml
# JobSet: distributed MNIST training with JAX on TPU.
#
# Prerequisites:
#   1. A GKE cluster with TPU node pools (e.g. v5e-lite with 2x4 topology).
#   2. JobSet CRD installed:
#        kubectl apply --server-side -f \
#          https://github.com/kubernetes-sigs/jobset/releases/download/v0.8.1/manifests.yaml
#   3. RBAC for JAX peer discovery:
#        kubectl apply -f svc-acct.yaml
#   4. Container image built and pushed:
#        docker build -t <REGISTRY>/jax-mnist:latest .
#        docker push <REGISTRY>/jax-mnist:latest
#
# Usage:
#   kubectl apply -f jobset.yaml
#   kubectl logs -l jobset.sigs.k8s.io/jobset-name=jax-mnist-training
#
# This creates a single-slice TPU v5e training job with 2 workers.
# Each worker gets 4 TPU chips (2x4 topology = 8 chips total).
# Adjust replicas, topology, and nodeSelector for your cluster.
---
apiVersion: jobset.x-k8s.io/v1alpha2
kind: JobSet
metadata:
  name: jax-mnist-training
spec:
  failurePolicy:
    maxRestarts: 2
  replicatedJobs:
    - name: workers
      replicas: 1  # Number of TPU slices. Set >1 for multislice.
      template:
        spec:
          parallelism: 2    # Workers per slice (must match TPU topology).
          completions: 2
          backoffLimit: 0
          template:
            spec:
              serviceAccountName: jax-job-sa
              hostNetwork: true
              dnsPolicy: ClusterFirstWithHostNet
              restartPolicy: Never
              nodeSelector:
                cloud.google.com/gke-tpu-accelerator: tpu-v5-lite-podslice
                cloud.google.com/gke-tpu-topology: 2x4
              containers:
                - name: jax-mnist
                  image: <REGISTRY>/jax-mnist:latest  # <-- Replace with your image
                  args: ["--epochs", "5", "--batch-size", "128"]
                  ports:
                    - containerPort: 8471  # JAX distributed coordinator
                    - containerPort: 8080  # Health check (optional)
                  resources:
                    limits:
                      google.com/tpu: 4    # TPU chips per worker
                  securityContext:
                    privileged: true       # Required for TPU device access
```

### `examples/k8s/mnist-training/README.md`

```markdown
# Distributed MNIST training with JAX on Kubernetes

This example demonstrates data-parallel training of a ConvNet on MNIST using
JAX, orchestrated by a Kubernetes
[JobSet](https://github.com/kubernetes-sigs/jobset). Each JAX process trains
on a shard of the data and gradients are averaged across all processes via
`jax.lax.pmean`.

The example runs on TPU slices in GKE, but the same pattern works for GPU
clusters — swap the container image, node selector, and resource limits.

## Files

| File | Description |
|------|-------------|
| `train_mnist.py` | JAX training script with distributed initialization |
| `Dockerfile` | Container image (installs `jax[tpu]`, `jax[k8s]`, optax) |
| `jobset.yaml` | JobSet manifest for TPU training on GKE |
| `svc-acct.yaml` | RBAC ServiceAccount for JAX pod-based peer discovery |

## Prerequisites

- A GKE cluster with a TPU node pool (e.g. `tpu-v5-lite-podslice`, topology
  `2x4`). See
  [Deploy TPU workloads on GKE](https://cloud.google.com/kubernetes-engine/docs/how-to/tpus)
  for setup instructions.
- The JobSet CRD installed:
  \```bash
  VERSION=v0.8.1
  kubectl apply --server-side -f \
    https://github.com/kubernetes-sigs/jobset/releases/download/$VERSION/manifests.yaml
  \```

## Quick start

1. **Build and push the container image:**

   \```bash
   export REGISTRY=gcr.io/<YOUR_PROJECT>
   docker build -t $REGISTRY/jax-mnist:latest .
   docker push $REGISTRY/jax-mnist:latest
   \```

2. **Create the RBAC resources:**

   JAX uses the Kubernetes API to discover peer pods for distributed
   initialization. The ServiceAccount needs permission to list pods.

   \```bash
   kubectl apply -f svc-acct.yaml
   \```

3. **Update `jobset.yaml`** with your container registry:

   Replace `<REGISTRY>/jax-mnist:latest` with your actual image path.

4. **Launch the training job:**

   \```bash
   kubectl apply -f jobset.yaml
   \```

5. **Monitor progress:**

   \```bash
   # Watch pod status
   kubectl get pods -l jobset.sigs.k8s.io/jobset-name=jax-mnist-training -w

   # View training logs (from the first worker)
   kubectl logs -l jobset.sigs.k8s.io/jobset-name=jax-mnist-training \
     --max-log-requests=10
   \```

6. **Clean up:**

   \```bash
   kubectl delete jobset jax-mnist-training
   \```

## How it works

**Peer discovery:** When running inside a Kubernetes JobSet, each pod calls
`jax.distributed.initialize()` with no arguments. JAX's built-in Kubernetes
coordinator (enabled by `jax[k8s]`) uses the Kubernetes API to list sibling
pods and automatically determines the coordinator address, process count, and
process ID. The RBAC in `svc-acct.yaml` grants the minimum permissions needed
for this discovery.

**Data parallelism:** The training data is split across processes — each
process trains on `60000 / num_processes` samples. After computing gradients
locally, `jax.lax.pmean` averages them across all devices before the optimizer
update. This is functionally equivalent to training on the full dataset with a
larger batch size.

**JobSet:** The manifest creates one replicated Job with 2 workers per TPU
slice. Each worker gets 4 TPU chips. The `hostNetwork: true` and
`privileged: true` settings are required for TPU device access and
inter-chip communication on GKE.

## Adapting for GPU

To run on GPUs instead of TPUs:

1. Change the Dockerfile base image and install `jax[cuda12]` instead of
   `jax[tpu]`.
2. In `jobset.yaml`, replace the `nodeSelector` and `resources` with your
   GPU node selector (e.g. `nvidia.com/gpu: 1`) and remove `hostNetwork`
   and `privileged`.
3. Ensure NCCL is available in the container image.

## Scaling up

- **More workers per slice:** increase `parallelism` and `completions` in the
  JobSet.
- **Multi-slice (multislice):** increase `replicas` under `replicatedJobs` and
  add the `alpha.jobset.sigs.k8s.io/exclusive-topology` annotation.
- **Queue management:** add [Kueue](https://kueue.sigs.k8s.io/) for
  multi-tenant clusters with shared TPU capacity.
```

## PR description

Title: `Add distributed MNIST training example for Kubernetes (JobSet + TPU)`

Body:

```
## Summary

Adds an end-to-end example under `examples/k8s/mnist-training/` showing
distributed MNIST training with JAX on TPU, orchestrated by a Kubernetes
JobSet. This goes beyond the existing hello-world peer discovery example
by demonstrating actual data-parallel training with gradient averaging
via `jax.lax.pmean`.

## What's included

- `train_mnist.py`: Pure JAX ConvNet (no Flax dependency) with
  `jax.distributed.initialize()` for auto peer discovery on Kubernetes.
- `Dockerfile`: Minimal image with `jax[tpu]`, `jax[k8s]`, and optax.
- `jobset.yaml`: JobSet manifest for TPU v5e training on GKE.
- `svc-acct.yaml`: RBAC for JAX's pod-based peer discovery.
- `README.md`: Setup instructions, how-it-works explanation, and
  adaptation guide for GPU clusters.

## Design choices

- **Pure JAX, no Flax**: keeps the example self-contained with zero
  framework dependencies beyond JAX + optax.
- **JobSet only, no Kueue**: minimal orchestration footprint. The README
  mentions Kueue as a scaling-up option.
- **Data sharding by process**: each process trains on 60K/N samples
  with gradient averaging, the simplest correct data-parallel pattern.
- **Fallback data loading**: tries tensorflow_datasets first, falls
  back to keras.datasets for environments without tfds.

## Testing

- [ ] Syntax check: `python -c "import ast; ast.parse(...)"`
- [ ] Local single-process: `python train_mnist.py --dry-run`
- [ ] GKE TPU v5e with JobSet (2 workers, 2x4 topology)
```

## Commit message

```
Add distributed MNIST training example for Kubernetes

Demonstrates data-parallel JAX training on TPU slices orchestrated by
a Kubernetes JobSet. Includes training script, Dockerfile, JobSet
manifest, RBAC config, and documentation.
```

## Instructions

1. Fork `jax-ml/jax` to my GitHub account
2. Clone it locally
3. Create branch `examples-k8s-mnist-training`
4. Create directory `examples/k8s/mnist-training/`
5. Write all five files above into that directory
6. Run `python -c "import ast; ast.parse(open('examples/k8s/mnist-training/train_mnist.py').read())"` to verify syntax
7. If ruff is available, run `ruff check examples/k8s/mnist-training/train_mnist.py` and fix issues
8. Commit with the message above
9. Show me the final diff before pushing
