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
  ```bash
  VERSION=v0.8.1
  kubectl apply --server-side -f \
    https://github.com/kubernetes-sigs/jobset/releases/download/$VERSION/manifests.yaml
  ```

## Quick start

1. **Build and push the container image:**

   ```bash
   export REGISTRY=gcr.io/<YOUR_PROJECT>
   docker build -t $REGISTRY/jax-mnist:latest .
   docker push $REGISTRY/jax-mnist:latest
   ```

2. **Create the RBAC resources:**

   JAX uses the Kubernetes API to discover peer pods for distributed
   initialization. The ServiceAccount needs permission to list pods.

   ```bash
   kubectl apply -f svc-acct.yaml
   ```

3. **Update `jobset.yaml`** with your container registry:

   Replace `<REGISTRY>/jax-mnist:latest` with your actual image path.

4. **Launch the training job:**

   ```bash
   kubectl apply -f jobset.yaml
   ```

5. **Monitor progress:**

   ```bash
   # Watch pod status
   kubectl get pods -l jobset.sigs.k8s.io/jobset-name=jax-mnist-training -w

   # View training logs (from the first worker)
   kubectl logs -l jobset.sigs.k8s.io/jobset-name=jax-mnist-training \
     --max-log-requests=10
   ```

   You should see output like:
   ```
   JAX distributed initialized: 2 process(es)
   Global devices: 8 — [TpuDevice(id=0), ..., TpuDevice(id=7)]
   Training on 30000 samples per process, 60000 total
   Epoch 1/5 — loss: 0.3521, time: 12.3s
   Epoch 2/5 — loss: 0.0842, time: 4.1s
   ...
   Test accuracy: 9850/10000 (98.5%)
   ```

6. **Clean up:**

   ```bash
   kubectl delete jobset jax-mnist-training
   ```

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
