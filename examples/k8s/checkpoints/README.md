# JAX on Kubernetes: Probes and Checkpointing (Example #3)

This example demonstrates how to implement robust JAX training on Kubernetes using `JobSet`, focusing on **liveness probes**, **readiness probes**, and **resilient checkpointing** with Orbax and Google Cloud Storage (GCS).

## Overview

Distributed training in JAX requires careful orchestration to handle node failures and preemptions. This recipe provides a template for:
1.  **Health Monitoring**: Using Kubernetes probes to detect when a JAX process is stuck.
2.  **Persistence**: Using Orbax to save checkpoints directly to GCS via the Cloud Storage API.
3.  **Observability**: Using GCSFuse to mount the checkpoint bucket for manual inspection.
4.  **Security**: Using GKE Workload Identity to securely manage GCS access.

## Features

- **Liveness Probes**: The training loop updates a local heartbeat file (`/tmp/healthy`). Kubernetes monitors this file to ensure the training is making progress.
- **Readiness Probes**: Ensures a pod is only considered "ready" once `jax.distributed.initialize()` has successfully completed.
- **Orbax Checkpointing**: Configured to use native `gs://` paths for high-performance, asynchronous checkpointing.
- **Graceful Preemption**: A `SIGTERM` handler catches termination signals and triggers a final synchronous "emergency" checkpoint.
- **JobSet Integration**: Uses the Kubernetes `JobSet` API to manage the group of pods as a single logical unit.

## Prerequisites

- A GKE Cluster (Standard or Autopilot).
- The `JobSet` controller installed (`v0.8.0+`).
- A GCS bucket (e.g., `jax-examples-checkpoints`).

## Setup and Installation

### 1. Configure Security (GKE Workload Identity)

GKE Workload Identity is the recommended way for your pods to access Google Cloud services (like GCS) securely. It avoids the need for sensitive service account keys by "binding" a Kubernetes Service Account to a Google IAM Service Account.

#### Why is this necessary?
By default, a pod has no permissions to access GCS. We create a Google Service Account (GSA) with the necessary IAM roles and then tell GKE that our Kubernetes Service Account (KSA) is allowed to "act as" that GSA.

Follow these steps to configure access:

| Step | Action | Command |
| :--- | :--- | :--- |
| **1** | **Create GSA** | `gcloud iam service-accounts create jax-k8s-sa` |
| **2** | **Grant Storage Access** | `gcloud projects add-iam-policy-binding YOUR_PROJECT_ID --member="serviceAccount:jax-k8s-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com" --role="roles/storage.admin"` |
| **3** | **Allow Impersonation** | `gcloud iam service-accounts add-iam-policy-binding jax-k8s-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com --role="roles/iam.workloadIdentityUser" --member="serviceAccount:YOUR_PROJECT_ID.svc.id.goog[default/default]"` |
| **4** | **Annotate KSA** | `kubectl annotate serviceaccount default iam.gke.io/gcp-service-account=jax-k8s-sa@YOUR_PROJECT_ID.iam.gserviceaccount.com` |

> **Note:** Replace `YOUR_PROJECT_ID` with your actual GCP project ID. Step 3 assumes you are using the `default` namespace and `default` Kubernetes service account.

### 2. Create Artifact Registry (GAR)

Google Artifact Registry is the recommended place to store your training images. Create a repository with the following command:

```bash
gcloud artifacts repositories create jax-k8s-repo \
    --repository-format=docker \
    --location=us-central1 \
    --description="JAX Training Images"
```

### 3. Build and Push the Container

```bash
# 1. Build the image
docker build -t us-docker.pkg.dev/YOUR_PROJECT_ID/jax-k8s-repo/jax-train:latest .

# 2. Configure Docker to authenticate with GCP
gcloud auth configure-docker us-central1-docker.pkg.dev

# 3. Push the image
docker push us-docker.pkg.dev/YOUR_PROJECT_ID/jax-k8s-repo/jax-train:latest
```

### 4. Deploy the JobSet

Update `job-set.yaml` with your image path and bucket name, then run:

```bash
kubectl apply -f job-set.yaml
```

## Architecture Details

### Training Loop (`training.py`)
The script uses a `touch_liveness()` function that writes to `/tmp/healthy` at regular intervals. This signals to Kubernetes that the training step is executing correctly. If the script hangs (e.g., due to a deadlocked distributed call), Kubernetes will restart the pod.

### Checkpointing Path
- **Primary**: `gs://jax-examples-checkpoints/...` (Native API via Orbax)
- **Secondary (Mount)**: `/data` (GCSFuse mount of the same bucket)

### Preemption Strategy
When GKE sends a `SIGTERM` (preemption), the script catches the signal, sets an `exit_now` flag, and ensures `mngr.save(step, force=True)` is called. This guarantees the current state is saved before the pod is terminated.

## Verification

To verify the setup is working:
1.  Check pod status: `kubectl get pods -l jobset.sigs.k8s.io/jobset-name=jax-checkpointing-job`
2.  Inspect logs: `kubectl logs -l jobset.sigs.k8s.io/jobset-name=jax-checkpointing-job -c jax-training`
3.  Simulate failure: Manually delete a pod and verify the `JobSet` automatically restarts it and resumes from the latest checkpoint.
