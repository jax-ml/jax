# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "line" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""PCA from scratch in JAX.

This example demonstrates:
- Using jax.numpy.linalg.svd for classic linear algebra
- JIT compilation with static arguments
- vmap for batched execution across multiple datasets
- Proper timing and visualization

Run with: python examples/pca_from_scratch.py
"""

import time
import jax
import jax.numpy as jnp
from jax import jit, vmap
from functools import partial

# 1. Use @partial to mark n_components as static so JIT knows the output shape
@partial(jit, static_argnames=['n_components'])
def pca(X: jnp.ndarray, n_components: int = 2):
    """PCA from scratch using JAX.

    Args:
        X: Array of shape (n_samples, n_features)
        n_components: Number of principal components to keep (must be static)

    Returns:
        X_projected: shape (n_samples, n_components)
        components: shape (n_components, n_features)
        explained_variance_ratio: shape (n_components,)
    """
    if n_components > min(X.shape):
        raise ValueError(
            f"n_components={n_components} must be <= min(n_samples, n_features)={min(X.shape)}"
        )
    mean = jnp.mean(X, axis=0)
    X_centered = X - mean

    # SVD via XLA
    U, S, Vt = jnp.linalg.svd(X_centered, full_matrices=False)

    # Truncate components
    components = Vt[:n_components]

    # Project data
    X_projected = X_centered @ components.T

    # Calculate explained variance
    explained_variance = S**2 / (X.shape[0] - 1)
    explained_variance_ratio = explained_variance[:n_components] / jnp.sum(explained_variance)

    return X_projected, components, explained_variance_ratio

# 2. Vectorizing over a batch of distinct datasets
batched_pca = vmap(pca, in_axes=(0, None))

def main():
    print("--- JAX PCA Example ---")
    key = jax.random.PRNGKey(42)

    # Scenario 1: Standard JIT PCA
    n_samples, n_features = 10_000, 50

    print("\n1. Compiling and running standard PCA...")

    # Generate structured data instead of a uniform blob
    # scale the first two features to have massive variance, then randomly rotate
    base_data = jax.random.normal(key, (n_samples, n_features))
    scales = jnp.array([10.0, 5.0] + [1.0] * (n_features - 2))
    scaled_data = base_data * scales

    # Create a random rotation matrix to hide the variance across all 50 dimensions
    key, subkey = jax.random.split(key)
    random_matrix = jax.random.normal(subkey, (n_features, n_features))
    rotation, _ = jnp.linalg.qr(random_matrix) # Get an orthogonal matrix

    X = scaled_data @ rotation

    # Start (Warmup)
    _ = pca(X, n_components=2)

    start = time.time()
    X_proj, comp, evr = pca(X, n_components=2)
    X_proj.block_until_ready() # Correct synchronization
    print(f"Standard PCA Execution Time: {time.time() - start:.4f}s")
    print(f"Explained variance ratio (first 2 PCs): {evr}")
    # Scenario 2: Batched PCA

    batch_size = 10
    print(f"\n2. Running Batched PCA on {batch_size} datasets simultaneously...")

    # Generate structured batched data so the plot shows the variance capture
    key, subkey = jax.random.split(key)
    base_batch = jax.random.normal(subkey, (batch_size, 5000, n_features))
    scaled_batch = base_batch * scales
    X_batch = scaled_batch @ rotation

    # Start(Warmup)
    _ = batched_pca(X_batch, 2)

    start = time.time()
    X_proj_batch, comp_batch, evr_batch = batched_pca(X_batch, 2)
    X_proj_batch.block_until_ready()
    print(f"Batched PCA Execution Time: {time.time() - start:.4f}s")
    print(f"Output shape of projected batch: {X_proj_batch.shape}")

    # Visualization
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 5))
        # Plot only the first dataset from the batch
        plt.scatter(X_proj_batch[0, :, 0], X_proj_batch[0, :, 1], alpha=0.5, s=2)
        plt.title("PCA Projection (Dataset 0 in Batch)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.tight_layout()
        plt.savefig("pca_batched_projection.png")
        print("\nPlot saved successfully to 'pca_batched_projection.png'.")
    except ImportError:
        print("\nMatplotlib not installed. Skipping visualization.")

if __name__ == "__main__":
    main()
