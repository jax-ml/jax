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
