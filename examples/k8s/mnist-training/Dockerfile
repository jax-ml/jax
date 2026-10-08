FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    "jax[tpu]" \
    "jax[k8s]" \
    optax

COPY train_mnist.py .

ENTRYPOINT ["python", "train_mnist.py"]