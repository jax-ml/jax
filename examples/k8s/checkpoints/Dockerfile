# Use a base image with JAX and TPU/GPU support
# For TPU, consider: us-docker.pkg.dev/cloud-tpu-images/jax-tpu/jax:latest
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY training.py .

# Command to run the training
CMD ["python", "training.py"]
