# JAX Cloud TPU Preview

JAX now runs on Cloud TPUs! **This is a preview**, and we're still working on it. Help us out by kicking the tires, and letting us know if you run into any problems (see the [Reporting issues](#reporting-issues) section below).

## Example Cloud TPU notebooks

The following notebooks showcase how to use and what you can do with Cloud TPUs on Colab:

### [Pmap Cookbook](https://colab.research.google.com/github/google/jax/blob/master/cloud_tpu_colabs/Pmap_Cookbook.ipynb)
A guide to getting started with `pmap`, a transform for easily distributing SPMD
computations across devices.

### [Lorentz ODE Solver](https://colab.research.google.com/github/google/jax/blob/master/cloud_tpu_colabs/Lorentz_ODE_Solver.ipynb)
Contributed by Alex Alemi (alexalemi@)

Solve and plot parallel ODE solutions with `pmap`.

<img src="https://raw.githubusercontent.com/google/jax/master/cloud_tpu_colabs/images/lorentz.png" width=65%></image>

### [Wave Equation](https://colab.research.google.com/github/google/jax/blob/master/cloud_tpu_colabs/Wave_Equation.ipynb)
Contributed by Stephan Hoyer (shoyer@)

Solve the wave equation with `pmap`, and make cool movies! The spatial domain is partitioned across the 8 cores of a Cloud TPU.

![](https://raw.githubusercontent.com/google/jax/master/cloud_tpu_colabs/images/wave_movie.gif)

### [JAX Demo](https://colab.research.google.com/github/google/jax/blob/master/cloud_tpu_colabs/NeurIPS_2019_JAX_demo.ipynb)
An overview of JAX presented at the [Program Transformations for ML workshop at NeurIPS 2019](https://program-transformations.github.io/). Covers basic numpy usage, `grad`, `jit`, `vmap`, and `pmap`.

## Performance notes

The [guidance on running TensorFlow on TPUs](https://cloud.google.com/tpu/docs/performance-guide) applies to JAX as well, with the exception of TensorFlow-specific details. Here we highlight a few important details that are particularly relevant to using TPUs in JAX.

### Padding

One of the most common culprits for surprisingly slow code on TPUs is inadvertent padding:
- Arrays in the Cloud TPU are tiled. This entails padding one of the dimensions to a multiple of 8, and a different dimension to a multiple of 128.
- The matrix multiplication unit performs best with pairs of large matrices that minimize the need for padding.

### bfloat16 dtype

By default\*, matrix multiplication in JAX on TPUs [uses bfloat16](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus) with float32 accumulation. This can be controlled with the `precision` keyword argument on relevant `jax.numpy` functions (`matmul`, `dot`, `einsum`, etc). In particular:
- `precision=jax.lax.Precision.DEFAULT`: uses mixed bfloat16 precision (fastest)
- `precision=jax.lax.Precision.HIGH`: uses multiple MXU passes to achieve higher precision
- `precision=jax.lax.Precision.HIGHEST`: uses even more MXU passes to achieve full float32 precision

JAX also adds the `bfloat16` dtype, which you can use to explicitly cast arrays to bfloat16, e.g., `jax.numpy.array(x, dtype=jax.numpy.bfloat16)`.

\* We might change the default precision in the future, since it is arguably surprising. Please comment/vote on [this issue](https://github.com/google/jax/issues/2161) if it affects you!

## Running JAX on a Cloud TPU from a GCE VM

Creating a [Cloud TPU](https://cloud.google.com/tpu/docs/quickstart) involves creating the user GCE VM and the TPU node.

To create a user GCE VM, run the following command from your GCP console or your computer terminal where you have [gcloud installed](https://cloud.google.com/sdk/install).

```
export ZONE=us-central1-c
gcloud compute instances create $USER-user-vm-0001 \
   --machine-type=n1-standard-1 \
   --image-project=ml-images \
   --image-family=tf-1-14 \
   --boot-disk-size=200GB \
   --scopes=cloud-platform \
   --zone=$ZONE
```

To create a larger GCE VM, choose a different [machine type](https://cloud.google.com/compute/docs/machine-types).

Next, create the TPU node, following these [guidelines](https://cloud.google.com/tpu/docs/internal-ip-blocks) to choose a <TPU_IP_ADDRESS>.

```
export TPU_IP_ADDRESS=<TPU_IP_ADDRESS>
gcloud compute tpus create $USER-tpu-0001 \
      --zone=$ZONE \
      --network=default \
      --accelerator-type=v2-8 \
      --range=$TPU_IP_ADDRESS \
      --version=tpu_driver_nightly
```

Now that you have created both the user GCE VM and the TPU node, ssh to the GCE VM by executing the following command:

```
gcloud compute ssh $USER-user-vm-0001
```

Once you are in the VM, from your ssh terminal session, follow the example below to run a simple JAX program.

**Install jax and jaxlib wheels:**


```
pip install --user jax==0.1.54 jaxlib==0.1.37
```


**Create a program, simple_jax.py:**

**IMPORTANT**: Replace <TPU_IP_ADDRESS> below with the TPU node’s IP address. You can get the IP address from the GCP console: Compute Engine > TPUs.

```
from jax.config import config
from jax import random

# The following is required to use TPU Driver as JAX's backend.
config.FLAGS.jax_xla_backend = "tpu_driver"
config.FLAGS.jax_backend_target = "grpc://<TPU-IP-ADDRESS>:8470"

key = random.PRNGKey(0)
x = random.normal(key, (10,))
print(x)
```

**Run the program:**

```
python simple_jax.py
```

## Reporting issues

If you believe you’re experiencing a problem specific to using Cloud TPUs,
please create an issue in the [Cloud TPU issue
tracker](https://issuetracker.google.com/issues/new?component=507145). If you’re
unsure whether it’s problem with Cloud TPUs, JAX, Colab, or anything else, feel
free to create an issue in the [JAX issue
tracker](https://github.com/google/jax/issues) and we'll triage it
appropriately.

If you have any other questions or comments regarding JAX on Cloud TPUs, please
email jax-tpu@googlegroups.com. We’d like to hear from you!
