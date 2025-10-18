# XLA compiler flags

<!--* freshness: { reviewed: '2024-08-18' } *-->

## Introduction
This guide gives a brief overview of XLA and how XLA relates to Jax.
For in-depth details please refer to [XLA documentation](https://openxla.org/xla). 

## XLA: The Powerhouse Behind Jax
XLA (Accelerated Linear Algebra) is a domain-specific compiler for linear algebra that plays a pivotal role in Jax's performance and flexibility. It enables Jax to generate optimized code for various hardware backends (CPUs, GPUs, TPUs) by transforming and compiling your Python/NumPy-like code into efficient machine instructions.

Jax uses XLA's JIT compilation capabilities to transform your Python functions into optimized XLA computations at runtime.

## Configuring XLA in Jax:
You can influence XLA's behavior in Jax by setting XLA_FLAGS environment variables before running your Python script or colab notebook.

For the colab notebooks:

Provide flags using `os.environ['XLA_FLAGS']`:


```python
import os

# Set multiple flags separated by spaces
os.environ['XLA_FLAGS'] = '--flag1=value1 --flag2=value2'
```

For the python scripts:

Specify `XLA_FLAGS` as a part of cli command:

```bash
XLA_FLAGS='--flag1=value1 --flag2=value2'  python3 source.py
```

**Important Notes:**

* Set `XLA_FLAGS` before importing Jax or other relevant libraries. Changing `XLA_FLAGS` after backend initialization will have no effect and given backend initialization time is not clearly defined it is usually safer to set `XLA_FLAGS` before executing any Jax code.
* Experiment with different flags to optimize performance for your specific use case.


**For further information:**
* Complete and up to date documentation about XLA can be found in the official [XLA documentation](https://openxla.org/xla).

* For backends supported by open-source version of XLA (CPU, GPU), XLA flags are defined with their default values in [xla/debug_options_flags.cc](https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc), and a complete list of flags could be found [here](https://github.com/openxla/xla/blob/main/xla/xla.proto).

* A guide on how to use key XLA flags can be found [here](https://openxla.org/xla/flags_guidance).

**Additional reading:**
* [GPU performance tips](https://docs.jax.dev/en/latest/gpu_performance_tips.html#xla-performance-flags)
