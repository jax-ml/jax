Determinism on GPUs
======================

It can be tricky to make floating point computations on GPUs fully
deterministic, in the sense of computing precisely the same values
across separate program executions on the same hardware, and in general doing
so may require sacrificing some performance. Some sources of nondeterminism in
XLA:GPU are:

1. `XLA:GPU autotuning <https://docs.nvidia.com/deeplearning/frameworks/tensorflow-user-guide/index.html#xla-autotune>`_ automatically selects which cuBLAS and cuDNN kernels to call for best performance on your device, but exactly which gets picked for any given program execution may depend on exogenous factors;
2. `GPU atomics can cause reductions to be nondeterministic <https://github.com/tensorflow/tensorflow/commit/e31955d9fb34ae7273354dc2347ba99eea8c5280>`_.

One can remove some sources of nondeterminism by setting the environment
variable :code:`XLA_FLAGS='--xla_gpu_deterministic_reductions --xla_gpu_autotune_level=0'`
before importing the `jax` package, but other sources of nondeterminism may persist.
