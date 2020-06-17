# Profiling JAX programs

To profile JAX programs, there are currently two options: `nvprof` and XLA's
profiling features.

## nvprof

Nvidia's `nvprof` tool can be used to trace and profile JAX code on GPU. For
details, see the `nvprof` documentation.

## XLA profiling

XLA has some built-in support for profiling on both CPU and GPU. To use XLA's
profiling features from JAX, set the environment variables
`TF_CPP_MIN_LOG_LEVEL=0` and `XLA_FLAGS=--xla_hlo_profile`. XLA will log
profiling information about each computation JAX runs. For example:

```shell
$ TF_CPP_MIN_LOG_LEVEL=0 XLA_FLAGS=--xla_hlo_profile ipython
...
In [1]: from jax import lax
In [2]: lax.add(1, 2)
2019-08-08 20:47:52.659030: I external/org_tensorflow/tensorflow/compiler/xla/service/service.cc:168] XLA service 0x7fe2c719e200 executing computations on platform Host. Devices:
2019-08-08 20:47:52.659054: I external/org_tensorflow/tensorflow/compiler/xla/service/service.cc:175]   StreamExecutor device (0): Host, Default Version
/Users/phawkins/p/jax/jax/lib/xla_bridge.py:114: UserWarning: No GPU/TPU found, falling back to CPU.
  warnings.warn('No GPU/TPU found, falling back to CPU.')
2019-08-08 20:47:52.674813: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174] Execution profile for primitive_computation.4: (0.0324 us @ f_nom)
2019-08-08 20:47:52.674832: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174]              94 cycles (100.% 100Σ) ::          0.0 usec (         0.0 optimal) ::       30.85MFLOP/s ::                    ::    353.06MiB/s ::     0.128B/cycle :: [total] [entry]
2019-08-08 20:47:52.674838: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174]              94 cycles (100.00% 100Σ) ::          0.0 usec (         0.0 optimal) ::       30.85MFLOP/s ::                    ::    353.06MiB/s ::     0.128B/cycle :: %add.3 = s32[] add(s32[] %parameter.1, s32[] %parameter.2)
2019-08-08 20:47:52.674842: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174]
2019-08-08 20:47:52.674846: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174] ********** microseconds report **********
2019-08-08 20:47:52.674909: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174] There are 0 microseconds in total.
2019-08-08 20:47:52.674921: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174] There are 0 microseconds ( 0.00%) not accounted for by the data.
2019-08-08 20:47:52.674925: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174] There are 1 ops.
2019-08-08 20:47:52.674928: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174]
2019-08-08 20:47:52.674932: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174] ********** categories table for microseconds **********
2019-08-08 20:47:52.674935: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174]
2019-08-08 20:47:52.674939: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174]  0 (100.00% Σ100.00%)   non-fusion elementwise (1 ops)
2019-08-08 20:47:52.674942: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174]                               * 100.00% %add.3 = s32[] add(s32[], s32[])
2019-08-08 20:47:52.675673: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174]
2019-08-08 20:47:52.675682: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174]
2019-08-08 20:47:52.675688: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174] ********** MiB read+written report **********
2019-08-08 20:47:52.675692: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174] There are 0 MiB read+written in total.
2019-08-08 20:47:52.675697: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174] There are 0 MiB read+written ( 0.00%) not accounted for by the data.
2019-08-08 20:47:52.675700: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174] There are 3 ops.
2019-08-08 20:47:52.675703: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174]
2019-08-08 20:47:52.675812: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174] ********** categories table for MiB read+written **********
2019-08-08 20:47:52.675823: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174]
2019-08-08 20:47:52.675827: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174]  0 (100.00% Σ100.00%)   non-fusion elementwise (1 ops)
2019-08-08 20:47:52.675832: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174]                               * 100.00% %add.3 = s32[] add(s32[], s32[])
2019-08-08 20:47:52.675835: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174]  0 ( 0.00% Σ100.00%)   ... (1 more categories)
2019-08-08 20:47:52.675839: I external/org_tensorflow/tensorflow/compiler/xla/service/executable.cc:174]
Out[2]: DeviceArray(3, dtype=int32)
```
