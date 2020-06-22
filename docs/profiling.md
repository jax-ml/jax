# Profiling JAX programs

## TensorBoard profiling

[TensorBoard's
profiler](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras>)
can be used to profile JAX programs. This is a great way to easily time and
visualize different parts of your program, including what's happening on GPU or
TPU. The end result looks something like this:

![TensorBoard profiler example](_static/tensorboard_profiler.png)

### Installation

```shell
# Requires TensorFlow and TensorBoard version >= 2.2
pip install --upgrade tensorflow tensorboard_plugin_profile
```

### Usage

1. Start a TensorBoard server:

    ```shell
    tensorboard --logdir /tmp/tensorboard/
    ```

    You should be able to load TensorBoard at <http://localhost:6006/>. You can
    specify a different port with the `--port` flag. See {ref}`remote_profiling`
    below if running JAX on a remote server.<br /><br />

1. In the Python program or process you'd like to profile, add the following
   somewhere near the beginning:

   ```python
   import jax.profiler
   jax.profiler.start_server(9999)
   ```

    This starts the profiler server that TensorBoard connects to. The profiler
    server must be running before you move on to the next step.

    If you'd like to profile a snippet of a long-running program (e.g. a long
    training loop), you can put this at the beginning of the program and start
    your program as usual. If you'd like to profile a short program (e.g. a
    microbenchmark), one option is to start the profiler server in an IPython
    shell, and run the short program with `%run` after starting the capture in
    the next step. Another option is to start the profiler server at the
    beginning of the program and use `time.sleep()` to give you enough time to
    start the capture.<br /><br />

1. Open <http://localhost:6006/#profile>, and click the "CAPTURE PROFILE" button
   in the upper left. Enter "localhost:9999" as the profile service URL (this is
   the address of the profiler server you started in the previous step). Enter
   the number of milliseconds you'd like to profile for, and click "CAPTURE".<br
   /><br />

1. If the code you'd like to profile isn't already running (e.g. if you started
   the profiler server in a Python shell), run it while the capture is
   running.<br /><br />

1. After the capture finishes, TensorBoard should automatically refresh. (Not
   all of the TensorBoard profiling features are hooked up with JAX, so it may
   initially look like nothing was captured.) On the left under "Tools", select
   "trace_viewer".

   You should now see a timeline of the execution. You can use the WASD keys to
   navigate the trace, and click or drag to select events to see more details at
   the bottom. See [these TensorFlow
   docs](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras#use_the_tensorflow_profiler_to_profile_model_training_performance)
   for more details on using the trace viewer.<br /><br />

1. By default, the events in the trace viewer are mostly low-level internal JAX
   functions. You can add your own events and functions by using
   {func}`jax.profiler.TraceContext` and {func}`jax.profiler.trace_function` in
   your code and capturing a new profile.

### Troubleshooting

#### GPU profiling

Programs running on GPU should produce traces for the GPU streams near the top
of the trace viewer. If you're only seeing the host traces, check your program
logs and/or output for the following error messages.

**If you get an error like: `Could not load dynamic library 'libcupti.so.10.1'`**<br />
Full error:
```
W external/org_tensorflow/tensorflow/stream_executor/platform/default/dso_loader.cc:55] Could not load dynamic library 'libcupti.so.10.1'; dlerror: libcupti.so.10.1: cannot open shared object file: No such file or directory
2020-06-12 13:19:59.822799: E external/org_tensorflow/tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1422] function cupti_interface_->Subscribe( &subscriber_, (CUpti_CallbackFunc)ApiCallback, this)failed with error CUPTI could not be loaded or symbol could not be found.
```

Add the path to `libcupti.so` to the environment variable `LD_LIBRARY_PATH`.
(Try `locate libcupti.so` to find the path.) For example:
```shell
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/extras/CUPTI/lib64/:$LD_LIBRARY_PATH
```

**If you get an error like: `failed with error CUPTI_ERROR_INSUFFICIENT_PRIVILEGES`**<br />
Full error:
```shell
E external/org_tensorflow/tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1445] function cupti_interface_->EnableCallback( 0 , subscriber_, CUPTI_CB_DOMAIN_DRIVER_API, cbid)failed with error CUPTI_ERROR_INSUFFICIENT_PRIVILEGES
2020-06-12 14:31:54.097791: E external/org_tensorflow/tensorflow/core/profiler/internal/gpu/cupti_tracer.cc:1487] function cupti_interface_->ActivityDisable(activity)failed with error CUPTI_ERROR_NOT_INITIALIZED
```

Run the following commands (note this requires a reboot):
```shell
echo 'options nvidia "NVreg_RestrictProfilingToAdminUsers=0"' | sudo tee -a /etc/modprobe.d/nvidia-kernel-common.conf
sudo update-initramfs -u
sudo reboot now
```

(remote_profiling)=
#### Profiling on a remote machine

If the JAX program you'd like to profile is running on a remote machine, one
option is to run all the instructions above on the remote machine (in
particular, start the TensorBoard server on the remote machine), then use SSH
local port forwarding to access the TensorBoard web UI from your local
machine. Use the following SSH command to forward the default TensorBoard port
6006 from the local to the remote machine:

```shell
ssh -L 6006:localhost:6006 <remote server address>
```

## Nsight

Nvidia's `Nsight` tool can be used to trace and profile JAX code on GPU. For
details, see the `Nsight` documentation.

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
