# Profiling JAX programs

## TensorBoard profiling

[TensorBoard's
profiler](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras)
can be used to profile JAX programs. Tensorboard is a great way to acquire and
visualize performance traces and profiles of your program, including activity on
GPU and TPU. The end result looks something like this:

![TensorBoard profiler example](_static/tensorboard_profiler.png)

### Installation

The TensorBoard profiler is only available with the version of TensorBoard
bundled with TensorFlow.

```shell
pip install tensorflow tbp-nightly
```

If you already have TensorFlow installed, you only need to install the
`tbp-nightly` pip package. Be careful to only install one version of TensorFlow
or TensorBoard, otherwise you may encounter the "duplicate plugins" error
described {ref}`below <multiple_installs>`.

(We recommend `tbp-nightly` because `tensorboard-plugin-profile==2.4.0` is
incompatible with TensorBoard's experimental fast data loading logic. This
should be resolved with `tensorboard-plugin-profile==2.5.0` when it's
released. These instructions were tested with `tensorflow==2.4.1` and
`tbp-nightly==2.5.0a20210428`.)

### Programmatic capture

You can instrument your code to capture a profiler trace via the
{func}`jax.profiler.start_trace` and {func}`jax.profiler.stop_trace`
methods. Call {func}`~jax.profiler.start_trace` with the directory to write
trace files to. This should be the same `--logdir` directory used to start
TensorBoard. Then, you can use TensorBoard to view the traces.

For example, to take a profiler trace:

```python
import jax

jax.profiler.start_trace("/tmp/tensorboard")

# Run the operations to be profiled
key = jax.random.PRNGKey(0)
x = jax.random.normal(key, (5000, 5000))
y = x @ x
y.block_until_ready()

jax.profiler.stop_trace()
```

Note the {func}`block_until_ready` call. We use this to make sure on-device
execution is captured by the trace. See {ref}`async-dispatch` for details on why
this is necessary.

You can also use the {func}`jax.profiler.trace` context manager as an
alternative to `start_trace` and `stop_trace`:

```python
import jax

with jax.profiler.trace():
  key = jax.random.PRNGKey(0)
  x = jax.random.normal(key, (5000, 5000))
  y = x @ x
  y.block_until_ready()
```

To view the trace, first start TensorBoard if you haven't already:

```shell
$ tensorboard --logdir=/tmp/tensorboard
[...]
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.5.0 at http://localhost:6006/ (Press CTRL+C to quit)
```

You should be able to load TensorBoard at <http://localhost:6006/> in this
example. You can specify a different port with the `--port` flag. See
{ref}`remote_profiling` below if running JAX on a remote server.

Then, either select "Profile" in the upper-right dropdown menu, or go directly
to <http://localhost:6006/#profile>. Available traces appear in the "Runs"
dropdown menu on the left. Select the run you're interested in, and then under
"Tools", select "trace_viewer".  You should now see a timeline of the
execution. You can use the WASD keys to navigate the trace, and click or drag to
select events to see more details at the bottom. See [these TensorFlow
docs](https://www.tensorflow.org/tensorboard/tensorboard_profiling_keras#use_the_tensorflow_profiler_to_profile_model_training_performance)
for more details on using the trace viewer.

### Manual capture via TensorBoard

The following are instructions for capturing a manually-triggered N-second trace
from a running program.

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
   server = jax.profiler.start_server(9999)
   ```

    This starts the profiler server that TensorBoard connects to. The profiler
    server must be running before you move on to the next step. It will remain
    alive and listening until the object returned by `start_server()` is
    destroyed.

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

### Adding custom trace events

By default, the events in the trace viewer are mostly low-level internal JAX
functions. You can add your own events and functions by using
{class}`jax.profiler.TraceAnnotation` and {func}`jax.profiler.annotate_function` in
your code.

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

If you still get the `Could not load dynamic library` message after doing this,
check if the GPU trace shows up in the trace viewer anyway. This message
sometimes occurs even when everything is working, since it looks for the
`libcupti` library in multiple places.

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

See [NVIDIA's documentation on this
error](https://developer.nvidia.com/nvidia-development-tools-solutions-err-nvgpuctrperm-cupti)
for more information.

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

#### Profiling on a Cloud TPU VM

Cloud TPU VMs come with a special version of TensorFlow pre-installed, so
there's no need to explicitly install it, and doing so can cause TensorFlow to
stop working on TPU. Just `pip install tbp-nightly`.

(multiple_installs)=
#### Multiple TensorBoard installs

**If starting TensorBoard fails with an error like: `ValueError: Duplicate
plugins for name projector`**

It's often because there are two versions of TensorBoard and/or TensorFlow
installed (e.g. the `tensorflow`, `tf-nightly`, `tensorboard`, and `tb-nightly`
pip packages all include TensorBoard). Uninstalling a single pip package can
result in the `tensorboard` executable being removed which is then hard to
replace, so it may be necessary to uninstall everything and reinstall a single
version:

```shell
pip uninstall tensorflow tf-nightly tensorboard tb-nightly
pip install tensorflow
```

## Nsight

NVIDIA's `Nsight` tools can be used to trace and profile JAX code on GPU. For
details, see the [`Nsight`
documentation](https://developer.nvidia.com/tools-overview).
