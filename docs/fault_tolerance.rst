.. raw:: html

    <link href="_static/fault_tolerance/fault_tolerance.css" rel="stylesheet" />
    <script src="_static/fault_tolerance/fault_tolerance.js"></script>


Fault Tolerant Distributed JAX
==============================

Recall that `multi-controller JAX`_ allows you to run a JAX program distributed
across multiple machines. By default, if *any* of these machines fail, then
*every* machine will fail. That is, multi-controller JAX is not
**fault-tolerant** by default.

This article has three parts. In the first part, we'll explain the basics of
how to write fault tolerant multi-controller JAX programs. In the second part,
we'll show some example fault-tolerant multi-controller JAX programs. In the
third part, we'll take a look under the covers at how multi-controller JAX
implements fault tolerance.

.. warning::

    JAX's support for fault tolerance is still experimental. It currently only
    works fully on GPUs. It has rough edges, is probably buggy, and is subject
    to change. Use at your own risk.


.. _part1:

Part 1: Fault Tolerance Basics
------------------------------

Fault Intolerant By Default
^^^^^^^^^^^^^^^^^^^^^^^^^^^

By default, multi-controller JAX programs are not fault tolerant. If *any*
process crashes, then *all* other processes will also intentionally crash. To
make this concrete, consider the following trivial script, ``example.py``, that
initializes multi-controller JAX by calling ``jax.distributed.initialize`` and
then enters an infinite loop.

.. literalinclude:: _static/fault_tolerance/while_loop.py
    :language: python
    :emphasize-lines: 12-18
    :lines: 15-
    :linenos:
    :caption: ``example.py``

Run ``example.py`` across four processes on a VM with four GPUs by running
the following four commands, each in a different terminal. The
``local_device_ids`` argument to ``jax.distributed.initialize`` ensures each
process is assigned only one of the four GPUs. We'll explain the
``heartbeat_timeout_seconds`` argument in just a second.

.. code-block:: shell

    python example.py --i=0 --n=4  # in terminal 1
    python example.py --i=1 --n=4  # in terminal 2
    python example.py --i=2 --n=4  # in terminal 3
    python example.py --i=3 --n=4  # in terminal 4

When you run these commands, you'll see the processes dutifully printing out
the current time every second. Next, fail the fourth process: ``pkill -9 -f
'python example.py --i=3 --n=4'``. After about ten seconds, the other
processes will also terminate and spit out error messages that look something
like this:

.. code-block::

   E0926 17:26:32.075402  157988 coordination_service_agent.cc:332] Polled an error from coordination service (this can be an error from this or another task).
   F0926 17:26:32.075587  157988 client.h:77] Terminating process because the JAX distributed service detected fatal errors. This most likely indicates that another task died; see the other task logs for more details. Disable Python buffering, i.e. `python -u`, to be sure to see all the previous output. absl::Status: UNAVAILABLE: The following tasks are unhealthy (stopped sending heartbeats):
   /job:jax_worker/replica:0/task:3
   The tasks have crashed. Check the task logs for an earlier error, or scheduler events (e.g. preemption, eviction) to debug further.

   RPC: /tensorflow.CoordinationService/PollForError [type.googleapis.com/tensorflow.CoordinationServiceError='']

When a process in a multi-controller JAX program notices that a peer process
has crashed, it decides to crash as well. The processes `share fate`_. The
``heartbeat_timeout_seconds`` argument to ``jax.distributed.initialize``
determines how long a process waits before concluding a peer process has died.
The first three processes crash about ten seconds after you kill the fourth
because we passed ``heartbeat_timeout_seconds=10`` as an argument to
``jax.distributed.initialize``.

Surviving Faults
^^^^^^^^^^^^^^^^

We can disable fate-sharing by adding the
``--xla_gpu_nccl_terminate_on_error=false`` flag and the
``jax_enable_recoverability`` configuration option to ``example.py``, as shown
below:

.. literalinclude:: _static/fault_tolerance/dont_fail.py
    :language: python
    :emphasize-lines: 1-2,15
    :linenos:
    :lines: 15-

Again run the script across four processes and then kill the fourth. Notice
that now, the other three processes happily continue executing.

Next try failing process 0. Notice that all four processes terminate with
error messages that look something like the following:

.. code-block::

   E0929 17:42:48.594192 1044529 coordination_service_agent.cc:332] Polled an error from coordination service (this can be an error from this or another task).
   F0929 17:42:48.594200 1044529 client.h:77] Terminating process because the JAX distributed service detected fatal errors. This most likely indicates that another task died; see the other task logs for more details. Disable Python buffering, i.e. `python -u`, to be sure to see all the previous output. absl::Status: UNAVAILABLE: Failed to send RPC to coordination service. Either the leader task was preempted/died/restarted unexpectedly or this task is experiencing network issues. Check earlier logs from 1) this task, 2) the leader (usually slice 0 task 0), and 3) cluster scheduler to debug further.
   Additional GRPC error information from remote target coordination_service while calling /tensorflow.CoordinationService/PollForError:
   :UNKNOWN:Error received from peer  {grpc_message:"Socket closed", grpc_status:14}

Process 0 is special. If process 0 fails, every process will fail, even with
fate-sharing disabled. Why? Process 0 runs an RPC service called the
coordination service that all processes use to coordination with each other. If
the coordination service fails, all other processes have no choice but to fail.
See :ref:`part3` for more details.

Getting Stuck in Collectives
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``example.py`` is now able to survive faults, but the processes do not
communicate with each other at all. Any realistic multi-controller JAX program
would involve communication between the processes (otherwise, what's the point
of using multi-controller JAX?). Let's edit ``example.py`` so that the
processes perform a collective ``jnp.sum`` in every iteration of the loop.

.. literalinclude:: _static/fault_tolerance/collectives.py
    :language: python
    :emphasize-lines: 27-32
    :linenos:
    :lines: 15-

In the highlighted code above, the processes create an array ``x`` sharded
across the four processes and then perform a distributed ``jnp.sum``. Again run
the program and fail the fourth process. You'll notice that the first three
process do not crash, but they do get *stuck*. By default, if a process fails
while participating in a distributed computation (like ``jnp.sum``), then the
rest of the processes participating in the computation will get stuck
*forever*.

.. _`canceling_collectives`:

Cancelling Collectives
^^^^^^^^^^^^^^^^^^^^^^

We can avoid getting stuck by cancelling collectives with a failed participant.
We can enable collective cancelling by providing a few more flags and
environment variables, highlighted below.

.. literalinclude:: _static/fault_tolerance/cancel_collectives.py
    :language: python
    :emphasize-lines: 1-8,22,33-35
    :linenos:
    :lines: 15-

We also need to insert a call to
``jax.experimental.multihost_utils._live_devices`` to make the script work. You
should normally not do this. You should instead use the ``live_devices`` API
that we'll introduce momentarily. For now, ``_live_devices`` is a hack to get
the script working before we explain the proper API.

Again run the script and fail the fourth process. The first three processes
will be stuck in their call to ``jnp.sum``, but after about ten seconds, the
call will be cancelled and ``jnp.sum`` will raise an exception that looks
something like this:

.. code-block::

    jaxlib._jax.XlaRuntimeError: FAILED_PRECONDITION: Task with incarnation id 3446767950926952685 is not connected


Knowing Who's Alive
^^^^^^^^^^^^^^^^^^^

After a process dies, the remaining *alive* procesess need to learn who is dead
and who is alive. For this, we can use the core JAX fault tolerance API:
``live_devices``. ``live_devices`` is a context manager that takes a list of
devices as an argument and returns the subset of these devices that are alive.
Below, we edit ``example.py`` to call ``live_devices``.

.. literalinclude:: _static/fault_tolerance/live_devices.py
    :language: python
    :emphasize-lines: 34-46
    :linenos:
    :lines: 15-

In the highlighted code above, we call ``live_devices`` with all devices
(``jax.devices()``) to get the set ``devices`` of live devices. We then shard
array ``x`` over these devices and perform a ``jnp.sum``. If a process fails
while executing the ``jnp.sum``, then ``jnp.sum`` will be cancelled and raise
an exception on the remaining live devices. Technically, the collective is not
guaranteed to fail. We'll revisit this in :ref:`atomicity`. For now, assume it
will fail.

.. note::

    ``jax.devices()`` always returns the set of *all* devices, even if some of
    these devices are on failed processes. Use
    ``jax.experimental.multihost_utils.live_devices`` to learn which of these
    devices are live.

Again run the script and fail the fourth process. Notice that the remaining
three alive processes catch the exception raised by ``jnp.sum`` and continue to
the next iteration of the while loop. In this next iteration, ``devices`` does
not include the device on the failed fourth process. The three alive processes
continue to execute correctly even though the fourth process is dead.

Next, restart the fourth process. Notice that after the fourth process
restarts, its device is again included in the set of alive devices returned by
``live_devices``. All four processes then continue executing normally.

At first blush, ``live_devices`` seems trivial. You give it a list of devices,
and it returns the ones that are alive. How complicated can that be?
Unfortunately, as with `many things in distributed systems`_, there are a lot
subtleties to iron out. Next, we explain the **barrier** semantics and
**atomicity** properties of ``live_devices``.

Barrier Semantics
^^^^^^^^^^^^^^^^^

Recall that every process in a `multi-controller JAX`_ program should run in
lockstep. The processes should execute the same instructions in the same order.
Failing to do so will *almost certainly* lead to deadlocks, crashes, or
anomalous behavior.

In the context of ``live_devices``, we need to ensure that every process agrees
on which processes are currently alive. This is difficult to ensure because
every process is executing independently at potentially different speeds and
processes can fail at any time. Consider again the ``example.py`` script from
above running on four processes. Imagine process 1 and 2 call ``live_devices``,
then process 4 fails, and then process 3 calls ``live_devices``. Process 1 and
2 might think process 4 is alive while process 3 thinks it is dead.

To avoid situations like these, ``live_devices`` guarantees that it returns the
same set of live devices to every process. It accomplishes this using a
barrier. A call to ``live_devicess(devices)`` blocks until every live process
hosting a device in ``devices`` has also called ``live_devices``. Once every
live process is in the ``live_devices`` barrier, ``live_devices`` returns the
same set of live devices to every process.

.. important::

    ``live_devices`` uses a barrier to ensure that it will *always* return the
    same set of live devices to every live process.

Because ``live_devices`` implements a barrier it is susceptible to deadlock if
used improperly. We recommend only having a single ``with live_devices`` block
in a program. Multiple calls to ``live_devices`` is hard to reason about and
can lead to deadlock.

See :ref:`part3` for details on how the ``live_devices`` barrier is implemented
as well as a formal semantics based on `linearizability`_.

.. _atomicity:

Atomicity
^^^^^^^^^

A distributed computation is **atomic** if every participant in the computation
agrees on whether the operation succeeds or fails. In the ``example.py`` script
above, we saw that when a process failed during the execution of a ``jnp.sum``,
then ``jnp.sum`` would abort and raise an exception on the remaining live
processes. So ``jnp.sum`` is atomic?

Unfortunately, it's not.

When a process fails during the execution of a collective operation (like
``jnp.sum``), the remaining processes may cancel the operation and raise an
exception or they may complete the operation successfully. Collective
operations in JAX do not have any inherent atomicity properties.

If collective operations are not atomic, however, then multi-controller JAX
processes might diverge. For example, if a process fails during a training step
of a machine learning model, some processes might detect the failure and roll
the model back to a checkpoint while other processes might think the step
succeeded and keep training.

To avoid the complexities of non-atomic execution, ``live_devices`` provides
its own atomicity guarantees despite the fact that collectives are not atomic.
Specifically, the body of a ``with live_devices`` block is guaranteed to either
complete successfully on all processes or raise an exception on all processes.
More concretely, if we consider the code snippet below, either every process
executes branch A or every process executes branch B. It is impossible for some
processes to execute A while others execute B.

.. code-block:: python

    try:
      with live_devices(jax.live_devices()) as devices:
        ...
    except Exception as e:
      ... # Branch A
    else:
      ... # Branch B

.. warning::

    A ``with live_devices`` block does not guarantee atomicity if the code
    block non-deterministically raises exceptions for reasons other than
    collectives that fail because of a crashed process. For example, if one
    process raises an exception because it runs out of memory, this exception
    will not be propagated to the other processes.

Recall that JAX uses `asynchronous dispatch`_. Operations like ``jnp.sum`` do
not block until the operation is complete. Instead, they return ``jax.Arrays``
that act as futures. This asynchrony can interact with ``live_devices`` in
unexpected ways. For example, consider the following code that performs a
``jnp.sum``, assigns the result to ``y``, and then prints ``y``:

.. code-block:: python

    x = ...
    y = ...
    try:
      with live_devices(jax.live_devices()) as devices:
        y = jnp.sum(x)
    except Exception as e:
      ... # Branch A
    else:
      ... # Branch B
    print(y)

Imagine that the ``with live_devices`` block executes successfully on all
processes. That is, all processes execute branch B. This only guarantees that
every process successfully created a future and assigned it to ``y``. The
actual computation of the ``jnp.sum`` may be delayed until outside the block.
Thus, some processes might successfully complete the ``jnp.sum`` and print the
value of ``y`` while other processes fail to complete the ``jnp.sum`` and raise
an exception when trying to print ``y``.

To avoid this, use ``jax.block_until_ready`` to ensure that computations are
performed within the ``with live_devices`` block. The code snippet below, which
now calls ``jax.block_until_ready`` when assigning to ``y``, guarantees that
every process will successfully execute the ``jnp.sum`` or every process will
raise an exception.

.. code-block:: python

    x = ...
    y = ...
    try:
      with live_devices(jax.live_devices()) as devices:
        y = jax.block_until_ready(jnp.sum(x))
    except Exception as e:
      ... # Branch A
    else:
      ... # Branch B
    print(y)

See :ref:`part3` for details on how atomicity is implemented.

Part 2: Examples
----------------

``live_devices`` is not a panacea; it is a tool. It does not magically make
multi-controller JAX programs fault tolerant. Rather, it allows you to
implement fault tolerance yourself in the way that is best for your
application.

The exact details of how you implement fault-tolerance will vary greatly based
on the nature of your application. In this section, we present some examples of
how to use ``live_devices``. The examples are meant to be illustrative but not
prescriptive. There are many other ways to implement fault tolerance.

Example 1: Fault Tolerant Data Parallel Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we train a trivial single-parameter linear model (:math:`y =
\alpha x`) with data parallelism across four processes. The example is
contrived---you would never train a model with a single parameter across four
machines---but we intentionally keep the model simple to focus on fault
tolerance.

Data parallelism makes implementing fault tolerance relatively straightforward.
Because every process has a full copy of the model weights, if a process fails,
we can simply ignore it and continue training. This example tolerates an
arbitrary number of process failures (excluding process 0), but once a process
fails, we assume it does not recover. The next example shows how to handle
process recovery.

First, we set some flags to disable fate-sharing and enable collective
cancelling. We also make the necessary imports and define some flags.

.. literalinclude:: _static/fault_tolerance/data_parallelism.py
    :language: python
    :lines: 15-33
    :lineno-start: 1

Next, we define a ``replicated`` function that returns an array replicated
across a set of devices. Note that ``replicated`` doesn't actually move any
data. It assumes the argument ``x`` already has equal value across all
processes. It simply returns a new view of that data, in a process-spanning
`jax.Array` with a replicated sharding.

.. literalinclude:: _static/fault_tolerance/data_parallelism.py
    :language: python
    :lines: 35-49
    :lineno-start: 21

We define a similar ``sharded`` function that returns an array sharded across a
set of devices. Again, ``sharded`` is not actually moving any data between
processes.

.. literalinclude:: _static/fault_tolerance/data_parallelism.py
    :language: python
    :lines: 52-64
    :lineno-start: 38

Now, we're ready to start writing our training loop. We begin by initializing
multi-controller JAX by calling ``jax.distributed.initialize``.

.. literalinclude:: _static/fault_tolerance/data_parallelism.py
    :language: python
    :lines: 67-76
    :lineno-start: 53

Then, we define our simple linear model, generate some random training data,
and initialize some basic hyperparameters.

.. literalinclude:: _static/fault_tolerance/data_parallelism.py
    :language: python
    :lines: 78-97
    :lineno-start: 64

Finally, we enter the main training loop.

.. literalinclude:: _static/fault_tolerance/data_parallelism.py
    :language: python
    :lines: 99-125
    :lineno-start: 85

- Every iteration of the loop, we call ``live_devices`` to learn which devices
  are currently alive.
- We then ensure that the model weights are replicated across these devices and
  ensure that the training data is sharded across these devices. Note that this
  doesn't actually move any data between the devices; it simply creates JAX
  arrays with the appropriate replication and sharding metadata.
- We call ``loss_and_grad`` to compute the gradient of the weights with respect
  to the current batch of data and then compute the new weights. Notice that we
  assign the new weights to ``new_weights`` rather than assigning to
  ``weights`` in case the training step fails. We also call
  ``jax.block_until_ready`` to ensure that every process has computed the new
  weights when we exit the ``live_devices`` block.
- If no processes failed during the execution of the training step, then the
  ``else`` branch is taken. The step is incremented, and ``weights`` is
  updated. Otherwise, an exception will be raised and the ``except`` branch is
  taken. In this case, we do not update ``step`` or ``weights`` and retry the
  step on the next iteration with the new set of live devices.

Here is the full example:

.. literalinclude:: _static/fault_tolerance/data_parallelism.py
    :language: python
    :linenos:
    :lines: 15-

Example 2: Fault Tolerant Data Parallel Training With Recovery
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Now, we modify the example above to allow failed processes to recover. When a
process recovers, it needs to receive the current step and model weights.
Because we assume process 0 never fails---recall that if process 0 fails, every
process will fail---we have process 0 send the current step and weights to
recovering processes.

First, we define ``send`` and ``recv`` functions that use a ``shard_map`` to
send data from one device to another. The sender calls ``send``, and the
receiver calls ``recv``.

.. literalinclude:: _static/fault_tolerance/data_parallelism_with_recovery.py
    :language: python
    :lines: 69-90
    :lineno-start: 55

``allgather`` performs an AllGather of a single float across a set of devices.

.. literalinclude:: _static/fault_tolerance/data_parallelism_with_recovery.py
    :language: python
    :lines: 93-100
    :lineno-start: 79

Finally, we modify the training loop to handle recovering processes, as shown
in the highlighted code below.

.. literalinclude:: _static/fault_tolerance/data_parallelism_with_recovery.py
    :language: python
    :lines: 135-178
    :lineno-start: 121
    :emphasize-lines: 7-22

Recovery is a two-step process. First, we need to detect which processes are
recovering. Second, we need process 0 to send the step and weights to the
recovering processes.

1. To detect which processes are recovering, we perform an AllGather on all
   live processes' steps. When a failed process recovers, its ``step`` will be
   ``0``, while the ``step`` on process ``0`` will be some positive number, so
   if a process' step is not equal to process 0's step, then it is recovering.
2. Then, we call the ``send`` and ``recv`` functions we defined above to
   transfer the current step and model weights from process 0 to the recovering
   processes.

Here is the full example:

.. literalinclude:: _static/fault_tolerance/data_parallelism_with_recovery.py
    :language: python
    :linenos:
    :lines: 15-

.. _part3:


Part 3: Implementation Details
------------------------------

We now take a deep dive into the architecture of multi-controller JAX and the
semantics and implementation of ``live_devices``. If you're only interested in
writing fault-tolerant multi-controller JAX programs, the first two parts of
this article suffice.

The Coordination Service
^^^^^^^^^^^^^^^^^^^^^^^^

When you launch a multi-controller JAX program, the first process (i.e. process
0) runs a standalone RPC server called the **coordination service**. Moreover,
all processes (including process 0) create an RPC client to the coordination
service. Concretely, the ``coordinator_address`` argument of
:func:`jax.distributed.initialize` is the address of the coordination service.
This argument lets process 0 know on what address to run the server, and it
lets all processes know which address to connect to.

The coordination service implements the multi-controller JAX **control plane**.
For example, it can perform a distributed barrier across all processes, and it
implements a key-value store that processes can use to exchange small amounts
of metadata. Note, however, that the **data plane** (e.g., all collective
operations on program data) is implemented directly between the processes and
does not involve the coordination service.

One of the most important functionalities of the coordination service is health
checking. Every process periodically sends a heartbeat to the coordination
service. If a process fails, it stops sending heartbeats. If the coordination
service hasn't received a heartbeat from a process for a while, it assumes the
process has failed.

This is shown in the interactive visualization below. The coordination service
is shown at the top and three multi-controller JAX processes are shown at the
bottom. Note how the processes periodically send heartbeats to the controller,
and the controller keeps track of the health of each process based on when it
last received a heartbeat. Try failing process 2 by clicking the "Fail" button.
Observe how the process stops sending heartbeats and the coordination service
eventually considers the process dead.

.. raw:: html

    <div class="cluster" id="cluster_1"></div>
    <script>
      init_cluster("cluster_1", {
        share_fate: false,
        live_devices: false,
        barrier: false,
      });
    </script>

By default, when the coordination service detects that a process has failed, it
sends a message to all other processes requesting that they self-terminate. In
other words, all processes in a multi-controller JAX program `share fate`_.
Again fail process 2 in the visualization below by clicking the "Fail" button
and observe how the coordination service notifies the other processes to fail.

.. raw:: html

    <div class="cluster" id="cluster_2"></div>
    <script>
      init_cluster("cluster_2", {
        share_fate: true,
        live_devices: false,
        barrier: false,
      });
    </script>

This fate sharing means that multi-controller JAX programs are not at all
fault-tolerant. They are fault-*intolerant*. To enable fault-tolerance, we
need to do two things:

- First, we need to remove fate sharing and allow processes to continue
  executing even when a peer process has died. This can be enabled using the
  ``jax_enable_recoverability`` option, as described in :ref:`part1`. We'll
  assume that this option is set.
- Second, we need to provide an API that processes can use to learn which
  processes are alive and which have failed. This is the ``live_devices`` API
  introduced in :ref:`part1`.

There is a surprising amount of technical depth and subtlety in implementing
the ``live_devices`` API. We'll walk through the design and implementation of
the API step-by-step. We'll begin by introducing a simpler ``live_processes``
API and slowly improve it until we arrive at the ``live_devices`` API.

Live Processes
^^^^^^^^^^^^^^

Let's try to design a new hypothetical JAX API: ``jax.live_processes``. As the
name suggests, we want ``jax.live_processes()`` to return the set of all
currently alive processes.  Here is a naive but (as we'll see momentarily)
incorrect implementation. When a process calls ``jax.live_processes()``, it
sends an RPC request to the coordination service. Remember that the
coordination service already uses heartbeats to keep track of which processes
are dead and which are alive, so when it receives a ``jax.live_processes``
request, it responds with the set of processes it thinks are alive.

This is illustrated below. Below each process is a "Call live_processes"
button. You can click this button to make the process call
``jax.live_processes``. Note how the coordination service replies to a
``live_processess`` request with the set of alive processes. Fail process 2 by
clicking the "Fail" button and see how it affects later calls to
``jax.live_processes``.

.. raw:: html

    <div class="cluster" id="cluster_3"></div>
    <script>
      init_cluster("cluster_3", {
        share_fate: false,
        live_devices: true,
        barrier: false,
      });
    </script>

This naive implementation is simple but incorrect. It is crucial that all
processes in a multi-controller JAX job execute the same instructions in the
same order. If the processes start to diverge, by executing different code
paths in the JAX program, the job will behave erratically. Most likely, it will
crash or hang or produce garbage values, and most certainly it will be very
hard to reason about.

Our naive implementation of ``jax.live_processes`` can very easily lead to
divergence. For example, consider a multi-controller JAX job with three
processes. If process 0 and 1 both call ``jax.live_processes`` around the same
time that process 2 fails, the coordination service might report to process 0
that all processes are alive but report to process 1 that only processes 0 and
1 are alive. Try to produce this scenario in the visualization below:

.. raw:: html

    <div class="cluster" id="cluster_4"></div>
    <script>
      init_cluster("cluster_4", {
        share_fate: false,
        live_devices: true,
        barrier: false,
      });
    </script>

If processes disagree on which processes are alive, they will almost certainly
diverge. Thankfully, we can avoid this divergence by augmenting
``jax.live_processes`` with barrier semantics.

Barrier Semantics
^^^^^^^^^^^^^^^^^

Let's change the implementation of ``jax.live_processes`` so that when the
coordination service receives a ``jax.live_processes()`` request, it does not
reply right away. Instead, the coordination service only replies once *every*
live process has called ``jax.live_processes()``. Once every alive process has
entered the ``jax.live_processess()`` barrier, the coordination service returns
the set of live processes. Crucially, the coordination service returns the
*same* set of live processes to all processes, which prevents the processes
from diverging.

This is illustrated below. Note that coordination server now keeps track of
which devices are in the ``live_processes`` barrier.  Try calling
``live_processes`` from every process.  Notice how the coordination service
doesn't respond until every process has entered the barrier. Then fail process
2 and call ``live_processes`` from process 0 and process 1.

.. raw:: html

    <div class="cluster" id="cluster_5"></div>
    <script>
      init_cluster("cluster_5", {
        share_fate: false,
        live_devices: true,
        barrier: true,
      });
    </script>

Formal Semantics
^^^^^^^^^^^^^^^^

Distributed systems are notoriously complex. Machines can fail at arbitrary
times, and network messages can be dropped, delayed, and reordered. In this
section, we introduce a formal semantics of the ``jax.live_processes`` API to
help tame this complexity. Thinking rigorously about the semantics of
``jax.live_processes`` will help us understand the behavior of the API even in
pathological executions.

We'll base the formal semantics of ``jax.live_processes`` on
`linearizability`_: a popular formalism used to define the semantics of many
distributed APIs. Concretely, we model our distributed system as a number of
processes. Each process serially performs a number of events. There are four
types of events:

1. A process can **start** (👶). We'll assume that when a process starts, it
   connects to the coordination service, so the coordination service is aware
   that is has started.
2. A process can **fail** (💀). Unlike starting, the coordination service may
   not immediately be aware that a process has failed.
3. A process can **send** a ``jax.live_processes`` request to the coordination
   service.
4. A process can **receive** a reply to a ``jax.live_processes`` request from
   the coordination service.

Below is a diagram of an execution of three processes: 0, 1, and 2. Time
progresses from left to right. First, all three processes start. This is shown
with the baby emojis. Then all three processes send ``jax.live_processes``
requests to the coordination service. This is shown as the start of the thick
colored regions. Later, all three processes receive a reply from the
coordination service with ``0,1,2`` as the set of live devices.

.. raw:: html

    <div class="svgbox">
      <svg viewBox="-10 -30 325 160">
        <!-- Process names -->
        <text x="0" y="0" class="proc">0</text>
        <text x="0" y="50" class="proc">1</text>
        <text x="0" y="100" class="proc">2</text>

        <!-- Process axes -->
        <line x1="10" y1="0" x2="300" y2="0" class="proc-axis"></line>
        <line x1="10" y1="50" x2="300" y2="50" class="proc-axis"></line>
        <line x1="10" y1="100" x2="300" y2="100" class="proc-axis"></line>

        <!-- Process 1 -->
        <text x="25" y="0" class="event">👶</text>
        <line x1="50" y1="0" x2="250" y2="0" class="rpc p0-color"></line>
        <text x="250" y="-15" class="reply">0,1,2</text>

        <!-- Process 2 -->
        <text x="25" y="50" class="event">👶</text>
        <line x1="50" y1="50" x2="250" y2="50" class="rpc p1-color"></line>
        <text x="250" y="35" class="reply">0,1,2</text>

        <!-- Process 2 -->
        <text x="25" y="100" class="event">👶</text>
        <line x1="50" y1="100" x2="250" y2="100" class="rpc p2-color"></line>
        <text x="250" y="85" class="reply">0,1,2</text>
      </svg>
    </div>

In this simple execution, it is clear that ``jax.live_processes`` is behaving
correctly. We can formalize this intuition with the following formal semantics.

.. attention::

   An execution is valid if whenever ``jax.live_processes`` returns a set ``P``
   of live processes, there exists an instantaneous moment in time at which
   every process in ``P`` was in the ``live_processes`` barrier and every other
   process was dead. An implementation of ``live_processes`` is correct if
   it only allows for valid executions.

Later, we will amend these formal semantics to cover some subtle corner cases,
but assume this simplified semantics for now.

In the example above, ``live_processes`` returns ``0,1,2``. In the
visualization below, we show that there does exist an instantaneous moment of
time in which processes 0, 1, and 2 are all in the barrier and all other
processes (there are none) are dead. The moment in time is drawn as a vertical
red bar.

.. raw:: html

    <div class="svgbox">
      <svg viewBox="-10 -30 325 160">
        <!-- Process names -->
        <text x="0" y="0" class="proc">0</text>
        <text x="0" y="50" class="proc">1</text>
        <text x="0" y="100" class="proc">2</text>

        <!-- Process axes -->
        <line x1="10" y1="0" x2="300" y2="0" class="proc-axis"></line>
        <line x1="10" y1="50" x2="300" y2="50" class="proc-axis"></line>
        <line x1="10" y1="100" x2="300" y2="100" class="proc-axis"></line>

        <!-- Process 1 -->
        <text x="25" y="0" class="event">👶</text>
        <line x1="50" y1="0" x2="250" y2="0" class="rpc p0-color"></line>
        <text x="250" y="-15" class="reply">0,1,2</text>

        <!-- Process 2 -->
        <text x="25" y="50" class="event">👶</text>
        <line x1="50" y1="50" x2="250" y2="50" class="rpc p1-color"></line>
        <text x="250" y="35" class="reply">0,1,2</text>

        <!-- Process 2 -->
        <text x="25" y="100" class="event">👶</text>
        <line x1="50" y1="100" x2="250" y2="100" class="rpc p2-color"></line>
        <text x="250" y="85" class="reply">0,1,2</text>

        <!-- Snapshot -->
        <line x1="150" y1="-20" x2="150" y2="120" class="snapshot"></line>
      </svg>
    </div>

There is nothing special about the specific moment in time we chose in the
visualization above. All that's important is that *there exists some* moment in
time where all processes in `P` are in the barrier and all other processes are
dead. There are many moments in time that satisfy this property, as shown
below.

.. raw:: html

    <div class="svgbox">
      <svg viewBox="-10 -30 325 160">
        <!-- Process names -->
        <text x="0" y="0" class="proc">0</text>
        <text x="0" y="50" class="proc">1</text>
        <text x="0" y="100" class="proc">2</text>

        <!-- Process axes -->
        <line x1="10" y1="0" x2="300" y2="0" class="proc-axis"></line>
        <line x1="10" y1="50" x2="300" y2="50" class="proc-axis"></line>
        <line x1="10" y1="100" x2="300" y2="100" class="proc-axis"></line>

        <!-- Process 1 -->
        <text x="25" y="0" class="event">👶</text>
        <line x1="50" y1="0" x2="250" y2="0" class="rpc p0-color"></line>
        <text x="250" y="-15" class="reply">0,1,2</text>

        <!-- Process 2 -->
        <text x="25" y="50" class="event">👶</text>
        <line x1="50" y1="50" x2="250" y2="50" class="rpc p1-color"></line>
        <text x="250" y="35" class="reply">0,1,2</text>

        <!-- Process 2 -->
        <text x="25" y="100" class="event">👶</text>
        <line x1="50" y1="100" x2="250" y2="100" class="rpc p2-color"></line>
        <text x="250" y="85" class="reply">0,1,2</text>

        <!-- Snapshot -->
        <line x1="150" y1="-20" x2="150" y2="120" class="snapshot">
          <animate attributeName="x1" values="50;250;50" dur="4s" repeatCount="indefinite"/>
          <animate attributeName="x2" values="50;250;50" dur="4s" repeatCount="indefinite"/>
        </line>
      </svg>
    </div>

In the next example, processes 0 and 1 start, call ``jax.live_devices``, and
receive ``0,1`` as a reply. Process 2 is dead throughout the execution.

.. raw:: html

    <div class="svgbox">
      <svg viewBox="-10 -30 325 160">
        <!-- Process names -->
        <text x="0" y="0" class="proc">0</text>
        <text x="0" y="50" class="proc">1</text>
        <text x="0" y="100" class="proc">2</text>

        <!-- Process axes -->
        <line x1="10" y1="0" x2="300" y2="0" class="proc-axis"></line>
        <line x1="10" y1="50" x2="300" y2="50" class="proc-axis"></line>
        <line x1="10" y1="100" x2="300" y2="100" class="proc-axis"></line>

        <!-- Process 1 -->
        <text x="25" y="0" class="event">👶</text>
        <line x1="50" y1="0" x2="250" y2="0" class="rpc p0-color"></line>
        <text x="250" y="-15" class="reply">0,1</text>

        <!-- Process 2 -->
        <text x="25" y="50" class="event">👶</text>
        <line x1="50" y1="50" x2="250" y2="50" class="rpc p1-color"></line>
        <text x="250" y="35" class="reply">0,1</text>

        <!-- Process 2 -->
        <text x="25" y="100" class="event">💀</text>
      </svg>
    </div>

This is a valid execution under our formal semantics because there exists a
moment a time in which processes 0 and 1 are in the barrier and process 2 is
dead.

.. raw:: html

    <div class="svgbox">
      <svg viewBox="-10 -30 325 160">
        <!-- Process names -->
        <text x="0" y="0" class="proc">0</text>
        <text x="0" y="50" class="proc">1</text>
        <text x="0" y="100" class="proc">2</text>

        <!-- Process axes -->
        <line x1="10" y1="0" x2="300" y2="0" class="proc-axis"></line>
        <line x1="10" y1="50" x2="300" y2="50" class="proc-axis"></line>
        <line x1="10" y1="100" x2="300" y2="100" class="proc-axis"></line>

        <!-- Process 1 -->
        <text x="25" y="0" class="event">👶</text>
        <line x1="50" y1="0" x2="250" y2="0" class="rpc p0-color"></line>
        <text x="250" y="-15" class="reply">0,1</text>

        <!-- Process 2 -->
        <text x="25" y="50" class="event">👶</text>
        <line x1="50" y1="50" x2="250" y2="50" class="rpc p1-color"></line>
        <text x="250" y="35" class="reply">0,1</text>

        <!-- Process 2 -->
        <text x="25" y="100" class="event">💀</text>

        <!-- Snapshot -->
        <line x1="150" y1="-20" x2="150" y2="120" class="snapshot"></line>
      </svg>
    </div>

In the following execution, process 0 calls ``jax.live_processes`` and receives
a reply of ``0``. Process 1 calls ``jax.live_processes``, but dies before
receiving a reply.

.. raw:: html

    <div class="svgbox">
      <svg viewBox="-10 -30 325 110">
        <!-- Process names -->
        <text x="0" y="0" class="proc">0</text>
        <text x="0" y="50" class="proc">1</text>

        <!-- Process axes -->
        <line x1="10" y1="0" x2="300" y2="0" class="proc-axis"></line>
        <line x1="10" y1="50" x2="300" y2="50" class="proc-axis"></line>

        <!-- Process 0 -->
        <text x="25" y="0" class="event">👶</text>
        <line x1="50" y1="0" x2="250" y2="0" class="rpc p0-color"></line>
        <text x="250" y="-15" class="reply">0</text>

        <!-- Process 1 -->
        <text x="25" y="50" class="event">👶</text>
        <line x1="50" y1="50" x2="150" y2="50" class="rpc p1-color"></line>
        <text x="150" y="50" class="event">💀</text>
      </svg>
    </div>

Is this a valid execution? Yes. There exists a moment in time at which process
0 is in the barrier and process 1 is dead, as shown below. Even though process
1 called ``jax.live_processes``, it is not guaranteed that process 1 will be
included in the coordination service's response.

For example, process 1's ``jax.live_processes`` request may have been dropped
by the network and never received by the coordination service. So from the
coordination service's perspective, process 1 is thoroughly dead and never even
entered the ``live_processes`` barrier.

.. raw:: html

    <div class="svgbox">
      <svg viewBox="-10 -30 325 110">
        <!-- Process names -->
        <text x="0" y="0" class="proc">0</text>
        <text x="0" y="50" class="proc">1</text>

        <!-- Process axes -->
        <line x1="10" y1="0" x2="300" y2="0" class="proc-axis"></line>
        <line x1="10" y1="50" x2="300" y2="50" class="proc-axis"></line>

        <!-- Process 0 -->
        <text x="25" y="0" class="event">👶</text>
        <line x1="50" y1="0" x2="250" y2="0" class="rpc p0-color"></line>
        <text x="250" y="-15" class="reply">0</text>

        <!-- Process 1 -->
        <text x="25" y="50" class="event">👶</text>
        <line x1="50" y1="50" x2="150" y2="50" class="rpc p1-color"></line>
        <text x="150" y="50" class="event">💀</text>

        <!-- Snapshot -->
        <line x1="200" y1="-20" x2="200" y2="70" class="snapshot"></line>
      </svg>
    </div>

What about the same exact execution, except that process 0 now receives the
reply ``0,1`` from the coordination service?

.. raw:: html

    <div class="svgbox">
      <svg viewBox="-10 -30 325 110">
        <!-- Process names -->
        <text x="0" y="0" class="proc">0</text>
        <text x="0" y="50" class="proc">1</text>

        <!-- Process axes -->
        <line x1="10" y1="0" x2="300" y2="0" class="proc-axis"></line>
        <line x1="10" y1="50" x2="300" y2="50" class="proc-axis"></line>

        <!-- Process 0 -->
        <text x="25" y="0" class="event">👶</text>
        <line x1="50" y1="0" x2="250" y2="0" class="rpc p0-color"></line>
        <text x="250" y="-15" class="reply">0,1</text>

        <!-- Process 1 -->
        <text x="25" y="50" class="event">👶</text>
        <line x1="50" y1="50" x2="150" y2="50" class="rpc p1-color"></line>
        <text x="150" y="50" class="event">💀</text>
      </svg>
    </div>

Again, this is a valid execution, as witnessed below. Intuitively, the
coordination service could have received ``jax.live_processes`` requests from
both processes 0 and 1 and sent the reply ``0,1`` to both. While this reply was
in the network, process 1 failed. Thus, even though process 1 is dead when
process 0 receives a reply, the execution is still valid.

.. raw:: html

    <div class="svgbox">
      <svg viewBox="-10 -30 325 110">
        <!-- Process names -->
        <text x="0" y="0" class="proc">0</text>
        <text x="0" y="50" class="proc">1</text>

        <!-- Process axes -->
        <line x1="10" y1="0" x2="300" y2="0" class="proc-axis"></line>
        <line x1="10" y1="50" x2="300" y2="50" class="proc-axis"></line>

        <!-- Process 0 -->
        <text x="25" y="0" class="event">👶</text>
        <line x1="50" y1="0" x2="250" y2="0" class="rpc p0-color"></line>
        <text x="250" y="-15" class="reply">0,1</text>

        <!-- Process 1 -->
        <text x="25" y="50" class="event">👶</text>
        <line x1="50" y1="50" x2="150" y2="50" class="rpc p1-color"></line>
        <text x="150" y="50" class="event">💀</text>

        <!-- Snapshot -->
        <line x1="100" y1="-20" x2="100" y2="70" class="snapshot"></line>
      </svg>
    </div>

This point bears repeating. If ``jax.live_processes`` returns a set ``P`` of
processes, it does not mean that all processes in ``P`` are *currently* alive
and all other processes are *currently* dead. It only means that *there existed
a point in time* when this was true.

In the following execution, process 1 calls ``jax.live_processes`` and fails.
Later, process 0 starts, calls ``jax.live_processes``, and receives ``0,1`` as
a reply.

.. raw:: html

    <div class="svgbox">
      <svg viewBox="-10 -30 325 110">
        <!-- Process names -->
        <text x="0" y="0" class="proc">0</text>
        <text x="0" y="50" class="proc">1</text>

        <!-- Process axes -->
        <line x1="10" y1="0" x2="300" y2="0" class="proc-axis"></line>
        <line x1="10" y1="50" x2="300" y2="50" class="proc-axis"></line>

        <!-- Process 0 -->
        <text x="175" y="0" class="event">👶</text>
        <line x1="200" y1="0" x2="250" y2="0" class="rpc p0-color"></line>
        <text x="250" y="-15" class="reply">0,1</text>

        <!-- Process 1 -->
        <text x="25" y="50" class="event">👶</text>
        <line x1="50" y1="50" x2="100" y2="50" class="rpc p1-color"></line>
        <text x="100" y="50" class="event">💀</text>
      </svg>
    </div>

Using the formal semantics described thus far, this is *not* a valid execution.
There is never a point in time where process 0 and 1 are both alive. However,
this *should* be a valid execution.

The reason has to do with the unavoidable fact that in a distributed system, it
is impossible to detect failures with 100% accuracy. If the coordination
service hasn't received heartbeats from a process in a while, it considers the
process dead. But, the coordination service cannot determine with 100%
certainty when the process died or if the process is actually dead at all.
Maybe the process died a long time ago, or maybe it died very recently, or
maybe it is alive but on the other side of a network partition.

Let's return to the execution above for a concrete example. Imagine the
coordination service successfully received process 1's ``live_processes``
request. Then, process 1 failed but the coordination service didn't detect the
failure immediately. In the meantime, the coordination service received process
0's ``live_processes`` request. At this point, the coordination service thought
both processes were alive and saw that both processes were in the barrier, so
it naturally returned ``0,1`` to both processes (though only process 0 received
the reply because process 1 was dead).

The coordination service thought process 1 was alive when it was dead. And
sometimes the coordination service might think a process is dead when it is
alive. Though not ideal, we need to accommodate executions like this because
they are unavoidable.

We amend our formal semantics and allow ourselves to move a failure either
earlier or later in time, though we cannot move a failure past a different
event from the same process. Intuitively, we can move a failure from when it
actually happened to the point in time when the coordination service thought it
happened. Continuing the example above, we can delay the failure of process 1
to create a moment in time in which both processes 0 and 1 are in the barrier,
witnessing the fact that the execution is valid.

.. raw:: html

    <div class="svgbox">
      <svg viewBox="-10 -30 325 110">
        <!-- Process names -->
        <text x="0" y="0" class="proc">0</text>
        <text x="0" y="50" class="proc">1</text>

        <!-- Process axes -->
        <line x1="10" y1="0" x2="300" y2="0" class="proc-axis"></line>
        <line x1="10" y1="50" x2="300" y2="50" class="proc-axis"></line>

        <!-- Process 0 -->
        <text x="175" y="0" class="event">👶</text>
        <line x1="200" y1="0" x2="250" y2="0" class="rpc p0-color"></line>
        <text x="250" y="-15" class="reply">0,1</text>

        <!-- Process 1 -->
        <text x="25" y="50" class="event">👶</text>
        <line x1="50" y1="50" x2="100" y2="50" class="rpc p1-color">
          <animate attributeName="x2" values="100;275;100" dur="4s" repeatCount="indefinite"/>
        </line>
        <text x="100" y="50" class="event">
          <animate attributeName="x" values="100;275;100" dur="4s" repeatCount="indefinite"/>
          💀
        </text>

        <!-- Snapshot -->
        <line x1="225" y1="-20" x2="225" y2="70" class="snapshot">
          <animate attributeName="opacity" values="0;0;0;1;0;0;0" dur="4s" repeatCount="indefinite"/>
        </line>
      </svg>
    </div>

Consider a similar execution below.

.. raw:: html

    <div class="svgbox">
      <svg viewBox="-10 -30 325 110">
        <!-- Process names -->
        <text x="0" y="0" class="proc">0</text>
        <text x="0" y="50" class="proc">1</text>

        <!-- Process axes -->
        <line x1="10" y1="0" x2="300" y2="0" class="proc-axis"></line>
        <line x1="10" y1="50" x2="300" y2="50" class="proc-axis"></line>

        <!-- Process 0 -->
        <text x="25" y="0" class="event">👶</text>
        <line x1="100" y1="0" x2="200" y2="0" class="rpc p0-color"></line>
        <text x="200" y="-15" class="reply">0</text>

        <!-- Process 1 -->
        <text x="25" y="50" class="event">👶</text>
        <line x1="50" y1="50" x2="250" y2="50" class="rpc p1-color"></line>
        <text x="250" y="50" class="event">💀</text>
      </svg>
    </div>

As is, there is no moment in time in which process 0 is alive and process 1 is
dead. However, if we move the failure of process 1 leftwards, there is. How
might such an execution arise? Imagine process 1 is partitioned from the
coordination service. The coordination service doesn't receive any messages
from process 1, including its heartbeats. This leads the coordination service
to conclude that process 1 is dead, even though it isn't. Then, the
coordination service receives process 0's ``live_processes`` request and
responds with ``0``.

.. raw:: html

    <div class="svgbox">
      <svg viewBox="-10 -30 325 110">
        <!-- Process names -->
        <text x="0" y="0" class="proc">0</text>
        <text x="0" y="50" class="proc">1</text>

        <!-- Process axes -->
        <line x1="10" y1="0" x2="300" y2="0" class="proc-axis"></line>
        <line x1="10" y1="50" x2="300" y2="50" class="proc-axis"></line>

        <!-- Process 0 -->
        <text x="25" y="0" class="event">👶</text>
        <line x1="100" y1="0" x2="200" y2="0" class="rpc p0-color"></line>
        <text x="200" y="-15" class="reply">0</text>

        <!-- Process 1 -->
        <text x="25" y="50" class="event">👶</text>
        <line x1="50" y1="50" x2="250" y2="50" class="rpc p1-color">
          <animate attributeName="x2" values="250;100;250" dur="4s" repeatCount="indefinite"/>
        </line>
        <text x="250" y="50" class="event">
          <animate attributeName="x" values="250;100;250" dur="4s" repeatCount="indefinite"/>
          💀
        </text>

        <!-- Snapshot -->
        <line x1="150" y1="-20" x2="150" y2="70" class="snapshot">
          <animate attributeName="opacity" values="0;0;0;1;0;0;0" dur="4s" repeatCount="indefinite"/>
        </line>
      </svg>
    </div>

We cannot move a process failure past the process' other events, however. For
example, the following execution is *invalid* because no matter where we move
the failure of process 1, there is never a moment in time where both processes
are in the barrier.

.. raw:: html

    <div class="svgbox">
      <svg viewBox="-10 -30 325 110">
        <!-- Process names -->
        <text x="0" y="0" class="proc">0</text>
        <text x="0" y="50" class="proc">1</text>

        <!-- Process axes -->
        <line x1="10" y1="0" x2="300" y2="0" class="proc-axis"></line>
        <line x1="10" y1="50" x2="300" y2="50" class="proc-axis"></line>

        <!-- Process 0 -->
        <text x="175" y="0" class="event">👶</text>
        <line x1="200" y1="0" x2="250" y2="0" class="rpc p0-color"></line>
        <text x="250" y="-15" class="reply">0,1</text>

        <!-- Process 1 -->
        <text x="25" y="50" class="event">👶</text>
        <text x="175" y="50" class="event">👶</text>
        <line x1="50" y1="50" x2="100" y2="50" class="rpc p1-color">
          <animate attributeName="x2" values="100;150;100" dur="2s" repeatCount="indefinite"/>
        </line>
        <text x="100" y="50" class="event">
          <animate attributeName="x" values="100;150;100" dur="2s" repeatCount="indefinite"/>
          💀
        </text>
      </svg>
    </div>

With these formal semantics, we can make sense of even complex executions. For
example, consider the following execution.

.. raw:: html

    <div class="svgbox">
      <svg viewBox="-10 -30 325 160">
        <!-- Process names -->
        <text x="0" y="0" class="proc">0</text>
        <text x="0" y="50" class="proc">1</text>
        <text x="0" y="100" class="proc">2</text>

        <!-- Process axes -->
        <line x1="10" y1="0" x2="300" y2="0" class="proc-axis"></line>
        <line x1="10" y1="50" x2="300" y2="50" class="proc-axis"></line>
        <line x1="10" y1="100" x2="300" y2="100" class="proc-axis"></line>

        <!-- Process 0 -->
        <text x="25" y="0" class="event">👶</text>
        <line x1="40" y1="0" x2="150" y2="0" class="rpc p0-color"></line>
        <text x="150" y="-15" class="reply">0</text>
        <line x1="175" y1="0" x2="275" y2="0" class="rpc p0-color"></line>
        <text x="275" y="-15" class="reply">0,2</text>

        <!-- Process 1 -->
        <text x="50" y="50" class="event">👶</text>
        <line x1="65" y1="50" x2="125" y2="50" class="rpc p1-color"></line>
        <text x="125" y="50" class="event">💀</text>
        <text x="150" y="50" class="event">👶</text>
        <line x1="165" y1="50" x2="290" y2="50" class="rpc p1-color"></line>
        <text x="290" y="50" class="event">💀</text>

        <!-- Process 2 -->
        <text x="100" y="100" class="event">👶</text>
        <line x1="115" y1="100" x2="150" y2="100" class="rpc p2-color"></line>
        <text x="150" y="100" class="event">💀</text>
      </svg>
    </div>


After moving some process failures, we see the execution is valid.

.. raw:: html

    <div class="svgbox">
      <svg viewBox="-10 -30 325 160">
        <!-- Process names -->
        <text x="0" y="0" class="proc">0</text>
        <text x="0" y="50" class="proc">1</text>
        <text x="0" y="100" class="proc">2</text>

        <!-- Process axes -->
        <line x1="10" y1="0" x2="300" y2="0" class="proc-axis"></line>
        <line x1="10" y1="50" x2="300" y2="50" class="proc-axis"></line>
        <line x1="10" y1="100" x2="300" y2="100" class="proc-axis"></line>

        <!-- Process 0 -->
        <text x="25" y="0" class="event">👶</text>
        <line x1="40" y1="0" x2="150" y2="0" class="rpc p0-color"></line>
        <text x="150" y="-15" class="reply">0</text>
        <line x1="175" y1="0" x2="275" y2="0" class="rpc p0-color"></line>
        <text x="275" y="-15" class="reply">0,2</text>

        <!-- Process 1 -->
        <text x="50" y="50" class="event">👶</text>
        <line x1="65" y1="50" x2="75" y2="50" class="rpc p1-color"></line>
        <text x="75" y="50" class="event">💀</text>
        <text x="150" y="50" class="event">👶</text>
        <line x1="165" y1="50" x2="200" y2="50" class="rpc p1-color"></line>
        <text x="200" y="50" class="event">💀</text>

        <!-- Process 2 -->
        <text x="100" y="100" class="event">👶</text>
        <line x1="115" y1="100" x2="275" y2="100" class="rpc p2-color"></line>
        <text x="275" y="100" class="event">💀</text>

        <!-- Snapshot -->
        <line x1="87" y1="-20" x2="87" y2="120" class="snapshot"></line>
        <line x1="225" y1="-20" x2="225" y2="120" class="snapshot"></line>
      </svg>
    </div>

The following execution, on the other hand, is invalid.

.. raw:: html

    <div class="svgbox">
      <svg viewBox="-10 -30 325 160">
        <!-- Process names -->
        <text x="0" y="0" class="proc">0</text>
        <text x="0" y="50" class="proc">1</text>
        <text x="0" y="100" class="proc">2</text>

        <!-- Process axes -->
        <line x1="10" y1="0" x2="300" y2="0" class="proc-axis"></line>
        <line x1="10" y1="50" x2="300" y2="50" class="proc-axis"></line>
        <line x1="10" y1="100" x2="300" y2="100" class="proc-axis"></line>

        <!-- Process 0 -->
        <text x="165" y="0" class="event">👶</text>
        <line x1="180" y1="0" x2="275" y2="0" class="rpc p0-color"></line>
        <text x="275" y="-15" class="reply">0,2</text>

        <!-- Process 1 -->
        <text x="25" y="50" class="event">👶</text>
        <line x1="40" y1="50" x2="125" y2="50" class="rpc p1-color"></line>
        <text x="125" y="35" class="reply">1</text>
        <text x="140" y="50" class="event">💀</text>

        <!-- Process 2 -->
        <text x="25" y="100" class="event">👶</text>
        <line x1="40" y1="100" x2="275" y2="100" class="rpc p2-color"></line>
        <text x="275" y="100" class="event">💀</text>
      </svg>
    </div>


Atomicity
^^^^^^^^^

Equipped with ``jax.live_processes``, let's try to write some fault-tolerant
multi-controller JAX code.

.. code-block:: python

   step = 0
   while True:
       # Get the devices on all live processes.
       procs = jax.live_processes()
       devices = [d for d in jax.devices() if d.process_index in procs]

       # Shard array x over these devices.
       mesh = jax.make_mesh((len(devices),), ("i",), devices=devices)
       spec = jax.sharding.PartitionSpec("i")
       sharding = jax.sharding.NamedSharding(mesh, spec)
       x = jax.make_array_from_process_local_data(sharding, np.ones(1))

       # Try to perform a jnp.sum.
       try:
           print(jnp.sum(x))
       except:
           # jnp.sum failed.
           pass
       else:
           # jnp.sum succeeded.
           step += 1

The code repeatedly

- calls ``jax.live_processes`` to learn which processes are alive,
- computes the set of devices on the healthy processes,
- shards an array across these healthy devices,
- performs a ``jnp.sum`` (i.e. AllReduce) on the array, and
- increments ``step`` if the ``jnp.sum`` succeeds.

This code *looks* correct, but it has a very subtle bug. Assume the ``jnp.sum``
is being performed across a set of processes ``P``. If one (or more) of the
processes in ``P`` fails during the execution of the ``jnp.sum``, then
``jnp.sum`` can behave differently on different processes. Some processes in
``P`` might see ``jnp.sum`` return the correct result. Other processes might
see ``jnp.sum`` raise an exception. Others might see ``jnp.sum`` return an
incorrect result.

.. warning::

   If a process fails during a collective operation, the operation may behave
   differently on different processes.

This means that the processes executing the code example above might diverge.
Some might increment ``step``, and some might not. In the trivial code example
above, this divergence is benign, but in a real program, the divergence would
likely lead to a crash, a deadlock, or garbage outputs. For example, if a
multi-controller JAX program is training a model with data parallelism and
starts to diverge, some processes might roll back their model weights to a
previous checkpoint while others continue training, leading to a
"franken-model" where nobody agrees on what the model weights are supposed to
be.

To write fault-tolerant code that does not diverge, we want **atomicity**. When
executing a block of code (like the ``jnp.sum`` above), we either want *every*
process to run the code successfully, or *every* process to learn that the code
failed to execute successfully. We don't want some processes succeeding and
others failing.

Thankfully, we can achieve atomicity with a very simple trick: call
``live_processes`` twice, once before a code block and once after. If all the
processes that were alive before the block are also alive after the block, then
the code block executed successfully on all live processes. On the other hand,
if any process died, then all remaining processes can agree the code block
failed to execute properly. Here's a sketch of what that might look like:

.. code-block:: python

        # Get the set of live processes before the code block.
        procs_before = jax.live_processes()

        # Execute the code block.
        ...

        # Get the set of live processes after the code block
        procs_after = jax.live_processes()
        if procs_before == procs_after:
            # The code block executed successfully on all processes in
            # procs_before.
            pass
        else:
            # The code block did not execute successfully. All processes will
            # agree it failed.
            pass

The code above should give you a rough idea of how to use two calls to
``live_processes`` to achieve atomicity, but there are still a handful of small
issues we need to address before it is fully correct. For example,

- What if the code block throws an exception? We need to catch the exception
  and still call ``live_processess`` the second time and then re-raise the
  exception.
- What if a process fails after the first call to ``live_processes`` and
  recovers before the second call? Wouldn't the code block fail but the
  processes before and after be the same? Every time a process starts, it
  generates a random **incarnation id**. In addition to checking that the set
  of processes hasn't changed, we also check that their incarnation ids haven't
  changed.
- What if a process recovers and its first call to ``live_processes`` matches
  up with a different process' second call to ``live_processes``? Couldn't this
  lead to a deadlock? Yes. We can avoid the problem by only calling
  ``live_processes`` at a single program point. We can be clever and use a
  single call to ``live_processes`` for two purposes. It can be used to check
  that the set of processes hasn't changed since the previous call to
  ``live_processes``, and it can be used to generate the set of live processes
  that should be used the next time the atomic code block is executed.

All these details are handled and abstracted away by the ``jax.live_devices``
API introduced in :ref:`part1`. ``jax.live_devices`` is a context manager that
guarantees the atomic execution of a block of code. In the code snippet below,
``devices`` is a list of the devices on all live processes. The code block
``A`` will execute atomically across these processes. That is, either every
process will see the code raise an exception (branch ``B``) or every process
will see the code succeed (branch ``C``).

.. code-block:: python

    try:
      with live_devices() as devices:
        pass # A
    except Exception as e:
      pass # B
    else:
      pass # C

Cancelling Collectives
^^^^^^^^^^^^^^^^^^^^^^

As mentioned in :ref:`canceling_collectives`, if a process participating in a
collective fails, then the other participating processes get stuck forever. We
need to explicitly cancel these collectives to allow the alive participants to
make progress. While the ``live_devices`` API is supported on all JAX backends
(i.e. CPU, GPU, TPU), cancelling collectives is only supported by the GPU
backend. Here, we briefly explain some of the implementation details behind
collective cancelling.

The GPU backend implements collectives using `NCCL`_, NVIDIA's collective
communication library. When a set of processes wants to perform a collective,
they form a **NCCL communicator**. Processes can then repeatedly perform
collectives using this communicator. Creating a communicator is expensive---it
requires network communication---so the JAX backend caches communicators keyed
by the set of participating processes and their incarnation ids.

Internally, a JAX client polls the coordination service for the current status
of every process. If a client ever detects that a process is dead or has
restarted with a new incarnation id, then the client aborts all communicators
with the failed incarnation id in its cache key.

.. _asynchronous dispatch: https://docs.jax.dev/en/latest/async_dispatch.html
.. _linearizability: https://cs.brown.edu/~mph/HerlihyW90/p463-herlihy.pdf
.. _many things in distributed systems: https://en.wikipedia.org/wiki/Fallacies_of_distributed_computing
.. _multi-controller JAX: https://docs.jax.dev/en/latest/multi_process.html
.. _NCCL: https://developer.nvidia.com/nccl
.. _reference: https://docs.jax.dev/en/latest/config_options.html#jax_enable_recoverability
.. _share fate: https://en.wikipedia.org/wiki/Fate-sharing
