=====================
The Training Cookbook
=====================

Traditionally, machine learning codebases rely on libraries to perform much of the bookkeeping and parameter wrangling necessary for training large, complex models. While convenient, these libraries can abstract the key functionality and core APIs offered in JAX. The purpose of this cookbook, therefore, is to demonstrate best practices (or "recipes") for writing simple yet high-performance machine learning training code directly in JAX. Following the patterns documented below will prepare your machine learning workloads to maximally leverage our compiler (XLA) for performance and tractability. Most training scripts adhere roughly to the following structure:

.. tagged-block:: the-training-cookbook.py train-loop

For each line of code above, we will explain the best practices and showcase the core technologies we have assembled to empower you to write simple, yet unbelievably performant code in JAX. The code above is a segment of a self-contained, completely functional `companion script <https://github.com/jax-ml/jax/blob/main/docs/the-training-cookbook.py>`_ in which we initialize a `Vaswani et al. (2017) <https://arxiv.org/abs/170.03762>`_ Transformer decoder, define the training loss for next-token prediction, and `Adam optimizer <https://arxiv.org/abs/1412.6980>`_, in pure JAX. The code therein is suited to TPUs, CPUs, and GPUs, as well as single- and multi-host systems. For that reason, we use the terms *device* or *accelerator* to refer interchangeably to the hardware JAX is primarily performing arithmetic on—whether it be a TPU, GPU, or CPU—and *host system* to refer to operations performed exclusively using the host CPU. In this guide, there are many aspects of the JAX APIs we will gloss over for the sake of expediency. These are available for you to peruse at your leisure in our API documentation. However, there is a central JAX concept that one must confront in detail for much of what follows to cohere.

Device Mesh and Shardings
-------------------------

JAX employs the `Single Program, Multiple Data (SPMD) <https://en.wikipedia.org/wiki/Single_program,_multiple_data>`_ model of parallelism. This means we write a single program that runs on multiple devices, using annotations to specify which part of the data each device is responsible for. The two primary concepts for this are the :class:`jax.sharding.Mesh` and :class:`jax.P`.

Device Mesh
~~~~~~~~~~~
A :class:`jax.sharding.Mesh` is an arrangement of all our accelerators into a NumPy ``ndarray``, together with string labels for the axes of the device array. The reason for using an array is that this allows for a very convenient annotation for how arrays should be partitioned across devices. For this introduction, we will use the notation of an ordered dictionary [#ordered]_, so that ``{"x": 2, "y": 4}`` refers to a device mesh of shape ``(2, 4)`` with labeled axes ``"x"`` and ``"y"``. To shard an array ``param``, we decorate it with a :class:`jax.P`, which is a tuple of ``str | None`` elements of the same length as the dimensions of the array. The ``jax.P`` specifies which axes of our array are to be sharded over which axes of devices. A more thorough account of the notation of shardings and sharded computations is available in :ref:`sharded-computation`. Some common sharding strategies such as data parallel, fully sharded data parallel, and basic tensor parallelism will be covered in :ref:`achieving-high-performance`.

.. admonition:: Example

    Suppose we have a device mesh of ``{"x": 2, "y": 4}`` and an array ``param`` of shape ``(32, 64, 64, 128)``. If we shard this array with `jax.P(None, "x", "y", None) `, we end up with shards of size ``(32, 32, 16, 128)`` distributed across the devices. The ``None`` indicates that an axis should not be sharded. JAX implicitly broadcasts trailing axes, so an identical sharding can be achieved more concisely with `jax.P(None, "x", "y")`. As a result, the shorthand for a fully replicated array (of any dimension) is `jax.P()`.

.. admonition:: Example

    More advanced mesh geometries are convenient when aligned with the communication hierarchy of our devices. Host-to-host communication is typically slower than accelerator-to-accelerator communication. Suppose we have two host machines, each with eight attached GPUs. One might arrange the devices into a mesh of ``{"host": 2, "gpu": 8}``. Then we can shard a parameter as follows:

    .. code-block:: python

        param = jnp.zeros((256, 192), out_sharding=jax.P("gpu", None))

    The whole of ``param`` will be replicated twice, but within each host, it will be spread across the eight locally attached GPUs, with each GPU storing a shard of shape ``(32, 192)`` in HBM. This is particularly useful for :ref:`fsdp-sharding`.


Train State Initialization
--------------------------

.. tagged-block:: the-training-cookbook.py get-train-state
   :hl_lines: 4

Before we can get started, the first thing we need to do is set up the train state. The train state encapsulates (unsurprisingly) all the *stateful* aspects of the training process. This typically includes, at a minimum, the model parameters and the optimizer state. The way we have structured this function (though you may choose to do otherwise) is to:

1. Create a series of nested dictionaries to house the model parameters, and then

2. :func:`jax.tree.map` over those parameters to produce a similar set of nested dictionaries to house the accompanying optimizer states. (More on this `below <#optimizer-initialization>`_.)

Parameter Initialization
~~~~~~~~~~~~~~~~~~~~~~~~
.. tagged-block:: the-training-cookbook.py get-train-state
   :hl_lines: 4

To initialize our parameters, we build a series of nested dictionaries that correspond to the semantic sections of the neural network. If we were using a layer-based library such as PyTorch or Flax, these might correspond to neural network layers. For this example, we could, in fact, get by with a completely flattened dictionary, but the nested approach is convenient both for working with some of the APIs in JAX and for structuring our code.

.. tagged-block:: the-training-cookbook.py get-param-state

Our ``get_param_state`` function makes use of the ``constant`` and ``he_normal`` factories provided in :mod:`jax.nn.initializers`. These factories return an *initializer*, which is a function conforming to the following protocol:

.. code-block:: python

    class Initializer(Protocol):
        def __call__(self, key, shape, dtype, out_sharding) -> jax.Array:
            ...

The functional flavor of JAX requires explicit handling of all stochasticity (viz. :ref:`pseudorandom-numbers`), so we set up a little iterator that yields PRNG keys. Then, to build our parameters, we initialize them at their respective positions in the ``params`` nested dictionary, supplying the parameter shape, dtype, and sharding from the ``Config`` class.

.. note::

    By specifying the shardings here, we initialize each shard of each parameter directly on the correct device in the device mesh where it needs to be, preventing the need for needless host-to-device transfers or, in the case of a model that does not fit in system memory, avoiding out-of-memory errors.

Optimizer Initialization
~~~~~~~~~~~~~~~~~~~~~~~~
.. tagged-block:: the-training-cookbook.py get-train-state
   :hl_lines: 5

When it comes to setting up the optimizer state, things are a little less straightforward than when we built the model parameters. The `Adam optimizer <https://arxiv.org/abs/1412.6980>`_ requires that, for each parameter, we keep track of three optimization states: ``mu``, ``nu``, and ``count``. The simplest of these is ``count``, which stores the number of training steps we have performed. This is just a scalar used to de-bias the Adam updates. The ``mu`` and ``nu`` states will be arrays of the same shape, dtype, and sharding as the accompanying parameter ``param`` [#zeros_like]_

.. tagged-block:: the-training-cookbook.py get-adam-state

When we use :func:`jax.tree.map`, it iterates over the items in ``train_state.params``. For each parameter, it creates a corresponding Adam state, resulting in a new nested dictionary that mirrors the structure of ``train_state.params``. Each leaf in this new structure contains the optimizer state for the corresponding parameter.

The Train Step (Functional Transformations)
-------------------------------------------

.. tagged-block:: the-training-cookbook.py train-step

The train step is where we calculate the gradient of the model with respect to the current parameters and use the gradient, together with the optimizer, to update the parameters. To do this in JAX, we define the forward pass of the model, then we leverage JAX's functional transformations to automatically generate the backward pass, which we use to calculate the gradients and perform the update.

Model Forward Pass
~~~~~~~~~~~~~~~~~~

.. tagged-block:: the-training-cookbook.py model-apply

The model's forward pass is mostly unremarkable, aside from the ``out_sharding`` annotations we have supplied. These annotations declare what the result-sharding should be after the operation executes. The compiler uses these activation shardings, together with the parameter shardings we supplied when we `initialized the model <#parameter-initialization>`_, to dynamically insert `communication collectives <https://en.wikipedia.org/wiki/Collective_operation>`_ that ferry parameters and activations alike between devices. By choosing a good sharding strategy, we can achieve highly performant training (and inference) code. We will cover some standard strategies that serve most use cases in the section titled :ref:`achieving-high-performance`. For a detailed discussion of the principles underpinning the design of sharding strategies, see `The Scaling Cookbook <https://jax-ml.github.io/scaling-book/>`_.

Gradient and Optimizer Update
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
.. tagged-block:: the-training-cookbook.py train-step
   :hl_lines: 3-6

In order to calculate the gradient, we define the training loss. This is a function of the parameters that returns a scalar which summarizes how well our model, with the current ``train_state`` parameters, is explaining the data.

.. tagged-block:: the-training-cookbook.py train-step 8

By supplying this function to :func:`jax.value_and_grad`, we transform it into a function that returns both the scalar value and the gradient of ``loss_fn`` evaluated at ``params`` (the *value* and *grad*). Since we have defined our parameters in terms of a series of nested dictionaries, the gradient will also be a series of nested dictionaries, mirroring the parameters. Recall that, unlike the parameters, the optimizer states contain some extra, deeper nested dictionaries corresponding to the optimizer state per parameter. Take a moment, before reading the explanation, to ponder what the semantics of the following function call might be:

.. tagged-block:: the-training-cookbook.py train-step 9

Examining the call signature of the function ``adam_apply`` gives us a hint:

.. tagged-block:: the-training-cookbook.py adam-apply

Because ``train_state.params`` is the first argument, :func:`jax.tree.map` uses its tree structure to guide the mapping process.[#prefix_tree]_ This means that ``train_state.opt`` is traversed only as deep as the leaves of ``train_state.params``. The optimizer state for each parameter is therefore passed in as a complete subtree, which allows us to easily access all relevant states (like ``mu`` and ``nu``) for a given ``param`` inside ``adam_apply``.

.. tip::

    If we wished to use different optimization algorithms and states on different parameters in our model (or freeze some parameters), we could achieve this by modifying the body of ``adam_apply`` and replacing :func:`jax.tree.map` with :func:`jax.tree_util.tree_map_with_path`, which allows the operand function to customize its behavior depending on the parameter.

The Training Loop
-----------------
.. tagged-block:: the-training-cookbook.py train-loop
   :hl_lines: 11-13

During training, we have to orchestrate the flow of data between two key players: the host system and the accelerator. Ensuring smooth interplay between these systems is key to writing highly performant training code. The Python `GIL <https://en.wikipedia.org/wiki/Global_interpreter_lock>`_ would ordinarily pose a significant obstacle here, but to work around this, the paradigm of :ref:`Asynchronous Dispatch <async-dispatch>` adopted by JAX makes this orchestration easy to accomplish. But, in order to leverage this paradigm, we need to be mindful of how our code will be executed when structuring our training step.

Efficiency via Asynchronous Dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
One of the most important tasks performed by the host system is to fetch data and place it on the accelerators so that the accelerators are never waiting for data. The time when accelerators are waiting idle between train steps is referred to as the *step bubble*. We can leverage asynchronous dispatch to minimize the step bubble. Let's see how this works with our training loop, discarding, for the moment, the line concerning the ``record_writer``.

.. tagged-block:: the-training-cookbook.py train-loop 5:7

When this code executes, Python will first query the range iterator, get ``step`` (with value ``0``), then call ``next(batch)``, which will take some time to retrieve the batch. Then, ``train_step`` gets called. So far, nothing out of the ordinary.

What happens next is interesting. Because :func:`jax.jit`-decorated calls are non-blocking, the call to ``train_step`` returns to the Python interpreter immediately. While the computation is enqueued on the accelerator, no work is actually performed yet. The Python loop continues, advancing the step counter and calling ``next(batch)`` for the *next* iteration. Once the second call to ``train_step`` is made, its inputs are now the mutated reference to ``train_state`` from the previous JIT call and a fresh batch of data. The runtime is clever and sees that in order to execute the second call to ``train_step``, we first need to realize the ``train_state`` result of step ``0`` to perform the mutation. And so it fires off the computation for the first step, and, crucially, while this happens, ``train_step``, once again, returns immediately, and the loop skips over again. Python now runs ahead until it encounters the ``next(batch)`` function at step 3, which proceeds to execute in Python, loading data, *while* the first train step is executing (for real this time). And just like that, we can simultaneously load data and perform math on the accelerator, without any traditional multiprocessing. [#sleep]_

.. mermaid::

    ---
    displayMode: compact
    ---
    gantt
        title Synchronous Dispatch: No Overlap
        axisFormat %

        section Host
        next(batch) :gb0, 0, 1000s
        next(batch) :gb1, after ajc0, 1000s
        next(batch) :gb2, after ajc1, 1000s

        section Accelerator

        train_step 0 :ajc0, after gb0, 2000s
        train_step 1 :ajc1, after gb1, 2000s


.. mermaid::

    ---
    displayMode: compact
    ---
    gantt
        title JAX Asynchronous Dispatch: Host-Device Overlap
        axisFormat %

        section Host
        %% Task: id, name, start, duration_or_end
        next(batch) :gb0, 0, 1000s
        next(batch) :gb1, after gb0, 1000s
        next(batch) :gb2, after gb1, 1000s
        next(batch) :gb3, after jc0, 1000s
        next(batch) :gb4, after jc1, 1000s

        section Accelerator
        %% Task: id, name, start, duration_or_end
        train_step 0 :jc0, after gb1, 2000s
        train_step 1 :jc1, after jc0, 2000s
        train_step 2 :jc2, after jc1, 2000s

Common Mistakes
~~~~~~~~~~~~~~~
When writing asynchronous dispatch code in Python, there are two primary mistakes one should be wary of so as not to interrupt our careful orchestration of compute.

Requesting device-to-host transfers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Up until now, we have ignored what happens to the variable ``metrics``. Indeed, if this is left dangling, nothing will happen, and we will achieve good overlap just as advertised. However, more often than not, we would like to observe telemetry from our train step, such as the current loss, gradient statistics, and so on. Suppose we were to insert code such as:

.. code-block:: python

    metrics = train_step(config, train_state, next(batch))
    print({"step": step} | metrics)

Instead of the loop ticking over, ``print`` will incur a device-to-host transfer of whatever on-device arrays are in ``metrics``. This interrupts the Python interpreter, and the code is forced to execute synchronously, producing a step bubble. The solution is slightly counterintuitive: at each step, we gather the telemetry for the *previous* step.

.. tagged-block:: the-training-cookbook.py record-writer

and

.. tagged-block:: the-training-cookbook.py train-loop 6:7

A small helper function like this is essential to achieve good overlap and make the most of the resources of our host system and our accelerator. Of course, the simple ``print`` statement here can be swapped out for any Python operation that requests data from the accelerator.

Interrupting the accelerator
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The other common way in which we can waste spectacular amounts of cloud compute money is by unintentionally enqueuing math operations on the accelerator outside of the train step. Suppose we are using a cosine learning rate schedule.

.. code-block:: python

    def learning_rate(count, init_value: float = 1e-4, decay_steps: int = 10_000, alpha: float = 1e-6):
        cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(count, decay_steps) / decay_steps))
        return init_value * (1 - alpha) * cosine_decay

A common pattern is to want to visualize the schedule alongside the other metrics we're gathering. However, even if we use the clever ``record_writer`` class we defined earlier, the following code will create a bubble on the accelerator.

.. code-block:: python

    metrics = train_step(config, train_state, next(batch))
    record_writer({"step": step, "learning_rate": learning_rate(step)} | metrics)


This is because we have used :mod:`jax.numpy` in our calculations. When :func:`jax.numpy.minimum` is called, the Python integer ``step`` is promoted to a :class:`jax.Array` and transferred to the accelerator (a host-to-device transfer). The calculation is now enqueued on the accelerator, outside our main ``train_step``. To ``print`` the result, the value must be transferred back to the host (a device-to-host transfer). This round-trip forces the accelerator to synchronize with the host, and we have thrown away money by creating a performance bubble. The two ways to avoid this are to use NumPy for these calculations or to use the :func:`jax.default_device` context manager.

.. code-block:: python

    metrics = train_step(config, train_state, next(batch))
    with jax.default_device('cpu'):
      record_writer({"step": step, "learning_rate": learning_rate(step)} | metrics)


Data Loading
~~~~~~~~~~~~
In addition to overlapping the actual loading of the data (that is, retrieving it from network storage to the host), JAX also allows us to overlap the host-to-device transfer of the data itself with the computation of the train step. The special function :func:`jax.device_put` is carefully designed to be non-blocking, executing asynchronously, which makes it perfectly fine to use in the context of our train step. However, there is a more convenient function specifically designed for the task of loading data. In the following code, ``dataset`` is an ordinary Python iterator that yields a ``dict`` of batched data. By mapping over this iterator with :func:`jax.make_array_from_process_local_data`, we generate a new iterator. Yielding from this new iterator will generate data placed on the device, ready for consumption by our train step. Internally, it will :func:`jax.tree.map` to create :class:`jax.Array` objects and queue them to be transferred to the device. Provided the data can be batched fast enough, on both TPUs and GPUs, these transfers will be overlapped with the train step computation.

.. tagged-block:: the-training-cookbook.py get-dataset-on-device


.. _achieving-high-performance:

Achieving High Performance
--------------------------

In this section, we will describe the three primary forms of model parallelism that are useful for training. During training, *throughput* is of paramount importance; that is, we wish to maximize the average number of operations per second. This contrasts with inference, where the goal is to minimize *latency* by ensuring all the operations happen in as little time as possible. Keeping throughput in mind as our ultimate goal for training, this section introduces the three primary strategies for sharding during training. For each strategy, we outline the JAX shardings that implement it and describe the collectives involved so that when studying program traces, you'll have landmarks to look for to confirm that the program is behaving as expected. The sharding variables we define in the code blocks below correspond to their uses in the `initialization <#train-state-initialization>`_ and `model forward pass <#model-forward-pass>`_. But in the companion script these and other aspects of the training code are set conveniently using the global `Config` class.

.. tagged-block:: the-training-cookbook.py config


Data Parallel
~~~~~~~~~~~~~
Data parallel is the most common and easy-to-understand form of parallelism. In this scheme, each accelerator stores a complete copy of the model parameters, and we shard activations along the batch axis to split the computation of the gradients. To compute the gradients, each accelerator performs an individual forward and backward pass. Then, before the parameters are updated, XLA inserts an ``AllReduce`` to share the updates and keep the models in sync.

*Mesh:*

.. code-block:: python

    mesh = jax.sharding.Mesh(jax.devices(), ('devices',))

*Parameter Shardings:*

.. code-block:: python

    pos_embed = jax.P(None, None)
    att_qkv = jax.P(None, None, None, None)
    att_out = jax.P(None, None, None)
    mlp_in = jax.P(None, None)
    mlp_out = jax.P(None, None)
    in_kernel = jax.P(None, None)
    in_bias = jax.P(None)
    out_kernel = jax.P(None, None)
    out_bias = jax.P(None)

*Activation Shardings:*

.. code-block:: python

    act_ids = jax.P("devices")
    act_seq = jax.P("devices", None, None)
    act_att = jax.P("devices", None, None, None)
    act_hidden = jax.P("devices", None, None)


.. _fsdp-sharding:

Fully-Sharded Data Parallel (FSDP)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The drawback of data-parallel sharding is that we have to keep multiple, full, redundant copies of the model parameters in HBM. This is a very performant strategy for small models, but since HBM is in short supply, we need to shard the model parameters as well. In the *Fully-Sharded Data Parallel (FSDP)* strategy, we shard both the model and the parameters. Now, as the forward pass happens, the parameters are, one-by-one, unsharded (via ``AllGather``) into whole arrays before they are applied to the activations. This unsharding is brief and temporary, however, leading to a large saving in HBM. In the backward pass, each ``AllGather`` becomes a ``ReduceScatter``. Then there is a final ``ReduceScatter`` at the optimizer update to synchronize gradients. Compared with Data parallelism, the total communication traffic is 50% highter, but we our HBM pressure is reduced by the size of the model divided by the number of devices.

*Mesh:*

.. code-block:: python

    mesh = jax.sharding.Mesh(jax.devices(), ('fsdp',))

*Parameter Shardings:*

.. code-block:: python

    pos_embed = jax.P(None, None)
    att_qkv = jax.P(None, "fsdp", None, None)
    att_out = jax.P("fsdp", None, None)
    mlp_in = jax.P("fsdp", None)
    mlp_out = jax.P(None, "fsdp")
    in_kernel = jax.P(None, None)
    in_bias = jax.P(None)
    out_kernel = jax.P("fsdp", None)
    out_bias = jax.P(None)

*Activation Shardings:*

.. code-block:: python

    act_ids = jax.P("fsdp")
    act_seq = jax.P("fsdp", None, None)
    act_att = jax.P("fsdp", None, None, None)
    act_hidden = jax.P("fsdp", None, None)


.. note::

    While FSDP entails a great deal more communication than data parallel, in practice we are able to overlap the communication with the compute, thereby hiding it and achieving the same throughput at a drastically improved HBM budget.

Tensor Parallel
~~~~~~~~~~~~~~~
If our model is large enough and structured appropriately, it becomes beneficial to partition the computation within a single example across our accelerators. Using a matrix multiplication as an example, we can spread the large matrix multiplications over two or four accelerators. This entails significantly more communication, and so this strategy only works for computations with a very high arithmetic intensity, such as extremely large matrix multiplications. With multi-head self-attention, we opt to shard along the heads with a replicated sequence axis, since this offers the most natural amount of parallelism. If the MLP is large enough we can also efficiently shard the matrix multiplications.

*Mesh:*

.. code-block:: python

    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(128, 4), ("fsdp", "tensor"))

*Parameter Shardings:*

.. code-block:: python

    pos_embed = jax.P(None, "tensor")
    att_qkv = jax.P(None, "fsdp", "tensor", None)
    att_out = jax.P("fsdp", None, None)
    mlp_in = jax.P("fsdp", "tensor")
    mlp_out = jax.P("tensor", "fsdp")
    in_kernel = jax.P(None, None)
    in_bias = jax.P(None)
    out_kernel = jax.P("fsdp", None)
    out_bias = jax.P(None)

*Activation Shardings:*

.. code-block:: python

    act_ids = jax.P("fsdp")
    act_seq = jax.P("fsdp", None, None)
    act_att = jax.P("fsdp", None, "tensor", None)
    act_hidden = jax.P("fsdp", None, "tensor")

.. [#ordered] Of course, all dictionaries are order-preserving in modern Python, so this is somewhat redundant.
.. [#zeros_like] This is accomplished by using the ``zeros_like`` constructor, but we could have specified the sharding manually using the ``devices`` argument of many of the :mod:`jax.numpy` functions.
.. [#prefix_tree] We could have achieved the same behavior equivalently by ordering ``grad`` first.
.. [#sleep] For the purposes of this explanation, you can think of ``next(batch)`` as just a sleep.
