Transfer guard
==============

JAX may transfer data between the host and devices and between devices during
type conversion and input sharding. To log or disallow any unintended
transfers, the user may configure a JAX transfer guard.

JAX transfer guards distinguish between two types of transfers:

* Explicit transfers: ``jax.device_put*()`` and ``jax.device_get()`` calls.
* Implicit transfers: Other transfers (e.g., printing a ``DeviceArray``).

A transfer guard can take an action based on its guard level:

* ``"allow"``: Silently allow all transfers (default).
* ``"log"``: Log and allow implicit transfers. Silently allow explicit
  transfers.
* ``"disallow"``: Disallow implicit transfers. Silently allow explicit
  transfers.
* ``"log_explicit"``: Log and allow all transfers.
* ``"disallow_explicit"``: Disallow all transfers.

JAX will raise a ``RuntimeError`` when disallowing a transfer.

The transfer guards use the standard JAX configuration system:

* A ``--jax_transfer_guard=GUARD_LEVEL`` command-line flag and
  ``jax.config.update("jax_transfer_guard", GUARD_LEVEL)`` will set the global
  option.
* A ``with jax.transfer_guard(GUARD_LEVEL): ...`` context manager will set the
  thread-local option within the scope of the context manager.

Note that similar to other JAX configuration options, a newly spawned thread
will use the global option instead of any active thread-local option of the
scope where the thread was spawned.

The transfer guards can also be applied more selectively, based on the
direction of transfer. The flag and context manager name is suffixed with a
corresponding transfer direction (e.g., ``--jax_transfer_guard_host_to_device``
and ``jax.config.transfer_guard_host_to_device``):

* ``"host_to_device"``: Converting a Python value or NumPy array into a JAX
  on-device buffer.
* ``"device_to_device"``: Copying a JAX on-device buffer to a different device.
* ``"device_to_host"``: Fetching a JAX on-device buffer.

Fetching a buffer on a CPU device is always allowed regardless of the transfer
guard level.

The following shows an example of using the transfer guard.

.. code-block:: python

   >>> jax.config.update("jax_transfer_guard", "allow")  # This is default.
   >>>
   >>> x = jnp.array(1)
   >>> y = jnp.array(2)
   >>> z = jnp.array(3)
   >>>
   >>> print("x", x)  # All transfers are allowed.
   x 1
   >>> with jax.transfer_guard("disallow"):
   ...   print("x", x)  # x has already been fetched into the host.
   ...   print("y", jax.device_get(y))  # Explicit transfers are allowed.
   ...   try:
   ...     print("z", z)  # Implicit transfers are disallowed.
   ...     assert False, "This line is expected to be unreachable."
   ...   except:
   ...     print("z could not be fetched")  # doctest: +SKIP
   x 1
   y 2
   z could not be fetched
