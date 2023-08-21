JAX Infrequently Asked Questions (IAQ)
======================================

.. _JAX - Frequently Asked Questions (FAQ): https://jax.readthedocs.io/en/latest/faq.html
.. _JAX - The Sharp Bits: https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html

This document represents a collection of *infrequently asked questions (IAQ)*. Contributions welcome!

How to apply `jax.device_put` on an `jax.Array`` that has different values per host into a 'data' mesh axis?
------------------------------------------------------------------------------------------------------------

Here's an example from :meth:`jax.Array` (Line 647 <https://github.com/google/jax/blob/main/jax/_src/array.py#647>`_):

>>> def make_array_from_callback(
...     shape: Shape, sharding: Sharding,
...         data_callback: Callable[[Index | None], ArrayLike]) -> ArrayImpl:
...       """Returns a ``jax.Array`` via data fetched from ``data_callback``.
...       ...
...       """
...       device_to_index_map = sharding.devices_indices_map(shape)
...       ...
...       arrays = [
...           api.device_put(data_callback(device_to_index_map[device]), device)
...           for device in sharding.addressable_devices
...       ]
...       aval = core.ShapedArray(shape, arrays[0].dtype, weak_type=False)
...       if dtypes.issubdtype(aval.dtype, dtypes.extended):
...         return aval.dtype._rules.make_sharded_array(aval, sharding, arrays, committed=True)
...       return ArrayImpl(aval, sharding, arrays, committed=True)

References:

- :meth:`jax.device_put` API docs
- :meth: `jax.`
- Tutorial: `Distributed arrays and automatic parallelization <https://jax.readthedocs.io/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html>`_