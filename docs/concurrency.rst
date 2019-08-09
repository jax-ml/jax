Concurrency
===========

JAX has some limited support for Python concurrency.

Concurrency support is experimental and only lightly tested; please report any
bugs.

Clients may call JAX APIs (e.g., :fun:`jax.jit` or :fun:`jax.grad`) concurrently
from separate Python threads.

It is not permitted to manipulate JAX trace values concurrently from multiple
threads. In other words, while it is permissible to call functions that use JAX
tracing (e.g., :fun:`jax.jit`) from multiple threads, you must not use threading
to manipulate JAX values inside the implementation of the function `f` that is
passed to :fun:`jax.jit`. The most likely outcome if you do this is a mysterious
error from JAX.
