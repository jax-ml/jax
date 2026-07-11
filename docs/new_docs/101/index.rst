:orphan:
:nosearch:

JAX 101
=======

JAX is a Python library for high-performance numerical computing and machine
learning. Its interface, centered on :mod:`jax.numpy`, will look familiar if
you've used NumPy. What sets JAX apart is what it can do with the functions
you write: transform them, to compute gradients or to vectorize over batches;
and compile them, to run fast on CPU, GPU, and TPU, at any scale.

These pages cover the first half of that story: how to *express* computations
in JAX. They're meant to be read in order:

1. :doc:`arrays` — JAX's array type and the :mod:`jax.numpy` API: what's the
   same as NumPy, what's different, and why.
2. :doc:`transformations` — computing gradients with :func:`jax.grad` and
   vectorizing with :func:`jax.vmap`, plus the tracing model that underlies
   every JAX transformation.
3. :doc:`pytrees` — how JAX handles structured data, like nested dictionaries
   and lists of arrays.
4. :doc:`random` — pseudorandom numbers with explicit PRNG keys: pure
   functions of key values, with no hidden generator state.
5. :doc:`state` — stateful computations: threading state through pure
   functions, and in-place mutation with refs, JAX's mutable array type.
6. :doc:`errors` — common JAX errors, explained: most arise from expressing
   something in a way that's incompatible with tracing, so the tracing model
   from :doc:`transformations` is the key to fixing them.

The second half of the story is making these computations fast: compilation
with :func:`jax.jit`, sharded arrays and parallelism, and profiling. Those are
the subject of the performance and scaling docs: :doc:`/new_docs/201/index`.

.. toctree::
   :hidden:
   :maxdepth: 1

   arrays
   transformations
   pytrees
   random
   state
   errors
