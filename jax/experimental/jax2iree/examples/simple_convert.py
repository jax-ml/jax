import numpy as np
import timeit

from jax import core
from jax import numpy as jnp
from jax.experimental import jax2iree


def f(x, y):
  return jnp.add(x, y)


def fabs(x, y):
  z = jnp.add(x, y)
  return (jnp.abs(z),)


def trace_and_compile():
  builder = jax2iree.Builder()
  out_avals = jax2iree.trace_flat_function(
      # Trace function.
      fabs,
      builder=builder,
      in_avals=[
          core.ShapedArray([1, 4], jnp.float32),
          core.ShapedArray([4, 1], jnp.float32)
      ])
  binary = builder.compile_module_to_binary()
  return binary


def load_and_run(binary):
  compiled = jax2iree.Builder.load_compiled_binary(binary)

  # TODO: If we emitted reflection metadata, we would get better automatic
  # type conversion. Also, since IREE can natively represent pytrees, we
  # could/should map those as-is (which would be better, especially for
  # offline users who want to invoke the function later and would like a
  # structured API in their language vs a flat list of args). But that
  # requires a bit more work to leverage.
  # TODO: Without metadata, we aren't recognizing
  # jaxlib.xla_extension.DeviceArray as an array.
  return compiled.fabs(np.asarray([[0.5, -0.5, 3.5, 2.0]]),
                       np.asarray([[-1.0], [0.0], [1.0], [2.0]]))


# TODO: The IREE VM mutates the binary. To date this hasn't been a problem
# because we load or anonymously map it from a file. For pure in-memory use,
# we either need to be smarter or COW map/something. Here we just make
# defensive copies.
original_binary = trace_and_compile()
print(load_and_run(bytes(original_binary)))

total_time = timeit.timeit(trace_and_compile, number=20)
print(f"Compile time/iteration = {total_time / 20}s")

total_time = timeit.timeit(lambda: load_and_run(bytes(original_binary)),
                           number=20)
print(f"Run time/iteration = {total_time / 20}s")
