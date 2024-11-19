r"""Sample code to show how to serialize and deserialize a JAX executable.

To invoke:
  $ bazel run -c opt examples/cpp_exec/serialized_fn:serialize -- /tmp/serialized_fn
"""

import jax
from jax.experimental import serialize_executable
import jax.numpy as jnp
import sys


@jax.jit
def my_function(x, y):
  return x + y


def main(argv):
  lowered = jax.jit(my_function).lower(1.0, 2.0)

  compiled = lowered.compile()

  serialized = serialize_executable.serialize_raw(compiled)

  if len(argv) > 1:
    fn = argv[1]
    with open(fn, 'wb') as f:
      f.write(serialized)

  deserialized = serialize_executable.deserialize_raw(serialized)

  result = deserialized.execute((jnp.array(8.0), jnp.array(2.0)))
  print('The result is:', result)  # Output: 10.


if __name__ == '__main__':
  main(sys.argv)
