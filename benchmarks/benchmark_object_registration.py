import time
from dataclasses import dataclass
import jax
import jax.numpy as jnp
from jax import tree_util

class ObjectRegistered:
  def __init__(self, x, y):
    self.x = x
    self.y = y
    self.__mapping__ = {'x': True, 'y': False, '__mapping__': False}

  def __eq__(self, other):
    return self.y == other.y and jnp.array_equal(self.x, other.x)

jax.tree_util.register_object(ObjectRegistered, '__mapping__')

class PyTreeNodeRegistered:
  def __init__(self, x, y):
    self.x = x
    self.y = y

  def __eq__(self, other):
    return self.y == other.y and jnp.array_equal(self.x, other.x)

  def tree_flatten(self):
    return ([self.x], self.y)

  @classmethod
  def tree_unflatten(cls, y, xs):
    return cls(xs[0], y)

tree_util.register_pytree_node(
    PyTreeNodeRegistered,
    PyTreeNodeRegistered.tree_flatten,
    PyTreeNodeRegistered.tree_unflatten
)

@tree_util.register_dataclass
@dataclass
class DataclassRegistered:
  x: jax.Array
  y: str

  def __eq__(self, other):
    return self.y == other.y and jnp.array_equal(self.x, other.x)

def benchmark_roundtrip(obj, iterations=100000):
  start = time.time()
  for _ in range(iterations):
    xs, tree = tree_util.tree_flatten(obj)
    tree_util.tree_unflatten(tree, xs)
  end = time.time()
  return end - start

if __name__ == '__main__':
  obj_registered = ObjectRegistered(jnp.arange(5), "hello")
  pytree_registered = PyTreeNodeRegistered(jnp.arange(5), "hello")
  dataclass_registered = DataclassRegistered(jnp.arange(5), "hello")

  print("Benchmarking flatten...")
  time_obj = benchmark_roundtrip(obj_registered)
  time_pytree = benchmark_roundtrip(pytree_registered)
  time_dataclass = benchmark_roundtrip(dataclass_registered)

  print(f"register_object: {time_obj:.4f}s")
  print(f"register_pytree_node: {time_pytree:.4f}s")
  print(f"register_dataclass: {time_dataclass:.4f}s")
