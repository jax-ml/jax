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

def benchmark_flatten(obj, iterations=100000):
  start = time.time()
  for _ in range(iterations):
    tree_util.tree_flatten(obj)
  end = time.time()
  return end - start

def benchmark_unflatten(tree, xs, iterations=100000):
  start = time.time()
  for _ in range(iterations):
    tree_util.tree_unflatten(tree, xs)
  end = time.time()
  return end - start

if __name__ == '__main__':
  obj_registered = ObjectRegistered(jnp.arange(5), "hello")
  pytree_registered = PyTreeNodeRegistered(jnp.arange(5), "hello")
  dataclass_registered = DataclassRegistered(jnp.arange(5), "hello")

  xs_obj, tree_obj = tree_util.tree_flatten(obj_registered)
  xs_pytree, tree_pytree = tree_util.tree_flatten(pytree_registered)
  xs_dataclass, tree_dataclass = tree_util.tree_flatten(dataclass_registered)

  print("Benchmarking flatten...")
  time_obj_flatten = benchmark_flatten(obj_registered)
  time_pytree_flatten = benchmark_flatten(pytree_registered)
  time_dataclass_flatten = benchmark_flatten(dataclass_registered)

  print(f"register_object flatten: {time_obj_flatten:.4f}s")
  print(f"register_pytree_node flatten: {time_pytree_flatten:.4f}s")
  print(f"register_dataclass flatten: {time_dataclass_flatten:.4f}s")
  print(f"Speedup over pytree_node: {time_pytree_flatten / time_obj_flatten:.2f}x")
  print(f"Speedup over dataclass: {time_dataclass_flatten / time_obj_flatten:.2f}x")
  print()

  print("Benchmarking unflatten...")
  time_obj_unflatten = benchmark_unflatten(tree_obj, xs_obj)
  time_pytree_unflatten = benchmark_unflatten(tree_pytree, xs_pytree)
  time_dataclass_unflatten = benchmark_unflatten(tree_dataclass, xs_dataclass)

  print(f"register_object unflatten: {time_obj_unflatten:.4f}s")
  print(f"register_pytree_node unflatten: {time_pytree_unflatten:.4f}s")
  print(f"register_dataclass unflatten: {time_dataclass_unflatten:.4f}s")
  print(f"Speedup over pytree_node: {time_pytree_unflatten / time_obj_unflatten:.2f}x")
  print(f"Speedup over dataclass: {time_dataclass_unflatten / time_obj_unflatten:.2f}x")
