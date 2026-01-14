import time
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from jax import tree_util

class ObjectRegistered:
  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z
    self.__mapping__ = {'x': True, 'y': False, 'z': False, '__mapping__': False}

  def __eq__(self, other):
    return self.y == other.y and self.z == other.z and jnp.array_equal(self.x, other.x)

jax.tree_util.register_object(ObjectRegistered, '__mapping__')

class PyObjectRegistered:
  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z
    self.__mapping__ = {'x': True, 'y': False, 'z': False, '__mapping__': False}

  def __eq__(self, other):
    return self.y == other.y and self.z == other.z and jnp.array_equal(self.x, other.x)

  def tree_flatten(self):
    statics = []
    data = []
    for (k,v) in self.__mapping__.items():
      (data if v else statics).append(getattr(self, k))
    return (data, (statics, self.__mapping__))

  @classmethod
  def tree_unflatten(cls, stored, xs):
    statics, mapping = stored
    obj = object.__new__(cls)
    static_iter = iter(statics)
    data_iter = iter(xs)
    for (k,v) in mapping.items():
      object.__setattr__(obj, k, next(data_iter if v else static_iter))
    return obj

tree_util.register_pytree_node(
    PyObjectRegistered,
    PyObjectRegistered.tree_flatten,
    PyObjectRegistered.tree_unflatten
)

class PyTreeNodeRegistered:
  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z

  def __eq__(self, other):
    return self.y == other.y and self.z == other.z and jnp.array_equal(self.x, other.x)

  def tree_flatten(self):
    return ([self.x], [self.y, self.z])

  @classmethod
  def tree_unflatten(cls, statics, xs):
    return cls(*xs, *statics)

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
  z: str = field(metadata=dict(static=True))

  def __eq__(self, other):
    return self.y == other.y and self.z == other.z and jnp.array_equal(self.x, other.x)

def benchmark_roundtrip(obj, iterations=100000):
  start = time.time()
  for _ in range(iterations):
    xs, tree = tree_util.tree_flatten(obj)
    tree_util.tree_unflatten(tree, xs)
  end = time.time()
  return end - start

if __name__ == '__main__':
  obj_registered = ObjectRegistered(jnp.arange(5), "hello", "world")
  pytree_registered = PyTreeNodeRegistered(jnp.arange(5), "hello", "world")
  dataclass_registered = DataclassRegistered(jnp.arange(5), "hello", "world")
  pyobj_registered = PyObjectRegistered(jnp.arange(5), "hello", "world")

  print("Benchmarking flatten...")
  time_obj = benchmark_roundtrip(obj_registered)
  time_pytree = benchmark_roundtrip(pytree_registered)
  time_dataclass = benchmark_roundtrip(dataclass_registered)
  time_pyobject = benchmark_roundtrip(pyobj_registered)

  print(f"register_object: {time_obj:.4f}s")
  print(f"register_pytree_node: {time_pytree:.4f}s")
  print(f"register_dataclass: {time_dataclass:.4f}s")
  print(f"register_pyobject: {time_pyobject:.4f}s")
