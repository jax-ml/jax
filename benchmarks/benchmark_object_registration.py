import time
from dataclasses import dataclass, field
import jax
import jax.numpy as jnp
from jax import tree_util
import numpy as np

from jax._src import lib as jaxlib

jaxlib.pytree.StringSet({'hello': True, 'world': True})

class ObjectRegistered:
  def __init__(self, x, y, z):
    self.x = x
    self.y = y
    self.z = z
    self.__mapping__ = jaxlib.pytree.StringSet({'x': True})

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
    for (k,v) in vars(self).items():
      if k in self.__mapping__ and self.__mapping__[k]:
        data.append(v)
      else:
        self.__mapping__[k] = False
        statics.append(v)
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

def ci(a):
  return f"{np.mean(a):.4f}s ± {2 * np.std(a):.4f}s"

if __name__ == '__main__':
  obj_registered = ObjectRegistered(jnp.arange(5), "hello", "world")
  pytree_registered = PyTreeNodeRegistered(jnp.arange(5), "hello", "world")
  dataclass_registered = DataclassRegistered(jnp.arange(5), "hello", "world")
  pyobj_registered = PyObjectRegistered(jnp.arange(5), "hello", "world")

  print("Benchmarking flatten...")
  time_obj = [benchmark_roundtrip(obj_registered) for _ in range(10)]
  time_pytree = [benchmark_roundtrip(pytree_registered) for _ in range(10)]
  time_dataclass = [benchmark_roundtrip(dataclass_registered) for _ in range(10)]
  time_pyobject = [benchmark_roundtrip(pyobj_registered) for _ in range(10)]

  print(f"register_object: {ci(time_obj)}")
  print(f"register_pytree_node: {ci(time_pytree)}")
  print(f"register_dataclass: {ci(time_dataclass)}")
  print(f"register_pyobject: {ci(time_pyobject)}")

  print(f"Improvement : {np.mean(time_obj) / np.mean(time_pyobject):.4f}")
