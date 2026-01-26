import time
from dataclasses import dataclass, field, make_dataclass
import jax
import jax.numpy as jnp
from jax import tree_util
import numpy as np
from typing import Any

from jax._src import lib as jaxlib

# TODO: If meta_data and meta_keys were python arrays instead, would there be a penalty?

@jax.tree_util.register_static
class StaticDict(dict):
  pass

class DictWrap(dict):
  pass

class ObjectRegistered:
  def __init__(self, data_children=[], static_children=[]):
    for i, child in enumerate(data_children):
      setattr(self, f'data{i+1}', child)
    for i, child in enumerate(static_children):
      setattr(self, f'static{i+1}', child)
    mapping = {f'data{i+1}': True for i in range(len(data_children))}
    self.__mapping__ = jaxlib.pytree.StringSet(mapping)

  def __eq__(self, other):
    return vars(self) == vars(other)

jax.tree_util.register_object(ObjectRegistered, '__mapping__')

class PyObjectRegistered(ObjectRegistered):

  def tree_flatten(self):
    statics = {}
    data = []
    data_keys = []
    for (k,v) in vars(self).items():
      if k in self.__mapping__:
        data.append(v)
        data_keys.append(k)
      else:
        statics[k] = v
    return (data, (data_keys, statics))

  @classmethod
  def tree_unflatten(cls, saved, xs):
    data_keys, statics = saved
    obj = object.__new__(cls)
    for k,v in statics.items():
      object.__setattr__(obj, k, v)
    for k,v in zip(data_keys, xs):
      object.__setattr__(obj, k, v)
    return obj

tree_util.register_pytree_node(
    PyObjectRegistered,
    PyObjectRegistered.tree_flatten,
    PyObjectRegistered.tree_unflatten
)

class PyTreeNodeRegistered:
  def __init__(self, data_children=[], static_children=[]):
    self._num_data = len(data_children)
    self._num_static = len(static_children)
    for i, child in enumerate(data_children):
      setattr(self, f'data{i+1}', child)
    for i, child in enumerate(static_children):
      setattr(self, f'static{i+1}', child)

  def __eq__(self, other):
    return vars(self) == vars(other)

  def tree_flatten(self):
    data_list = [getattr(self, f'data{i+1}') for i in range(self._num_data)]
    static_list = [getattr(self, f'static{i+1}') for i in range(self._num_static)]
    return (data_list, static_list)

  @classmethod
  def tree_unflatten(cls, aux, data_children):
    static_children = aux
    return cls(data_children=data_children, static_children=static_children)

tree_util.register_pytree_node(
    PyTreeNodeRegistered,
    PyTreeNodeRegistered.tree_flatten,
    PyTreeNodeRegistered.tree_unflatten
)

def make_dataclass_registered(num_data, num_static):
  data_fields = [f'data{i+1}' for i in range(num_data)]
  static_fields = [(f'static{i+1}', Any, field(metadata=dict(static=True))) for i in range(num_static)]
  DataclassRegistered = make_dataclass('DataclassRegistered', data_fields + static_fields)
  tree_util.register_dataclass(DataclassRegistered)
  return DataclassRegistered

def make_tree(cls, depth, num_data, num_static):
  if depth == 0:
    return cls(data_children=[], static_children=[])
  data_children = [make_tree(cls, depth - 1, num_data, num_static) for _ in range(num_data)]
  static_children = ["hello world" for _ in range(num_static)]
  return cls(data_children=data_children, static_children=static_children)

def make_dict(data_children, static_children):
  return ({f'data{i+1}': d for i, d in enumerate(data_children)},
    StaticDict({f'static{i+1}': d for i,d in enumerate(static_children)}))

def make_dataclass_tree(cls, depth, num_data, num_static):
  if depth == 0:
    kwargs = {}
    for i in range(num_data):
      kwargs[f'data{i+1}'] = None
    for i in range(num_static):
      kwargs[f'static{i+1}'] = None
    return cls(**kwargs)
  data_children = [make_dataclass_tree(cls, depth - 1, num_data, num_static) for _ in range(num_data)]
  static_children = ["hello world" for _ in range(num_static)]
  kwargs = {}
  for i, child in enumerate(data_children):
    kwargs[f'data{i+1}'] = child
  for i, child in enumerate(static_children):
    kwargs[f'static{i+1}'] = child
  return cls(**kwargs)

def benchmark_roundtrip(obj, iterations=1000):
  start = time.time()
  for _ in range(iterations):
    xs, tree = tree_util.tree_flatten(obj)
    tree_util.tree_unflatten(tree, xs)
  end = time.time()
  return end - start

def ci(a):
  return f"{np.mean(a):.4f}s ± {2 * np.std(a):.4f}s"

# For reasons I don't understand, Jax's register_dataclass is substantially faster
# on this manually defined class compared to one created by make_dataclass_registered
@jax.tree_util.register_dataclass
@dataclass
class DataclassRegistered:
  data1: Any
  data2: Any
  data3: Any
  data4: Any
  data5: Any
  static1: Any = field(metadata=dict(static=True))
  static2: Any = field(metadata=dict(static=True))
  static3: Any = field(metadata=dict(static=True))
  static4: Any = field(metadata=dict(static=True))
  static5: Any = field(metadata=dict(static=True))

if __name__ == '__main__':
  width = 10
  depth = 5
  num_data = width // 2
  num_static = width - num_data

  obj_registered = make_tree(ObjectRegistered, depth, num_data, num_static)
  pytree_registered = make_tree(PyTreeNodeRegistered, depth, num_data, num_static)
  # DataclassRegistered = make_dataclass_registered(num_data, num_static)
  dataclass_registered = make_dataclass_tree(DataclassRegistered, depth, num_data, num_static)
  pyobj_registered = make_tree(PyObjectRegistered, depth, num_data, num_static)
  dict_registered = make_tree(make_dict, depth, num_data, num_static)
  vals, _ = jax.tree.flatten(dataclass_registered)

  print("Benchmarking object...")
  time_obj = [benchmark_roundtrip(obj_registered) for _ in range(3)]
  print("Benchmarking pytree")
  time_pytree = [benchmark_roundtrip(pytree_registered) for _ in range(3)]
  print("Benchmarking dict")
  time_dict = [benchmark_roundtrip(dict_registered) for _ in range(3)]
  print("Benchmarking pre-flattened")
  time_flat = [benchmark_roundtrip(vals) for _ in range(3)]
  print("Benchmarking dataclass")
  time_dataclass = [benchmark_roundtrip(dataclass_registered) for _ in range(3)]
  print("Benchmarking pyobject")
  time_pyobject = [benchmark_roundtrip(pyobj_registered) for _ in range(3)]

  print(f"pre flattened: {ci(time_flat)}")
  print(f"register_object: {ci(time_obj)}")
  print(f"register_pytree_node: {ci(time_pytree)}")
  print(f"register_dict: {ci(time_dict)}")
  print(f"register_dataclass: {ci(time_dataclass)}")
  print(f"register_pyobject: {ci(time_pyobject)}")

  print(f"C++ Speedup : {np.mean(time_obj) / np.mean(time_pyobject):.4f}")
