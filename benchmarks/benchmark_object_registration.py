import time
from dataclasses import dataclass, field, make_dataclass
import jax
import jax.numpy as jnp
from jax import tree_util
import numpy as np
from typing import Any
from jax._src import lib as jaxlib

@jax.tree_util.register_static
class StaticDict(dict):
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

class ObjectRegisteredInt(ObjectRegistered):
  def __init__(self, data_children=[], static_children=[]):
    for i, child in enumerate(data_children):
      setattr(self, f'{i+1}', child)
    for i, child in enumerate(static_children):
      setattr(self, f'static{i+1}', child)
    mapping = {f'{i+1}': True for i in range(len(data_children))}
    self.__mapping__ = jaxlib.pytree.StringSet(mapping)

jax.tree_util.register_object(ObjectRegisteredInt, '__mapping__', True)

def key_fn(k):
  if k.isdigit():
    return (0, int(k))
  return (1, k)

class PyObjectRegistered(ObjectRegistered):

  @classmethod
  def tree_unflatten(cls, meta, data):
    (meta_fields, meta_data, data_fields) = meta
    module = object.__new__(cls)
    for name, value in zip(meta_fields, meta_data):
      object.__setattr__(module, name, value)
    for name, value in zip(data_fields, data):
      object.__setattr__(module, name, value)
    return module

  def tree_flatten(self):
    m = self.__mapping__
    data = []
    data_fields = []
    meta_data = []
    meta_fields = []
    sorted_keys = sorted(self.__dict__.keys(), key=key_fn)
    for k in sorted_keys:
      if k in m and m[k]:
        data.append(getattr(self, k))
        data_fields.append(k)
      else:
        meta_data.append(getattr(self, k))
        meta_fields.append(k)
    return data, (meta_fields, meta_data, data_fields)

tree_util.register_pytree_node(
    PyObjectRegistered,
    PyObjectRegistered.tree_flatten,
    PyObjectRegistered.tree_unflatten
)

def make_dataclass_registered(num_data, num_static):
  lines = ['@jax.tree_util.register_dataclass', '@dataclass', 'class DataclassRegistered:']
  for i in range(num_data):
    lines.append(f'  data{i+1}: Any')
  for i in range(num_static):
    lines.append(f'  static{i+1}: Any = field(metadata=dict(static=True))')
  code = '\n'.join(lines)
  namespace = {'jax': jax, 'dataclass': dataclass, 'Any': Any, 'field': field}
  exec(code, namespace)
  return namespace['DataclassRegistered']

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

def benchmark_roundtrip(obj, iterations=100):
  start = time.time()
  for _ in range(iterations):
    xs, tree = tree_util.tree_flatten(obj)
    result = tree_util.tree_unflatten(tree, xs)
  end = time.time()
  assert result == obj, "Roundtrip failed"
  return end - start

def ci(a):
  return f"{np.mean(a):.4f}s ± {2 * np.std(a):.4f}s"



if __name__ == '__main__':
  width = 10
  depth = 5
  num_data = width // 2
  num_static = width - num_data

  obj_registered = make_tree(ObjectRegistered, depth, num_data, num_static)
  obj_registered_int = make_tree(ObjectRegisteredInt, depth, num_data, num_static)
  DataclassRegistered = make_dataclass_registered(num_data, num_static)
  dataclass_registered = make_dataclass_tree(DataclassRegistered, depth, num_data, num_static)
  pyobj_registered = make_tree(PyObjectRegistered, depth, num_data, num_static)
  dict_registered = make_tree(make_dict, depth, num_data, num_static)
  vals, _ = jax.tree.flatten(dataclass_registered)

  print("Object time ", ci([benchmark_roundtrip(obj_registered) for _ in range(3)]))
  print("Int object time ", ci([benchmark_roundtrip(obj_registered_int) for _ in range(3)]))
  print("Dict time ", ci([benchmark_roundtrip(dict_registered) for _ in range(3)]))
  print("Pre-flattened time ", ci([benchmark_roundtrip(vals) for _ in range(3)]))
  print("Dataclass time ", ci([benchmark_roundtrip(dataclass_registered) for _ in range(3)]))
  print("Pyobject time ", ci([benchmark_roundtrip(pyobj_registered) for _ in range(3)]))
