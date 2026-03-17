# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from absl.testing import absltest
import jax
from jax._src import test_util as jtu
from jax._src.core import _pyobj_registry
import numpy as np


class Tree:
  def __init__(self, children=()):
    self.children = children

  def add_child(self, child):
    return Tree(self.children + (child,))

  def __eq__(self, other):
    return isinstance(other, Tree) and self.children == other.children

  def __repr__(self):
    return f"Tree({self.children!r})"


class PyObjCallbackTest(jtu.JaxTestCase):

  def setUp(self):
    super().setUp()
    if not jtu.test_device_matches(["cpu", "gpu"]):
      self.skipTest("PyObject callbacks only supported on CPU/GPU")

  def test_pure_callback_returns_pyobj(self):
    @jax.jit
    def f():
      return jax.pure_callback(lambda: Tree(), jax.PyObjectType)

    result = f()
    ptr = int(np.asarray(result))
    obj = _pyobj_registry[ptr]
    self.assertIsInstance(obj, Tree)
    self.assertEqual(obj.children, ())

  def test_pure_callback_pyobj_passthrough(self):
    @jax.jit
    def f():
      root = jax.pure_callback(lambda: Tree(), jax.PyObjectType)
      child = jax.pure_callback(lambda: Tree(), jax.PyObjectType)
      return jax.pure_callback(
          lambda r, c: r.add_child(c), jax.PyObjectType, root, child)

    result = f()
    ptr = int(np.asarray(result))
    obj = _pyobj_registry[ptr]
    self.assertIsInstance(obj, Tree)
    self.assertEqual(len(obj.children), 1)
    self.assertIsInstance(obj.children[0], Tree)

  def test_py_traced_argnums_basic(self):
    @jax.jit(py_traced_argnums=(0,))
    def f(tree):
      return jax.pure_callback(
          lambda t: t.add_child(Tree()), jax.PyObjectType, tree)

    root = Tree()
    result = f(root)
    ptr = int(np.asarray(result))
    obj = _pyobj_registry[ptr]
    self.assertIsInstance(obj, Tree)
    self.assertEqual(len(obj.children), 1)

  def test_py_traced_argnums_full_example(self):
    @jax.jit(py_traced_argnums=(0,))
    def foo(bar):
      baz = jax.pure_callback(lambda: Tree(), jax.PyObjectType)
      return jax.pure_callback(
          lambda r, z: r.add_child(z), jax.PyObjectType, bar, baz)

    bar = Tree()
    result = foo(bar)
    ptr = int(np.asarray(result))
    obj = _pyobj_registry[ptr]
    self.assertIsInstance(obj, Tree)
    self.assertEqual(len(obj.children), 1)
    self.assertIsInstance(obj.children[0], Tree)

  def test_py_traced_argnums_no_recompile(self):
    compile_count = [0]

    @jax.jit(py_traced_argnums=(0,))
    def f(tree):
      compile_count[0] += 1
      return jax.pure_callback(
          lambda t: t.add_child(Tree()), jax.PyObjectType, tree)

    f(Tree())
    f(Tree((Tree(),)))
    self.assertEqual(compile_count[0], 1,
                     "Function should compile once for all Tree instances")

  def test_pure_callback_mixed_results(self):
    import jax.numpy as jnp

    @jax.jit
    def f(x):
      arr_result = jnp.sum(x)
      obj_result = jax.pure_callback(lambda: Tree(), jax.PyObjectType)
      return arr_result, obj_result

    arr, obj_handle = f(jnp.array([1.0, 2.0, 3.0]))
    self.assertAlmostEqual(float(arr), 6.0)
    ptr = int(np.asarray(obj_handle))
    obj = _pyobj_registry[ptr]
    self.assertIsInstance(obj, Tree)

  def test_pyobjecttype_is_sentinel(self):
    self.assertIs(jax.PyObjectType, jax.PyObjectType)

  def test_pyobj_registry_keeps_objects_alive(self):
    import gc
    results = []

    @jax.jit
    def f():
      return jax.pure_callback(lambda: Tree((Tree(),)), jax.PyObjectType)

    for _ in range(3):
      r = f()
      results.append(r)
      gc.collect()

    for r in results:
      ptr = int(np.asarray(r))
      obj = _pyobj_registry[ptr]
      self.assertIsInstance(obj, Tree)
      self.assertEqual(len(obj.children), 1)


if __name__ == "__main__":
  absltest.main(testLoader=jtu.JaxTestLoader())
