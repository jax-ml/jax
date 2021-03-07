# Copyright 2020 Google LLC
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

from absl.testing import absltest, parameterized

import numpy as np

from jax import test_util as jtu
import jax.numpy as jnp
from jax import core, jit, lax, lazy, make_jaxpr
from jax.interpreters import xla
from jax.lib import xla_client
xops = xla_client.ops

from jax.config import config
config.parse_flags_with_absl()

# TODO(jakevdp): use a setup/teardown method to populate and unpopulate all the
# dictionaries associated with the following objects.

# Define a sparse array data structure. The important feature here is that
# it is a jaxpr object that is backed by two device buffers.
class SparseArray:
  """Simple sparse COO array data structure."""
  def __init__(self, aval, data, indices):
    self.aval = aval
    self.shape = aval.shape
    self.data = data
    self.indices = indices

  @property
  def index_dtype(self):
    return self.indices.dtype

  @property
  def dtype(self):
    return self.data.dtype

  @property
  def nnz(self):
    return self.data.shape[0]

  def __repr__(self):
    return repr(list((tuple(ind), d) for ind, d in zip(self.indices, self.data)))


class AbstractSparseArray(core.ShapedArray):
  __slots__ = ['index_dtype', 'nnz', 'data_aval', 'indices_aval']
  _num_buffers = 2

  def __init__(self, shape, dtype, index_dtype, nnz, weak_type=False,
               named_shape={}):
    super(AbstractSparseArray, self).__init__(shape, dtype)
    self.index_dtype = index_dtype
    self.nnz = nnz
    self.data_aval = core.ShapedArray((nnz,), dtype, weak_type, named_shape)
    self.indices_aval = core.ShapedArray((nnz, len(shape)), index_dtype,
                                         named_shape)

  def update(self, shape=None, dtype=None, index_dtype=None, nnz=None,
             weak_type=None, named_shape=None):
    if shape is None:
      shape = self.shape
    if dtype is None:
      dtype = self.dtype
    if index_dtype is None:
      index_dtype = self.dtype
    if nnz is None:
      nnz = self.nnz
    if weak_type is None:
      weak_type = self.weak_type
    if named_shape is None:
      named_shape = self.named_shape
    return AbstractSparseArray(
        shape, dtype, index_dtype, nnz, weak_type, named_shape)

  def strip_weak_type(self):
    return self

  @core.aval_property
  def data(self):
    return sp_data_p.bind(self)

  @core.aval_property
  def indices(self):
    return sp_indices_p.bind(self)

class ConcreteSparseArray(AbstractSparseArray):
  pass

def sparse_array_result_handler(device, aval):
  def build_sparse_array(data_buf, indices_buf):
    data = xla.make_device_array(aval.data_aval, device, lazy.array(aval.data_aval.shape), data_buf)
    indices = xla.make_device_array(aval.indices_aval, device, lazy.array(aval.indices_aval.shape), indices_buf)
    return SparseArray(aval, data, indices)
  return build_sparse_array

def sparse_array_shape_handler(a):
  return (
    xla.xc.Shape.array_shape(a.data_aval.dtype, a.data_aval.shape),
    xla.xc.Shape.array_shape(a.indices_aval.dtype, a.indices_aval.shape),
  )

def sparse_array_device_put_handler(a, device):
  return (
    xla.xb.get_device_backend(device).buffer_from_pyval(a.data, device),
    xla.xb.get_device_backend(device).buffer_from_pyval(a.indices, device)
  )

core.pytype_aval_mappings[SparseArray] = lambda x: x.aval
core.raise_to_shaped_mappings[AbstractSparseArray] = lambda aval, _: aval
xla.pytype_aval_mappings[SparseArray] = lambda x: x.aval
xla.canonicalize_dtype_handlers[SparseArray] = lambda x: x
xla.device_put_handlers[SparseArray] = sparse_array_device_put_handler
xla.xla_result_handlers[AbstractSparseArray] = sparse_array_result_handler
xla.xla_shape_handlers[AbstractSparseArray] = sparse_array_shape_handler


sp_indices_p = core.Primitive('sp_indices')

@sp_indices_p.def_impl
def _sp_indices_impl(mat):
  return mat.indices

@sp_indices_p.def_abstract_eval
def _sp_indices_abstract_eval(mat):
  return mat.indices_aval

def _sp_indices_translation_rule(c, data, indices):
  return indices

# Note: cannot use lower_fun to define attribute access primitives
# because it leads to infinite recursion.
xla.translations[sp_indices_p] = _sp_indices_translation_rule

sp_data_p = core.Primitive('sp_data')

@sp_data_p.def_impl
def _sp_data_impl(mat):
  return mat.data

@sp_data_p.def_abstract_eval
def _sp_data_abstract_eval(mat):
  return mat.data_aval

def _sp_data_translation_rule(c, data, indices):
  return data

# Note: cannot use lower_fun to define attribute access primitives
# because it leads to infinite recursion.
xla.translations[sp_data_p] = _sp_data_translation_rule

def identity(x):
  return identity_p.bind(x)

identity_p = core.Primitive('identity')

@identity_p.def_impl
def _identity_impl(mat):
  return mat

@identity_p.def_abstract_eval
def _identity_abstract_eval(mat):
  return AbstractSparseArray(mat.shape, mat.dtype, mat.index_dtype, mat.nnz)

xla.translations_with_avals[identity_p] = xla.lower_fun(_identity_impl, multiple_results=False, with_avals=True)

def split(x):
  return split_p.bind(x)

split_p = core.Primitive('split')
split_p.multiple_results = True

@split_p.def_impl
def _split_impl(mat):
  return mat, mat

@split_p.def_abstract_eval
def _split_abstract_eval(mat):
  m = AbstractSparseArray(mat.shape, mat.dtype, mat.index_dtype, mat.nnz)
  return m, m

xla.translations_with_avals[split_p] = xla.lower_fun(_split_impl, multiple_results=True, with_avals=True)

def make_sparse_array(rng, shape, dtype, nnz=0.2):
  mat = rng(shape, dtype)
  size = int(np.prod(shape))
  if 0 < nnz < 1:
    nnz = nnz * size
  nnz = int(nnz)
  if nnz == 0:
    mat = np.zeros_like(mat)
  elif nnz < size:
    # TODO(jakevdp): do we care about duplicates?
    cutoff = np.sort(mat.ravel())[nnz]
    mat[mat >= cutoff] = 0
  nz = (mat != 0)
  data = jnp.array(mat[nz])
  indices = jnp.array(np.where(nz)).T
  aval = AbstractSparseArray(shape, data.dtype, indices.dtype, len(indices))
  return SparseArray(aval, data, indices)

def matvec(mat, v):
  v = jnp.asarray(v)
  assert v.ndim == 1
  assert len(mat.shape) == 2
  assert v.shape[0] == mat.shape[1]
  rows = mat.indices[:, 0]
  cols = mat.indices[:, 1]
  dv = mat.data * v[cols]
  return jnp.zeros(mat.shape[0], dtype=dv.dtype).at[rows].add(dv)


class Empty:
  def __init__(self, aval):
    self.aval = aval

class AbstractEmpty(core.AbstractValue):
  _num_buffers = 0

  def join(self, other):
    assert isinstance(other, self.__class__), other
    return self

  def __hash__(self):
    return hash(())

  def __eq__(self, other):
    return isinstance(other, AbstractEmpty)

class ConcreteEmpty(AbstractEmpty):
  pass


core.pytype_aval_mappings[Empty] = lambda x: ConcreteEmpty()
core.raise_to_shaped_mappings[AbstractEmpty] = lambda aval, _: aval
xla.pytype_aval_mappings[Empty] = lambda x: AbstractEmpty()
xla.canonicalize_dtype_handlers[Empty] = lambda x: x
xla.device_put_handlers[Empty] = lambda _, __: ()
xla.xla_result_handlers[AbstractEmpty] = lambda _, __: lambda: Empty(AbstractEmpty())
xla.xla_shape_handlers[AbstractEmpty] = lambda _: ()


class CustomObjectTest(jtu.JaxTestCase):

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_compile={}_primitive={}".format(compile, primitive),
       "compile": compile, "primitive": primitive}
      for primitive in [True, False]
      for compile in [True, False]))
  def testSparseIdentity(self, compile, primitive):
    f = identity if primitive else (lambda x: x)
    f = jit(f) if compile else f
    rng = jtu.rand_default(self.rng())
    M = make_sparse_array(rng, (10,), jnp.float32)
    M2 = f(M)

    jaxpr = make_jaxpr(f)(M).jaxpr
    core.check_jaxpr(jaxpr)

    self.assertEqual(M.dtype, M2.dtype)
    self.assertEqual(M.index_dtype, M2.index_dtype)
    self.assertAllClose(M.data, M2.data)
    self.assertAllClose(M.indices, M2.indices)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_compile={}".format(compile),
       "compile": compile}
      for compile in [True, False]))
  def testSparseSplit(self, compile):
    f = jit(split) if compile else split
    rng = jtu.rand_default(self.rng())
    M = make_sparse_array(rng, (10,), jnp.float32)
    M2, M3 = f(M)

    jaxpr = make_jaxpr(f)(M).jaxpr
    core.check_jaxpr(jaxpr)

    for MM in M2, M3:
      self.assertEqual(M.dtype, MM.dtype)
      self.assertEqual(M.index_dtype, MM.index_dtype)
      self.assertArraysEqual(M.data, MM.data)
      self.assertArraysEqual(M.indices, MM.indices)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_compile={}_primitive={}".format(compile, primitive),
       "compile": compile, "primitive": primitive}
      for primitive in [True, False]
      for compile in [True, False]))
  def testSparseLaxLoop(self, compile, primitive):
    rng = jtu.rand_default(self.rng())
    f = identity if primitive else (lambda x: x)
    f = jit(f) if compile else f
    body_fun = lambda _, A: f(A)
    M = make_sparse_array(rng, (10,), jnp.float32)
    lax.fori_loop(0, 10, body_fun, M)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_attr={}".format(attr), "attr": attr}
      for attr in ["data", "indices"]))
  def testSparseAttrAccess(self, attr):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [make_sparse_array(rng, (10,), jnp.float32)]
    f = lambda x: getattr(x, attr)
    self._CompileAndCheck(f, args_maker)

  @parameterized.named_parameters(jtu.cases_from_list(
      {"testcase_name": "_{}".format(
         jtu.format_shape_dtype_string(shape, dtype)),
       "shape": shape, "dtype": dtype}
      for shape in [(3, 3), (2, 6), (6, 2)]
      for dtype in jtu.dtypes.floating))
  def testSparseMatvec(self, shape, dtype):
    rng = jtu.rand_default(self.rng())
    args_maker = lambda: [make_sparse_array(rng, shape, dtype), rng(shape[-1:], dtype)]
    self._CompileAndCheck(matvec, args_maker)

  def testLowerToNothing(self):
    empty = Empty(AbstractEmpty())
    jaxpr = make_jaxpr(jit(lambda e: e))(empty).jaxpr
    core.check_jaxpr(jaxpr)

    # cannot return a unit, because CompileAndCheck assumes array output.
    testfunc = lambda e: None
    args_maker = lambda: [empty]
    self._CompileAndCheck(testfunc, args_maker)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())
