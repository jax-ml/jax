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

"""
Sparse XLA
----------
This file contains primitive definitions for sparse matrix functionality.
"""

import abc

from jax import core, lax, lazy, vmap, xla
from jax.lib import xla_client
import jax.numpy as jnp
from jax.numpy.lax_numpy import _promote_args

xops = xla_client.ops


class SparseArray(abc.ABC):
  @abc.abstractclassmethod
  def fromdense(cls, x):
    ...

  @abc.abstractmethod
  def todense(self):
    ...

  @abc.abstractmethod
  def tobsr(self, blocksize=None):
    ...

  @abc.abstractmethod
  def tocoo(self):
    ...

  @abc.abstractmethod
  def tocsr(self):
    ...

  @abc.abstractmethod
  def toell(self):
    ...

  @abc.abstractproperty
  def dtype(self):
    ...

  @abc.abstractproperty
  def index_dtype(self):
    ...

  @abc.abstractproperty
  def nnz(self):
    ...

  @abc.abstractproperty
  def shape(self):
    ...

  @abc.abstractmethod
  def matvec(self, v):
    ...

  @property
  def ndim(self):
    return len(self.shape)

  def __repr__(self):
    return f"{self.__class__.__name__}({self.dtype.name}{list(self.shape)}, nnz={self.nnz})"


class COO(SparseArray):
  """JAX-based sparse array stored in COO format."""
  def __init__(self, coords, data, shape=None):
    self.coords = jnp.atleast_2d(coords)
    self.data = jnp.asarray(data)
    if shape is None:
      shape = tuple(1 + self.coords.max(1))
    self._shape = shape

    assert self.data.ndim == 1
    assert self.coords.ndim == 2
    assert self.coords.shape == (len(shape), self.data.shape[0])

  @property
  def index_dtype(self):
    return self.coords.dtype

  @property
  def dtype(self):
    return self.data.dtype

  @property
  def nnz(self):
    return self.data.shape[0]

  @property
  def shape(self):
    return tuple(self._shape)

  @classmethod
  def fromdense(cls, x):
    return coo_fromdense(cls, x)

  def todense(self):
    return coo_todense(self)

  def matvec(self, v):
    return coo_matvec(self, v)

  def tocoo(self):
    return self

  def tobsr(self, blocksize=None):
    # TODO(jakevdp): specialize this
    return BSR.fromdense(self.todense(), blocksize=blocksize)

  def tocsr(self):
    assert self.ndim == 2
    row, col, data = lax.sort(tuple(self.coords) + (self.data,))
    if len(row) == 0:
      return CSR(row, jnp.zeros(self.shape[0] + 1, row.dtype), data, self.shape)
    indices = jnp.ravel(col)
    indptr = jnp.cumsum(jnp.bincount(row))
    indptr = jnp.concatenate([
      jnp.zeros(1, indptr.dtype),
      indptr,
      jnp.full(self.shape[0] - len(indptr), indptr[-1])
    ])
    return CSR(indices, indptr, data, self.shape)

  def toell(self):
    # TODO(jakevdp): implement this more directly.
    return self.tocsr().toell()


class AbstractCOO(core.ShapedArray):
  __slots__ = ['index_dtype', 'nnz', 'data_aval', 'coords_aval']
  _num_buffers = 2

  def __init__(self, shape, dtype, index_dtype, nnz):
    super().__init__(shape, dtype)
    self.index_dtype = index_dtype
    self.nnz = nnz
    self.data_aval = core.ShapedArray((nnz,), dtype)
    self.coords_aval = core.ShapedArray((len(shape), nnz), index_dtype)

  @core.aval_property
  def data(self):
    return coo_buffers_p.bind(self)[0]

  @core.aval_property
  def coords(self):
    return coo_buffers_p.bind(self)[1]

def abstract_coo(arr):
  return AbstractCOO(arr.shape, arr.dtype, arr.index_dtype, arr.nnz)

def coo_result_handler(device, aval):
  def build_coo_array(coords_buf, data_buf):
    data = xla.DeviceArray(aval.data_aval, device, lazy.array(aval.data_aval.shape), data_buf)
    coords = xla.DeviceArray(aval.coords_aval, device, lazy.array(aval.coords_aval.shape), coords_buf)
    return  COO(coords, data, shape=aval.shape)
  return build_coo_array

def coo_shape_handler(a):
  return (
    xla.xc.Shape.array_shape(a.data_aval.dtype, a.data_aval.shape),
    xla.xc.Shape.array_shape(a.coords_aval.dtype, a.coords_aval.shape),
  )

def coo_device_put_handler(a, device):
  return (
    xla.xb.get_device_backend(device).buffer_from_pyval(a.data, device),
    xla.xb.get_device_backend(device).buffer_from_pyval(a.coords, device),
  )

core.pytype_aval_mappings[COO] = abstract_coo
core.raise_to_shaped_mappings[AbstractCOO] = lambda aval, _: aval
xla.pytype_aval_mappings[COO] = abstract_coo
xla.canonicalize_dtype_handlers[COO] = lambda x: x
xla.device_put_handlers[COO] = coo_device_put_handler
xla.xla_result_handlers[AbstractCOO] = coo_result_handler
xla.xla_shape_handlers[AbstractCOO] = coo_shape_handler


class CSR(SparseArray):
  """JAX-based sparse array stored in CSR format."""
  def __init__(self, indices, indptr, data, shape=None):
    self.indices, self.indptr = _promote_args("CSR", indices, indptr)
    self.data = jnp.array(data)
    if shape is None:
      shape = (len(self.indptr) - 1, self.indices.max() + 1)
    self._shape = shape

    assert len(shape) == 2
    assert self.data.ndim == 1
    assert self.indices.shape == self.data.shape
    assert shape[0] == len(self.indptr) - 1
    assert shape[1] > self.indices.max()

  @property
  def aval(self):
    return abstract_csr(self)

  @property
  def index_dtype(self):
    return self.indices.dtype

  @property
  def dtype(self):
    return self.data.dtype

  @property
  def nnz(self):
    return self.data.shape[0]

  @property
  def shape(self):
    return tuple(self._shape)

  @classmethod
  def fromdense(cls, x):
    return csr_fromdense(cls, x)

  def todense(self):
    return csr_todense(self)

  def matvec(self, v):
    return csr_matvec(self, v)

  def tocsr(self):
    return self

  def tobsr(self, blocksize=None):
    # TODO(jakevdp): specialize this
    return BSR.fromdense(self.todense(), blocksize=blocksize)

  def tocoo(self):
    row = jnp.repeat(jnp.arange(self.shape[0]), jnp.diff(self.indptr))
    col = self.indices
    return COO(jnp.vstack([row, col]), self.data, self.shape)

  def toell(self):
    rownz = jnp.diff(self.indptr)
    shape = (self.shape[0], rownz.max())
    columns = jnp.zeros(shape, dtype=self.index_dtype)
    data = jnp.zeros(shape, dtype=self.dtype)
    row = jnp.repeat(jnp.arange(self.shape[0]), rownz)
    # TODO(jakevdp): faster way to do this?
    col = jnp.concatenate([jnp.arange(n) for n in rownz])
    data = data.at[row, col].set(self.data)
    columns = columns.at[row, col].set(self.indices)
    return ELL(rownz, columns, data, self.shape)


class AbstractCSR(core.ShapedArray):
  __slots__ = ['index_dtype', 'nnz', 'data_aval', 'indices_aval', 'indptr_aval']
  _num_buffers = 3

  def __init__(self, shape, dtype, index_dtype, nnz):
    super().__init__(shape, dtype)
    self.index_dtype = index_dtype
    self.nnz = nnz
    self.data_aval = core.ShapedArray((nnz,), dtype)
    self.indices_aval = core.ShapedArray((nnz,), index_dtype)
    self.indptr_aval = core.ShapedArray((shape[0] + 1,), index_dtype)

  @core.aval_property
  def data(self):
    return csr_buffers_p.bind(self)[0]

  @core.aval_property
  def indices(self):
    return csr_buffers_p.bind(self)[1]

  @core.aval_property
  def indptr(self):
    return csr_buffers_p.bind(self)[2]

def abstract_csr(arr):
  return AbstractCSR(arr.shape, arr.dtype, arr.index_dtype, arr.nnz)

def csr_result_handler(device, aval):
  def build_csr_array(data_buf, indices_buf, indptr_buf):
    data = xla.DeviceArray(aval.data_aval, device, lazy.array(aval.data_aval.shape), data_buf)
    indices = xla.DeviceArray(aval.indices_aval, device, lazy.array(aval.indices_aval.shape), indices_buf)
    indptr = xla.DeviceArray(aval.indptr_aval, device, lazy.array(aval.indptr_aval.shape), indices_buf)
    return CSR(data, indices, indptr, shape=aval.shape)
  return build_csr_array

def csr_shape_handler(a):
  return (
    xla.xc.Shape.array_shape(a.data_aval.dtype, a.data_aval.shape),
    xla.xc.Shape.array_shape(a.indices_aval.dtype, a.indices_aval.shape),
    xla.xc.Shape.array_shape(a.indptr_aval.dtype, a.indptr_aval.shape),
  )

def csr_device_put_handler(a, device):
  return (
    xla.xb.get_device_backend(device).buffer_from_pyval(a.data, device),
    xla.xb.get_device_backend(device).buffer_from_pyval(a.indices, device),
    xla.xb.get_device_backend(device).buffer_from_pyval(a.indptr, device),
  )

core.pytype_aval_mappings[CSR] = abstract_csr
core.raise_to_shaped_mappings[AbstractCSR] = lambda aval, _: aval
xla.pytype_aval_mappings[CSR] = abstract_csr
xla.canonicalize_dtype_handlers[CSR] = lambda x: x
xla.device_put_handlers[CSR] = csr_device_put_handler
xla.xla_result_handlers[AbstractCSR] = csr_result_handler
xla.xla_shape_handlers[AbstractCSR] = csr_shape_handler


class ELL(SparseArray):
  """JAX-based sparse array stored in ELL format."""
  def __init__(self, rownz, columns, data, shape=None):
    self.rownz = jnp.asarray(rownz)
    self.columns = jnp.asarray(columns)
    self.data = jnp.asarray(data)
    if shape is None:
      shape = (self.columns.shape[0], self.columns.max() + 1)
    self._shape = shape
    assert self.data.ndim == 2
    assert self.rownz.shape == self.data.shape[:1]
    assert self.data.shape == self.columns.shape

  @property
  def index_dtype(self):
    return self.columns.dtype

  @property
  def dtype(self):
    return self.data.dtype

  @property
  def nnz(self):
    return self.rownz.sum()

  @property
  def shape(self):
    return tuple(self._shape)

  @classmethod
  def fromdense(cls, x):
    return ell_fromdense(cls, x)

  def todense(self):
    return ell_todense(self)

  def matvec(self, v):
    return ell_matvec(self, v)

  def toell(self):
    return self

  def tobsr(self, blocksize=None):
    # TODO(jakevdp): specialize this
    return BSR.fromdense(self.todense(), blocksize=blocksize)

  def tocsr(self):
    valid = (jnp.arange(self.data.shape[1]) < self.rownz[:, None])
    indices = self.columns[valid]
    indptr = jnp.cumsum(jnp.concatenate([jnp.zeros(1, dtype=indices.dtype), valid.sum(1)]))
    data = self.data[valid]
    return CSR(indices, indptr, data, self.shape)

  def tocoo(self):
    valid = (jnp.arange(self.data.shape[1]) < self.rownz[:, None])
    col = self.columns[valid]
    row = jnp.where(valid)[0]
    data = self.data[valid]
    return COO(jnp.vstack([row, col]), data, self.shape)


class AbstractELL(core.ShapedArray):
  __slots__ = ['index_dtype', 'nnz', 'data_aval', 'rownz_aval', 'columns_aval']
  _num_buffers = 3

  def __init__(self, shape, dtype, index_dtype, nnz, row_max_nz):
    super().__init__(shape, dtype)
    self.index_dtype = index_dtype
    self.nnz = nnz
    self.data_aval = core.ShapedArray((shape[0], row_max_nz), dtype)
    self.rownz_aval = core.ShapedArray(shape[:1], index_dtype)
    self.columns_aval = core.ShapedArray((shape[0], row_max_nz), index_dtype)

  @core.aval_property
  def data(self):
    return ell_buffers_p.bind(self)[0]

  @core.aval_property
  def rownz(self):
    return ell_buffers_p.bind(self)[1]

  @core.aval_property
  def columns(self):
    return ell_buffers_p.bind(self)[2]

def abstract_ell(arr):
  return AbstractELL(arr.shape, arr.dtype, arr.index_dtype, arr.nnz, arr.data.shape[1])

def ell_result_handler(device, aval):
  def build_ell_array(columns_buf, data_buf):
    data = xla.DeviceArray(aval.data_aval, device, lazy.array(aval.data_aval.shape), data_buf)
    columns = xla.DeviceArray(aval.columns_aval, device, lazy.array(aval.columns_aval.shape), columns_buf)
    return  ELL(rownz=aval.rownz, columns=columns, data=data, shape=aval.shape)
  return build_ell_array

def ell_shape_handler(a):
  return (
    xla.xc.Shape.array_shape(a.data_aval.dtype, a.data_aval.shape),
    xla.xc.Shape.array_shape(a.rownz_aval.dtype, a.rownz_aval.shape),
    xla.xc.Shape.array_shape(a.columns_aval.dtype, a.columns_aval.shape),
  )

def ell_device_put_handler(a, device):
  return (
    xla.xb.get_device_backend(device).buffer_from_pyval(a.data, device),
    xla.xb.get_device_backend(device).buffer_from_pyval(a.rownz, device),
    xla.xb.get_device_backend(device).buffer_from_pyval(a.columns, device),
  )

core.pytype_aval_mappings[ELL] = abstract_ell
core.raise_to_shaped_mappings[AbstractELL] = lambda aval, _: aval
xla.pytype_aval_mappings[ELL] = abstract_ell
xla.canonicalize_dtype_handlers[ELL] = lambda x: x
xla.device_put_handlers[ELL] = ell_device_put_handler
xla.xla_result_handlers[AbstractELL] = ell_result_handler
xla.xla_shape_handlers[AbstractELL] = ell_shape_handler


class BSR(SparseArray):
  """JAX-based sparse array stored in BSR format."""
  def __init__(self, indices, indptr, data, shape=None):
    self.indices, self.indptr = _promote_args("CSR", indices, indptr)
    self.data = jnp.array(data)
    assert self.data.ndim == 3
    assert self.indices.shape == self.data.shape[:1]

    if shape is None:
      shape = (self.blocksize[0] * (len(self.indptr) - 1),
               self.blocksize[1] * (self.indices.max() + 1))
    self._shape = shape

    assert len(shape) == 2
    assert shape[0] % self.blocksize[0] == 0
    assert shape[1] % self.blocksize[1] == 0
    assert shape[0] // self.blocksize[0] == (len(self.indptr) - 1)
    assert shape[1] // self.blocksize[1] > self.indices.max()

  @property
  def blocksize(self):
    return self.data.shape[1:]

  @property
  def blockshape(self):
    return (self.shape[0] // self.blocksize[0], self.shape[1] // self.blocksize[1])

  def __repr__(self):
    return f"{self.__class__.__name__}({self.dtype.name}{list(self.shape)}, blocksize={self.blocksize}, nnz={self.nnz})"

  @property
  def index_dtype(self):
    return self.indices.dtype

  @property
  def dtype(self):
    return self.data.dtype

  @property
  def nnz(self):
    return self.data.size

  @property
  def shape(self):
    return tuple(self._shape)

  @classmethod
  def fromdense(cls, x, blocksize=None):
    if blocksize is None:
      blocksize = (1, 1)
    return bsr_fromdense(cls, x, blocksize=blocksize)

  def todense(self):
    return bsr_todense(self)

  def matvec(self, v):
    return bsr_matvec(self, v)

  def tobsr(self, blocksize=None):
    if blocksize is not None and blocksize != self.blocksize:
      # TODO(jakevdp): specialize this
      return self.fromdense(self.todense(), blocksize=blocksize)
    return self

  def tocoo(self):
    # TODO(jakevdp): specialize this
    return COO.fromdense(self.todense())

  def tocsr(self):
    # TODO(jakevdp): specialize this
    return CSR.fromdense(self.todense())

  def toell(self):
    # TODO(jakevdp): specialize this
    return ELL.fromdense(self.todense())


class AbstractBSR(core.ShapedArray):
  __slots__ = ['index_dtype', 'nnz', 'blocksize', 'data_aval', 'indices_aval', 'indptr_aval']
  _num_buffers = 3

  def __init__(self, shape, dtype, index_dtype, nnz, blocksize):
    super().__init__(shape, dtype)
    self.index_dtype = index_dtype
    self.nnz = nnz
    self.blocksize = blocksize
    self.data_aval = core.ShapedArray((nnz,) + blocksize, dtype)
    self.indices_aval = core.ShapedArray((nnz,), index_dtype)
    self.indptr_aval = core.ShapedArray((shape[0] + 1,), index_dtype)

  @core.aval_property
  def data(self):
    return bsr_buffers_p.bind(self)[0]

  @core.aval_property
  def indices(self):
    return bsr_buffers_p.bind(self)[1]

  @core.aval_property
  def indptr(self):
    return bsr_buffers_p.bind(self)[2]

def abstract_bsr(arr):
  return AbstractBSR(arr.shape, arr.dtype, arr.index_dtype, arr.nnz, arr.blocksize)

def bsr_result_handler(device, aval):
  def build_bsr_array(data_buf, indices_buf, indptr_buf):
    data = xla.DeviceArray(aval.data_aval, device, lazy.array(aval.data_aval.shape), data_buf)
    indices = xla.DeviceArray(aval.indices_aval, device, lazy.array(aval.indices_aval.shape), indices_buf)
    indptr = xla.DeviceArray(aval.indptr_aval, device, lazy.array(aval.indptr_aval.shape), indices_buf)
    return BSR(data, indices, indptr, shape=aval.shape)
  return build_bsr_array

def bsr_shape_handler(a):
  return (
    xla.xc.Shape.array_shape(a.data_aval.dtype, a.data_aval.shape),
    xla.xc.Shape.array_shape(a.indices_aval.dtype, a.indices_aval.shape),
    xla.xc.Shape.array_shape(a.indptr_aval.dtype, a.indptr_aval.shape),
  )

def bsr_device_put_handler(a, device):
  return (
    xla.xb.get_device_backend(device).buffer_from_pyval(a.data, device),
    xla.xb.get_device_backend(device).buffer_from_pyval(a.indices, device),
    xla.xb.get_device_backend(device).buffer_from_pyval(a.indptr, device),
  )

core.pytype_aval_mappings[BSR] = abstract_bsr
core.raise_to_shaped_mappings[AbstractBSR] = lambda aval, _: aval
xla.pytype_aval_mappings[BSR] = abstract_bsr
xla.canonicalize_dtype_handlers[BSR] = lambda x: x
xla.device_put_handlers[BSR] = bsr_device_put_handler
xla.xla_result_handlers[AbstractBSR] = bsr_result_handler
xla.xla_shape_handlers[AbstractBSR] = bsr_shape_handler

#--------------------------------------------------------------------------
# Primitives

coo_buffers_p = core.Primitive('coo_buffers')
coo_buffers_p.multiple_results = True

@coo_buffers_p.def_impl
def _coo_buffers_impl(mat):
  return (mat.data, mat.coords)

@coo_buffers_p.def_abstract_eval
def _coo_buffers_impl(mat):
  return (mat.data_aval, mat.coords_aval)

def _coo_buffers_translation_rule(c, data, coords):
  return xops.Tuple(c, (data, coords))

xla.translations[coo_buffers_p] = _coo_buffers_translation_rule

def coo_fromdense(cls, mat):
  mat = jnp.asarray(mat)
  nz = (mat != 0)
  return cls(jnp.where(nz), mat[nz], mat.shape)

def coo_todense(mat):
  d = jnp.zeros(mat.shape, mat.dtype)
  return d.at[tuple(mat.coords)].add(mat.data)

def coo_matvec(mat: COO, v: jnp.ndarray):
  v = jnp.asarray(v)
  assert v.ndim == 1
  assert mat.ndim == 2
  assert v.shape[0] == mat.shape[1]
  rows, cols = mat.coords
  dv = mat.data * v[cols]
  return jnp.zeros(mat.shape[0], dtype=dv.dtype).at[rows].add(dv)

csr_buffers_p = core.Primitive('csr_buffers')
csr_buffers_p.multiple_results = True

@csr_buffers_p.def_impl
def _csr_buffers_impl(mat):
  return (mat.data, mat.indices, mat.indptr)

@csr_buffers_p.def_abstract_eval
def _csr_buffers_impl(mat):
  return (mat.data_aval, mat.indices_aval, mat.indptr_aval)

def _csr_buffers_translation_rule(c, data, indices, indptr):
  return xops.Tuple(c, (data, indices, indptr))

xla.translations[csr_buffers_p] = _csr_buffers_translation_rule

def csr_fromdense(cls, mat):
  mat = jnp.asarray(mat)
  nz = (mat != 0)
  data = mat[nz]
  row, col = jnp.where(nz)
  if len(row) == 0:
    return cls(row, jnp.zeros(mat.shape[0] + 1, row.dtype), data, mat.shape)
  row, col = lax.sort_key_val(row, col)
  indices = jnp.ravel(col)
  indptr = jnp.cumsum(jnp.bincount(row))
  indptr = jnp.concatenate([
    jnp.zeros(1, indptr.dtype),
    indptr,
    jnp.full(mat.shape[0] - len(indptr), indptr[-1])
  ])
  return cls(indices, indptr, data, mat.shape)

def csr_todense(mat):
  d = jnp.zeros(mat.shape, mat.dtype)
  row = jnp.repeat(jnp.arange(mat.shape[0]), jnp.diff(mat.indptr))
  col = mat.indices
  return d.at[row, col].add(mat.data)

def csr_matvec(mat: CSR, v: jnp.ndarray):
  v = jnp.asarray(v)
  assert v.ndim == 1
  assert v.shape[0] == mat.shape[1]
  dv = mat.data * v[mat.indices]
  ind = jnp.cumsum(jnp.zeros_like(mat.indices).at[mat.indptr].add(1))
  return jnp.zeros(mat.shape[0], dv.dtype).at[ind - 1].add(dv)

ell_buffers_p = core.Primitive('ell_buffers')
ell_buffers_p.multiple_results = True

@ell_buffers_p.def_impl
def _ell_buffers_impl(mat):
  return (mat.data, mat.rownz, mat.columns)

@ell_buffers_p.def_abstract_eval
def _ell_buffers_impl(mat):
  return (mat.data_aval, mat.rownz_aval, mat.columns_aval)

def _ell_buffers_translation_rule(c, data, rownz, columns):
  return xops.Tuple(c, (data, rownz, columns))

xla.translations[ell_buffers_p] = _ell_buffers_translation_rule

def ell_fromdense(cls, mat):
  mat = jnp.asarray(mat)
  nz = (mat != 0)
  rownz = nz.sum(1)
  shape = (mat.shape[0], int(rownz.max()))

  col = nz.cumsum(1)[nz] - 1
  row = jnp.broadcast_to(jnp.arange(nz.shape[0])[:, None], nz.shape)[nz]
  data = jnp.zeros(shape, dtype=mat.dtype).at[row, col].set(mat[nz])
  columns = jnp.zeros(shape, dtype=col.dtype).at[row, col].set(jnp.where(nz)[1])

  return cls(rownz, columns, data, mat.shape)

def ell_todense(mat):
  valid = jnp.arange(mat.columns.shape[1]) < mat.rownz[:, None]
  rows = jnp.broadcast_to(jnp.arange(mat.columns.shape[0])[:, None], mat.columns.shape)
  return jnp.zeros(mat.shape, mat.dtype).at[rows[valid], mat.columns[valid]].add(mat.data[valid])


def ell_matvec(mat: ELL, v: jnp.ndarray):
  v = jnp.asarray(v)
  assert v.ndim == 1
  assert v.shape[0] == mat.shape[1]
  invalid = (jnp.arange(mat.data.shape[1]) >= mat.rownz[:, None])
  dv = mat.data * v[mat.columns]
  return dv.at[invalid].set(0).sum(1, dtype=dv.dtype)

bsr_buffers_p = core.Primitive('bsr_buffers')
bsr_buffers_p.multiple_results = True

@bsr_buffers_p.def_impl
def _bsr_buffers_impl(mat):
  return (mat.data, mat.indices, mat.indptr)

@bsr_buffers_p.def_abstract_eval
def _bsr_buffers_impl(mat):
  return (mat.data_aval, mat.indices_aval, mat.indptr_aval)

def _bsr_buffers_translation_rule(c, data, indices, indptr):
  return xops.Tuple(c, (data, indices, indptr))

xla.translations[bsr_buffers_p] = _bsr_buffers_translation_rule

def bsr_fromdense(cls, mat, blocksize):
  mat = jnp.asarray(mat)
  blocksize = tuple(blocksize)
  assert len(blocksize) == 2
  assert mat.ndim == 2
  assert all(i % j == 0 for i, j in zip(mat.shape, blocksize))
  blockshape = (mat.shape[0] // blocksize[0], mat.shape[1] // blocksize[1])
  data = mat.reshape(blockshape[0], blocksize[0], blockshape[1], blocksize[1])
  data = data.transpose((0, 2, 1, 3))

  nz = (data != 0).any(-1).any(-1)
  row, col = jnp.where(nz)
  dataflat = data[nz]
  if len(row) == 0:
    return cls(row, jnp.zeros(data.shape[0] + 1, row.dtype), dataflat, mat.shape)
  row, col = lax.sort_key_val(row, col)
  indices = jnp.ravel(col)
  indptr = jnp.cumsum(jnp.bincount(row))
  indptr = jnp.concatenate([
    jnp.zeros(1, indptr.dtype),
    indptr,
    jnp.full(data.shape[0] - len(indptr), indptr[-1])
  ])
  return cls(indices, indptr, dataflat, mat.shape)

def bsr_todense(mat):
  d = jnp.zeros(mat.blockshape + mat.blocksize, mat.dtype)
  row = jnp.repeat(jnp.arange(mat.blockshape[0]), jnp.diff(mat.indptr))
  col = mat.indices
  return d.at[row, col].add(mat.data).transpose((0, 2, 1, 3)).reshape(mat.shape)

def bsr_matvec(mat: BSR, v: jnp.ndarray):
  v = jnp.asarray(v)
  assert v.ndim == 1
  assert v.shape[0] == mat.shape[1]
  v = v.reshape(-1, mat.blocksize[1])
  dv = vmap(jnp.dot)(mat.data, v[mat.indices])
  ind = jnp.cumsum(jnp.zeros_like(mat.indices).at[mat.indptr].add(1))
  return jnp.zeros((mat.blockshape[0], mat.blocksize[0]), dv.dtype).at[ind - 1].add(dv).ravel()
