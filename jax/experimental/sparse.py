import abc

from jax import lax, vmap
import jax.numpy as jnp
from jax.numpy.lax_numpy import _promote_args


class SparseArray(abc.ABC):
  @abc.abstractclassmethod
  def fromdense(cls, x):
    ...

  @abc.abstractmethod
  def todense(self):
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

  @classmethod
  def fromdense(cls, x):
    x = jnp.asarray(x)
    nz = (x != 0)
    return cls(jnp.where(nz), x[nz], x.shape)

  def todense(self):
    d = jnp.zeros(self.shape, self.dtype)
    return d.at[tuple(self.coords)].add(self.data)

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

  def matvec(self, v):
    v = jnp.asarray(v)
    assert v.ndim == 1
    assert self.ndim == 2
    assert v.shape[0] == self.shape[1]
    rows, cols = self.coords
    dv = self.data * v[cols]
    return jnp.zeros(self.shape[0], dtype=dv.dtype).at[rows].add(dv)

  def tocoo(self):
    return self

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

  @classmethod
  def fromdense(cls, x):
    x = jnp.asarray(x)
    nz = (x != 0)
    data = x[nz]
    row, col = jnp.where(nz)
    if len(row) == 0:
      return cls(row, jnp.zeros(x.shape[0] + 1, row.dtype), data, x.shape)
    row, col = lax.sort_key_val(row, col)
    indices = jnp.ravel(col)
    indptr = jnp.cumsum(jnp.bincount(row))
    indptr = jnp.concatenate([
      jnp.zeros(1, indptr.dtype),
      indptr,
      jnp.full(x.shape[0] - len(indptr), indptr[-1])
    ])
    return cls(indices, indptr, data, x.shape)

  def todense(self):
    d = jnp.zeros(self.shape, self.dtype)
    row = jnp.repeat(jnp.arange(self.shape[0]), jnp.diff(self.indptr))
    col = self.indices
    return d.at[row, col].add(self.data)

  def tocsr(self):
    return self

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

  def matvec(self, v):
    v = jnp.asarray(v)
    assert v.ndim == 1
    assert v.shape[0] == self.shape[1]
    dv = self.data * v[self.indices]
    ind = jnp.cumsum(jnp.zeros_like(self.indices).at[self.indptr].add(1))
    return jnp.zeros(self.shape[0], dv.dtype).at[ind - 1].add(dv)


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

  @classmethod
  def fromdense(cls, x):
    x = jnp.asarray(x)
    nz = (x != 0)
    rownz = nz.sum(1)
    shape = (x.shape[0], int(rownz.max()))

    col = nz.cumsum(1)[nz] - 1
    row = jnp.broadcast_to(jnp.arange(nz.shape[0])[:, None], nz.shape)[nz]
    data = jnp.zeros(shape, dtype=x.dtype).at[row, col].set(x[nz])
    columns = jnp.zeros(shape, dtype=col.dtype).at[row, col].set(jnp.where(nz)[1])

    return cls(rownz, columns, data, x.shape)

  def todense(self):
    valid = jnp.arange(self.columns.shape[1]) < self.rownz[:, None]
    rows = jnp.broadcast_to(jnp.arange(self.columns.shape[0])[:, None], self.columns.shape)
    return jnp.zeros(self.shape, self.dtype).at[rows[valid], self.columns[valid]].add(self.data[valid])

  def toell(self):
    return self

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

  def matvec(self, v):
    v = jnp.asarray(v)
    assert v.ndim == 1
    assert v.shape[0] == self.shape[1]
    invalid = (jnp.arange(self.data.shape[1]) >= self.rownz[:, None])
    dv = self.data * v[self.columns]
    return dv.at[invalid].set(0).sum(1, dtype=dv.dtype)


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
    x = jnp.asarray(x)
    if blocksize is None:
      blocksize = (1, 1)
    blocksize = tuple(blocksize)
    assert len(blocksize) == 2
    assert x.ndim == 2
    assert all(i % j == 0 for i, j in zip(x.shape, blocksize))
    blockshape = (x.shape[0] // blocksize[0], x.shape[1] // blocksize[1])
    data = x.reshape(blockshape[0], blocksize[0], blockshape[1], blocksize[1])
    data = data.transpose((0, 2, 1, 3))

    nz = (data != 0).any(-1).any(-1)
    row, col = jnp.where(nz)
    dataflat = data[nz]
    if len(row) == 0:
      return cls(row, jnp.zeros(data.shape[0] + 1, row.dtype), dataflat, x.shape)
    row, col = lax.sort_key_val(row, col)
    indices = jnp.ravel(col)
    indptr = jnp.cumsum(jnp.bincount(row))
    indptr = jnp.concatenate([
      jnp.zeros(1, indptr.dtype),
      indptr,
      jnp.full(data.shape[0] - len(indptr), indptr[-1])
    ])
    return cls(indices, indptr, dataflat, x.shape)

  def tocoo(self):
    # TODO(jakevdp): specialize this
    return COO.fromdense(self.todense())

  def tocsr(self):
    # TODO(jakevdp): specialize this
    return CSR.fromdense(self.todense())

  def todense(self):
    d = jnp.zeros(self.blockshape + self.blocksize, self.dtype)
    row = jnp.repeat(jnp.arange(self.blockshape[0]), jnp.diff(self.indptr))
    col = self.indices
    return d.at[row, col].add(self.data).transpose((0, 2, 1, 3)).reshape(self.shape)

  def toell(self):
    # TODO(jakevdp): specialize this
    return ELL.fromdense(self.todense())

  def matvec(self, v):
    v = jnp.asarray(v)
    assert v.ndim == 1
    assert v.shape[0] == self.shape[1]
    v = v.reshape(-1, self.blocksize[1])
    dv = vmap(jnp.dot)(self.data, v[self.indices])
    ind = jnp.cumsum(jnp.zeros_like(self.indices).at[self.indptr].add(1))
    return jnp.zeros((self.blockshape[0], self.blocksize[0]), dv.dtype).at[ind - 1].add(dv).ravel()
