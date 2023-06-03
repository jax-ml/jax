# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from jax import jit
import jax.numpy as jnp
import jax.lax as lax
from jax._src.numpy.util import promote_dtypes_inexact

__all__ = [
  'as_series', 'trimseq', 'trimcoef', 'getdomain', 'mapdomain', 'mapparms'
]

@jit
def _as_series(alist):
  ret = [jnp.array(a, ndmin=1) for a in alist]
  if min([a.size for a in ret]) == 0:
    raise ValueError("Coefficient array is empty")
  if any(a.ndim != 1 for a in ret):
    raise ValueError("Coefficient array is not 1-d")
  return ret

def as_series(alist, trim=True):
  if trim:
    return [trimseq(s) for s in _as_series(alist)]
  else:
    return _as_series(alist)

#Not compatible with JIT
def trimseq(seq):
  if len(seq) == 0:
    return seq
  else:
    for i in range(len(seq)-1, -1, -1):
      if seq[i] != 0:
        break
    return seq[:i+1]

#Not compatible with JIT
def trimcoef(c, tol=0):
  if tol < 0:
    raise ValueError("tol must be non-negative")
  [c] = _as_series([c])
  c, = promote_dtypes_inexact(c)
  [ind] = jnp.nonzero(jnp.abs(c) > tol)
  if len(ind) == 0:
    return c[:1]*0
  else:
    return c[:ind[-1] + 1]

@jit
def getdomain(x):
  [x] = _as_series([x])
  if jnp.iscomplexobj(x):
    rmin, rmax = x.real.min(), x.real.max()
    imin, imax = x.imag.min(), x.imag.max()
    return jnp.array((lax.complex(rmin, imin), lax.complex(rmax, imax)), dtype=x.dtype)
  return jnp.array((x.min(), x.max()), dtype=x.dtype)

@jit
def mapparms(old, new):
  oldlen = old[1] - old[0]
  newlen = new[1] - new[0]
  off = (old[1]*new[0] - old[0]*new[1])/oldlen
  scl = newlen/oldlen
  return off, scl

@jit
def mapdomain(x, old, new):
  x = jnp.asarray(x)
  x, = promote_dtypes_inexact(x)
  off, scl = mapparms(old, new)
  return off + scl*x
