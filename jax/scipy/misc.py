# Copyright 2018 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as onp
import scipy.misc as osp_misc

from .. import lax


def _wraps(fun):
  """Like functools.wraps but works with numpy.ufuncs."""
  docstr = """
  LAX-backed implementation of {fun}. Corresponding Scipy docstring below.

  {np_doc}
  """.format(fun=fun.__name__, np_doc=fun.__doc__)
  def wrap(op):
    try:
      op.__name__ = fun.__name__
      op.__doc__ = docstr
    finally:
      return op
  return wrap


@_wraps(osp_misc.logsumexp)
def logsumexp(a, axis=None, b=None, keepdims=False, return_sign=False):
  if b is not None or return_sign:
    raise NotImplementedError("Only implemented for b=None, return_sign=False")
  dims = _reduction_dims(a, axis)
  shape = lax.subvals(onp.shape(a), zip(dims, (1,) * len(dims)))
  dimadd = lambda x: lax.reshape(x, shape)
  amax = lax.reduce(a, _constant_like(a, -onp.inf), lax.max, dims)
  amax_singletons = dimadd(amax)
  out = lax.add(lax.log(lax.reduce(lax.exp(lax.sub(a, amax_singletons)),
                                   _constant_like(a, 0), lax.add, dims)), amax)
  return dimadd(out) if keepdims else out


# TODO(mattjj): this is duplicated from lax_numpy.py
def _reduction_dims(a, axis):
  if axis is None:
    return onp.arange(onp.ndim(a))
  elif isinstance(axis, (onp.ndarray, tuple, list)):
    return onp.mod(onp.asarray(axis), onp.ndim(a))
  elif isinstance(axis, int):
    return onp.mod([axis], onp.ndim(a))
  else:
    raise TypeError("Unexpected type of axis argument: {}".format(type(axis)))


def _constant_like(x, const):
  return onp.array(const, dtype=lax._dtype(x))
