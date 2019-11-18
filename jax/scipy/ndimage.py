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

import functools
import itertools
import operator
import textwrap

import scipy.ndimage

from ..numpy import lax_numpy as np
from ..numpy.lax_numpy import _wraps


def _prod(values):
  out = 1
  for value in values:
    out *= value
  return out


# Note: these only hold for order=1
_INDEX_FIXERS = {
    'constant': lambda index, size: index,
    'nearest': lambda index, size: np.clip(index, 0, size - 1),
    'wrap': lambda index, size: index % size,
}


@_wraps(scipy.ndimage.map_coordinates, lax_description=textwrap.dedent("""\
    Only linear interpolation (``order=1``) and modes ``'constant'``,
    ``'nearest'`` and ``'wrap'`` are currently supported. Note that
    interpolation near boundaries differs from the scipy function, because we
    fixed an outstanding bug (https://github.com/scipy/scipy/issues/2640);
    this function interprets the ``mode`` argument as documented by SciPy, but
    not as implemented by SciPy.
    """))
def map_coordinates(
    input, coordinates, order, mode='constant', cval=0.0,
):
  if order != 1:
    raise NotImplementedError(
        'jax.scipy.ndimage.map_coordinates currently requires order=1')

  input = np.asarray(input)

  index_fixer = _INDEX_FIXERS.get(mode)
  if index_fixer is None:
    raise NotImplementedError(
        'jax.scipy.ndimage.map_coordinates does not yet support mode {}. '
        'Currently supported modes are {}.'.format(mode, set(_INDEX_FIXERS)))

  if mode == 'constant':
    is_valid = lambda index, size: (0 <= index) & (index < size)
  else:
    is_valid = lambda index, size: True

  all_indices_and_weights = []
  for coordinate, size in zip(coordinates, input.shape):
    lower = np.floor(coordinate)
    upper = np.ceil(coordinate)
    l_index = lower.astype(np.int32)
    u_index = upper.astype(np.int32)
    l_weight = 1 - (coordinate - lower)
    u_weight = 1 - l_weight  # handles the edge case lower==upper
    all_indices_and_weights.append(
        [(index_fixer(l_index, size), is_valid(l_index, size), l_weight),
         (index_fixer(u_index, size), is_valid(u_index, size), u_weight)]
    )

  outputs = []
  for items in itertools.product(*all_indices_and_weights):
    indices, validities, weights = zip(*items)
    if any(valid is not True for valid in validities):
      all_valid = functools.reduce(operator.and_, validities)
      contribution = np.where(all_valid, input[indices], cval)
    else:
      contribution = input[indices]
    outputs.append(_prod(weights) * contribution)
  result = sum(outputs)
  return result
