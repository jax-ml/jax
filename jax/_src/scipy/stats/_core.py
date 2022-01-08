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

from collections import namedtuple
import jax.numpy as jnp
import warnings
from numpy import  ma
from . import mstats_basic
import numpy as np

def _chk_asarray(a, axis):
    if axis is None:
        a = jnp.ravel(a)
        outaxis = 0
    else:
        a = jnp.asarray(a)
        outaxis = axis

    if a.ndim == 0:
        a = jnp.atleast_1d(a)

    return a, outaxis

ModeResult = namedtuple('ModeResult', ('mode', 'count'))

def _contains_nan(a, nan_policy='propagate'):
    policies = ['propagate', 'raise', 'omit']
    if nan_policy not in policies:
        raise ValueError("nan_policy must be one of {%s}" %
                         ', '.join("'%s'" % s for s in policies))
    try:
        # Calling jnp.sum to avoid creating a huge array into memory
        # e.g. jnp.isnan(a).any()
        with np.errstate(invalid='ignore'):                                               #check
            contains_nan = jnp.isnan(jnp.sum(a))
    except TypeError:
        # This can happen when attempting to sum things which are not
        # numbers (e.g. as in the function `mode`). Try an alternative method:
        try:
            contains_nan = jnp.nan in set(a.ravel())
        except TypeError:
            # Don't know what to do. Fall back to omitting nan values and
            # issue a warning.
            contains_nan = False
            nan_policy = 'omit'
            warnings.warn("The input array could not be properly "
                          "checked for nan values. nan values "
                          "will be ignored.", RuntimeWarning)

    if contains_nan and nan_policy == 'raise':
        raise ValueError("The input contains nan values")

    return contains_nan, nan_policy

def mode(a, axis=0, nan_policy='propagate'):

    """
    Return an array of the modal (most common) value in the passed array.
    If there is more than one such value, only the smallest is returned.
    The bin-count for the modal bins is also returned.
    Parameters
    ----------
    a : array_like
        n-dimensional array of which to find mode(s).
    axis : int or None, optional
        Axis along which to operate. Default is 0. If None, compute over
        the whole array `a`.
    nan_policy : {'propagate', 'raise', 'omit'}, optional
        Defines how to handle when input contains nan.
        The following options are available (default is 'propagate'):
          * 'propagate': returns nan
          * 'raise': throws an error
          * 'omit': performs the calculations ignoring nan values
    Returns
    -------
    mode : ndarray
        Array of modal values.
    count : ndarray
        Array of counts for each mode.
    Examples
    --------
    >>> a = jnp.array([[6, 8, 3, 0],
    ...               [3, 2, 1, 7],
    ...               [8, 1, 8, 4],
    ...               [5, 3, 0, 5],
    ...               [4, 7, 5, 9]])
    >>> from scipy import stats
    >>> stats.mode(a)
    ModeResult(mode=array([[3, 1, 0, 0]]), count=array([[1, 1, 1, 1]]))
    To get mode of whole array, specify ``axis=None``:
    >>> stats.mode(a, axis=None)
    ModeResult(mode=array([3]), count=array([3]))
    """

    a, axis = _chk_asarray(a, axis)
    if a.size == 0:
        return ModeResult(jnp.array([]), jnp.array([]))

    contains_nan, nan_policy = _contains_nan(a, nan_policy)

    if contains_nan and nan_policy == 'omit':
        a = ma.masked_invalid(a)
        return mstats_basic.mode(a, axis)

    if a.dtype == object and jnp.nan in set(a.ravel()):
        # Fall back to a slower method since jnp.unique does not work with NaN
        scores = set(jnp.ravel(a))  # get ALL unique values
        testshape = list(a.shape)
        testshape[axis] = 1
        oldmostfreq = jnp.zeros(testshape, dtype=a.dtype)
        oldcounts = jnp.zeros(testshape, dtype=int)

        for score in scores:
            template = (a == score)
            counts = jnp.sum(template, axis, keepdims=True)
            mostfrequent = jnp.where(counts > oldcounts, score, oldmostfreq)
            oldcounts = jnp.maximum(counts, oldcounts)
            oldmostfreq = mostfrequent

        return ModeResult(mostfrequent, oldcounts)

    def _mode1D(a):
        vals, cnts = jnp.unique(a, return_counts=True)
        return vals[cnts.argmax()], cnts.max()

    # jnp.apply_along_axis will convert the _mode1D tuples to a numpy array,
    # casting types in the process.
    # This recreates the results without that issue
    # View of a, rotated so the requested axis is last
    in_dims = list(range(a.ndim))
    a_view = jnp.transpose(a, in_dims[:axis] + in_dims[axis+1:] + [axis])

    inds = np.ndindex(a_view.shape[:-1])                                    #star
    modes = jnp.empty(a_view.shape[:-1], dtype=a.dtype)
    counts = jnp.empty(a_view.shape[:-1], dtype=jnp.int_)
    for ind in inds:
        modes[ind], counts[ind] = _mode1D(a_view[ind])
    newshape = list(a.shape)
    newshape[axis] = 1
    return ModeResult(modes.reshape(newshape), counts.reshape(newshape))
