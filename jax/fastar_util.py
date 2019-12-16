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


def true_mask(val):
  return onp.full(onp.shape(val), True, dtype=bool)


def false_mask(val):
  return onp.full(onp.shape(val), False, dtype=bool)


def mask_all(parray):
  _, mask = parray
  return onp.all(mask)


class HashableMask(object):
  def __init__(self, mask):
    self.mask = mask

  def __hash__(self):
    return hash(self.mask.tostring())

  def __eq__(self, other):
    return onp.all(self.mask == other.mask)


def _to_tree(idxs):
  tree = {}
  for idx in idxs:
    branch = tree
    for i in idx:
      branch = branch.setdefault(i, {})
  return tree


def _contains_rectangle(idx_tree, rectangle):
  """
  Return True if rectangle is contained in idx_tree, else False.
  """
  (start, stop), rectangle = rectangle[0], rectangle[1:]
  return all(
    n in idx_tree
    and (not rectangle or _contains_rectangle(idx_tree[n], rectangle))
    for n in range(start, stop))


def _remove_rectangle(idx_tree, rectangle):
  (start, stop), rectangle = rectangle[0], rectangle[1:]
  new_tree = {}
  for root, branch in idx_tree.items():
    if start <= root < stop:
      if rectangle:
        new_branch = _remove_rectangle(branch, rectangle)
        if new_branch:
          new_tree[root] = new_branch
    else:
      new_tree[root] = branch
  return new_tree


def _find_rectangle(idx_tree):
  """
  Greedily find a rectangle in idx_tree.
  """
  start = min(idx_tree.keys())
  stop = start + 1
  branch = idx_tree[start]
  if branch:
    rect = _find_rectangle(branch)
    while stop in idx_tree and _contains_rectangle(idx_tree[stop], rect):
      stop += 1
    return ((start, stop),) + rect
  else:
    while stop in idx_tree:
      stop += 1
    return (start, stop),


def mask_to_slices(mask):
  """
  Greedily search for rectangular slices in mask.
  """
  if onp.shape(mask) == ():
    return [()] if mask else []

  rectangles = []
  idx_tree = _to_tree(onp.argwhere(mask))
  while idx_tree:
    rect = _find_rectangle(idx_tree)
    rectangles.append(rect)
    idx_tree = _remove_rectangle(idx_tree, rect)
  return [tuple(slice(s, e) for s, e in rect) for rect in rectangles]
