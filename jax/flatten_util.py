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

from .tree_util import tree_flatten, tree_unflatten
from .linear_util import transformation_with_aux
from .util import safe_zip

import jax.numpy as np
from jax.api import vjp

zip = safe_zip


def ravel_pytree(pytree):
  leaves, treedef = tree_flatten(pytree)
  flat, unravel_list = vjp(ravel_list, *leaves)
  unravel_pytree = lambda flat: tree_unflatten(treedef, unravel_list(flat))
  return flat, unravel_pytree

def ravel_list(*lst):
  return np.concatenate([np.ravel(elt) for elt in lst]) if lst else np.array([])


@transformation_with_aux
def ravel_fun(unravel_inputs, flat_in, **kwargs):
  pytree_args = unravel_inputs(flat_in)
  ans = yield pytree_args, {}
  yield ravel_pytree(ans)
