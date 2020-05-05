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


from .tree_util import tree_flatten, tree_unflatten
from . import linear_util as lu
from .util import safe_zip

import jax.numpy as jnp
from jax.api import vjp

zip = safe_zip


def ravel_pytree(pytree):
  leaves, treedef = tree_flatten(pytree)
  flat, unravel_list = vjp(ravel_list, *leaves)
  unravel_pytree = lambda flat: tree_unflatten(treedef, unravel_list(flat))
  return flat, unravel_pytree

def ravel_list(*lst):
  return jnp.concatenate([jnp.ravel(elt) for elt in lst]) if lst else jnp.array([])


@lu.transformation_with_aux
def ravel_fun(unravel_inputs, flat_in, **kwargs):
  pytree_args = unravel_inputs(flat_in)
  ans = yield pytree_args, {}
  yield ravel_pytree(ans)
