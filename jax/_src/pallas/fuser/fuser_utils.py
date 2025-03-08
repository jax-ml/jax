# Copyright 2025 The JAX Authors.
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

"""Basic utils for fuser internals."""
from jax._src import api_util
from jax._src import core
from jax._src import linear_util as lu
from jax._src import tree_util
from jax._src.interpreters import partial_eval as pe



def make_jaxpr(f, *args, **kwargs):
  flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
  flat_avals = [core.get_aval(x) for x in flat_args]
  debug_info = api_util.debug_info('make_jaxpr', f, args, kwargs)
  flat_fun, out_tree_thunk = api_util.flatten_fun(
      lu.wrap_init(f, debug_info=debug_info), in_tree
  )
  jaxpr, _, consts, _ = pe.trace_to_jaxpr_dynamic(flat_fun, flat_avals)
  out_tree = out_tree_thunk()
  return jaxpr, consts, in_tree, out_tree
