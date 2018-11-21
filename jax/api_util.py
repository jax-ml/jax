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

from .core import pack
from .tree_util import build_tree, process_pytree
from .linear_util import transformation_with_aux
from .util import safe_map, unzip2, partial

map = safe_map

@transformation_with_aux
def flatten_fun(in_trees, *args, **kwargs):
  py_args = map(build_tree, in_trees, args)
  ans = yield py_args
  yield process_pytree(pack, ans)


def unflatten_fun(fun, io_tree, *py_args):
  in_trees_expected, out_tree = io_tree
  args, in_trees = unzip2(map(tree_to_jaxtuples, py_args))
  for i, (in_tree, expected) in enumerate(zip(in_trees, in_trees_expected)):
    if in_tree != expected:
      raise TypeError("Expected {}, got {}".format(expected, in_tree))

  ans = fun(*args)
  return build_tree(out_tree, ans)


tree_to_jaxtuples = partial(process_pytree, pack)
