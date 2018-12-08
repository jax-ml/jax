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

import os

import ray

from .. import core
from ..util import partial
from ..core import AbstractTuple, JaxTuple
from ..abstract_arrays import make_shaped_array, array_types
from .partial_eval import trace_to_subjaxpr, merge_pvals, JaxprTrace, PartialVal

def abstractify(x):
  try:
    return pytype_aval_mappings[type(x)](x)
  except KeyError:
    raise TypeError("No abstraction handler for type: {}".format(type(x)))

pytype_aval_mappings = {}

def abstractify_tuple(tup):
  return AbstractTuple(tuple(map(abstractify, tup)))
pytype_aval_mappings[JaxTuple] = abstractify_tuple

for t in array_types:
  pytype_aval_mappings[t] = make_shaped_array


def ray_call_impl(fun, *args):
  compiled_fun = ray_callable(fun, *map(abstractify, args))
  return ray.get(ray.remote(compiled_fun).remote(*args))

def ray_callable(fun, *abstract_args):
  with core.new_master(JaxprTrace, True) as master:
    pvals = [PartialVal((aval, core.unit)) for aval in abstract_args]
    jaxpr, (pval, consts, env) = trace_to_subjaxpr(fun, master).call_wrapped(pvals)
    assert not env
    return lambda *args: ray_fun(jaxpr, consts, *args)

def ray_fun(jaxpr, consts, *args):
  import jax.numpy as np
  print(os.getpid())
  return core.eval_jaxpr(jaxpr, consts, (), *args)

ray_call_p = core.Primitive('ray_call')
ray_call = partial(core.call_bind, ray_call_p)
ray_call_p.def_custom_bind(ray_call)
ray_call_p.def_impl(ray_call_impl)
