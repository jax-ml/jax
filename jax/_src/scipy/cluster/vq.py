# Copyright 2022 The JAX Authors.
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

import operator

import scipy.cluster.vq
import textwrap

from jax import vmap
import jax.numpy as jnp
from jax._src.numpy.util import _wraps, check_arraylike, promote_dtypes_inexact


_no_chkfinite_doc = textwrap.dedent("""
Does not support the Scipy argument ``check_finite=True``,
because compiled JAX code cannot perform checks of array values at runtime
""")


@_wraps(scipy.cluster.vq.vq, lax_description=_no_chkfinite_doc, skip_params=('check_finite',))
def vq(obs, code_book, check_finite=True):
    check_arraylike("scipy.cluster.vq.vq", obs, code_book)
    if obs.ndim != code_book.ndim:
        raise ValueError("Observation and code_book should have the same rank")
    obs, code_book = promote_dtypes_inexact(obs, code_book)
    if obs.ndim == 1:
        obs, code_book = obs[..., None], code_book[..., None]
    if obs.ndim != 2:
        raise ValueError("ndim different than 1 or 2 are not supported")

    # explicitly rank promotion
    dist = vmap(lambda ob: jnp.linalg.norm(ob[None] - code_book, axis=-1))(obs)
    code = jnp.argmin(dist, axis=-1)
    dist_min = vmap(operator.getitem)(dist, code)
    return code, dist_min
