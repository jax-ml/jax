# Copyright 2018 The JAX Authors.
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

# The following are a subset of the full jax.numpy functionality used by
# internal imports.

from jax._src.numpy.array_constructors import (
    asarray as asarray,
    array as array,
)

from jax._src.numpy.array_creation import (
    full as full,
    ones as ones,
    zeros as zeros,
)

from jax._src.numpy.indexing import (
    take as take,
)

from jax._src.numpy.lax_numpy import (
    arange as arange,
    argmax as argmax,
    argmin as argmin,
    argsort as argsort,
    broadcast_to as broadcast_to,
    clip as clip,
    diag as diag,
    expand_dims as expand_dims,
    eye as eye,
    nonzero as nonzero,
    permute_dims as permute_dims,
    reshape as reshape,
    round as round,
    squeeze as squeeze,
    trace as trace,
    transpose as transpose,
    where as where,
)

from jax._src.numpy.reductions import (
    max as max,
)

from jax._src.numpy.tensor_contractions import (
    matmul as matmul,
)

from jax._src.numpy.ufuncs import (
    abs as abs,
    logical_and as logical_and,
)
