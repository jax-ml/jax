# Copyright 2026 The JAX Authors.
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

from jax._src.random import philox2x32 as philox2x32
from jax._src.random import philox4x32 as philox4x32
from jax._src.random import rbg as rbg
from jax._src.random import threefry2x32 as threefry2x32
from jax._src.random import threefry4x32 as threefry4x32
from jax._src.random.core import (
    PRNGKey as PRNGKey,
    ball as ball,
    bernoulli as bernoulli,
    binomial as binomial,
    beta as beta,
    bits as bits,
    categorical as categorical,
    cauchy as cauchy,
    chisquare as chisquare,
    choice as choice,
    clone as clone,
    dirichlet as dirichlet,
    double_sided_maxwell as double_sided_maxwell,
    exponential as exponential,
    f as f,
    fold_in as fold_in,
    gamma as gamma,
    generalized_normal as generalized_normal,
    geometric as geometric,
    gumbel as gumbel,
    key as key,
    key_data as key_data,
    key_dtype as key_dtype,
    key_impl as key_impl,
    laplace as laplace,
    logistic as logistic,
    loggamma as loggamma,
    lognormal as lognormal,
    maxwell as maxwell,
    multinomial as multinomial,
    multivariate_normal as multivariate_normal,
    normal as normal,
    orthogonal as orthogonal,
    pareto as pareto,
    permutation as permutation,
    poisson as poisson,
    rademacher as rademacher,
    randint as randint,
    random_gamma_p as random_gamma_p,
    rayleigh as rayleigh,
    split as split,
    t as t,
    triangular as triangular,
    truncated_normal as truncated_normal,
    uniform as uniform,
    wald as wald,
    weibull_min as weibull_min,
    wrap_key_data as wrap_key_data,
    # Type aliases
    Shape as Shape,
    RealArray as RealArray,
    IntegerArray as IntegerArray,
    DTypeLikeInt as DTypeLikeInt,
    DTypeLikeUInt as DTypeLikeUInt,
    DTypeLikeFloat as DTypeLikeFloat,
    KeyDTypeLike as KeyDTypeLike,
    # Internal APIs
    resolve_prng_impl as resolve_prng_impl,
    PRNGSpec as PRNGSpec,
    PRNGSpecDesc as PRNGSpecDesc,
    default_prng_impl as default_prng_impl,
    random_clone_p as random_clone_p,
)
