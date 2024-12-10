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
"""Module for the control flow primitives."""
# Private utilities used elsewhere in JAX
# TODO(sharadmv): lift them into a more common place
from jax._src.lax.control_flow.common import (
    _check_tree_and_avals as _check_tree_and_avals,
    _initial_style_jaxpr as _initial_style_jaxpr,
    _initial_style_jaxprs_with_common_consts as _initial_style_jaxprs_with_common_consts,
    _initial_style_open_jaxpr as _initial_style_open_jaxpr,
)
from jax._src.lax.control_flow.conditionals import (
    cond_p as cond_p,
    cond as cond,
    platform_dependent as platform_dependent,
    platform_index_p as platform_index_p,
    switch as switch,
)
from jax._src.lax.control_flow.loops import (
    _scan_impl as _scan_impl,
    associative_scan as associative_scan,
    cumlogsumexp_p as cumlogsumexp_p,
    cumlogsumexp as cumlogsumexp,
    cummax_p as cummax_p,
    cummax as cummax,
    cummin_p as cummin_p,
    cummin as cummin,
    cumprod_p as cumprod_p,
    cumprod as cumprod,
    cumred_reduce_window_impl as cumred_reduce_window_impl,
    cumsum_p as cumsum_p,
    cumsum as cumsum,
    fori_loop as fori_loop,
    map as map,
    scan_p as scan_p,
    scan as scan,
    while_loop as while_loop,
    while_p as while_p,
)
from jax._src.lax.control_flow.solves import (
    _custom_linear_solve_impl as _custom_linear_solve_impl,
    custom_linear_solve as custom_linear_solve,
    custom_root as custom_root,
    linear_solve_p as linear_solve_p,
)
# TODO(mattjj): fix dependent library which expects optimization_barrier_p here
from jax._src.lax.lax import optimization_barrier_p as optimization_barrier_p
