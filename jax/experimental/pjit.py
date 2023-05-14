# Copyright 2021 The JAX Authors.
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

# flake8: noqa

from jax._src.pjit import (
  hashable_pytree as hashable_pytree,
  pjit as pjit,
  pjit_p as pjit_p,
  with_sharding_constraint as with_sharding_constraint,
)
from jax._src.sharding_impls import (
  AUTO as AUTO,
  UNSPECIFIED as _UNSPECIFIED,
  ParsedPartitionSpec as ParsedPartitionSpec,
  get_array_mapping as get_array_mapping,
  prepare_axis_resources as _prepare_axis_resources,
  parse_flatten_op_sharding as parse_flatten_op_sharding,
)

from jax._src.pjit import (_get_op_sharding_from_executable,
                           _get_pspec_from_executable, _pjit_lower_cached,
                           _pjit_lower, _pjit_jaxpr,
                           _process_in_axis_resources)

from jax._src.sharding_impls import (
  NamedSharding as _deprecated_NamedSharding,
)
from jax._src.partition_spec import (
  PartitionSpec as _deprecated_PartitionSpec,
)

import typing
if typing.TYPE_CHECKING:
  from jax._src.sharding_impls import NamedSharding as NamedSharding
  from jax._src.partition_spec import PartitionSpec as PartitionSpec
del typing

_deprecations = {
    # Added Feb 13, 2023:
    "NamedSharding": (
        (
            "jax.experimental.pjit.NamedSharding is deprecated. Use "
            "jax.sharding.NamedSharding."
        ),
        _deprecated_NamedSharding,
    ),
    "PartitionSpec": (
        (
            "jax.experimental.pjit.PartitionSpec is deprecated. Use "
            "jax.sharding.PartitionSpec."
        ),
        _deprecated_PartitionSpec,
    ),
    "FROM_GDA": (
        (
            "jax.experimental.pjit.FROM_GDA has been removed. Please pass in"
            " sharded jax.Arrays as input and remove the in_shardings argument"
            " to pjit since pjit will infer the shardings from jax.Array."
        ),
        None,
    ),
}

from jax._src.deprecations import deprecation_getattr as _deprecation_getattr
__getattr__ = _deprecation_getattr(__name__, _deprecations)
del _deprecation_getattr
