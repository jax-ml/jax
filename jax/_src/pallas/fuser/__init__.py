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

from jax._src.pallas.fuser.block_spec import get_fusion_values as get_fusion_values
from jax._src.pallas.fuser.block_spec import make_scalar_prefetch_handler as make_scalar_prefetch_handler
from jax._src.pallas.fuser.block_spec import pull_block_spec as pull_block_spec
from jax._src.pallas.fuser.block_spec import push_block_spec as push_block_spec
from jax._src.pallas.fuser.custom_evaluate import evaluate as evaluate
from jax._src.pallas.fuser.fusible import fusible as fusible
from jax._src.pallas.fuser.fusion import Fusion as Fusion
from jax._src.pallas.fuser.jaxpr_fusion import fuse as fuse
