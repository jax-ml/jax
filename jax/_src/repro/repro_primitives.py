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

# Collect all the primitives we know about

from typing import Any


# Initialize the table of primitives from the lowering rule tables
primitives: dict[Any, str] = {}  # Map primitives to the name to use in repro
primitives_by_name: dict[str, Any] = {}


def _add_primitive(p: Any):
  if p in primitives: return
  use_name = p.name
  if use_name in primitives_by_name:
    # In some rare cases, we reuse names of primitives
    p_module = p.abstract_eval.__module__.split(".")[-1]
    use_name = f"{p_module}.{p.name}"
    if use_name in primitives_by_name:
      raise ValueError(f"Cannot pick a unique name for primitive {p}")

  primitives[p] = use_name
  primitives_by_name[use_name] = p


def populate():
  if primitives: return

  from jax._src.interpreters import mlir  # type: ignore  # noqa: F401

  # Import modules to populate the lowering rules

  # Some primitives do not have lowering rules
  from jax._src.interpreters import ad  # type: ignore
  for prim in ad.primitive_jvps:
    _add_primitive(prim)
  for prim in ad.primitive_linearizations:
    _add_primitive(prim)
  del ad

  for prim in mlir._lowerings:
    _add_primitive(prim)
  for platform_lowerings in mlir._platform_specific_lowerings.values():
    for prim in platform_lowerings:
      _add_primitive(prim)

  try:
    from jax._src.pallas import primitives as pallas_primitives # type: ignore  # noqa: F401
  except ImportError:
    pass
  try:
    from jax._src.pallas.mosaic_gpu import primitives as mosaic_gpu_primitives  # type: ignore  # noqa: F401
    _add_primitive(mosaic_gpu_primitives.wgmma_ref_p)  # No lowering?
  except ImportError:
    pass

  try:
    from jax._src.pallas.mosaic import lowering as mosaic_lowering  # type: ignore  # noqa: F401
    for by_core_type in mosaic_lowering.lowering_rules.values():
      for prim in by_core_type.keys():
        _add_primitive(prim)
  except ImportError:
    pass

  try:
    from jax._src.pallas.mosaic_gpu import lowering as mosaic_gpu_lowering  # type: ignore  # noqa: F401
    for prim_rules in mosaic_gpu_lowering.mosaic_lowering_rules.values():
      for prim in prim_rules.keys():
        _add_primitive(prim)
  except ImportError:
    pass
