# Lint as: python3
# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utils for building a device mesh."""

import itertools
from typing import Any, Dict, List, Sequence, Tuple

from absl import logging
import jax
import numpy as np

_TPU_V3 = 'TPU v3'
_TPU_V4 = 'TPU v4'

# Maps physical topology -> mesh shape -> transpose to use for jekbradbury's
# famous contiguous mesh trick.
#
# The trick only works for certain topologies and mesh shapes, which are chosen
# to be representative of large language model workloads (the three axes are
# [replica, data, model]).
_TRANSPOSE_TRICKS: Dict[Tuple[int, ...],
                        Dict[Tuple[int, ...], Tuple[int, ...]]] = {
    (2, 2, 1): {
        (1, 2, 2): (0, 1, 2),
        (2, 2): (0, 1, 2),
    },
    (2, 2, 4): {
        (1, 4, 4): (0, 1, 2),
        (4, 4): (0, 1, 2),
    },
    (4, 4, 4): {
        (1, 16, 4): (0, 2, 1),
        (16, 4): (0, 2, 1),
    },
    (4, 8, 8): {
        (1, 64, 4): (0, 2, 1),
        (1, 4, 64): (0, 2, 1),
        (64, 4): (0, 2, 1),
        (4, 64): (0, 2, 1),
    },
    (8, 8, 8): {
        (1, 64, 8): (0, 2, 1),
        (64, 8): (0, 2, 1),
    },
    (8, 16, 16): {
        (1, 256, 8): (0, 2, 1),
        (1, 8, 256): (0, 2, 1),
        (256, 8): (0, 2, 1),
        (8, 256): (0, 2, 1),
    },
}


def _create_device_mesh_for_tpu_v4(
    physical_mesh: np.ndarray, mesh_shape: Sequence[int]
) -> Tuple[np.ndarray, Tuple[Tuple[int, ...], ...]]:
  """Creates a performant device mesh for jax.experimental.maps.mesh.

  Creates a device mesh for a given physical mesh and logical mesh that allows
  for fast XLA collectives for use with jax.experimental.maps.mesh.

  Given a logical mesh `mesh_shape` and a physical topology `physical_mesh`, we
  need to map each logical mesh axis to one or more physical axes. We want to
  preferentially map the logical axes with the highest network intensity (the
  highest collective communication costs) to the physical axes with the highest
  network bandwidth.

  Let's use a concrete example to explain the concepts and considerations.

  As an example, suppose the mesh_shape is [replica, data, mdl] where batch dim
  is split over [replica, data] and model dims are split over either data or
  mdl axis, then replica has the least network intensity and mdl the highest.

  For a TPU pod, due to uniform ICI bandwidth, a physical mesh of 4x4x16 will
  have uniform bandwidth on any of the three axes. However, the 2D x-y plane of
  4x4 may have faster XLA collective implementations. Suppose the mesh_shape is
  [1, 16, 16], we may want the mdl axis to be mapped to 2x2 x-y plane rather
  than the single axis of size 16. To account for this, we preferentially map to
  2D plane first when considering higher network intensity logical axis.

  Args:
    physical_mesh: a np.ndarray with the shape of the physical topology.
    mesh_shape: shape of logical mesh, ordered by increasing network-intensity
      e.g. [replica, data, mdl] or [data, mdl] where mdl has the most network
      communication requirements.

  Returns:
    A device mesh with mesh_shape as its shape.
    An assignment map each logical mesh axis to a subset of physical axes.
  """
  # Remaining physical axes to be assigned to logical axes.
  assignable_physical_mesh = list(physical_mesh.shape)
  # Map each logical axis to a subsets of physical axes.
  assignment: List[Tuple[int, ...]] = [() for _ in mesh_shape]

  # Assign logical axes from highest network intensity to lowest.
  # `mesh_shape` is assumed to ordered by lowest network intensity first, so
  # reverse it first.
  for logical_axis_index, logical_axis_size in reversed(
      list(enumerate(mesh_shape))):
    # Preferentially map to 2D subplane first for higher bandwidth.
    for num_axes in range(2, 0, -1):
      # Try assign to any subset of size num_axes. Generate all candidates.
      axes = itertools.combinations(assignable_physical_mesh, num_axes)
      indices = itertools.combinations(
          range(len(assignable_physical_mesh)), num_axes)
      # Go through all candidates, 2D plane first.
      for c_axes, c_indices in zip(axes, indices):
        # TODO(zhangqiaorjc): Due to limitations in XLA, 2D collectives only
        # implemented for square 2D plane. Mapping a physical axis to two
        # logical axes might be slower for non-square 2D plane, e.g., map 32 to
        # 4x8 or a single axis. If XLA 2D collectives support non-square plane
        # soon, we can continue to preferentially map to 2D plane in general,
        # otherwise, we should avoid non-square 2D plane.
        if np.product(c_axes) == logical_axis_size:
          assignment[logical_axis_index] = c_indices
          # Zero the assigned physical axes.
          assignable_physical_mesh = [
              0 if i in c_indices else v
              for i, v in enumerate(assignable_physical_mesh)
          ]
          break
      if assignment[logical_axis_index]:
        # We already found an assignment from one candidate above.
        break
    else:
      # If the num_axes for loop did not break, i.e. none of the candidates work
      # goto here with this while-else construct.
      if logical_axis_size > 1:
        raise NotImplementedError(f'Failed to find assignment for '
                                  f'logical_axis_index {logical_axis_index}')
  # Flatten the assignment, e.g., [(), (2,), (0, 1)] -> (2, 0, 1).
  transpose: List[int] = []
  for x in assignment:
    for y in x:
      transpose.append(int(y))
  return physical_mesh.transpose(transpose).reshape(mesh_shape), tuple(
      assignment)


def _bounds_from_last_device(last_device) -> Sequence[int]:
  """Gets the bound from the given last device."""
  # Must be passed the device at the highest-coordinate corner of the
  # relevant mesh, which is a requirement we know is satisfied by the last
  # device in jax.devices().
  assert hasattr(last_device, 'coords'), 'Only TPU supported'
  x, y, z = last_device.coords
  return x + 1, y + 1, z + 1, last_device.core_on_chip + 1


def _jax_devices_order_normalized(
    jax_local_devices_from_process_0: Sequence[Any],
    jax_devices: Sequence[Any]) -> np.ndarray:
  r"""Normalize jax.devices() to an order untiled by host and minor in z.

  Args:
    jax_local_devices_from_process_0: A list of jax devices, which is a
      flattened list from jax.local_devices(process_index=0).
    jax_devices: A list of jax devices, which is a flattened list from
      jax.devices().

  Returns:
    A np.ndarray of jax devices with shape [global_x, global_y, global_z].

  """
  local_topology = _bounds_from_last_device(
      jax_local_devices_from_process_0[-1])
  # h_x, h_y can be 2x2 or 1x1 depending on tasks_per_host=4 or 1
  h_x, h_y, _, cores_per_chip = local_topology
  assert cores_per_chip == 1
  physical_topology = _bounds_from_last_device(jax_devices[-1])
  g_x, g_y, g_z, cores_per_chip = physical_topology
  assert cores_per_chip == 1
  assert g_x % h_x == 0 and g_y % h_y == 0

  jax_devices_array = np.array(jax_devices).reshape(
      (g_z, g_y // h_y, g_x // h_x, h_y, h_x, cores_per_chip))
  jax_devices_array = jax_devices_array.transpose(0, 1, 3, 2, 4, 5)
  jax_devices_array = jax_devices_array.reshape((g_z, g_y, g_x))
  # Transpose to be [global_x, global_y, global_z]
  return jax_devices_array.transpose()


# jekbradbury's famous trick for creating contiguous submeshes (where available)
def _transpose_trick(physical_mesh: np.ndarray,
                     mesh_shape: Sequence[int]) -> np.ndarray:
  mesh_shape = tuple(mesh_shape)
  topology = physical_mesh.shape
  if topology not in _TRANSPOSE_TRICKS:
    raise ValueError(
        f"create_device_mesh cannot create contiguous submeshes for "
        f"physical mesh topology {topology}")

  if mesh_shape not in _TRANSPOSE_TRICKS[topology]:
    raise ValueError(
        f"create_device_mesh cannot create contiguous submeshes for "
        f"mesh_shape {mesh_shape} and physical mesh topology {topology}. "
        f"Available mesh_shapes: {list(_TRANSPOSE_TRICKS[topology].keys())}")

  return physical_mesh.transpose(*_TRANSPOSE_TRICKS[topology][mesh_shape])


def create_device_mesh(mesh_shape: Sequence[int],
                       contiguous_submeshes: bool = False) -> np.ndarray:
  """Creates a performant device mesh for jax.experimental.maps.mesh.

  Args:
    mesh_shape: shape of logical mesh, ordered by increasing network-intensity
      e.g. [replica, data, mdl] where mdl has the most network communication
      requirements.
    contiguous_submeshes: if True, this function will attempt to create a mesh
      where each process's local devices form a contiguous submesh. This is
      required when passing non-GlobalDeviceArrays to `pjit` (see the
      "Multi-process platforms" note of the [pjit
      documentation](https://jax.readthedocs.io/en/latest/jax.experimental.pjit.html)
      for more information on this constraint). A ValueError will be raised if
      this function can't produce a suitable mesh.

  Returns:
    A np.ndarray of jax global devices with mesh_shape as its shape that can be
    fed into jax.experimental.maps.mesh with good collective performance.

  """
  process_0_devices = jax.local_devices(process_index=0)
  global_devices = jax.devices()
  device_kind = global_devices[-1].device_kind
  return _create_device_mesh(process_0_devices, global_devices, device_kind,
                             mesh_shape, contiguous_submeshes)

# Break out core logic for easier testing
def _create_device_mesh(process_0_devices, global_devices, device_kind,
                        mesh_shape: Sequence[int],
                        contiguous_submeshes: bool = False) -> np.ndarray:
  # TODO(zhangqiaorjc): Handle TPU versions other than v4 more generally.
  if device_kind == _TPU_V3:
    device_mesh = np.asarray(global_devices).reshape(mesh_shape)
    if mesh_shape[-1] == 8:
      logging.info('Re-order TPUv3 device mesh for better performance.')
      perm = np.array([0, 1, 2, 3, 6, 7, 4, 5])
      device_mesh = device_mesh[:, :, perm]
    return device_mesh
  elif device_kind == _TPU_V4:
    physical_mesh = _jax_devices_order_normalized(
        process_0_devices, global_devices)
    if contiguous_submeshes:
      physical_mesh = _transpose_trick(physical_mesh, mesh_shape)
    device_mesh, assignment = _create_device_mesh_for_tpu_v4(
        physical_mesh, mesh_shape)
    logging.info('_create_device_mesh_for_tpu_v4 assignment: %s', assignment)
    return device_mesh
  else:
    device_mesh = np.asarray(global_devices).reshape(mesh_shape)
    return device_mesh
