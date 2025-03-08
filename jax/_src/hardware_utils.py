# Copyright 2023 The JAX Authors.
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

import enum
import os
import pathlib
import glob

_GOOGLE_PCI_VENDOR_ID = '0x1ae0'

_NVIDIA_GPU_DEVICES = [
    '/dev/nvidia0',
    '/dev/nvidiactl',  # Docker/Kubernetes
    '/dev/dxg',  # WSL2
]


class TpuVersion(enum.IntEnum):
  # TPU v2, v3
  v2 = 0
  v3 = 1
  # No public name (plc)
  plc = 2
  # TPU v4
  v4 = 3
  # TPU v5p
  v5p = 4
  # TPU v5e
  v5e = 5
  # TPU v6e
  v6e = 6


_TPU_PCI_DEVICE_IDS = {
    '0x0027': TpuVersion.v3,
    '0x0056': TpuVersion.plc,
    '0x005e': TpuVersion.v4,
    '0x0062': TpuVersion.v5p,
    '0x0063': TpuVersion.v5e,
    '0x006f': TpuVersion.v6e,
}

def num_available_tpu_chips_and_device_id():
  """Returns the device id and number of TPU chips attached through PCI."""
  num_chips = 0
  tpu_version = None
  for vendor_path in glob.glob('/sys/bus/pci/devices/*/vendor'):
    vendor_id = pathlib.Path(vendor_path).read_text().strip()
    if vendor_id != _GOOGLE_PCI_VENDOR_ID:
      continue

    device_path = os.path.join(os.path.dirname(vendor_path), 'device')
    device_id = pathlib.Path(device_path).read_text().strip()
    if device_id in _TPU_PCI_DEVICE_IDS:
      tpu_version = _TPU_PCI_DEVICE_IDS[device_id]
      num_chips += 1

  return num_chips, tpu_version


def has_visible_nvidia_gpu() -> bool:
  """True if there's a visible nvidia gpu available on device, False otherwise."""

  return any(os.path.exists(d) for d in _NVIDIA_GPU_DEVICES)


def transparent_hugepages_enabled() -> bool:
  # See https://docs.kernel.org/admin-guide/mm/transhuge.html for more
  # information about transparent huge pages.
  path = pathlib.Path('/sys/kernel/mm/transparent_hugepage/enabled')
  return path.exists() and path.read_text().strip() == '[always] madvise never'
