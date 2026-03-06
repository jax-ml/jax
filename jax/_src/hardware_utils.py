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
from itertools import islice
import logging
import os
import pathlib
import glob
import re
import sys


logger = logging.getLogger(__name__)

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
  # TPU7x
  tpu7x = 7


_TPU_PCI_DEVICE_IDS = {
    '0x0027': TpuVersion.v3,
    '0x0056': TpuVersion.plc,
    '0x005e': TpuVersion.v4,
    '0x0062': TpuVersion.v5p,
    '0x0063': TpuVersion.v5e,
    '0x006f': TpuVersion.v6e,
    '0x0076': TpuVersion.tpu7x,
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


def count_amd_gpus(stop_at: int = sys.maxsize) -> int:
    """Count AMD GPUs available via KFD kernel driver.

    This function checks for the presence of AMD GPUs by examining KFD kernel
    driver entities as a proxy. In WSL setups, we detect the environment by
    checking for /dev/dxg (WSL indicator) and return 1 GPU. On native Linux,
    we count GPUs via sysfs KFD topology. This approach provides a good compromise
    between performance, reliability and simplicity.
    Presence of such entities doesn't guarantee that the GPUs are
    usable through HIP and PJRT, however, we can't do much better without
    spawning an additional process with a potentially complicated setup to run
    actual HIP code. And we don't want to initialize HIP right now inside the
    current process, because doing so might spoil a proper initialization of
    the rocprofiler-sdk later during PJRT startup.

    Args:
        stop_at: If provided, stop counting once this many GPUs are found.
                 This allows early exit when only checking for thresholds.

    Returns:
        The number of AMD GPUs detected (up to stop_at if provided).
    """
    gpu_count = 0
    try:
        # WSL detection: /dev/dxg indicates WSL, /dev/kfd indicates AMD GPU
        if os.path.exists("/dev/dxg"):
            return 1

        kfd_nodes_path = pathlib.Path("/sys/class/kfd/kfd/topology/nodes/")
        if not kfd_nodes_path.exists():
            return 0

        # Pattern matches "simd_count ##" - we use non-zero simd_count as GPU trait
        # following KFD implementation:
        # https://github.com/torvalds/linux/blob/ea1013c1539270e372fc99854bc6e4d94eaeff66/drivers/gpu/drm/amd/amdkfd/kfd_topology.c#L941
        r_simd_count = re.compile(r"\bsimd_count\s+(\d+)\b", re.MULTILINE)

        def is_valid_gpu_node(props_file):
            """Check if a properties file represents a valid GPU node."""
            try:
                file_size = props_file.stat().st_size
                # 16KB is more than a reasonable limit
                if file_size <= 0 or file_size > 16 * 1024:
                    return False
                content = props_file.read_text(encoding="ascii")
                match = r_simd_count.search(content)
                return match and int(match.group(1)) > 0
            except Exception as e:  # pylint: disable=broad-exception-caught
                logger.debug("Failed to read KFD node file '%s': %s", props_file, e)
                return False

        # Pipeline: list nodes -> get properties paths -> filter existing -> filter valid GPUs -> limit -> count
        props_files = (node_dir / "properties" for node_dir in kfd_nodes_path.iterdir())
        existing_files = filter(lambda p: p.exists(), props_files)
        valid_gpus = filter(is_valid_gpu_node, existing_files)
        limited_gpus = islice(valid_gpus, stop_at)
        gpu_count = sum(1 for _ in limited_gpus)

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Failed to count AMD GPUs: %s", e)
    return gpu_count


def get_shm_size():
    """Get /dev/shm size in MB.

    Returns:
        Size in MB if /dev/shm exists, None if it doesn't exist, 0 on error.
    """
    try:
        shm_path = "/dev/shm"
        if not os.path.exists(shm_path):
            return None

        stat = os.statvfs(shm_path)
        # Total size in bytes
        shm_size_bytes = stat.f_blocks * stat.f_frsize
        shm_size_mb = shm_size_bytes / (1024 * 1024)

        return shm_size_mb

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.warning("Failed to get shm size", e)
        return 0
