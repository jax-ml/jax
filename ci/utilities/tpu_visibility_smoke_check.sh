#!/bin/bash
# Copyright 2026 The JAX Authors.
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

function run_tpu_visibility_smoke_check() {
  local visibility_mode="${1:?}"
  local worker_count="${2:?}"
  local python_bin="${3:?}"

  if [[ "$visibility_mode" != "devices" ]]; then
    return 0
  fi

  echo "Checking per-worker TPU_VISIBLE_DEVICES visibility..."
  local i
  for ((i = 0; i < worker_count; i++)); do
    echo "=== TPU visibility smoke check: TPU_VISIBLE_DEVICES=$i ==="
    if ! env -u TPU_VISIBLE_CHIPS \
      TPU_VISIBLE_DEVICES="$i" \
      TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1,1 \
      TPU_PROCESS_BOUNDS=1,1,1,1 \
      ALLOW_MULTIPLE_LIBTPU_LOAD=true \
      JAX_PLATFORMS="${JAX_PLATFORMS:-tpu,cpu}" \
      "$python_bin" - <<'PY'; then
import os
import pprint

import numpy as np
import jax


_DEVICE_ATTRS = (
    'id',
    'process_index',
    'platform',
    'device_kind',
    'coords',
    'core_on_chip',
    'slice_index',
)


def _safe_value(value):
  if callable(value):
    try:
      return value()
    except TypeError:
      return '<callable>'
  return value


def _device_info(device):
  info = {'repr': repr(device)}
  for attr in _DEVICE_ATTRS:
    if hasattr(device, attr):
      info[attr] = _safe_value(getattr(device, attr))
  return info


def _array_device(array):
  device = getattr(array, 'device', None)
  if device is None:
    return None
  return _safe_value(device)


def _print_device_list(label, devices):
  print(f'{label}:')
  pprint.pp([_device_info(device) for device in devices], sort_dicts=True)


print('TPU_VISIBLE_DEVICES:', os.environ.get('TPU_VISIBLE_DEVICES'))
print('TPU_VISIBLE_CHIPS:', os.environ.get('TPU_VISIBLE_CHIPS'))
print(
    'TPU_CHIPS_PER_PROCESS_BOUNDS:',
    os.environ.get('TPU_CHIPS_PER_PROCESS_BOUNDS'),
)
print('TPU_PROCESS_BOUNDS:', os.environ.get('TPU_PROCESS_BOUNDS'))
print('default backend:', jax.default_backend())
print('process count:', jax.process_count())
print('process index:', jax.process_index())
print('device count:', jax.device_count())
devices = jax.devices()
local_devices = jax.local_devices()
print('len(jax.devices()):', len(devices))
print('len(jax.local_devices()):', len(local_devices))
_print_device_list('jax.devices()', devices)
_print_device_list('jax.local_devices()', local_devices)
addressable_devices = getattr(jax, 'addressable_devices', None)
if addressable_devices is None:
  print('jax.addressable_devices(): unavailable')
else:
  _print_device_list('jax.addressable_devices()', addressable_devices())
local_device_count = jax.local_device_count()
print('local device count:', local_device_count)
if local_device_count != 1:
  raise SystemExit(
      f'Expected exactly one local TPU device; got {local_device_count}'
  )
if len(local_devices) != 1:
  raise SystemExit(
      f'Expected exactly one local TPU device object; got {len(local_devices)}'
  )
probe = jax.device_put(np.arange(4, dtype=np.int32), local_devices[0])
probe.block_until_ready()
print('device_put local array device:', _device_info(_array_device(probe)))
print('device_put local array value:', np.asarray(probe).tolist())
PY
      echo "TPU visibility smoke check failed for TPU_VISIBLE_DEVICES=$i"
      return 1
    fi
  done

  echo "Checking concurrent TPU_VISIBLE_DEVICES visibility..."
  local tmp_dir
  tmp_dir="$(mktemp -d)"
  local ready_dir="${tmp_dir}/ready"
  mkdir -p "$ready_dir"
  local -a pids=()
  for ((i = 0; i < worker_count; i++)); do
    (
      env -u TPU_VISIBLE_CHIPS \
        TPU_VISIBLE_DEVICES="$i" \
        TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1,1 \
        TPU_PROCESS_BOUNDS=1,1,1,1 \
        ALLOW_MULTIPLE_LIBTPU_LOAD=true \
        JAX_PLATFORMS="${JAX_PLATFORMS:-tpu,cpu}" \
        TF_CPP_MIN_LOG_LEVEL=0 \
        TPU_STDERR_LOG_LEVEL=0 \
        JAX_TPU_CORE_SPLIT_READY_DIR="$ready_dir" \
        JAX_TPU_CORE_SPLIT_WORKER_COUNT="$worker_count" \
        "$python_bin" - <<'PY'
import os
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np


def _device_info(device):
  return {
      'repr': repr(device),
      'id': getattr(device, 'id', None),
      'process_index': getattr(device, 'process_index', None),
      'coords': getattr(device, 'coords', None),
      'core_on_chip': getattr(device, 'core_on_chip', None),
  }


visible_devices = os.environ['TPU_VISIBLE_DEVICES']
worker_count = int(os.environ['JAX_TPU_CORE_SPLIT_WORKER_COUNT'])
ready_dir = pathlib.Path(os.environ['JAX_TPU_CORE_SPLIT_READY_DIR'])
print('concurrent smoke TPU_VISIBLE_DEVICES:', visible_devices, flush=True)
print('TPU_VISIBLE_CHIPS:', os.environ.get('TPU_VISIBLE_CHIPS'), flush=True)
print(
    'TPU_CHIPS_PER_PROCESS_BOUNDS:',
    os.environ.get('TPU_CHIPS_PER_PROCESS_BOUNDS'),
    flush=True,
)
print('TPU_PROCESS_BOUNDS:', os.environ.get('TPU_PROCESS_BOUNDS'), flush=True)
print('default backend:', jax.default_backend(), flush=True)
print('process count:', jax.process_count(), flush=True)
print('process index:', jax.process_index(), flush=True)
local_devices = jax.local_devices()
print('local devices:', [_device_info(d) for d in local_devices], flush=True)
if len(local_devices) != 1:
  raise SystemExit(
      f'Expected exactly one local TPU device; got {len(local_devices)}'
  )


@jax.jit
def _compute(x):
  return jnp.sum((x + 1.0) * (x + 2.0))


x = jax.device_put(np.arange(16, dtype=np.float32), local_devices[0])
result = _compute(x).block_until_ready()
actual = float(jax.device_get(result))
expected = float(sum((i + 1) * (i + 2) for i in range(16)))
print('concurrent smoke compute result:', actual, flush=True)
if actual != expected:
  raise SystemExit(f'Expected compute result {expected}; got {actual}')

(ready_dir / visible_devices).write_text('ready', encoding='utf-8')
deadline = time.time() + 60
while len(list(ready_dir.iterdir())) < worker_count:
  if time.time() > deadline:
    raise SystemExit('Timed out waiting for other TPU workers')
  time.sleep(0.1)

time.sleep(5)
print('concurrent smoke worker finished', flush=True)
PY
    ) >"${tmp_dir}/worker_${i}.log" 2>&1 &
    pids[$i]=$!
  done

  local failed=0
  for ((i = 0; i < worker_count; i++)); do
    if ! wait "${pids[$i]}"; then
      failed=1
    fi
  done

  for ((i = 0; i < worker_count; i++)); do
    echo "=== Concurrent TPU visibility smoke check log: TPU_VISIBLE_DEVICES=$i ==="
    if [[ -f "${tmp_dir}/worker_${i}.log" ]]; then
      while IFS= read -r line; do
        echo "[TPU_VISIBLE_DEVICES=$i] $line"
      done <"${tmp_dir}/worker_${i}.log"
    fi
  done

  rm -rf "$tmp_dir"
  if [[ "$failed" -ne 0 ]]; then
    echo "Concurrent TPU visibility smoke check failed"
    return 1
  fi
}
