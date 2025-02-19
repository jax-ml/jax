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
"""Utilities to launch multi-process JAX examples on a single host."""

from functools import partial
import multiprocessing


def init_multi_process_local(num_processes, process_id, num_devices):
    assert 0 <= process_id < num_processes

    # Assume all processes run on a single node
    assert num_devices % num_processes == 0
    num_devices_per_process = num_devices // num_processes
    local_device_ids = [
        process_id*num_devices_per_process + i
        for i in range(num_devices_per_process)
    ]

    import jax
    jax.distributed.initialize(
        coordinator_address="localhost:1234",
        num_processes=num_processes,
        process_id=process_id,
        local_device_ids=local_device_ids,
    )


# Needs to be a top-level function to pickle as part of multiprocessing.
def _entrypoint(num_processes, process_id, user_main, num_devices):
    # Only import these in the subprocess, not the launcher process.
    import jax.experimental.multihost_utils
    from jax.experimental._private_mm import profile_utils

    init_multi_process_local(num_processes, process_id, num_devices)
    jax.experimental.multihost_utils.sync_global_devices("start_user_main")
    user_main(num_processes, process_id)
    profile_utils.maybe_stop_profile()


def launch_example(num_processes, user_main, num_devices=8):
    """
    A launcher for examples running across multiple processes on a single node.
    Returns true iff all processes exited successfully.

    Example code my_example.py:
        def my_example(num_processes, process_id):
            # Do some distributed JAX stuff.
            ...

        if __name__ == '__main__':
            import sys
            num_processes = int(sys.argv[1])
            launch_utils.launch_example(num_processes, my_example)

    Usage:
        # Run without profiling
        my_example.py 4
        # Run with jax.profiler + annotations
        PROFILE=jax python3 my_example.py 4
        # Run with nsys profiling + annotations
        PROFILE=nsys nsys profile --output my_example.nsys-rep --cpuctxsw=none --trace=cublas,cuda,cudnn,cusolver,nvtx,osrt,python-gil --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node --python-sampling=true python3 my_example.py 4
    """
    assert num_processes > 0
    # Spawn subprocesses to avoid timeouts when profiling using nsys.
    ctx = multiprocessing.get_context('spawn')
    ps = [
        ctx.Process(
            target=partial(
                _entrypoint,
                num_processes,
                process_id,
                user_main,
                num_devices,
            ),
            name=f'example_proc{process_id}',
        )
        for process_id in range(num_processes)
    ]
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    return all(p.exitcode == 0 for p in ps)
