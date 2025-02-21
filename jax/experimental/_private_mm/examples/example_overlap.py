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
"""An example showcasing overlap on a (forward-only) PP-like workload."""

from dataclasses import dataclass
from typing import Any, Callable
import time

import numpy as np

import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax.experimental import _private_mm as mm
from jax.experimental._private_mm import profile_utils
from jax.experimental._private_mm.examples import launch_utils


@dataclass(frozen=True)
class Stage:
    fwd: Callable[[Any, Any], Any]  # (params, acts) -> acts
    mesh: Mesh


def transfer(arr, stage):
    sharding = NamedSharding(stage.mesh, P())  # just replicate
    return mm.device_put(arr, device=sharding)


def stages_step_fn(stages, num_mubatches, params_by_stage, xs):
    # One task per mubatch and stage (e.g. forward stages only)
    tasks = [
        (mubatch_idx, stage_idx)
        for stage_idx in range(len(stages))
        for mubatch_idx in range(num_mubatches)
    ]
    # We want to be careful with the order in which we enqueue work, since
    # a single process is managing multiple devices.
    # Assuming a GPipe-like schedule we traverse tasks in the following order:
    #          t=0 t=1 t=2 t=3 t=4 t=5 t=6
    # stage=0    1   2   4   7
    # stage=1        3   5   8  11
    # stage=2            6   9  12  14
    # stage=3               10  13  15  16
    def task_key(task):
        mubatch_idx, stage_idx = task
        return (mubatch_idx + stage_idx, stage_idx)
    tasks.sort(key=task_key)

    input = {
        (mubatch_idx, 0): xs
        for mubatch_idx in range(num_mubatches)
    }

    for task_id in tasks:
        mubatch_idx, stage_idx = task_id
        stage = stages[stage_idx]
        params = params_by_stage[stage_idx]
        with profile_utils.annotate(
            f'mub{mubatch_idx}/F{stage_idx}',
            color='cyan',
        ):
            # Invoke the stage and immediately enqueue the transfer of the
            # result to the next stage. We want the transfer to be overlapped
            # with subsequent computation on the same stage.
            local_output = stage.fwd(params, input[task_id])
            if stage_idx + 1 < len(stages):
                with profile_utils.annotate(
                    f'Tx mub{mubatch_idx} to {stage_idx+1}',
                    color='yellow',
                ):
                    input[(mubatch_idx, stage_idx+1)] = transfer(
                        local_output,
                        stages[stage_idx+1],
                    )

    return local_output


### Example usage

def example_overlap(num_processes, process_id):
    assert jax.device_count() == 8

    NUM_STAGES = 4
    NUM_MUBATCHES = 4

    # FIXME: Support stages spread across multiple processes.
    assert NUM_STAGES % num_processes == 0

    # Takes ~5ms/stage/microbatch on H100s:
    LAYER_SIZE = 8192
    # # a) Several layers per stage, little communication (32MB activations)
    # NUM_LAYERS = NUM_STAGES * 16
    # BATCH_SIZE = 1024
    # b) One layer per stage, more communication (512MB activations)
    NUM_LAYERS = NUM_STAGES
    BATCH_SIZE = 1024 * 16


    def mlp(params, xs):
        for W in params:
            xs = xs @ W
        return xs

    def init_params(key):
        params = []
        for _ in range(NUM_LAYERS):
            key, key_W = jax.random.split(key)
            params.append(jax.random.normal(key_W, (LAYER_SIZE, LAYER_SIZE)))
        return params, key


    # Two devices per stage (running fully-replicated)
    num_devices_per_stage = 2
    stages = []
    for i in range(NUM_STAGES):
        devices = jax.devices()[
            num_devices_per_stage*i : num_devices_per_stage*(i+1)
        ]
        assert all(d.process_index == devices[0].process_index for d in devices)
        mesh = Mesh(np.asarray(devices), ('repl',))
        jitted_fun = mm.jit(
            mlp,
            in_shardings=(NamedSharding(mesh, P()), NamedSharding(mesh, P())),
            out_shardings=NamedSharding(mesh, P()),
        )
        stages.append(Stage(jitted_fun, mesh))

    def step_fn(params_by_stage, xs):
        return stages_step_fn(stages, NUM_MUBATCHES, params_by_stage, xs)


    def shard_params_by_stage(params):
        num_per_stage, rem = divmod(len(params), NUM_STAGES)
        assert num_per_stage > 0
        assert rem == 0
        params_by_stage = [
            jax.tree.map(
                lambda arr: transfer(arr, stages[stage_idx]),
                params[num_per_stage*stage_idx:num_per_stage*(stage_idx+1)],
            )
            for stage_idx in range(NUM_STAGES)
        ]
        return params_by_stage


    key = jax.random.PRNGKey(0)
    params, key = init_params(key)
    params_by_stage = shard_params_by_stage(params)

    key, key_xs = jax.random.split(key)
    xs_batch = jax.random.uniform(key_xs, (BATCH_SIZE, LAYER_SIZE))

    NUM_STEPS = 50
    NUM_STEPS_PROFILED = 3
    for i in range(NUM_STEPS):
        print(f'===== STEP {i} {process_id=} =====')
        if i == 1:
            # The overhead from compilations during warm-up ends up
            # staggering executions on devices of the same stage. The sleep
            # below allows them to catch up. In a real model collectives
            # within each stage would likely have the same effect of keeping
            # devices in sync.
            time.sleep(0.2)
        if i == NUM_STEPS - NUM_STEPS_PROFILED:
            profile_utils.maybe_start_profile(f"overlap_trace/p{process_id}")

        xs_batch = transfer(xs_batch, stages[0])
        with profile_utils.annotate(f'step{i}', color='white'):
            xs_batch = step_fn(params_by_stage, xs_batch)


if __name__ == '__main__':
    import sys
    num_processes = 4
    if len(sys.argv) >= 2:
        num_processes = int(sys.argv[1])
    success = launch_utils.launch_example(num_processes, example_overlap)
    sys.exit(0 if success else 1)
