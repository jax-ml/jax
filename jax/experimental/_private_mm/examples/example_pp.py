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
"""A toy model with MPMD pipeline parallelism."""

from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any, Callable

import numpy as np

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from jax.experimental import _private_mm as mm
from jax.experimental._private_mm import profile_utils
from jax.experimental._private_mm.examples import launch_utils


LR = 0.01


@dataclass(frozen=True)
class Stage:
    raw_fwd: Callable[[Any, Any, Any], Any]  # (params, acts, ys) -> acts
    mesh: Mesh
    params_specs: Any  # pytree of PartitionSpecs

    def sharding(self, spec):
        return NamedSharding(self.mesh, spec)

    def params_shardings(self):
        return jax.tree.map(self.sharding, self.params_specs)

    @cached_property
    def fwd(self):
        raw_fwd = self.raw_fwd

        @partial(
            mm.jit,
            in_shardings=(
                self.params_shardings(),
                self.sharding(P()),
                self.sharding(P()),
            ),
            out_shardings=self.sharding(P()),
        )
        def _fwd(params, acts, ys):
            return raw_fwd(params, acts, ys)

        return _fwd

    @cached_property
    def grad_init(self):
        @partial(
            mm.jit,
            in_shardings=(self.params_shardings(),),
            out_shardings=self.params_shardings(),
        )
        def _grad_init(params):
            return jax.tree.map(jnp.zeros_like, params)

        return _grad_init

    @cached_property
    def bwd_and_grad_acc(self):
        raw_fwd = self.raw_fwd

        @partial(
            mm.jit,
            in_shardings=(
                self.params_shardings(),
                self.sharding(P()),
                self.sharding(P()),
                self.params_shardings(),
                self.sharding(P()),
            ),
            out_shardings=(
                self.params_shardings(),
                self.sharding(P()),
            ),
        )
        def _bwd_and_grad_acc(params, fwd_activation, ys, grads_acc, activation):
            with jax.named_scope('bwd'):
                fwd_with_ys = lambda params, xs: raw_fwd(params, xs, ys)
                _, bwd = jax.vjp(fwd_with_ys, params, fwd_activation)
                grads, activation = bwd(activation)
            with jax.named_scope('grad-acc'):
                grads = jax.tree.map(jnp.add, grads_acc, grads)
            return grads, activation

        return _bwd_and_grad_acc

    @cached_property
    def update(self):
        @partial(
            mm.jit,
            in_shardings=(
                self.params_shardings(),
                self.params_shardings(),
            ),
            out_shardings=self.params_shardings(),
        )
        def _update(params, grads):
            return jax.tree.map(lambda v, dv: v - dv * LR, params, grads)

        return _update


def print_sharding(prefix, arr):
    sharding_str = str(arr.sharding)
    if hasattr(arr.sharding, 'mesh'):
        mesh_str = str(arr.sharding.mesh.devices).replace('\n', ' ')
        sharding_str += f' / {mesh_str}'
    print(f'{prefix} {sharding_str}')


def transfer(arr, stage, spec):
    sharding = stage.sharding(spec)
    return mm.device_put(arr, device=sharding)


def _mpmd_constant(stage, spec, shape, value, dtype=jnp.float32):
    # TODO: Better support for constants in mm (bake into jit executable?)
    return transfer(jnp.full(shape, value, dtype=dtype), stage, spec)


def stages_step_fn(stages, num_mubatches, params_by_stage, xs, ys):
    num_stages = len(stages)

    ### Schedule
    tasks = [
        (mubatch_idx, stage_idx, is_fwd)
        for stage_idx in range(num_stages)
        for mubatch_idx in range(num_mubatches)
        for is_fwd in (False, True)
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
        mubatch_idx, stage_idx, is_bwd = task
        if is_bwd:
            stage_idx = -stage_idx
        return (is_bwd, mubatch_idx + stage_idx, stage_idx)
    tasks.sort(key=task_key)

    ### State
    # fwd_input : (mubatch_idx, stage_idx) -> input/activation
    # TODO: Actually slice the input data into separate microbatches
    fwd_input = {
        (mubatch_idx, 0): xs
        for mubatch_idx in range(num_mubatches)
    }
    # bwd_input : (mubatch_idx, stage_idx) -> activation
    bwd_input = {
        (mubatch_idx, num_stages-1): _mpmd_constant(
            stages[-1], P(), shape=(), value=1.0)
        for mubatch_idx in range(num_mubatches)
    }
    # grads_by_stage : stage_idx -> grads
    grads_by_stage = []
    for stage_idx, stage in enumerate(stages):
        with profile_utils.annotate(f'grad-init{stage_idx}', color='red'):
            grads_by_stage.append(stage.grad_init(params_by_stage[stage_idx]))
    # loss : mubatch_idx -> loss
    # TODO: Add a leading mubatch dim to loss instead of making it a list
    loss = [None] * num_mubatches

    def maybe_ys(stage_idx):
        if stage_idx == num_stages-1:
            return ys
        else:
            return _mpmd_constant(stages[stage_idx], P(), shape=(), value=jnp.nan)

    ### Microbatched forward+backward
    for mubatch_idx, stage_idx, is_bwd in tasks:
        stage = stages[stage_idx]
        params = params_by_stage[stage_idx]
        fwd_bwd_str = 'B' if is_bwd else 'F'
        with profile_utils.annotate(
            f'mub{mubatch_idx}/{fwd_bwd_str}{stage_idx}', color='cyan'
        ):
            curr_id = (mubatch_idx, stage_idx)
            if not is_bwd:
                ### Forward
                succ_id = (mubatch_idx, stage_idx+1)
                activation = stage.fwd(
                    params,
                    fwd_input[curr_id],
                    maybe_ys(stage_idx),
                )
                if stage_idx+1 < num_stages:
                    with profile_utils.annotate(
                        f'Tx mub{mubatch_idx} to {stage_idx+1}', color='yellow',
                    ):
                        fwd_input[succ_id] = transfer(
                            activation,
                            stages[stage_idx+1],
                            P(),
                        )
                else:
                    loss[mubatch_idx] = activation
            else:
                ### Backward
                succ_id = (mubatch_idx, stage_idx-1)
                grads_by_stage[stage_idx], activation = stage.bwd_and_grad_acc(
                    params,
                    fwd_input.pop(curr_id),  # NB: Frees activation afterwards.
                    maybe_ys(stage_idx),
                    grads_by_stage[stage_idx],
                    bwd_input.pop(curr_id),  # NB: Frees activation afterwards.
                )
                if stage_idx-1 >= 0:
                    with profile_utils.annotate(
                        f'Tx mub{mubatch_idx} to {stage_idx-1}', color='yellow',
                    ):
                        bwd_input[succ_id] = transfer(
                            activation,
                            stages[stage_idx-1],
                            P(),
                        )

    ### Update params
    for stage_idx, stage in enumerate(stages):
        with profile_utils.annotate(f'U{stage_idx}', color='green'):
            params_by_stage[stage_idx] = stage.update(
                params_by_stage[stage_idx],
                grads_by_stage[stage_idx],
            )

    return loss, params_by_stage


### Example usage

def example_pp(num_processes, process_id):
    assert jax.device_count() == 8

    NUM_STAGES = 4
    NUM_MUBATCHES = 4

    # FIXME: Support stages spread across multiple processes.
    assert NUM_STAGES % num_processes == 0

    LAYER_SIZE = 8192
    # a) Several layers per stage, little communication (32MB activations)
    # NUM_LAYERS = NUM_STAGES * 16
    # BATCH_SIZE = 1024
    # b) One layer per stage, more communication (512MB activations)
    NUM_LAYERS = NUM_STAGES
    BATCH_SIZE = 1024 * 16

    ENABLE_TP = True


    @jax.jit
    def mlp(params, xs):
        for WA, WB in params:
            xs = xs @ WA @ WB
        return xs

    @jax.jit
    def mse(act, ys):
        return jnp.mean(jnp.square(act - ys))

    def init_params(key):
        params = []
        for _ in range(NUM_LAYERS):
            key, key_WA, key_WB = jax.random.split(key, 3)
            WA = jax.random.normal(key_WA, (LAYER_SIZE, LAYER_SIZE))
            WB = jax.random.normal(key_WB, (LAYER_SIZE, LAYER_SIZE))
            params.append((WA, WB))
        return params, key

    def shard_params_by_stage(params, stages):
        num_per_stage, rem = divmod(len(params), len(stages))
        assert num_per_stage > 0
        assert rem == 0
        params_by_stage = [
            jax.tree.map(
                lambda arr, spec: transfer(arr, stage, spec),
                params[num_per_stage*stage_idx:num_per_stage*(stage_idx+1)],
                stage.params_specs,
            )
            for stage_idx, stage in enumerate(stages)
        ]
        return params_by_stage


    # Define stages -- two devices per stage (running fully-replicated).
    num_devices_per_stage = jax.device_count() // NUM_STAGES
    stages = []
    for i in range(NUM_STAGES):
        devices = jax.devices()[num_devices_per_stage*i : num_devices_per_stage*(i+1)]
        assert all(d.process_index == devices[0].process_index for d in devices)
        mesh = Mesh(np.asarray(devices), ('model',))
        if i == NUM_STAGES - 1:
            fwd = lambda params, xs, ys: mse(mlp(params, xs), ys)
        else:
            fwd = lambda params, xs, _ys: mlp(params, xs)
        num_layers_per_stage = NUM_LAYERS // NUM_STAGES
        if ENABLE_TP:
            params_specs = [(P(None, 'model'), P('model', None))] * num_layers_per_stage
        else:
            params_specs = [(P(), P())] * num_layers_per_stage
        stages.append(Stage(fwd, mesh, params_specs))

    def step_fn(params_by_stage, xs, ys):
        return stages_step_fn(stages, NUM_MUBATCHES, params_by_stage, xs, ys)


    key = jax.random.PRNGKey(0)
    params, key = init_params(key)
    params_by_stage = shard_params_by_stage(params, stages)

    # Just keep reusing one batch, so we don't have to worry about infeed.
    key, key_xs = jax.random.split(key)
    xs_batch = jax.random.uniform(
        key_xs,
        (BATCH_SIZE, LAYER_SIZE),
    )
    ys_batch = 7 * xs_batch

    xs_batch = transfer(xs_batch, stages[0], P())
    ys_batch = transfer(ys_batch, stages[-1], P())

    NUM_STEPS = 50
    NUM_STEPS_PROFILED = 3
    for i in range(NUM_STEPS):
        print(f'===== STEP {i} {process_id=} =====')
        if i == NUM_STEPS - NUM_STEPS_PROFILED:
            profile_utils.maybe_start_profile(f"pp_trace/p{process_id}")

        with profile_utils.annotate(f'step{i}', color='white'):
            loss, params_by_stage = step_fn(params_by_stage, xs_batch, ys_batch)


if __name__ == '__main__':
    import sys
    num_processes = 4
    if len(sys.argv) >= 2:
        num_processes = int(sys.argv[1])
    success = launch_utils.launch_example(num_processes, example_pp)
    sys.exit(0 if success else 1)
