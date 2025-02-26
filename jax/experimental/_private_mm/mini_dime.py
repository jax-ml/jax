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
"""
Explicit NCCL communicators (via CuPy) integrated with JAX. Based on the
communication library used in JaxPP (https://arxiv.org/abs/2412.14374).
Requires `pip install cupy-cuda12x`.
"""

import enum
import pickle
from collections import OrderedDict
from functools import cached_property

try:
    import cupy  # type: ignore[import-not-found]
    from cupy.cuda import nccl  # type: ignore[import-not-found]

    # CuPy NCCL utils from https://github.com/cupy/cupy/blob/118ade4a146d1cc68519f7f661f2c145f0b942c9/cupyx/distributed/_nccl_comm.py#L46-L55
    _nccl_dtypes = {
        "b": nccl.NCCL_INT8,
        "B": nccl.NCCL_UINT8,
        "i": nccl.NCCL_INT32,
        "I": nccl.NCCL_UINT32,
        "l": nccl.NCCL_INT64,
        "L": nccl.NCCL_UINT64,
        "q": nccl.NCCL_INT64,
        "Q": nccl.NCCL_UINT64,
        "e": nccl.NCCL_FLOAT16,
        "f": nccl.NCCL_FLOAT32,
        "d": nccl.NCCL_FLOAT64,
        # Size of array will be doubled
        "F": nccl.NCCL_FLOAT32,
        "D": nccl.NCCL_FLOAT64,
    }
except ImportError:
    cupy = None
    nccl = None

import jax
import jax.numpy as jnp
import jaxlib.xla_extension as xe
from jax._src import array
from jax._src.op_shardings import are_op_shardings_equal


def _get_nccl_dtype_and_count(arr, count=None):
    dtype = arr.dtype.char
    if dtype not in _nccl_dtypes:
        raise TypeError(f"Unknown dtype {arr.dtype} for NCCL")
    nccl_dtype = _nccl_dtypes[dtype]
    if count is None:
        count = arr.size
    if dtype in "FD":
        return nccl_dtype, 2 * count
    return nccl_dtype, count


def get_distributed_client() -> xe.DistributedRuntimeClient:
    from jax._src.distributed import global_state

    assert isinstance(global_state.client, xe.DistributedRuntimeClient)
    return global_state.client


class UniqueDevices(tuple[jax.Device, ...]):
    def __new__(cls, *args):
        return super().__new__(cls, sorted(set(args), key=lambda d: d.id))

    @cached_property
    def ranks(self):
        return OrderedDict((d, idx) for idx, d in enumerate(self))

    @property
    def leader(self):
        return self[0]

    @cached_property
    def key(self) -> str:
        return ",".join(str(d.id) for d in self)


local_comms: dict = {}


def get_or_create_comm(devs: UniqueDevices):
    TIMEOUT = 5_000

    comm = local_comms.get(devs)
    my_process_index = jax.process_index()
    if comm is None:
        if devs.leader.process_index == my_process_index:
            nccl_id = nccl.get_unique_id()
            get_distributed_client().key_value_set_bytes(
                devs.key, pickle.dumps(nccl_id)
            )
        else:
            nccl_id = get_distributed_client().blocking_key_value_get_bytes(
                devs.key, TIMEOUT
            )
            nccl_id = pickle.loads(nccl_id)

        nccl.groupStart()
        for d in devs:
            if d.process_index == my_process_index:
                with cupy.cuda.Device(d.local_hardware_id):
                    comm = nccl.NcclCommunicator(len(devs), nccl_id, devs.ranks[d])
        nccl.groupEnd()

        local_comms[devs] = comm
    return comm


local_streams: dict = {}


class OpT(enum.Enum):
    SEND = 0
    RECV = 1


def get_or_create_stream(op: OpT, local_device: jax.Device):
    # XXX: I think this can be one stream per local_device for this specific example.
    #  It depends on the use case
    stream = local_streams.get((op, local_device))
    if stream is None:
        with cupy.cuda.Device(local_device.local_hardware_id):
            stream = cupy.cuda.Stream(non_blocking=True)
        local_streams[local_device] = stream
    return stream


def shardings_are_compatible(
    self: jax.sharding.Sharding, other: jax.sharding.Sharding, ndim: int
):
    # NOTE: Variant of `jax.sharding.Sharding.is_equivalent_to` that skips _internal_device_list check
    return (
        are_op_shardings_equal(
            self._to_xla_hlo_sharding(ndim), other._to_xla_hlo_sharding(ndim)
        )
        # and self._internal_device_list == other._internal_device_list  # type: ignore
        and self.memory_kind == other.memory_kind
    )


## API


def send_or_recv(
    x: jax.Array,
    tgt_sharding: jax.sharding.Sharding,
    src_sharding: jax.sharding.Sharding | None = None,
):
    """
    When `src_sharding is None` this function corresponds to a send and
    `x.sharding` must be equal to `tgt_sharding`.
    When `src_sharding is not None` this function corresponds to a receive
    and `x` will be consumed, i.e. it's unsafe to use `x` after `send_or_recv(x, src_sharding=...)`.

    `x` can be a "global" array spanning multiple processes/hosts.
    In that case, this process will send/receive only its corresponding addressable_shards
    """

    if src_sharding is None:
        is_send = True
        other_sharding = tgt_sharding
    else:
        is_send = False
        other_sharding = src_sharding

    if not is_send:
        # XXX: x.sharding and tgt_sharding must be equal since this is a recv.
        # This seems redundant to me. Not sure what the final version from Skye
        # will look like.
        assert x.sharding == tgt_sharding

    # TODO: implement reshard for 4 devs -> 2 devs or 2->4 reshards
    assert shardings_are_compatible(x.sharding, other_sharding, x.ndim), \
        f'incompatible shardings: {x.sharding=} vs {other_sharding=}'

    # Create communicators lazily as needed. This can be a separate "setup function"
    for pair in zip(
        x.sharding._device_assignment,
        other_sharding._device_assignment,
        strict=True,
    ):
        if pair[0].process_index == jax.process_index():
            _ = get_or_create_comm(UniqueDevices(*pair))

    shards_by_device = {shard.device: shard for shard in x.addressable_shards}

    cpy_arrays_and_streams = []
    # FIXME: maybe narrow `nccl_group_{start,end}` scope by first accumulating
    # arguments in a list and then performing the operation
    nccl.groupStart()
    for x_device, other_device in zip(
        x.sharding._device_assignment,
        other_sharding._device_assignment,
        strict=True,
    ):
        if x_device.process_index == jax.process_index():
            shard = shards_by_device[x_device]
            stream = get_or_create_stream(OpT.SEND if is_send else OpT.RECV, x_device)
            # FIXME: cupy doesn't support bf16. Use capsule/ctype APIs
            cpy_arr = cupy.from_dlpack(
                jax.dlpack.to_dlpack(shard.data, stream=stream.ptr)
            )
            cpy_arrays_and_streams.append((cpy_arr, stream))

            nccl_dtype, count = _get_nccl_dtype_and_count(cpy_arr)

            key = UniqueDevices(x_device, other_device)
            comm = get_or_create_comm(key)

            with cupy.cuda.Device(x_device.local_hardware_id):
                op = comm.send if is_send else comm.recv
                op(
                    cpy_arr.data.ptr,
                    count,
                    nccl_dtype,
                    key.ranks[other_device],
                    stream.ptr,
                )

    nccl.groupEnd()
    # NOTE: since communicators are blocking, after the group_end operation
    #  above, all the send/recvs have been enqueued into the stream. Therefore,
    #  we can record events on the stream

    # XXX: I don't like different return types below, however I am not sure
    #  what's a better alternative given we want a "symmetric"
    # `send_or_recv` API
    if is_send:

        def wait():
            for _, stream in cpy_arrays_and_streams:
                stream.synchronize()
            # NOTE: Keep the objects below alive just in case they are not
            #  deleted/overwritten by XLA while in use
            return (x, cpy_arrays_and_streams)

        return wait
    else:

        def enqueue_wait():
            jax_single_arrays = []
            for x_device, (cpy_arr, stream) in zip(
                x.sharding._device_assignment, cpy_arrays_and_streams, strict=True
            ):
                with cupy.cuda.Device(x_device.local_hardware_id):
                    event = stream.record()
                    ready_events_stream = (
                        x_device.get_stream_for_external_ready_events()
                    )
                    cupy.cuda.ExternalStream(ready_events_stream).wait_event(event)
                    jax_sda = jnp.array(
                        jax._src.lib.xla_client._xla.dlpack_managed_tensor_to_buffer(
                            cpy_arr.toDlpack(),
                            x_device,
                            ready_events_stream,
                        ),
                        copy=True,  # XXX: Just to be safe
                    )
                    jax_single_arrays.append(jax_sda)
            return array.ArrayImpl(
                x.aval,
                x.sharding,
                jax_single_arrays,
                committed=True,
                # NOTE: _skip_checks can be set to True however since this happens
                #  asynchronously there's no perf harm to keep it False.
                _skip_checks=False,
            )

        return enqueue_wait
