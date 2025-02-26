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
"""Poor-man's MPMD for JAX."""

from dataclasses import dataclass
from functools import cached_property, lru_cache, partial, wraps

from typing import Callable

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding, Sharding, SingleDeviceSharding

from jax._src.tree_util import broadcast_prefix, prefix_errors, tree_leaves_with_path

from jax.experimental._private_mm import mini_dime


@dataclass
class MpmdArray:
    """A generalization of jax.Array that also supports fully remote arrays."""
    aval: jax.core.ShapedArray
    sharding: Sharding
    _complete: Callable[[], jax.Array | tuple] | None
    _result: jax.Array | tuple | None = None

    def __repr__(self):
        remote_str = ', fully-remote' if self.is_fully_remote else ''
        return (
            f'MpmdArray({self.aval}, sharding={self.sharding}, '
            f'devices={self.sharding.mesh.devices}{remote_str})'
        )

    def block_until_ready(self):
        if self._complete is None:
            # Already awaited.
            assert self._result is not None
            return
        result = self._complete()
        if isinstance(result, jax.Array):
            # Recv result, store array.
            self._result = result
        else:
            # No-op result or send result. Drop objects kept alive, but register
            # completion.
            self._result = ()
        # Drop the closure.
        self._complete = None
        return self

    @cached_property
    def is_fully_remote(self):
        return is_fully_remote_sharding(self.sharding)

    @property
    def jax_array(self):
        if self.is_fully_remote:
            raise ValueError('cannot convert fully-remote MpmdArray to jax.Array')
        self.block_until_ready()
        assert isinstance(self._result, jax.Array), (
            'expected non-fully-remote MpmdArray to hold some local data, but got: '
            f'{self._result} (mesh devices: {self.sharding.mesh.devices})'
        )
        return self._result

    @property
    def shape(self):
        return self.aval.shape

    @property
    def dtype(self):
        return self.aval.dtype


JaxOrMpmdArray = jax.Array | MpmdArray


def is_local_device(device) -> bool:
    return device.process_index == jax.process_index()


def is_fully_remote_sharding(sharding: Sharding) -> bool:
    # TODO: Handle shardings other than NamedSharding?
    assert isinstance(sharding, NamedSharding)
    return not any(map(is_local_device, sharding.mesh.devices.flat))


def is_fully_local_sharding(sharding: Sharding) -> bool:
    # TODO: Handle shardings other than NamedSharding?
    assert isinstance(sharding, NamedSharding)
    return all(map(is_local_device, sharding.mesh.devices.flat))


def is_fully_remote_array(arr: JaxOrMpmdArray) -> bool:
    return isinstance(arr, MpmdArray) and arr.is_fully_remote


def as_jax_array(arr: JaxOrMpmdArray) -> jax.Array:
    if isinstance(arr, MpmdArray):
        return arr.jax_array
    assert isinstance(arr, jax.Array)
    return arr


def fix_sharding(sharding: Sharding) -> Sharding:
    # FIXME: During jax.device_put(..., sharding) jaxlib/XLA fills in a memory
    # kind if none was explicitly given. We don't always call into
    # jax.device_put here, but we want to mirror this behavior so that even
    # processes that don't call jax.device_put end up with the exact same
    # metadata. (The bandaid below is likely incomplete.)
    if sharding.memory_kind is None:
        sharding = sharding.with_memory_kind('device')
    return sharding


@lru_cache
def recv_buf_factory(shape, dtype, tgt_sharding):
    @partial(jax.jit, out_shardings=tgt_sharding)
    def recv_buf_init():
        return jnp.zeros(shape, dtype)
    return recv_buf_init


# TODO: Generalize mm.device_put to mix jax.device_put, send and recv as
# needed. For the moment, we only allow cases that neatly fall into one of the
# above three cases, i.e. the present process either issue a jax.device_put,
# a NCCL send or a NCCL recv. This means that every submesh (e.g. a stage) needs
# to be managed by a single process for now.
def device_put(arr: JaxOrMpmdArray, device: Sharding) -> MpmdArray:
    assert isinstance(device, Sharding)
    tgt_sharding = fix_sharding(device)
    src_sharding = fix_sharding(arr.sharding)

    def complete_with(complete):
        return MpmdArray(
            aval=arr.aval,
            sharding=tgt_sharding,
            _complete=complete,
        )

    if is_fully_remote_array(arr):
        if is_fully_remote_sharding(tgt_sharding):
            # FullyRemote->FullyRemote: Nothing to be done.
            return complete_with(lambda: ())
        else:
            # FullyRemote->NonFullyRemote: Recv.
            # NOTE: We run the same jitted fun on each participating device,
            # rather than jax.device_put(jnp.zeros(...), tgt_sharding). The
            # latter produces jnp.zeros first on one local device and then P2P-
            # copies to the others, which anecdotally appears to be slower, but
            # also litters the profile, so we avoid it.
            recv_buf = recv_buf_factory(
                arr.aval.shape,
                arr.aval.dtype,
                tgt_sharding,
            )()
            return complete_with(
                mini_dime.send_or_recv(
                    recv_buf,
                    tgt_sharding,
                    src_sharding,
                )
            )

    # arr has some locally-addressable shards.
    jax_array = as_jax_array(arr)
    if jax_array.committed:
        if is_fully_remote_sharding(tgt_sharding):
            # NonFullyRemote->FullyRemote: Send.
            # FIXME: Should force completion at some point.
            return complete_with(
                mini_dime.send_or_recv(
                    jax_array,
                    tgt_sharding,
                )
            )
        elif (
            is_fully_local_sharding(src_sharding) and
            is_fully_local_sharding(tgt_sharding)
        ):
            # NonFullyRemote->NonFullyRemote: jax.device_put
            new_jax_array = jax.device_put(jax_array, tgt_sharding)
            return complete_with(lambda: new_jax_array)
        else:
            # NOTE: We exclude cases of NonFullyRemote -> NonFullyRemote
            # which would require a mix of jax.device_put, Send and Recv.
            raise NotImplementedError('unsupported transfer')
    else:
        # Uncommitted array.
        assert isinstance(jax_array.sharding, SingleDeviceSharding)
        if is_fully_remote_sharding(tgt_sharding):
            # Uncommitted->FullyRemote: Nothing to be done
            return complete_with(lambda: ())
        else:
            # Uncommitted->NonFullyRemote: jax.device_put
            # NOTE: Uncommitted arrays arise when the user hasn't yet specified
            # a device or sharding, so the current (single-device) sharding is
            # somewhat arbitrary.
            # An important assumption here is that, though said device will vary
            # from process to process, we expect all of the processes to have
            # the same values.
            #
            # Now we'd like to do something like
            #   new_jax_array = jax.device_put(jax_array, tgt_sharding)
            # where we'd expect jax.device_put to simply simply transfer from
            # the current local single device to all the other relevant local
            # devices.
            #
            # This unfortunately doesn't work, because jax.device_put will check
            # the above assumption of same-values-everywhere by introducing a
            # broadcast from process 0 to all others. But in an MPMD program
            # only a subset of processes will participate in any given
            # device_put, so this might lead to hangs!
            #
            # We could likely work around this by doing appropriate device_puts
            # with single-device shardings and subsequently using
            # jax.make_array_from_single_device_arrays to build a global array.
            if not is_fully_local_sharding(tgt_sharding):
                raise NotImplementedError('unsupported transfer')
            new_jax_array = jax.device_put(jax_array, tgt_sharding)
            return complete_with(lambda: new_jax_array)


def jit(*args, **kwargs):
    if (out_shardings := kwargs.get('out_shardings')) is None:
        raise ValueError('missing out_shardings')
    fun = jax.jit(*args, **kwargs)

    @wraps(fun)
    def wrapped(*in_vals):
        first_fully_remote_input = next(
            (
                (path, in_val)
                for path, in_val in tree_leaves_with_path(in_vals)
                if is_fully_remote_array(in_val)
            ),
            None,
        )

        # This computation does not concern us, return fully-remote arrays.
        if first_fully_remote_input is not None:
            out_shape_dtypes = jax.eval_shape(fun, *in_vals)
            # Allow out_shardings to be a prefix tree
            try:
                out_shardings_flat = broadcast_prefix(
                    out_shardings,
                    out_shape_dtypes,
                    is_leaf=lambda x: x is None,  # FIXME: Correct?
                )
            except ValueError:
                e, *_ = prefix_errors(out_shardings, out_shape_dtypes)
                raise e('mm.jit out_shardings') from None
            out_shardings_full = jax.tree.unflatten(
                jax.tree.structure(out_shape_dtypes),
                out_shardings_flat,
            )
            # Make an MpmdArray for every out value
            def make_fully_remote_output(shape_dtype, sharding):
                if not is_fully_remote_sharding(sharding):
                    path, in_val = first_fully_remote_input
                    raise ValueError(
                        'mm.jit produces a non-fully-remote output, but '
                        f'was invoked on fully-remote input: {in_val} @ {path}')
                return MpmdArray(
                    aval=jax.core.ShapedArray(
                        shape_dtype.shape,
                        shape_dtype.dtype,
                    ),
                    sharding=sharding,
                    _complete=lambda: (),
                )
            return jax.tree.map(
                make_fully_remote_output,
                out_shape_dtypes,
                out_shardings_full,
            )

        # This computations concerns us, run the jax.jit-ed function.
        in_vals = jax.tree.map(as_jax_array, in_vals)
        out_vals = fun(*in_vals)
        return jax.tree.map(
            lambda jax_array: MpmdArray(
                jax_array.aval,
                jax_array.sharding,
                lambda: jax_array,
            ),
            out_vals,
        )
    return wrapped
