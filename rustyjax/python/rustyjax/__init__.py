"""Thin Python shim over the Rust core.

Policy lives here (which trace is active, the XLA compile/execute session);
mechanism lives in Rust (_core): the IR, tracing, interpretation, and lowering
to StableHLO. XLA is reached through jaxlib's bundled CPU PJRT client.
"""
import contextvars
from types import SimpleNamespace

import numpy as np

from . import _core
from ._core import Jaxpr, TraceCtx, Tracer, int_type

_active = contextvars.ContextVar("rustyjax_trace", default=None)

_client = None


def _cpu_client():
    global _client
    if _client is None:
        from jaxlib import xla_client
        _client = xla_client.make_cpu_client()
    return _client


class Traced:
    def __init__(self, jaxpr):
        self.jaxpr = jaxpr

    def __repr__(self):
        return repr(self.jaxpr)

    def compile(self):
        from jaxlib import xla_client
        client = _cpu_client()
        exe = client.compile_and_load(
            self.jaxpr.to_stablehlo(),
            executable_devices=xla_client.DeviceList(tuple(client.local_devices())),
            compile_options=xla_client.CompileOptions(),
        )
        return Compiled(exe)


class Compiled:
    def __init__(self, exe):
        self.exe = exe

    def eval(self, args):
        from jaxlib import xla_client as xc

        # The C++ sharding base class is missing two members that the
        # device_put path expects the (normally jax-provided) subclass to add.
        class _Sharding(xc.SingleDeviceSharding):
            @property
            def memory_kind(self):
                return None

            def _to_xla_hlo_sharding(self, ndim):
                return xc.HloSharding.replicate()

        dev = _cpu_client().local_devices()[0]
        aval = SimpleNamespace(shape=(), dtype=np.dtype(np.int64), weak_type=False)
        bufs = [
            xc.batched_device_put(aval, _Sharding(dev), [np.asarray(a, np.int64)],
                                  [dev], enable_x64=True)
            for a in args
        ]
        [out] = self.exe.execute(bufs)
        return int(np.asarray(out))


def trace(f, in_types):
    ctx = TraceCtx(in_types)
    token = _active.set(ctx)
    try:
        out = f(*ctx.tracers())
    finally:
        _active.reset(token)
    return Traced(ctx.finish(out))


def eval(traced, args):
    return _core.eval(traced.jaxpr, args)


def add(x, y):
    return _core.add(_active.get(), x, y)


def mul(x, y):
    return _core.mul(_active.get(), x, y)
