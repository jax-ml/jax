"""Thin Python shim over the Rust core.

Policy lives here (which trace is active, the XLA compile/execute session);
mechanism lives in Rust (_core): the IR, tracing, interpretation, and lowering
to StableHLO. XLA is reached through jaxlib's bundled CPU PJRT client.
"""
import contextvars
from types import SimpleNamespace

import numpy as np

from . import _core
from ._core import Jaxpr, TraceCtx, Tracer, Type, int_type


def f32(*shape):
    return Type(list(shape), "f32")


def i32(*shape):
    return Type(list(shape), "i32")

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
        bufs = []
        for a in args:
            a = np.asarray(a)
            aval = SimpleNamespace(shape=a.shape, dtype=a.dtype, weak_type=False)
            bufs.append(xc.batched_device_put(aval, _Sharding(dev), [a], [dev],
                                              enable_x64=True))
        [out] = self.exe.execute(bufs)
        out = np.asarray(out)
        # plain int for scalar integer results (the original mini-language API)
        return int(out) if out.ndim == 0 and out.dtype.kind == "i" else out


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


def _op(name):
    f = getattr(_core, name)

    def wrapper(*args):
        return f(_active.get(), *args)

    wrapper.__name__ = name
    return wrapper


add = _op("add")
sub = _op("sub")
mul = _op("mul")
div = _op("div")
matmul = _op("matmul")
take = _op("take")
exp = _op("exp")
tanh = _op("tanh")
sqrt = _op("sqrt")
reshape = _op("reshape")
transpose = _op("transpose")


def reduce_sum(x, axis=-1, keepdims=False):
    return _core.reduce_sum(_active.get(), x, axis, keepdims)


def reduce_max(x, axis=-1, keepdims=False):
    return _core.reduce_max(_active.get(), x, axis, keepdims)
