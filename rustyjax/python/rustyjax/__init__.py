"""Thin Python shim over the Rust core.

Policy lives here (which trace is active, via a contextvars stack); mechanism
lives in Rust (_core), whose functions all take the trace context explicitly.
"""
import contextvars

from . import _core
from ._core import Jaxpr, TraceCtx, Tracer, eval, int_type

_active = contextvars.ContextVar("rustyjax_trace", default=None)


def trace(f, in_types):
    ctx = TraceCtx(in_types)
    token = _active.set(ctx)
    try:
        out = f(*ctx.tracers())
    finally:
        _active.reset(token)
    return ctx.finish(out)


def add(x, y):
    return _core.add(_active.get(), x, y)


def mul(x, y):
    return _core.mul(_active.get(), x, y)
