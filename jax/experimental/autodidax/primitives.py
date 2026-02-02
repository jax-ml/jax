"""User-facing primitive operations.

These functions provide the API that users call. Under the hood they
dispatch to the Rust core via the bind mechanism.
"""

from typing import Any

from jax.experimental.autodidax import autodidax_core as _rust
from jax.experimental.autodidax import core


def _maybe_wrap_jvp_or_jaxpr(prim_name: str, *args: Any) -> Any:
    """Handle regular values, JVP tracers, and Jaxpr tracers."""
    if any(isinstance(a, _rust.JaxprTracer) for a in args):
        jaxpr_args = []
        for a in args:
            if isinstance(a, _rust.JaxprTracer):
                jaxpr_args.append(a)
            else:
                jaxpr_args.append(float(a))
        return _rust.jaxpr_bind1(prim_name, jaxpr_args)

    if any(isinstance(a, _rust.JvpTracer) for a in args):
        tracers = [
            a if isinstance(a, _rust.JvpTracer) else core.make_jvp_tracer(float(a), 0.0)
            for a in args
        ]
        return _rust.jvp_bind1(prim_name, tracers)

    return _rust.bind1(prim_name, [float(a) for a in args])


def add(x: Any, y: Any) -> Any:
    """Add two values."""
    return _maybe_wrap_jvp_or_jaxpr("add", x, y)


def mul(x: Any, y: Any) -> Any:
    """Multiply two values."""
    return _maybe_wrap_jvp_or_jaxpr("mul", x, y)


def neg(x: Any) -> Any:
    """Negate a value."""
    return _maybe_wrap_jvp_or_jaxpr("neg", x)


def sin(x: Any) -> Any:
    """Compute sine of a value."""
    return _maybe_wrap_jvp_or_jaxpr("sin", x)


def cos(x: Any) -> Any:
    """Compute cosine of a value."""
    return _maybe_wrap_jvp_or_jaxpr("cos", x)


def greater(x: Any, y: Any) -> Any:
    """Check if x > y."""
    if isinstance(x, _rust.JvpTracer) or isinstance(y, _rust.JvpTracer):
        xc = x.get_concrete() if isinstance(x, _rust.JvpTracer) else float(x)
        yc = y.get_concrete() if isinstance(y, _rust.JvpTracer) else float(y)
        return xc > yc
    return _rust.bind1("greater", [float(x), float(y)])


def less(x: Any, y: Any) -> Any:
    """Check if x < y."""
    if isinstance(x, _rust.JvpTracer) or isinstance(y, _rust.JvpTracer):
        xc = x.get_concrete() if isinstance(x, _rust.JvpTracer) else float(x)
        yc = y.get_concrete() if isinstance(y, _rust.JvpTracer) else float(y)
        return xc < yc
    return _rust.bind1("less", [float(x), float(y)])
