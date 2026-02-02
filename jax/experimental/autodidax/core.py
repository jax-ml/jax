"""Core functionality bridging Python to Rust autodidax implementation."""

from typing import Any, Callable, Tuple
from jax.experimental.autodidax import autodidax_core as _rust


class Primitive:
    """A primitive operation that can be traced."""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"Primitive({self.name!r})"


add_p = Primitive("add")
mul_p = Primitive("mul")
neg_p = Primitive("neg")
sin_p = Primitive("sin")
cos_p = Primitive("cos")
reduce_sum_p = Primitive("reduce_sum")
greater_p = Primitive("greater")
less_p = Primitive("less")


def bind1(prim: Primitive, *args: float) -> float:
    """Bind a single-output primitive to its arguments."""
    return _rust.bind1(prim.name, list(args))


def bind(prim: Primitive, *args: float) -> list[float]:
    """Bind a primitive to its arguments, returns list of outputs."""
    return _rust.bind(prim.name, list(args))


JVPTracer = _rust.JvpTracer
JaxprTracer = _rust.JaxprTracer
Jaxpr = _rust.Jaxpr


def make_jvp_tracer(primal: Any, tangent: Any) -> JVPTracer:
    """Create a JVPTracer from primal and tangent values."""
    return _rust.JvpTracer(primal, tangent)



def jvp(
    f: Callable[..., Any], primals: Tuple[Any, ...], tangents: Tuple[Any, ...]
) -> Tuple[Any, Any]:
    """Compute the Jacobian-vector product of f.

    Args:
        f: A function to differentiate.
        primals: Primal inputs to f.
        tangents: Tangent vectors for each primal.

    Returns:
        A tuple (primal_out, tangent_out) where primal_out = f(*primals)
        and tangent_out is the JVP.
    """
    if len(primals) != len(tangents):
        raise ValueError("primals and tangents must have the same length")

    tracers_in = [make_jvp_tracer(p, t) for p, t in zip(primals, tangents)]

    out = f(*tracers_in)

    if isinstance(out, _rust.JvpTracer):
        return out.primal, out.tangent
    elif isinstance(out, (list, tuple)):
        primals_out = []
        tangents_out = []
        for o in out:
            if isinstance(o, _rust.JvpTracer):
                primals_out.append(o.primal)
                tangents_out.append(o.tangent)
            else:
                primals_out.append(o)
                tangents_out.append(0.0)
        return type(out)(primals_out), type(out)(tangents_out)
    else:
        return out, 0.0


def make_jaxpr(f: Callable[..., Any], num_inputs: int | None = None) -> Callable[..., Jaxpr]:
    """Create a function that traces f and returns a Jaxpr.

    Args:
        f: Function to trace.
        num_inputs: Number of inputs (optional, inferred from example call).

    Returns:
        A new function that when called returns the Jaxpr representation of f.
    """
    def make_jaxpr_impl(*example_args):
        n = len(example_args) if num_inputs is None else num_inputs
        _rust.start_jaxpr_trace()
        in_tracers = _rust.make_jaxpr_tracers(n)

        out = f(*in_tracers)

        if isinstance(out, _rust.JaxprTracer):
            out_tracers = [out]
        elif isinstance(out, (list, tuple)):
            out_tracers = list(out)
        else:
            _rust.start_jaxpr_trace()
            raise ValueError(f"Unexpected output type: {type(out)}")

        return _rust.finalize_jaxpr(in_tracers, out_tracers)

    return make_jaxpr_impl


def lower_to_stablehlo(jaxpr: Jaxpr) -> str:
    """Lower a Jaxpr to StableHLO MLIR text.

    Args:
        jaxpr: The Jaxpr to lower.

    Returns:
        StableHLO MLIR module as a string.
    """
    return _rust.lower_jaxpr_to_stablehlo(jaxpr)
