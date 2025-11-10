import jax
import jax.numpy as jnp
from jax._src.hijax import VJPHiPrimitive

# TODO remove these deps by abstracting a little bit
from jax._src import core
from jax._src.interpreters import ad

# OLD:
#   LinearizeTrace:
#     parent_trace: Trace  # fwd
#     tangent_trace: Trace  # tangent
#   LinearizeTracer:
#     primal: Array
#     tangent: Array
#   def rule(nzs_in, x, y):
#     ...
#     return nzs_out, z, lambda xdot, ydot: ...zdot...

# NEW, for N levels of remat
#   LinearizeNCNPTrace:
#     remat_traces: list[Trace]  # [fwd1, fwd2, ..., fwdN]
#     parent_trace: Trace        # the last guy
#     tangent_trace: Trace
#   LinearizeNCNPTracers:
#     remats: list[Array]  # [fwd1, fwd2, ..., fwdN]
#     primal: Array
#     tangent: Array

# NEW, for N levels of remat
#   LinearizeNCNPTrace:
#     parent_traces: list[Trace]  # parents[0] is primal, parents[-1] gives res
#     tangent_trace: Trace
#   LinearizeNCNPTracers:
#     primals: list[Array]
#     tangent: Array


# example of controlling NCNP rule
class S3(VJPHiPrimitive):
  def __init__(self, in_aval):
    self.in_avals = in_aval,
    self.out_aval = in_aval
    self.params = {}
    super().__init__()

  def expand(self, x):
    return jnp.sin(jnp.sin(jnp.sin(x)))

  def linearize_ncnp(self, trace, tracer):
    *remat_traces, fwd_trace = trace.parent_traces
    *remat_xs, fwd_x = tracer.primals
    x_dot = tracer.tangent
    remat_ys = [core.set_current_trace(t)(self.expand)(x)
                for t, x in zip(remat_traces, remat_xs)]
    with core.set_current_trace(fwd_trace):
      x1 = jnp.sin(fwd_x)
      x2 = jnp.sin(x1)
      fwd_y = jnp.sin(x2)
    with core.set_current_trace(trace.tangent_trace):
      y_dot = x_dot * jnp.cos(fwd_x) * jnp.cos(x1) * jnp.cos(x2)
    return [ad.LinearizeTracer(trace, [*remat_ys, fwd_y], y_dot)]

def s3(x):
  ty = jax.typeof(x)
  return S3(ty)(x)

y = s3(1.)  # doesn't crash
print(y)
y, g = jax.value_and_grad(s3)(1.)
print(y)
print(g)

# y, g = jax.value_and_grad(lambda x: jax.lax.sin(jax.lax.sin(jax.lax.sin(x))))(1.)
y, g = jax.value_and_grad(lambda x: jnp.sin(jnp.sin(jnp.sin(x))))(1.)
print(y)
print(g)
