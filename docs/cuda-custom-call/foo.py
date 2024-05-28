import jax
import jax.numpy as jnp
import numpy as np
from jax.lib import xla_client
from jax._src.lib.mlir import ir
from jaxlib.hlo_helpers import custom_call
from jax.interpreters import mlir

import ctypes

# XLA needs uppercase, "cuda" isn't recognized
XLA_PLATFORM = "CUDA"

# JAX needs lowercase, "CUDA" isn't recognized
JAX_PLATFORM = "cuda"

# 0 = original ("opaque"), 1 = FFI
XLA_CUSTOM_CALL_API_VERSION = 1

# version 4 accepts MLIR dictionaries in backend_config
STABLEHLO_CUSTOM_CALL_API_VERSION = 4

# this string is how we identify the kernel to XLA:
# - first we register a pointer to the kernel with XLA under this name
# - then we "tell" JAX to emit StableHLO specifying this name to XLA
XLA_CUSTOM_CALL_TARGET = "foo"

# independently, the corresponding JAX primitive must also be named,
# it can have a name different from the XLA target
JAX_PRIMITIVE = "foo"

SHARED_LIBRARY = "./foo.so"


library = ctypes.cdll.LoadLibrary(SHARED_LIBRARY)

# register the XLA FFI binding pointer with XLA
xla_client.register_custom_call_target(
    name=XLA_CUSTOM_CALL_TARGET,
    fn=jax.ffi.build_capsule(library.Foo),
    platform=XLA_PLATFORM,
    api_version=XLA_CUSTOM_CALL_API_VERSION
)

foo_p = jax.core.Primitive(JAX_PRIMITIVE)
def foo(a, b):
    return foo_p.bind(a, b)

def _foo_abstract_eval(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    return jax.core.ShapedArray(a.shape, a.dtype)

foo_p.def_abstract_eval(_foo_abstract_eval)

def _foo_lowering(ctx, a, b):
    typ = ir.RankedTensorType(a.type)
    shape = typ.shape
    u64 = ir.IntegerType.get_unsigned(64)
    n = np.prod(shape)
    # default XLA layout that just means row major
    layout = range(len(shape)-1, -1, -1)
    return custom_call(
        XLA_CUSTOM_CALL_TARGET,
        api_version=STABLEHLO_CUSTOM_CALL_API_VERSION,
        result_types=[typ],  # same type as inputs
        operands=[a,b],
        backend_config=dict(n=ir.IntegerAttr.get(u64, n)),
        operand_layouts=[layout, layout],
        result_layouts=[layout],
    ).results

mlir.register_lowering(foo_p, _foo_lowering, platform=JAX_PLATFORM)

a = 2. * jnp.ones((2,3))
b = 3. * jnp.ones((2,3))
assert (jax.jit(foo)(a, b) == (2*(3+1))).all()

#print(jax.grad(foo)(a, b))
