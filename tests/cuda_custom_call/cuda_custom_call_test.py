import jax
import jax.numpy as jnp
import numpy as np
from jax.extend import ffi
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

# these strings are how we identify kernels to XLA:
# - first we register a pointer to the kernel with XLA under this name
# - then we "tell" JAX to emit StableHLO specifying this name to XLA
XLA_CUSTOM_CALL_TARGET_FWD = "foo-fwd"
XLA_CUSTOM_CALL_TARGET_BWD = "foo-bwd"

# independently, corresponding JAX primitives must also be named,
# names can be different from XLA targets, here they are the same
JAX_PRIMITIVE_FWD = "foo-fwd"
JAX_PRIMITIVE_BWD = "foo-bwd"

# note paths to files in current working directory need to be prefixed with "./"
SHARED_LIBRARY = "tests/cuda_custom_call/libfoo.so"

library = ctypes.cdll.LoadLibrary(SHARED_LIBRARY)


#-----------------------------------------------------------------------------#
#                              Forward pass                                   #
#-----------------------------------------------------------------------------#

# register the XLA FFI binding pointer with XLA
xla_client.register_custom_call_target(
    name=XLA_CUSTOM_CALL_TARGET_FWD,
    fn=ffi.pycapsule(library.FooFwd),
    platform=XLA_PLATFORM,
    api_version=XLA_CUSTOM_CALL_API_VERSION
)


# our forward primitive will also return the intermediate output b+1
# so it can be reused in the backward pass computation
def _foo_fwd_abstract_eval(a, b):
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    shaped_array = jax.core.ShapedArray(a.shape, a.dtype)
    return (
        shaped_array,  # output c
        shaped_array,  # intermediate output b+1
    )


def _foo_fwd_lowering(ctx, a, b):
    typ = ir.RankedTensorType(a.type)
    shape = typ.shape
    u64 = ir.IntegerType.get_unsigned(64)
    n = np.prod(shape)
    # default XLA layout that just means row major
    layout = range(len(shape)-1, -1, -1)
    return custom_call(
        XLA_CUSTOM_CALL_TARGET_FWD,
        api_version=STABLEHLO_CUSTOM_CALL_API_VERSION,
        result_types=[typ, typ],  # c, b_plus_1, same types as inputs
        operands=[a,b],
        backend_config=dict(n=ir.IntegerAttr.get(u64, n)),
        operand_layouts=[layout, layout],  # a, b
        result_layouts=[layout, layout],   # c, b_plus_1
    ).results


# construct a new JAX primitive
foo_fwd_p = jax.core.Primitive(JAX_PRIMITIVE_FWD)
# register the abstract evaluation rule for the forward primitive
foo_fwd_p.def_abstract_eval(_foo_fwd_abstract_eval)
foo_fwd_p.multiple_results = True
mlir.register_lowering(foo_fwd_p, _foo_fwd_lowering, platform=JAX_PLATFORM)

#-----------------------------------------------------------------------------#
#                              Backward pass                                  #
#-----------------------------------------------------------------------------#

# register the XLA FFI binding pointer with XLA
xla_client.register_custom_call_target(
    name=XLA_CUSTOM_CALL_TARGET_BWD,
    fn=ffi.pycapsule(library.FooBwd),
    platform=XLA_PLATFORM,
    api_version=XLA_CUSTOM_CALL_API_VERSION
)


def _foo_bwd_abstract_eval(c_grad, a, b_plus_1):
    assert c_grad.shape == a.shape
    assert a.shape == b_plus_1.shape
    assert c_grad.dtype == a.dtype
    assert a.dtype == b_plus_1.dtype

    shaped_array = jax.core.ShapedArray(a.shape, a.dtype)
    return (
        shaped_array,  # a_grad
        shaped_array,  # b_grad
    )


def _foo_bwd_lowering(ctx, c_grad, a, b_plus_1):
    typ = ir.RankedTensorType(a.type)
    shape = typ.shape
    u64 = ir.IntegerType.get_unsigned(64)
    n = np.prod(shape)
    # default XLA layout that just means row major
    layout = range(len(shape)-1, -1, -1)
    return custom_call(
        XLA_CUSTOM_CALL_TARGET_BWD,
        api_version=STABLEHLO_CUSTOM_CALL_API_VERSION,
        result_types=[typ, typ],  # a_grad, b_grad
        operands=[c_grad, a, b_plus_1],
        backend_config=dict(n=ir.IntegerAttr.get(u64, n)),
        operand_layouts=[layout, layout, layout],  # c_grad, a, b_plus_1
        result_layouts=[layout, layout],   # a_grad, b_grad
    ).results

# construct a new JAX primitive
foo_bwd_p = jax.core.Primitive(JAX_PRIMITIVE_BWD)
# register the abstract evaluation rule for the backward primitive
foo_bwd_p.def_abstract_eval(_foo_bwd_abstract_eval)
foo_bwd_p.multiple_results = True
mlir.register_lowering(foo_bwd_p, _foo_bwd_lowering, platform=JAX_PLATFORM)


#-----------------------------------------------------------------------------#
#                              User facing API                                #
#-----------------------------------------------------------------------------#

def foo_fwd(a, b):
    c, b_plus_1 = foo_fwd_p.bind(a, b)
    return c, (a, b_plus_1)

def foo_bwd(res, c_grad):
    a, b_plus_1 = res
    return foo_bwd_p.bind(c_grad, a, b_plus_1)

@jax.custom_vjp
def foo(a, b):
    c, _ = foo_fwd(a, b)
    return c

foo.defvjp(foo_fwd, foo_bwd)

#-----------------------------------------------------------------------------#
#                                    Test                                     #
#-----------------------------------------------------------------------------#

a = 2. * jnp.ones((2,3))
b = 3. * jnp.ones((2,3))
assert (jax.jit(foo)(a, b) == (2*(3+1))).all()

def loss(a, b):
    return jnp.sum(foo(a, b))

da, db = jax.jit(jax.grad(loss, argnums=(0,1)))(a, b)

assert (da == (b+1)).all()
assert (db == a).all()
