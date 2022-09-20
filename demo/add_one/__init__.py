import jax
from add_one import add_one_lib

from jax._src import custom_call

add_one_call = custom_call.custom_call("add_one")

add_one_call.register(add_one_lib.get_function(), platform="cpu")

def f(x):
  descriptor = add_one_lib.get_descriptor(4)
  return add_one_call(x, descriptor=descriptor, out_shape_dtype=x)

print(jax.make_jaxpr(f)(4.))
print(jax.jit(f).lower(4.).compiler_ir())
print(jax.jit(f)(4.))
