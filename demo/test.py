import jax
import add_one

from jax._src import custom_call

add_one_call = custom_call.custom_call("add_one")

add_one_call.register(add_one.get_function(), platform="cpu")

def f(x):
  descriptor = add_one.get_descriptor(4.)
  return add_one_call(x, descriptor=descriptor, out_shape_dtype=x)

print(jax.jit(f)(4.))

def g(x):
  descriptor = add_one.get_descriptor(-1.)
  return add_one_call(x, descriptor=descriptor, out_shape_dtype=x)

print(jax.jit(g)(4.))
