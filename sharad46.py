from functools import partial
import jax
import jax.numpy as jnp
from jax._src import core
from jax._src.state import discharge
from jax._src.interpreters import mlir
from jax._src.state.primitives import dup, pud


###

fake_kernel_p = core.Primitive('fake_kernel')
fake_kernel_p.multiple_results = True
@fake_kernel_p.def_effectful_abstract_eval
def _(ref, *, name):
  return [], {core.array_ref_effect}

k1 = partial(fake_kernel_p.bind, name='k1')
k2 = partial(fake_kernel_p.bind, name='k2')
k3 = partial(fake_kernel_p.bind, name='k3')

@discharge.register_discharge_rule(fake_kernel_p)
def dis(in_avals, out_avals, val_and_tok, *, name):
  val, tok = val_and_tok
  tok_ = fake_kernel_call_p.bind(val, tok, name=name)
  return [(val, tok_)], []


fake_kernel_call_p = core.Primitive('fake_kernel_call')
@fake_kernel_call_p.def_abstract_eval
def _(val, tok, *, name):
  return tok
mlir.register_lowering(fake_kernel_call_p, lambda _, val, tok, name: [tok])

###

@jax.jit
def f():
  x_ref = jax.new_ref(jnp.arange(3.))
  x1_ref, x2_ref = dup(x_ref)

  k1(x1_ref)
  k2(x1_ref)

  k3(x2_ref)

  x_ref = pud(x1_ref, x2_ref)
  return jax.freeze(x_ref)

print(f.trace().jaxpr)
f()

