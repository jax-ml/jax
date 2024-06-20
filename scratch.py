import numpy as np
import jax
from jax._src import core
import jax.numpy as jnp

# ========= in jax =======

class JaxVal:
  @classmethod
  def new(cls, *args):
    return cls.from_rep(*args)

  def type_of(self):
    assert False, "subclass should implement"

  @staticmethod
  def constructor_bwd(t):
    assert False, "subclass should implement"

class JaxType:
  def tangent_type_of(self):
    assert False, "subclass should implement"

  def __eq__(self):
    assert False, "subclass should implement"

def register_fancy_type(cls, ty, attr_names):
  @jax.custom_vjp
  def from_rep(*args):
    # args will not have tracers *around FancyVal*
    val = cls.__new__(cls)
    val.__init__(*args)
    return val
  def fwd(*args): assert False
  def bwd(*args): assert False
  from_rep.defvjp(fwd, bwd)
  cls.from_rep = from_rep

  @jax.custom_vjp
  def splat(val):
    return tuple(getattr(val, attr_name) for attr_name in attr_names)
  splat.defvjp(fwd, bwd)
  cls.splat = splat

  core.pytype_aval_mappings[cls] = lambda x: x.type_of()
  core.raise_to_shaped_mappings[ty] = lambda x, _: x

  for i, attr in enumerate(attr_names):
    setattr(ty, attr, property(lambda self: splat(self)[i]))


# next:
#   case when tangent type is also fancy - need to define zeros/plus
#   defining derivative rules. two options
#      1. treat splat/new as ordinary primitives
#           * need to define other rules for them (some at user level?)
#             or lower after AD??
#      2. OR just use custom_vjp which has solved some version of those
#         problems already
#   vmap/scan? either we make use custom_vjp instead of
#    primitive or we some other way tell jax that vmap/vjp commute



# problem: we seem to need to have the user specify not just ad rules but vmap,
# xla lowering etc. How do we avoid that? We hope to avoid it because AD is all
# we care to override and once AD is "done" we should be able to expand
# everything in terms of underlying representations and use the already-existing
# rules on those.

# related problem: how to do things "after AD"

#   dex-style: make vmap a primitive. Do AD first (going under a vmap and
#   differentiating its body)
#   custom_vjp-style - somehow does the same thing?





# custom_vjp has three parts: primal, fwd, bwd
# we can view these as *two interpretations*
#    1. primal  - this is the lowering interpretation
#    2. fwd/bwd  - this is the AD interpretation















# ============== in library land ========

# wraps a float32
# tangent type is a float16
class FancyValCls(JaxVal):
  def __init__(self, val:float):
    assert jnp.shape(val) == ()
    assert jnp.result_type(val) == np.float32
    self.val = val

  def type_of(self):
    return FancyType()

  def to_rep(self):
    return self._val

  def splat_bwd(t:tuple):
    val_t, = t
    return val_t.astype(np.float16)

  def constructor_bwd(t): # :TangentType[FancyType]):
    return t.astype(np.float32)

class FancyType(JaxType):
  def tangent_type_of(self):
    return jax.ShapeDtypeStruct((), np.dtype("float16"))

register_fancy_type(FancyValCls, FancyType, ["val"])

FancyVal = FancyValCls.new

@jax.custom_vjp
def fancy_sin(x):
  return FancyVal(jnp.sin(x.val))

def fancy_sin_fwd(x):
  assert False

def fancy_sin_bwd(residuals, t):
  assert False

fancy_sin.defvjp(fancy_sin_fwd, fancy_sin_bwd)

# ============== in user land ========

def my_function(x:float):
  return fancy_sin(FancyVal(x)).val

print(my_function(1.0))
print(jax.jit(my_function)(1.0))
# print(jax.grad(my_function)(1.0))

# =======================

