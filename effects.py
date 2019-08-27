from __future__ import print_function
import numpy as onp
from contextlib import contextmanager

from jax import core
from jax.abstract_arrays import ConcreteArray, ShapedArray
from jax.interpreters import xla
from jax.lib import xla_client as xc

from jax import jit
import jax.numpy as np

def identity(x): return x


# Consider a linear array type, instances of which can only be read once.
# (Reasons for making 'read' a primitive will be clear below, but for now it
# doesn't hurt and is just a bit of indirection.)

class LinearArray(object):
  __slots__ = ["arr", "consumed"]
  def __init__(self, arr):
    self.arr = arr
    self.consumed = False
  def consume(self):
    if self.consumed: raise ReuseError
    self.consumed = True
  def read(self):
    return read_p.bind(self)

read_p = core.Primitive('read')

def read_impl(x):
  x.consume()  # side-effect on x
  return x.arr
read_p.def_impl(read_impl)

class ReuseError(Exception): pass


# We want these linear values to work in function arguments but also in function
# closures. (We can also make them work as function outputs, but we'll ignore
# that for now.) Here's how they look as arguments to regular Python functions:

# Case 1: Function that consumes its LinearArray input
x = LinearArray(onp.arange(3))
y = LinearArray(onp.arange(3))
def f(x):
  return x.read() + 1
f(x); print("okay")
f(y); print("okay")
try: f(x)
except ReuseError: print("okay")
else: raise Exception


# Case 2: Function that erroneously consumes its LinearArray input twice
x = LinearArray(onp.arange(3))
def f(x):
  return x.read() + x.read()
try: f(x)
except ReuseError: print("okay")
else: raise Exception


# Here's how they look in function closures:

# Case 3: Function that consumes a closed-over LinearArray
x = LinearArray(onp.arange(3))
def f(y):
  return y + x.read()
f(1); print("okay")
try: f(2)
except ReuseError: print("okay")
else: raise Exception


# Case 4: Function that erroneously consumes a closed-over LinearArray twice
x = LinearArray(onp.arange(3))
def f(y):
  return y + x.read() + x.read()
try: f(1)
except ReuseError: print("okay")
else: raise Exception


# Now we want to be able to put a `jit` on those functions and have them behave
# the same way. That means our staged out representation needs to recapitulate
# the effect of setting the "consumed" flag on instances of LinearArray.

# What if jit could record and then replay certain side-effects? We'd need to
# accumulate a list of side-effects during abstract interpretation (i.e. jit
# tracing), and those effects would need to know how to apply themselves to the
# right objects so that we can replay them. By "during abstract interpretation"
# we mean in the dynamic scope of a jit trace.

class EffectRecord(list): pass
class EffectRecorders(list): pass
effect_recorders = EffectRecorders()

def register_effect(thunk):
  if effect_recorders:
    for s in effect_recorders:
      s.append(thunk)
  else:
    thunk()

@contextmanager
def monitor_effects():
  r = EffectRecord()
  effect_recorders.append(r)
  try:
    yield r
  finally:
    effect_recorders.pop()


# One way to make use of this machinery is to handle the closure problems above.
# I put the above code in xla.py and integrated it with jit so that registered
# effects are recorded during Python tracing and replayed when a staged-out
# computation is executed.

class LinearArray(object):
  __slots__ = ["arr", "consumed"]
  def __init__(self, arr):
    self.arr = arr
    self.consumed = False
  def read(self):
    xla.register_effect(self.consume)  # new!
    return self.arr
  def consume(self):
    if self.consumed: raise ReuseError
    self.consumed = True


# It can handle those cases of the closure problem (both with and without jit):

# Case 3, with a jit
x = LinearArray(onp.arange(3))
@jit
def f(y):
  return y + x.read()
f(1); print("okay")
try: f(2)
except ReuseError: print("okay")
else: raise Exception


# Case 4, with a jit
x = LinearArray(onp.arange(3))
@jit
def f(y):
  return y + x.read() + x.read()
try: f(1)
except ReuseError: print("okay")
else: raise Exception


# Side note, this even works for things like print statements:

@jit
def f():
  xla.register_effect(lambda: print("hi"))
f()  # prints 'hi'
f()  # prints 'hi'


# How do we make LinearArrays work as arguments to jitted functions? At Python
# tracing time, arguments to jitted functions get abstracted, and those
# AbstractValue instances are what get propagated through the user Python code,
# rather than the underlying LinearArrays themselves. But we're not interested
# in applying effectful updates to those abstract values: rather, when during
# abstract interpretation we see an effectful operation on an abstract value,
# that means we ultimately want to apply that effect to the corresponding
# concrete argument value supplied to the function at concrete execution time.

# This seems pretty different from the other case: before we just had to apply
# the effect to a closed-over constant, like the stdout stream or a closed-over
# LinearArray instance, and so at trace time we could just store a reference to
# that constant in the effect thunk. Here we instead want to model an effect on
# an argument value, and hence need to get a reference to the concrete argument
# value itself to the effect function. (Inputs to a jit function are either
# closed-over constants or abstracted arguments, so these cases are exhaustive.)

# To model effects on arguments, we'll associate with each argument abstract
# value a sequence of effects, representing the effects to be applied to a
# concrete value at execution time.

class AbstractLinearArray(core.AbstractValue):
  __slots__ = ["shape", "dtype", "consumed", "effects"]

  def __init__(self, shape, dtype, consumed):
    self.shape = shape
    self.dtype = dtype
    self.consumed = consumed
    self.effects = []

  @core.aval_method
  def read(self):
    return read_p.bind(self)

  def __hash__(self):
    assert not self.effects
    return hash((self.shape, self.dtype, self.consumed))

  def __eq__(self, other):
    assert not self.effects
    return (type(other) is AbstractLinearArray and self.shape == other.shape
            and self.dtype == other.dtype and self.consumed == other.consumed)

def read_abstract_eval(aval):
  if aval.consumed: raise ReuseError
  aval.effects.append(lambda val: val.consume())
  aval.consumed = True
  return ShapedArray(aval.shape, aval.dtype)
read_p.def_abstract_eval(read_abstract_eval)


# To map computations on LinearArrays to XLA, we need to tell xla.py 3 things:
#   1. how to abstractify a LinearArray argument for abstract interpretation /
#      tracing (xla.pytype_aval_mappings),
#   2. how to map an AbstractLinearArray to an XLA shape (xla.xla_shape_handlers),
#   3. how to map a LinearArray to an XLA value (xla.device_put_handlers).

def shaped_linear_array(x):
  shape, dtype = onp.shape(x.arr), onp.result_type(x.arr)
  return AbstractLinearArray(shape, dtype, x.consumed)
xla.pytype_aval_mappings[LinearArray] = shaped_linear_array

xla.xla_shape_handlers[AbstractLinearArray] = \
    lambda a: xc.Shape.array_shape(a.dtype, a.shape)
xla.device_put_handlers[LinearArray] = \
    lambda x, n, backend: xla.device_put(x.arr, n, backend)
xla.canonicalize_dtype_handlers[LinearArray] = identity


# Finally, we need a translation rule for read_p:
xla.translations[read_p] = lambda c, x: x


# Case 1, with a jit
x = LinearArray(onp.arange(3, dtype=onp.int32))
y = LinearArray(onp.arange(3, dtype=onp.int32))
@jit
def f(x):
  return x.read() + 1
f(x); print("okay")
f(y); print("okay")
try: f(x)
except ReuseError: print("okay")
else: raise Exception

# Case 2, with a jit
x = LinearArray(onp.arange(3, dtype=onp.int32))
@jit
def f(x):
  return x.read() + x.read()
try: f(x)
except ReuseError: print("okay")
else: raise Exception


# A couple other cases we want to check, which work trivially when there's no
# jit involved: LinearArrays created and used under jit, and LinearArrays
# returned by a jit computation.


# Case 5: LinearArrays created and used under a jit

x = onp.arange(3, dtype=onp.int32)
@jit
def f(x):
  y = LinearArray(x)
  return 2 + y.read()
f(x); print("okay")

x = onp.arange(3, dtype=onp.int32)
@jit
def f(x):
  y = LinearArray(x)
  return y.read() + y.read()
try: f(x)
except ReuseError: print("okay")
else: raise Exception


# To return a data type from an XLA computation, we need to define one more
# mapping, from an abstract value to a result handler (which allows device
# persistence if we want it).

def linear_array_result_handler(aval):
  def handler(buf):
    out = LinearArray(buf.to_py())
    out.consumed = aval.consumed
    return out
  return handler
xla.xla_result_handlers[AbstractLinearArray] = linear_array_result_handler


# Case 6: returning un-consumed, already-consumed, and newly-consumed values

x = LinearArray(onp.arange(3, dtype=onp.int32))
@jit
def f(x):
  return x
x = f(x); x.read(); print("okay")
x = f(x); print("okay")
try: x.read()
except ReuseError: print("okay")
else: raise Exception

x = LinearArray(onp.arange(3, dtype=onp.int32))
@jit
def f(x):
  x.read()
  return x
x = f(x); print("okay")
try: f(x)
except ReuseError: print("okay")
else: raise Exception
try: x.read()
except ReuseError: print("okay")
else: raise Exception
