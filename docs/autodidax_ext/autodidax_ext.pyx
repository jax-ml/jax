# cython: language_level=2
# distutils: language = c++

from cpython.pycapsule cimport PyCapsule_New
from libc.stdint cimport uint8_t
cimport numpy as np
import numpy as np
np.import_array()

from jaxlib import xla_client
_ops = xla_client.ops
Shape = xla_client.Shape

cdef register_cpu_custom_call_target(fn_name, void* fn):
  cdef const char* name = "xla._CUSTOM_CALL_TARGET"
  xla_client.register_custom_call_target(
      fn_name, PyCapsule_New(fn, name, NULL))

cdef void caller(void *out, const void **data) nogil:
  with gil:
    f = <object> (<void**> data[1])[0]
    args = tuple(
        np.asarray(<const np.uint8_t[:s.size]> data[i+2]).view(s.dtype).reshape(s.shape)
        for i, s in enumerate(f.arg_shapes))
    f(*args)
register_cpu_custom_call_target(b"caller", <void*>(caller))

def pycallback(c, token, f, *args):
  persist_for_life_of_executable(f)
  return _ops.CustomCallWithLayout(
      c, b"caller", shape_with_layout=Shape.token_shape(),
      operands=(token, _ops.Constant(c, np.uint64(id(f))), *args),
      operand_shapes_with_layout=(
          Shape.token_shape(), Shape.array_shape(np.dtype(np.uint64), (), ()),
          *(c.get_shape(x) for x in args)),
      has_side_effect=True)

# TODO: jaxlib needs a way to attach object to executable
leaks = []
persist_for_life_of_executable = leaks.append
