# Copyright 2024 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import ctypes

from jax._src.lib import jaxlib

# Use Python stable C API function to construct a PyCapsule.
# From ctypes.pythonapi documentation:
#
#  ctypes.pythonapi
#    An instance of PyDLL that exposes Python C API functions as attributes.
#    Note that all these functions are assumed to return C int, which is of
#    course not always the truth, so you have to assign the correct restype
#    attribute to use these functions.
#
# Taken from: https://docs.python.org/3/library/ctypes.html.
#
# More context here: https://stackoverflow.com/questions/24377845/ctype-why-specify-argtypes.
#
# Following this advice we annotate argument and return types of PyCapsule_New
# before we call it. Example here:
# https://stackoverflow.com/questions/65056619/converting-ctypes-c-void-p-to-pycapsule
def build_capsule(funcptr):
  """Construct a PyCapsule out of a ctypes function pointer.

  A typical use for this is registering custom call targets with XLA:

    import ctypes
    import jax
    from jax.lib import xla_client

    fooso = ctypes.cdll.LoadLibrary('./foo.so')
    xla_client.register_custom_call_target(
      name="bar",
      fn=jax.ffi.build_capsule(fooso.bar),
      platform=PLATFORM,
      api_version=API_VERSION
    )
  """
  PyCapsule_Destructor = ctypes.CFUNCTYPE(None, ctypes.py_object)
  PyCapsule_New = ctypes.pythonapi.PyCapsule_New
  PyCapsule_New.restype = ctypes.py_object
  PyCapsule_New.argtypes = (ctypes.c_void_p, ctypes.c_char_p, PyCapsule_Destructor)
  return PyCapsule_New(funcptr, None, PyCapsule_Destructor(0))

def include_dir() -> str:
  """Get the path to the directory containing header files bundled with jaxlib"""
  jaxlib_dir = os.path.dirname(os.path.abspath(jaxlib.__file__))
  return os.path.join(jaxlib_dir, "include")
