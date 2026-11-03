``jax.ffi`` module
==================

.. automodule:: jax.ffi

.. autosummary::
  :toctree: _autosummary

  abstract_token
  ffi_call
  ffi_lowering
  pycapsule
  register_ffi_target
  register_ffi_type

.. _ffi-gpu-record:

GPU command-buffer recording
----------------------------

XLA can group GPU operations into a command buffer to reduce host launch
overhead. A GPU FFI target can provide an optional ``"record"`` handler to describe
its device operations directly when XLA creates or updates that command buffer.
This is useful for custom kernels that can describe their launches and
dependencies explicitly, avoiding stream capture of the ``"execute"`` handler.

Define the handler with ``ffi::Ffi::BindRecord()`` and include
``xla/ffi/api/record_ffi.h``. XLA's
`recording API <https://github.com/openxla/xla/blob/main/xla/ffi/api/record_api.h>`_
provides ``ffi::RecordExtension`` and ``ffi::RecordContext`` for creating commands
and updating their arguments. The handler records work for later execution;
recording itself must not read or modify device buffer contents, which may not
yet be initialized. Buffer addresses, shapes, and attributes are available for
describing the work.

The host handler can run again when the command buffer needs updating. Replaying
the command buffer executes the recorded device operations without calling the
host handler again. Keep an ``"execute"`` handler for execution outside command
buffers, including warm-up. If no ``"record"`` handler is provided, XLA can instead
record compatible ``"execute"`` handlers using stream capture.
