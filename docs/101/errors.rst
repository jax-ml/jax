
.. _jax-101-errors:

Errors
======

This page lists some of the errors you might encounter when using JAX, along
with representative examples of how to fix them. Many of them, including all
the ``Tracer...Error`` and ``ConcretizationTypeError`` entries, come from code
that needs a concrete value where tracing provides only an abstract one; the
tracing model in :ref:`jax-101-tracing` explains why.

.. currentmodule:: jax.errors
.. autoclass:: InconclusiveDimensionOperation
   :no-index:
.. autoclass:: JaxprTypeError
   :no-index:
.. autoclass:: JaxRuntimeError
   :no-index:
.. autoclass:: JAXTypeError
   :no-index:
.. autoclass:: JAXIndexError
   :no-index:
.. autoclass:: ConcretizationTypeError
   :no-index:
.. autoclass:: KeyReuseError
   :no-index:
.. autoclass:: NonConcreteBooleanIndexError
   :no-index:
.. autoclass:: TracerArrayConversionError
   :no-index:
.. autoclass:: TracerBoolConversionError
   :no-index:
.. autoclass:: TracerIntegerConversionError
   :no-index:
.. autoclass:: UnexpectedTracerError
   :no-index:
