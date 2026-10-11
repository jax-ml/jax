
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
.. autoclass:: JaxprTypeError
.. autoclass:: JaxRuntimeError
.. autoclass:: JAXTypeError
.. autoclass:: JAXIndexError
.. autoclass:: ConcretizationTypeError
.. autoclass:: KeyReuseError
.. autoclass:: NonConcreteBooleanIndexError
.. autoclass:: TracerArrayConversionError
.. autoclass:: TracerBoolConversionError
.. autoclass:: TracerIntegerConversionError
.. autoclass:: UnexpectedTracerError
