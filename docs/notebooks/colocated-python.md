---
jupytext:
  formats: ipynb,md:myst
  main_language: python
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
---

+++ {"id": "WKchP4VBBRgq"}

# Colocated Python

NOTE: Colocated Python is currently an experimental API. Its functionality and
interface are subject to change without following the standard JAX compatibility
policy.

Colocated Python provides a uniform way to run Python code on the hosts
associated with a set of JAX devices. If the JAX devices represent local
devices, the Python code will run on the local host. If the JAX devices
represent remote devices, the Python code will be shipped to run on the host of
these remote devices. This is useful when building a multi-host ML system on top
of JAX that is portable across multi-controller JAX environments (running JAX
code on each host with accelerators) as well as single-controller JAX
environments (running JAX code on a single host orchestrating other hosts with
accelerators).

+++ {"id": "B38uuH1ZBZmd"}

## Colocated CPU devices

To use colocated Python, the first step is to obtain CPU devices colocated with
target accelerator devices.
`jax.experimental.colocated_python.colocated_cpu_devices` provides a standard
way to do so.

```{code-cell}
:id: d7FHtd4wCYEf

import jax
import jax.experimental.colocated_python as colocated_python

devices = jax.devices()
cpu_devices = colocated_python.colocated_cpu_devices(devices)
print(cpu_devices)
```

+++ {"id": "Grfb7H4FCVsE"}

As usual, the CPU devices can be used with JAX APIs.

```{code-cell}
:id: 5RmWK-s4DQsl

cpu_mesh = jax.sharding.Mesh(cpu_devices, ["x"])
cpu_sharding = jax.sharding.NamedSharding(cpu_mesh, jax.P())
x = jax.device_put(1, cpu_sharding)
y = jax.jit(lambda x: x + 1)(x)
print(y)
```

+++ {"id": "7U1OScHaCjSC"}

## Colocated Python function

CPU devices can also be used to run Python code with colocated Python.

```{code-cell}
:id: PJbdHF8mDZNT

def f(x):
  return x + 1


f = colocated_python.colocated_python(f)
y = f(x)
assert y.sharding == x.sharding
print(y)
```

+++ {"id": "tpGdXqG9C5X3"}

Since colocated Python runs normal Python code, you can also perform I/O:

```{code-cell}
:id: MeWnKNlHDgs3

def f(x):
  with open('/tmp/foo', 'w') as f:
    f.write(str(x))
  return x


f = colocated_python.colocated_python(f)
jax.block_until_ready(f(x))
```

+++ {"id": "HOGQQ5IUC7Pe"}

Note the use of `jax.block_until_ready` to ensure the Python code has
completed. In principle, colocated Python calls may run asynchronously, similar
to jitted function calls; the calls would return JAX arrays and do not block
until their output is produced. Thus, you should block on an output from a
colocated Python call if the completion of the execution is significant.

There exist cases where a colocated Python call runs synchronously.

* If the colocated Python function is called without "specialization" (see
  below), the very first call will run synchronously. This is because the shape
  and sharding of the output must be known for asynchronous execution, and
  colocated Python has to run the Python code once to discover this information.

* Some JAX backends do not yet fully support asynchronous execution, and will
  fall back to synchronous execution.

The wrapped Python code must use exactly the same set of devices in the input
and the output. This is a requirement similar to jitted functions that represent
an SPMD execution.

+++ {"id": "uX8q-42tC8ia"}

## Specialization

Specialization in colocated Python is a mechanism to supply extra information
about the input, output, and execution of a colocated Python function, when the
information cannot be inferred in advance, or you would like to ensure the
colocated Python executions to happen precisely as specified.

First, functions wrapped in colocated Python has a `specialize` method.
This method is used to create another colocated Python wrapped function
specialized with the supplied information.

`out_specs_fn` is a function that takes a pytree of
`jax.ShapeDtypeStruct` of the call inputs and returns a pytree of
`jax.ShapeDtypeStruct` expected for the output. Calling this function is
analogous to jitted function tracing, but this function is separate from the
original Python code. This function runs on the caller side and not executed on
the devices.

```{code-cell}
:id: SWEuz68nDtXE

def f(x):
  return x + 1


f = colocated_python.colocated_python(f)
f = f.specialize(out_specs_fn=lambda x: x)
y = f(x)
assert y.sharding == x.sharding
```

+++ {"id": "HkQZwqUBC-QV"}

`in_specs` takes a concrete pytree (the top level is tuple) of
`jax.sharding.ShapeDtypeStruct` expected for the input to the colocated
Python function call. This is used if a certain input spec must be used, or the
output specs function can be computed only for a concrete input spec.

```{code-cell}
:id: E0SQPPHID1WU

import jax.numpy as jnp


def f(x):
  return x + 1


f = colocated_python.colocated_python(f)
f = f.specialize(
    in_specs=(
        # args
        (
            jax.ShapeDtypeStruct(
                shape=(), dtype=jnp.int32, sharding=cpu_sharding
            ),
        ),
        # kwargs
        {},
    ),
    out_specs_fn=lambda x: jax.ShapeDtypeStruct(
        shape=(), dtype=jnp.int32, sharding=cpu_sharding
    ),
)
f(x)  # `x` must match the input spec.
```

+++ {"id": "2L7aUBvsC_4m"}

`devices` specifies a list of devices that the colocated Python function
should run on. Having `devices` specialized lets a colocated Python function
without input arguments run.

```{code-cell}
:id: ZwWQRm_PDAll

def f():
  with open('/tmp/foo', 'w') as f:
    f.write('foo')
  return


f = colocated_python.colocated_python(f)
f = f.specialize(devices=cpu_devices)
f()  # Would be an error if `f` is not specialized with ``devices``.
```

+++ {"id": "xIjM-au9DBQL"}

## Colocated Python class

Colocated Python also supports wrapping Python classes. A real instance is
created on the hosts associated with the devices, and the caller side will get a
wrapper class that forwards all method calls to the real instance using
colocated Python.

```{code-cell}
:id: Ikb4Hh5iDB7Z

class Adder:

  def __init__(self, increment):
    print('Adder created')
    self.increment = increment

  def __del__(self):
    print('Adder destroyed')

  def add(self, x):
    return x + self.increment


Adder = colocated_python.colocated_python_class(Adder)
adder = Adder(1)
x = jax.device_put(1, cpu_sharding)
y = adder.add(x)
print(y)
```

+++ {"id": "t4i192BGDCw8"}

When the wrapper class instance is destroyed, the real instance is destroyed as
well. Note that this destruction will be asynchronous.

```{code-cell}
:id: j5g-NNYFDDln

del adder
```

+++ {"id": "UfQTjAu9DEV-"}

There are a few important semantic differences between colocated Python and
normal Python.

* A colocated Python class instance is created only on the hosts associated with
  the devices when any non-constructor method is called for the first time. In
  the above example, `Adder(1)` captures the constructor arguments
  `1`, but the actual constructor call `Adder(1)` on the hosts
  happens only when the first `adder.add(x)` call is made. This is because
  it is unknown what hosts the `Adder` instance should be created on until
  there is a call to its method.

* If the method(s) of the same wrapper class is called with inputs with
  different devices, the real instance may be created at different times on
  different hosts. If the first method call used CPU devices on host A, and the
  second method call used CPU devices on host B, the real instance will be
  created on host A during the first method call, and then on host B during the
  second method call.

* The methods of colocated Python classes are not yet specializable. The support
  will be added in the future.

+++ {"id": "YOsb92ChDFQd"}

## Execution order and concurrency

Colocated Python provides "program order" execution. Even if colocated Python
calls may be asynchronous (returning output JAX arrays without blocking), the
calls will be executed in the same order as the order the calls are made in the
user program. Thus, by default, colocated Python calls are sequentially
executed.

Several use cases of colocated Python will benefit from concurrent execution.
For example, one colocated Python call may take long time to return because it
may be doing expensive file reads, while another colocated Python call may need
to do file writes that are independent from the first one. This situation could
expect two calls to run concurrently without blocking each other.

Colocated Python provides concurrent execution if colocated Python calls are
made from different threads. For example, the below example would make two
colocated Python calls to run concurrently.

```{code-cell}
:id: l0L1-HaGDGHo

import concurrent.futures
import time


def f(x):
  time.sleep(1)
  return x + 1


f = colocated_python.colocated_python(f)
f = f.specialize(out_specs_fn=lambda x: x)  # Calls will be asynchronous.

with concurrent.futures.ThreadPoolExecutor(2) as executor:
  fut1 = executor.submit(f, x)
  fut2 = executor.submit(f, x)
  # Will finish in approximately 1 second instead of 2 seconds.
  jax.block_until_ready([fut1.result(), fut2.result()])
```

+++ {"id": "lRYja4_pDHFm"}

While calls from different threads run concurrently, on each thread, program
ordering will continue to apply.
