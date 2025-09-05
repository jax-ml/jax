(repro-note)=

# Repro implementation details 

The challenge for repro extraction for JAX, compared to a regular compiler,
is that JAX does not get the input as a data structure that we can save.
Instead, we have to augment the JAX tracing mechanism to **track** which
JAX API calls are being made by the user program, and what user functions
JAX calls while tracing the program. The repro tracker (in `tracker.py`)
constructs a representation of the call tree. Then the repro emitter
(in `emitter.py`) outputs a pure JAX program that would result in the
same call tree.

## How do we track?

First, we wrap the JAX API functions that take user functions as arguments,
e.g., `jax.jit`, `jax.vmap`. We do this by adding a `repro_api_name` to the
existing `traceback_util.api_boundary` annotation. This annotation was already
present in most places we needed it, but we had to add it in a few places
where it was missing, e.g., in `lax.loops.while_loop`.
Whenever we call one of these annotated APIs, we scan the arguments looking
for callables, and we wrap those as well. One goal would be to emit repro
code for these callables. 

We use the class `tracker.Func` to wrap callables of interest. They are of several
kinds:

  * JAX API functions. These are constructed for the JAX API entry points annotated
    with `repro_api_name`. 
  * USER functions. These are constructed for callables passed to JAX API functions.
  * JAX non-API functions. These are constructed for callables returned by JAX API
    functions, e.g., the returned value from `jax.jit`. Note: this kind of functions
    will go away, see below.

When one of the tracker functions is called, we construct a `tracker.Call` object
that has references to the `Func` that was called, the actual arguments and results
of the call (these would be actual tracers, or constants, or even non-JAX values
for the static arguments). The call objects for user functions have a body, which
is a list of calls to JAX functions that the user function makes.

Furthermore, we modified the `core.Primitive._true_bind` method to call into
the repro source code with the primitive and its arguments. If this call happens
while we are currently in a user call, we record the primitive.

Thus, the call objects for a user function will contain a list of calls to
JAX functions and to primitives.

### Dealing with JAX caches


## How do we emit?

TO EXPLAIN ...

## How do we reduce?

TO DO ...
