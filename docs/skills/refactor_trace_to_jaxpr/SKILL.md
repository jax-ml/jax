---
name: refactor-trace-to-jaxpr
description: >-
  Guides the refactoring of JAX tracing API from trace_to_jaxpr_dynamic
  to trace_to_jaxpr. Use when replacing trace_to_jaxpr_dynamic calls
  with trace_to_jaxpr and tree_util.FlatTree.
---

# Refactoring JAX Tracing API

This skill guides you through refactoring JAX Pallas tracing calls from the
legacy `pe.trace_to_jaxpr_dynamic` to the newer `pe.trace_to_jaxpr`.

## API Comparison

| Legacy API                           | Modern API (`pe.trace_to_jaxpr`)  |
: (`pe.trace_to_jaxpr_dynamic`)        :                                   :
| :----------------------------------- | :-------------------------------- |
| **Signature**:                       | **Signature**:                    |
: `pe.trace_to_jaxpr_dynamic(fun,      : `pe.trace_to_jaxpr(fun, in_avals, :
: in_avals)`                           : debug_info)`                      :
| **Input type (`in_avals`)**: Flat    | **Input type (`in_avals`)**:      |
: list of avals                        : `tree_util.FlatTree`              :
| **Return type**: `(core.Jaxpr, out_avals, | **Return type**: `(core.ClosedJaxpr,  |
: consts)`                             : out_avals_flat_tree)`                    :

## Refactoring Workflow

### 1. Direct Tracing (Preferred)

The easiest case to refactor is when you have `lu.wrap_init` followed by `api_util.flatten_fun` (or `api_util.flatten_fun_nokwargs`), followed by a call to `pe.trace_to_jaxpr_dynamic`.

In these cases, `pe.trace_to_jaxpr` can trace the original Python callable directly. You do not need to manually wrap or flatten the function using `lu.wrap_init` and `api_util.flatten_fun` (or `api_util.flatten_fun_nokwargs`).

**Before:**

```python
# Create a tuple representing the function's arguments: (args, kwargs)
args_kwargs = (args, kwargs)
flat_args, in_tree = tree_util.tree_flatten(args_kwargs)

flat_fun, out_tree_thunk = api_util.flatten_fun(
    lu.wrap_init(fun, debug_info=debug_info), in_tree
)
some_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, flat_args)
out_tree = out_tree_thunk()
```

**After:**

```python
# Create the input FlatTree directly from the (args, kwargs) tuple:
in_avals_ft = tree_util.FlatTree.flatten((args_avals, kwargs_avals))

# Trace the function directly:
some_jaxpr, out_avals_ft = pe.trace_to_jaxpr(
    fun,
    in_avals_ft,
    debug_info,
)
out_tree = out_avals_ft.tree
```

> [!IMPORTANT]
> The `FlatTree` passed to `pe.trace_to_jaxpr` as the second argument (`in_avals`) *must* represent the tuple `(args, kwargs)`. Under the hood, JAX uses this structure to map and unpack the arguments and keyword arguments before calling the traced function.

Be careful that `some_jaxpr` after refactoring is a `core.ClosedJaxpr` while
before it was a `core.Jaxpr`.

The `debug_info` passed to `pe.trace_to_jaxpr` can be the same that was
passed to `pe.trace_to_jaxpr_dynamic` (or to `lu.wrap_init`).

> [!TIP]
> If you are calling the tracing function for a locally defined function (e.g., a helper function defined nested inside another function or test method), it is recreated on each execution and is guaranteed to never hit the cache. In this case, use `pe.trace_to_jaxpr_nocache` instead of `pe.trace_to_jaxpr` to avoid cache lookup overhead and cache pollution.

#### Unflattening Output

To get the unflattened output avals:

```python
unflat_avals = out_avals_ft.unflatten()
```

#### Accessing Flat Output Avals

To iterate over flat output avals:

```python
for ov in out_avals_ft.vals:
  ...
```

### 2. Flattening and Mapping with `FlatTree`

When you need to adjust or transform the tracing arguments (e.g., getting abstract values for references) before tracing, do not use `tree_util.tree_flatten` to create a flat list. Instead, use `tree_util.FlatTree.flatten(tree)` to get a `FlatTree`, and use `.map(...)` to transform the values.

The new `pe.trace_to_jaxpr` has built-in support for flattening; it accepts the `FlatTree` directly.

**Before:**

```python
flat_args, in_tree = tree_util.tree_flatten((args, kwargs))
ref_avals = tuple(t.get_ref_aval() for t in flat_args)

flat_fun, out_tree_thunk = api_util.flatten_fun(
    lu.wrap_init(fun, debug_info=debug_info), in_tree
)
some_jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, ref_avals)
```

**After:**

```python
args_ft = tree_util.FlatTree.flatten((args_avals, kwargs_avals))
ref_avals_ft = args_ft.map(lambda x: x.get_ref_aval())

# trace_to_jaxpr has built-in handling of FlatTree:
closed_jaxpr, out_avals_ft = pe.trace_to_jaxpr(
    fun, ref_avals_ft, debug_info
)
```

If you have only positional arguments,
you can flatten them using `flatten_args`:
```python
in_avals_ft = tree_util.FlatTree.flatten_args(*args)
```

### 3. Replacing `linear_util` Transformations

If you have a `linear_util` transformation (defined with `@lu.transformation` or `@lu.transformation2`) wrapping the function before tracing, you should replace it with a plain Python function decorator/wrapper.

Because `pe.trace_to_jaxpr` is decorated with `@weakref_lru_cache`, tracing uses the function object itself as a cache key. Therefore, any custom wrapping function must have a stable identity:
* Define the wrapping function utilizing a caching decorator, such as `@util.weakref_lru_cache`.
* Inside the wrapper `wrapped(*args, **kwargs)`, use `tree_util.FlatTree.flatten((args, kwargs))` to access the flat arguments, map the transformation on the flat tree, call `.unflatten()` to retrieve the unpacked arguments, and then delegate to the nested function.

**Before:**

```python
@lu.transformation2
def wrap_with_transforms(f, transforms, *args):
  new_args = tuple(
      state_types.TransformedRef(a, t) if t else a
      for a, t in zip(args, transforms)
  )
  return f(*new_args)

# In tracer:
wrapped_fun, out_tree_thunk = api_util.flatten_fun(
    lu.wrap_init(fun, debug_info=debug_info), in_tree)
wrapped_fun = primitives.wrap_with_transforms(
    wrapped_fun, transforms
)
jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(
    wrapped_fun, flat_args
)
```

**After:**

```python
@util.weakref_lru_cache
def wrap_with_transforms(
    fun: Callable,
    transforms: tuple[tuple[state_types.Transform, ...], ...],
) -> Callable:
  def wrapped(*args, **kwargs):
    # Flatten the arguments/kwargs tuple
    args_ft = tree_util.FlatTree.flatten(
        (args, kwargs), registry=tree_util.default_registry
    )
    # Map transformation over the FlatTree
    transformed_ft = args_ft.map2(
        lambda a, t: state_types.TransformedRef(a, t) if t else a,
        transforms
    )
    # Unflatten back to positional/kw args tuple and call the function
    t_args, t_kwargs = transformed_ft.unflatten()
    return fun(*t_args, **t_kwargs)
  return wrapped

# In tracer:
fun_with_transforms = wrap_with_transforms(fun, transforms)
closed_jaxpr, out_avals_ft = pe.trace_to_jaxpr(
    fun_with_transforms, args_avals_ft, debug_info
)
```


## Gotchas & Failure Modes

### 1. Unhashable wrapper functions (`TypeError: unhashable type`)

*   **Symptom**: Calling `pe.trace_to_jaxpr` raises `TypeError: unhashable type:
    '...'` when passing the function wrapper.
*   **Cause**: `pe.trace_to_jaxpr` is decorated with `@weakref_lru_cache`, which
    requires all arguments to be hashable. Wrapper classes (like `_IndexMapFunc`
    in `core.py`) that implement `__eq__` but do not implement `__hash__` will
    be considered unhashable by Python.

    Ask the user for guidance if you have this situation.

