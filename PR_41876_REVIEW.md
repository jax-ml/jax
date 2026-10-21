# Review of #41876: RefAccum aliasing across trivial reshapes

PR: https://github.com/jax-ml/jax/pull/41876, reviewed at its current head,
911f4d4b7e. This branch is that commit plus one commit that adds this file and
`tests/ref_accum_reshape_review_test.py`, which has a repro for each finding
below:

```
python -m pytest -q tests/ref_accum_reshape_review_test.py
```

## Summary

On main, a squeeze, reshape or broadcast_in_dim whose input is a RefAccum
produces a ValAccum. With this PR it produces a RefAccum view of the same root
Ref. Several consumers of GradAccums handle a root RefAccum, or reject RefAccums
outright, but don't handle views. So programs that differentiated fine on main
now raise. The view only appears when the reshape is applied directly
(`lax.squeeze`, `x[0]`, `x[None]`, `x.reshape(...)`). `jnp.squeeze` is an
inlined jit, so under an eager `jax.vjp` it stays a jit eqn and doesn't alias.

Test results (same with `JAX_ENABLE_X64=0` and `=1`, and with
`JAX_CUSTOM_JVP3=1 JAX_CUSTOM_VJP3=1 JAX_REMAT3=1`):

| test | main | PR head |
|---|---|---|
| `test_vmap_batched_grad_ref_through_squeeze` (x3) | pass | fail |
| `test_vmapped_custom_vjp_with_accums_through_squeeze` | pass | fail |
| `test_shard_map_check_vma_false_reshaped_view` | pass | fail |
| `test_shard_map_root_and_reshaped_view` (x2) | pass | fail |
| `test_scan_over_x_closing_over_reshaped_x` | pass | fail |
| `test_custom_gradient_ref_accum_through_squeeze` | fail (clear error) | fail (internal error) |
| `test_jit_in_shardings_reshaped_view_is_aliased` (P(), P(None)) | fail | fail |
| `test_jit_in_shardings_reshaped_view_is_aliased` (unspecified) | fail | pass |

## Regressions

### 1. vmap of a backward pass with a batched gradient Ref

`test_vmap_batched_grad_ref_through_squeeze`

```python
f = lambda x: lax.squeeze(x, (0,))[1:3].sum()
def per_example(x):
  ref = jax.new_ref(jnp.zeros_like(x))
  jax.vjp(f, x)[1].with_refs(ref)(1.)
  return jax.freeze(ref)
jax.vmap(per_example)(jnp.ones((3, 1, 6)))
# TypeError: Cannot reshape VmapTracer(aval=Ref{float32[1,6]}, batched=Ref{float32[3,1,6]}) to (6,)
```

The view's `.ref` is `TransformedRef(raw_ref, (ReshapeTransform(...),))`.
Under vmap the raw ref is a batch tracer, and
`ReshapeTransform.transform_type` only matches `AbstractRef` and `ShapedArray`
([types.py:190](jax/_src/state/types.py#L190)). The same failure happens for
slice, gather and dynamic_slice
([slicing.py:1437](jax/_src/lax/slicing.py#L1437),
[:1559](jax/_src/lax/slicing.py#L1559),
[:2262](jax/_src/lax/slicing.py#L2262)), and for user rules that write
`x_acc.ref[i] += g`. If `transform_type` handled tracers, the state vmap rules
would still have to batch a ReshapeTransform followed by an NDIndexer, and
`_addupdate_vmap` rejects more than one indexer.

Suggestion: a view only inserts or removes size-1 axes, so an indexer on the
view can be rewritten as an indexer on the root by adding a `0` index for each
removed axis and dropping each added axis. Then no ReshapeTransform is needed,
and the existing batching rules apply.

### 2. vmapped custom_vjp with accumulators

`test_vmapped_custom_vjp_with_accums_through_squeeze`

```python
f = lambda x: jax.vmap(total)(lax.squeeze(x, (0,))).sum()  # total uses defvjp_with_accums
# ValueError: under vmap, the custom_vjp function total has a bwd rule that takes gradient accumulators, ...
```

`CustomBwdWithValAccums.call_with_accums`
([ad.py:1280](jax/_src/interpreters/ad.py#L1280)) and hijax `vjp_bwd` with
`disallow_ref_accums` ([hijax.py:894](jax/_src/hijax.py#L894)) reject every
RefAccum. On main the rule got a ValAccum for the squeezed input.

Suggestion: for a view, hand the rule a fresh ValAccum and then call
`view.accum(val.freeze())`. That is what main did through the squeeze
transpose. Keep the error for root RefAccums.

### 3. shard_map with check_vma=False and a replicated in_spec

`test_shard_map_check_vma_false_reshaped_view`

```python
g = jax.shard_map(lambda y: y.sum(keepdims=True), in_specs=P(), out_specs=P(), check_vma=False)
f = lambda w: g(lax.squeeze(w, (0,))).sum()
# NotImplementedError: Differentiating a shard_map with check_vma=False into a gradient Ref ...
```

A view is kept only when its ct spec is unpartitioned. `_check_grad_refs` then
uses `P()` for it ([shard_map.py:1928](jax/_src/shard_map.py#L1928)).
`_unmentioned2(mesh, P(), ...)` lists every manual axis, so with
`check_vma=False` this check always raises
([shard_map.py:1929](jax/_src/shard_map.py#L1929)). On main the view was a
ValAccum, and its cotangent was psummed and accumulated outside.

Suggestion: in the `args_for_project` fallback, also turn views into ValAccums
when `not check_vma`.

### 4. shard_map that gets both the root array and a view of it

`test_shard_map_root_and_reshaped_view`

```python
g = jax.shard_map(lambda a, b: a * b.sum(), mesh=mesh, in_specs=(P(None), P()), out_specs=P(None))
f = lambda w: g(w, w[None]).sum()
# NotImplementedError: Differentiating a shard_map into one gradient Ref through arguments with different in_specs.
```

The root records `sp.to_ct_spec()` and the view records `P()`, and the dedupe
check compares them with `!=` ([shard_map.py:1933](jax/_src/shard_map.py#L1933)).
That fails for a root spec of `P('x')`, and also for `P(None)` (`P(None) != P()`),
even though both specs are replicated.

Suggestion: when a view's spec differs from the one recorded for its root,
turn the view into a ValAccum instead of raising.

### 5. scan over x that also closes over a view of x

`test_scan_over_x_closing_over_reshaped_x`

```python
def f(x):
  x_row = x[None]
  c, _ = lax.scan(lambda c, xi: (c + xi * x_row[0, 0], None), 0., x)
  return c
# NotImplementedError: Differentiating a scan with respect to an array that is both a constant and a scanned-over input, ...
```

`x_row` is now a view that shares `x`'s root, and `_scan_accum_class` puts it in
`mut_consts_bar`. The const/xs overlap check then raises
([loops.py:1159](jax/_src/lax/control_flow/loops.py#L1159)). On main `x_row`
was a ValAccum const.

Suggestion: when a const view shares a root with an xs, classify the view as
Val, the same way extensive views with a mismatched leading dim already fall
back to Val.

## Other findings

### 6. custom_gradient with a declared RefAccum now crashes on a view

`test_custom_gradient_ref_accum_through_squeeze`

```python
@partial(jax.custom_gradient, with_accums=ad.RefAccum)
def second(y):
  def rule(g, y_acc):
    y_acc.ref[1] += g
  return y[1], rule
f = lambda x: second(lax.squeeze(x, (0,)))
# TransformedRefAvalError: TransformedRefs cannot be abstractified.
```

`_apply_accums_closure_residuals` passes `acc.inst().ref` to `eval_jaxpr`
([custom_derivatives.py:1589](jax/_src/custom_derivatives.py#L1589)). For a
view that's a TransformedRef, but the rule's jaxpr was traced with a plain
`AbstractRef` of the view's shape. On main this raised the documented
TypeError ("declared a RefAccum ... but its backward pass got a ValAccum").
Either keep that error for views, or make the rule work with the view.

### 7. pjit's aliasing path never fires with explicit in_shardings

`test_jit_in_shardings_reshaped_view_is_aliased`

`_can_keep_reshaped_ref_accum` compares a NamedSharding on a concrete `Mesh`
with the root aval's sharding, which uses an `AbstractMesh`
([pjit.py:1897](jax/_src/pjit.py#L1897)). They are never equal, so with any
explicit `in_shardings` the view falls back to a ValAccum, the transposed jit
doesn't take the Ref, and the `P()` rewrite in `_ref_in_sharding` never runs.
The test checks whether the transposed jit eqn has a Ref input. It does with
unspecified in_shardings, and it doesn't with `P()` or `P(None)`. This costs
performance, not correctness.

Suggestion: compare `s.mesh.abstract_mesh` with the aval's mesh, and check
that the spec is unpartitioned, instead of comparing whole shardings.

### 8. Fixed: reduced and unreduced specs

The review first ran against an earlier revision of the PR (78e5a50eba). That
revision iterated `for p in sharding.spec` in `_is_trivial_reshape`, in
`_can_keep_reshaped_ref_accum` and in shard_map. `PartitionSpec.__iter__`
raises when the spec has `reduced` or `unreduced` axes. The current head checks
`reduced` and `unreduced` and reads `.partitions` at all three sites, so this
is fixed.

### 9. Cleanup: one predicate for "is a reshaped view"

The check "`_root is not self`, both avals are ShapedArrays, and the shapes
differ" is written out in `RefAccum.ref`, `accum` and `freeze`
([ad.py:476-516](jax/_src/interpreters/ad.py#L476)). It also appears as
`_is_reshaped_ref_accum` in both pjit.py and shard_map.py, and as a variant in
`_scan_accum_class`. A single `RefAccum` property would keep these consistent.
It would also give findings 1-6 one condition to check: each consumer that
can't handle a view turns it into a ValAccum, as pjit and shard_map already do
in `args_for_project`.
