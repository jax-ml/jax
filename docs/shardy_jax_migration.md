---
orphan: true
---
(shardy-jax-migration)=
# Shardy JAX Migration

<!--* freshness: { reviewed: '2025-07-18' } *-->

## TL;DR

### What’s going on?

[Shardy](https://openxla.org/shardy) is a new partitioning system co-developed
by GDM Model Scaling (author of [PartIR](https://arxiv.org/abs/2401.11202)) and
XLA/CoreML teams (author of [GSPMD](https://arxiv.org/abs//2105.04663)). Shardy
aims to provide better usability and control to users, and will gradually
replace GSPMD and PartIR.

After the migration is complete in March 2026, Shardy will be the only
partitioner in JAX.

Until then, as a temporary workaround for any problems, Shardy
[can be disabled](#how-can-i-disable-shardy-for-now). Please file a
[JAX issue](https://github.com/jax-ml/jax/issues) if you encounter any problem.

### How do I know if Shardy broke my code?

The easiest way to tell if Shardy is responsible for any problems is to disable
Shardy and see if the issues go away. See
[What issues can arise when Shardy is switched on?](#what-issues-can-arise-when-shardy-is-switched-on)
section below.

You can tell that Shardy is enabled by looking for
`Using Shardy for XLA SPMD propagation in the logs`.

### How can I disable Shardy for now?

Until March, 2026 it will be possible to temporarily disable Shardy by:

  * setting the shell environment variable `JAX_USE_SHARDY_PARTITIONER` to
    something false-like (e.g., 0);

  * setting the boolean flag `jax_use_shardy_partitioner` to something
    false-like if your code parses flags with absl;

  * using this statement in your main file or anywhere before you call
    `jax.jit`:

    ``` python
    import jax
    jax.config.update('jax_use_shardy_partitioner', False)
    ```

To debug partitioning with Shardy enabled, you can enable MLIR dumps as follows:

```
--xla_dump_hlo_pass_re=shardy --xla_dump_to=<some_directory>
```

NOTE: Please disable only the specific use cases that are not working as
expected if possible, and file a [bug](https://github.com/jax-ml/jax/issues)
with a reproducer, so we can resolve it asap and re-enable Shardy.

### JAX export backwards compatibility

Enabling Shardy in JAX by default is maintaining the 6 months backwards
compatibility guarantee. This means that you will be able to load a model
exported with Shardy disabled for at least 6 months after Shardy becomes enabled
for your model. That old checkpointed model will run with GSPMD, and only when
re-exporting the model will it start running with Shardy.

However, if you still encounter an issue with loading an old checkpoint, please
contact us or file a [bug](https://github.com/jax-ml/jax/issues).

NOTE: exporting a model with Shardy enabled, then loading it with Shardy
disabled isn’t supported and will fail.

### How do I prepare for Shardy being enabled in March 2026 permanently?

Due to us falling back to GSPMD for any JAX export checkpoint for 6 months, to
help find any potential issues, please re-export any models you have with Shardy
enabled. Then you can see if it runs fine, or there is any bug we need to fix.

## What issues can arise when Shardy is switched on?

### Performance regression or OOM

While Shardy improves on the existing sharding propagation systems (GSPMD and
PartIR), it can sometimes output slightly different results due to different
propagation order or conflict resolution heuristics.

This doesn’t necessarily mean that Shardy is doing the wrong thing, but possibly
that there aren't enough sharding constraints in the program, so a small change
in propagation order can affect the final result. It can also hint that existing
sharding constraints were overfitted to GSPMD and require slight adjustments
with Shardy.

Therefore, it is possible that enabling Shardy will cause some models to have a
performance regression or OOM (especially if the model was already close to the
memory capacity). However, we have already migrated many use cases across
Alphabet, and have observed equivalent or better performance than GSPMD.

To resolve such issues, users can either:

1.  Disable Shardy temporarily and open a [bug](https://github.com/jax-ml/jax/issues)
    with a reproducer.
2.  Add additional sharding constraints to make sure Shardy does the desired
    thing.

### Compilation failure

We have done extensive testing across many JAX models. However, it’s possible
that there are certain edge cases or situations we don’t support/handle (because
we didn't know we needed to).

This means that although rare, it’s possible that you will get a compilation
failure in the form of a segfault, hard check, python value error, etc.

In such a case, please disable Shardy temporarily and open a
[bug](https://github.com/jax-ml/jax/issues) with a reproducer.

### Inconsistent value of the use Shardy flag

If Shardy is disabled somewhere in your code, but there are still paths that use
the default value of the JAX flag, this can cause issues. For example, exporting
a model with Shardy enabled, then loading it with Shardy disabled isn’t
supported and will fail (the other way is supported for
[backwards compatibility](#jax-export-backwards-compatibility)).

The symptom for an issue like this can be an error in JAX or in XLA/Shardy, or
just undefined behavior. You can try disabling Shardy globally in
[JAX config](https://github.com/jax-ml/jax/blob/main/jax/_src/config.py) to see
if the issue goes away.

NOTE: Please ensure that Shardy is disabled consistently if needed, or remove
any explicit modification of the flag, to have the default value apply
throughout.

### New way to use the JAX `jax.experimental.custom_partitioning` API

If you use this API, you may see the error

```
Shardy is used, but sharding propagation callbacks instead of sharding_rule are
provided. Need to provide sharding_rule to migrate to Shardy.
```

Instead of defining `infer_sharding_from_operands` and `propagate_user_sharding`
callbacks, define a `jax.experimental.SdyShardingRule` that specifies an einsum-like relationship between dimensions during propagation. Refer to the [`custom_partitioning` doc](https://docs.jax.dev/en/latest/jax.experimental.custom_partitioning.html#module-jax.experimental.custom_partitioning)
for more info on how to define a sharding rule.

### `jax.export` requires all inputs and outputs to have the same mesh

As part of the Shardy migration, `jax.export` now requires all input/output
shardings to live on the same mesh - same axis names and sizes.
