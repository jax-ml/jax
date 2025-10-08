---
orphan: true
---
(direct-linearize-migration)=
# JAX direct linearize

<!--* freshness: { reviewed: '2025-07-21 } *-->

### What’s going on?

We're changing the way JAX implements autodiff internally. Previously grad was
done by a three-stage process: JVP, partial eval, transposition. With this
change we've bundled together the first two steps, JVP and partial eval, into a
new transformation: linearization.

This should mostly not change user-visible behavior. Some exceptions:

  * you'll see LinearizeTracer instead of JVPTracer if you print out traced values during autodiff.

  * It's possible that some numerics will change, just for the usual reason that any perturbation to programs can slightly alter numerical results.


### Why?

The upgrade unlocks several new features, like:

  * differentiation involving Pallas-style mutable array references;

  * simpler and more flexible user-defined autodiff rules, like custom_vjp/jvp;

  * controlling the autodiff behavior on user-defined types.

### This change broke my stuff!

For now, you can still get the old behavior by unsetting the use_direct_linearize config option:

  * set the shell environment variable to something falsey, e.g. JAX_USE_DIRECT_LINEARIZE=0

  * set the config option jax.config.update('jax_use_direct_linearize', False)

  * if you parse flags with absl, you can pass the command-line flag --jax_use_direct_linearize=false

We plan to remove the config option on August 16th 2025.
