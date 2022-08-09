JAX Enhancement Proposals (JEPs)
================================

Most changes can be discussed with simple issues/discussions and pull requests.

Some changes though are a bit larger in scope or require more discussion, and
these should be implemented as JEP. This allows for writing longer documents
that can be discussed in a pull request themselves.

The structure of JEPs is kept as lightweight as possible to start and might
be extended later on.

When you should use a JEP
-------------------------

- When your change requires a design doc. We prefer collecting the designs as
  JEPs for better discoverability and further reference.

- When your change requires extensive discussion. It's fine to have relatively
  short discussions on issues or pull requests, but when the discussion gets
  longer this becomes unpractical for later digestion. JEPs allow to update the
  main document with a summary of the discussion and these updates can be
  discussed themselves in the pull request adding the JEP.

How to start a JEP
------------------

First, create an issue with the `JEP label`_. All pull requests that relate to
the JEP (i.e. adding the JEP itself as well as any implementing pull requests)
should be linked to this issue.

Then create a pull request that adds a file named
`%d-{short-title}.md` - with the number being the issue number.

.. _JEP label: https://github.com/google/jax/issues?q=label%3AJEP

.. toctree::
  :maxdepth: 1

  263: JAX PRNG Design <263-prng>
  2026: Custom JVP/VJP rules for JAX-transformable functions <2026-custom-derivatives>
  4410: Omnistaging <4410-omnistaging>
  9407: Design of Type Promotion Semantics for JAX <9407-type-promotion>
  9419: Jax and Jaxlib versioning <9419-jax-versioning>
  10657: Sequencing side-effects in JAX <10657-sequencing-effects>




