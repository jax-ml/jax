(about-the-project)=

# About the project

The JAX project is led by the JAX core team. We develop in the open,
and welcome open-source contributions from across the community. We
frequently see contributions from [Google
DeepMind](https://deepmind.google/), Alphabet more broadly,
[NVIDIA](https://docs.nvidia.com/deeplearning/frameworks/jax-release-notes/overview.html),
and elsewhere.

At the heart of the project is the [JAX
core](http://github.com/jax-ml/jax) library, which focuses on the
fundamentals of machine learning and numerical computing, at scale.

When [developing](#development) the core, we want to maintain agility
and a focused scope, so we lean heavily on a surrounding [modular
technology stack](#components). First, we design the `jax` module
to be
[composable](https://github.com/jax-ml/jax?tab=readme-ov-file#transformations)
and
[extensible](https://docs.jax.dev/en/latest/jax.extend.html), so
that a wide variety of domain-specific libraries can thrive outside of
it in a decentralized manner. Second, we lean heavily on a modular
backend stack (compiler and runtime) to target different
accelerators. Whether you are [writing a new domain-specific library
built with JAX](#upstack), or looking to [support
new hardware](#downstack), you can often
contribute these with *minimal to no modifications* to the JAX core
codebase.

Many of JAX's core contributors have roots in open-source software and
in research, in fields spanning computer science and the natural
sciences. We strive to continuously enable the cutting edge of machine
learning and numerical computing---across all compute platforms and
accelerators---and to discover the truths of array programming at
scale.

(development)=
## Open development

JAX's day-to-day development takes place in the open on GitHub, using
pull requests, the issue tracker, discussions, and [JAX Enhancement
Proposals
(JEPs)](https://docs.jax.dev/en/latest/jep/index.html). Reading
and participating in these is a good way to get involved. We also
maintain [developer
notes](https://docs.jax.dev/en/latest/contributor_guide.html)
that cover JAX's internal design.

The JAX core team determines whether to accept changes and
enhancements. Maintaining a simple decision-making structure currently
helps us develop at the speed of the research frontier. Open
development is a core value of ours, and we may adapt to a more
intricate decision structure over time (e.g. with designated area
owners) if/when it becomes useful to do so.

For more see [contributing to
JAX](https://docs.jax.dev/en/latest/contributing.html).

(components)=
## A modular stack

To enable (a) a growing community of users across numerical domains,
and (b) an advancing hardware landscape, we lean heavily on
**modularity**.

(upstack)=
### Libraries built on JAX

While the JAX core library focuses on the fundamentals, we want to
encourage domain-specific libraries and tools to be built on top of
JAX. Indeed, [many
libraries](https://docs.jax.dev/en/latest/#ecosystem) have
emerged around JAX to offer higher-level features and extensions.

How do we encourage such decentralized development? We guide it with
several technical choices. First, JAX's main API focuses on basic
building blocks (e.g. numerical primitives, NumPy operations, arrays,
and transformations), encouraging auxiliary libraries to develop
utilities as needed for their domain. In addition, JAX exposes a
handful of more advanced APIs for
[customization](https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html)
and
[extensibility](https://docs.jax.dev/en/latest/jax.extend.html). Libraries
can [lean on these
APIs](https://docs.jax.dev/en/latest/building_on_jax.html) in
order to use JAX as an internal means of implementation, to integrate
more with its transformations like autodiff, and more.

Projects across the JAX ecosystem are developed in a distributed and
often open fashion. They are not governed by the JAX core team, even
though sometimes team members contribute to them or maintain contact
with their developers.

(downstack)=
### A pluggable backend

We want JAX to run on CPUs, GPUs, TPUs, and other hardware platforms
as they emerge. To encourage unhindered support of JAX on new
platforms, the JAX core emphasizes modularity in its backend too.

To manage hardware devices and memory, and for compilation to such
devices, JAX calls out to the open [XLA
compiler](https://openxla.org/) and the [PJRT
runtime](https://github.com/openxla/xla/tree/main/xla/pjrt/c#pjrt---uniform-device-api). Both
of these are projects external to the JAX core, governed and
maintained by OpenXLA (again, with frequent contributions from and
discussion with the JAX core developers).

XLA aims for interoperability across accelerators (e.g. by ingesting
[StableHLO](https://openxla.org/stablehlo) as input) and PJRT offers
extensibility through a plug-in device API. Adding support for new
devices is done by implementing a backend lowering for XLA, and
implementing a plug-in device API defined by PJRT. If you're looking
to contribute to compilation, or to supporting new hardware, we
encourage you to contribute at the XLA and PJRT layers.

These open system components allow third parties to support JAX on new
accelerator platforms, *without requiring changes in the JAX
core*. There are several plug-ins in development today. For example, a
team at Apple is working on a PJRT plug-in to get [JAX running on
Apple Metal](https://developer.apple.com/metal/jax/).
