# Python and NumPy version support policy


JAX follows NumPy's [NEP-29 deprecation policy](https://numpy.org/neps/nep-0029-deprecation_policy.html). JAX supports at least:

* All minor versions of Python released 42 months prior to the project, and at minimum the two latest minor versions.

* All minor versions of numpy released in the 24 months prior to the project, and at minimum the last three minor versions.

JAX may support older versions of Python and NumPy, but support for older versions may be dropped at any time.