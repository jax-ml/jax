(version-support-policy)=
# Python and NumPy version support policy

For NumPy and SciPy version support, JAX follows the Python scientific community's
[SPEC 0](https://scientific-python.org/specs/spec-0000/).

For Python version support, we have heard from users that a 36-month support window can
be too short, for example due to the delays in propagation of new CPython releases
to Linux vendor releases. For this reason JAX supports Python versions for at least
nine months longer than SPEC-0 recommends.

This means we support at least:

* All minor Python releases in the 45 months prior to each JAX release.

* All minor NumPy releases in the 24 months prior to each JAX release.

* All minor SciPy releases in the 24 months prior to each JAX release, starting
  with SciPy version 1.9

JAX releases may support older versions of Python, NumPy, and SciPy, but support
for older versions may be dropped at any time.
