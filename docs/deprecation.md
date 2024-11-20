(version-support-policy)=
# Python and NumPy version support policy

<!--* freshness: { reviewed: '2024-05-02' } *-->

For NumPy and SciPy version support, JAX follows the Python scientific community's
[SPEC 0](https://scientific-python.org/specs/spec-0000/).

For Python version support, we have heard from users that a 36-month support window can
be too short, for example due to the delays in propagation of new CPython releases
to Linux vendor releases. For this reason JAX supports Python versions for at least
nine months longer than SPEC-0 recommends.

This means we support at least:

* All Python feature releases in the 45 months prior to each JAX release. For example:

  * **Python 3.10** was released October 2021, and will be supported in new JAX releases at least until **July 2025**.
  * **Python 3.11** was released October 2022, and will be supported in new JAX releases at least until **July 2026**.
  * **Python 3.12** was released October 2023, and will be supported in new JAX releases at least until **July 2027**.
  * **Python 3.13** was released October 2024, and will be supported in new JAX releases at least until **July 2028**.

* All NumPy feature releases in the 24 months prior to each JAX release. For example:

  * **NumPy 1.24** was released December 2022, and will be supported in new JAX releases at least until **December 2024**.
  * **NumPy 1.25** was released June 2023, and will be supported in new JAX releases at least until **June 2025**
  * **NumPy 1.26** was released September 2023, and will be supported in new JAX releases at least until **September 2025**
  * **NumPy 2.0** was released June 2024, and will be supported in new JAX releases at least until **June 2026**
  * **NumPy 2.1** was released August 2024, and will be supported in new JAX releases at least until **August 2026**

* All SciPy feature releases in the 24 months prior to each JAX release. For example:

  * **Scipy 1.10** was released January 2023, and will be supported in new JAX releases at least until **January 2025**.
  * **Scipy 1.11** was released June 2023, and will be supported in new JAX releases at least until **June 2025**.
  * **Scipy 1.12** was released January 2024, and will be supported in new JAX releases at least until **January 2026**.
  * **Scipy 1.13** was released April 2024, and will be supported in new JAX releases at least until **April 2026**.
  * **Scipy 1.14** was released June 2024, and will be supported in new JAX releases at least until **June 2026**.

JAX releases may support older versions of Python, NumPy, and SciPy than strictly required
by this policy, but support for older versions may be dropped at any time beyond the listed
dates.
