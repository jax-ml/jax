# Contributing to JAX

<!--* freshness: { reviewed: '2023-11-16' } *-->

Everyone can contribute to JAX, and we value everyone's contributions. There are several
ways to contribute, including:

- Answering questions on JAX's [discussions page](https://github.com/jax-ml/jax/discussions)
- Improving or expanding JAX's [documentation](https://docs.jax.dev)
- Contributing to JAX's [code-base](https://github.com/jax-ml/jax)
- Contributing in any of the above ways to the broader ecosystem of [libraries built on JAX](https://github.com/jax-ml/jax#neural-network-libraries)

The JAX project follows [Google's Open Source Community Guidelines](https://opensource.google/conduct/).

## Ways to contribute

We welcome pull requests, in particular for those issues marked with
[contributions welcome](https://github.com/jax-ml/jax/issues?q=is%3Aopen+is%3Aissue+label%3A%22contributions+welcome%22) or
[good first issue](https://github.com/jax-ml/jax/issues?q=is%3Aopen+is%3Aissue+label%3A%22good+first+issue%22).

For other proposals, we ask that you first open a GitHub
[Issue](https://github.com/jax-ml/jax/issues/new/choose) or
[Discussion](https://github.com/jax-ml/jax/discussions)
to seek feedback on your planned contribution.

## Can I contribute AI generated code?

All submissions to Google Open Source projects need to follow Google's [Contributor License
Agreement (CLA)](https://cla.developers.google.com/), in which contributors agree that their
contribution is an original work of authorship. This doesn’t prohibit the use of coding
assistance tools, but what’s submitted does need to be a contributor's original creation.

In the JAX project, a main concern with AI-generated contributions is that
**low-quality AI-generated code imposes a disproportionate review cost**.
Since the team's capacity for code review is limited, we have a higher bar
for accepting AI-generated contributions compared to those written by a human.

A loose rule of thumb: if the team needs to spend more time reviewing a
contribution than the contributor spends generating it, then the contribution
is probably not helpful to the project, and we will likely reject it.

## Contributing code using pull requests

We do all of our development using git, so basic knowledge is assumed.

Follow these steps to contribute code:

1. Sign the [Google Contributor License Agreement (CLA)](https://cla.developers.google.com/).
   For more information, see the {ref}`pr-checklist` below.

2. Fork the JAX repository by clicking the **Fork** button on the
   [repository page](https://github.com/jax-ml/jax). This creates
   a copy of the JAX repository in your own account.

3. Install Python >= 3.11 locally in order to run tests.

4. `pip` installing your fork from source. This allows you to modify the code
   and immediately test it out:

   ```bash
   git clone https://github.com/YOUR_USERNAME/jax
   cd jax
   pip install -r build/test-requirements.txt  # Installs all testing requirements.
   pip install -e ".[cpu]"  # Installs JAX from the current directory in editable mode.
   ```

5. Add the JAX repo as an upstream remote, so you can use it to sync your
   changes.

   ```bash
   git remote add upstream https://www.github.com/jax-ml/jax
   ```

6. Create a branch where you will develop from:

   ```bash
   git checkout -b name-of-change
   ```

   And implement your changes using your favorite editor (we recommend
   [Visual Studio Code](https://code.visualstudio.com/)).

7. Make sure your code passes JAX's lint and type checks, by running the following from
   the top of the repository:

   ```bash
   pip install pre-commit
   pre-commit run --all
   ```

   See {ref}`linting-and-type-checking` for more details.

8. Make sure the tests pass by running the following command from the top of
   the repository:

   ```bash
   pytest -n auto tests/
   ```

   Run them in 64-bit mode as well, by setting the environment variable `JAX_ENABLE_X64=True`:

   ```bash
   JAX_ENABLE_X64=True pytest -n auto tests/
   ```

   JAX's test suite is quite large, so if you know the specific test file that covers your
   changes, you can limit the tests to that; for example:

   ```bash
   pytest -n auto tests/lax_scipy_test.py
   ```

   You can narrow the tests further by using the `pytest -k` flag to match particular test
   names:

   ```bash
   pytest -n auto tests/lax_scipy_test.py -k testLogSumExp
   ```

   JAX also offers more fine-grained control over which particular tests are run;
   see {ref}`running-tests` for more information.

9. Once you are satisfied with your change, create a commit as follows (
   [how to write a commit message](https://chris.beams.io/posts/git-commit/)):

   ```bash
   git add file1.py file2.py ...
   git commit -m "Your commit message"
   ```

   Then sync your code with the main repo:

   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

   Finally, push your commit on your development branch and create a remote
   branch in your fork that you can use to create a pull request from:

   ```bash
   git push --set-upstream origin name-of-change
   ```

   Please ensure your contribution is a single commit (see {ref}`single-change-commits`)

10. Create a pull request from the JAX repository and send it for review.
    Check the {ref}`pr-checklist` for considerations when preparing your PR, and
    consult [GitHub Help](https://help.github.com/articles/about-pull-requests/)
    if you need more information on using pull requests.

(pr-checklist)=

## JAX pull request checklist

As you prepare a JAX pull request, here are a few things to keep in mind:

### Google contributor license agreement

Contributions to this project must be accompanied by a Google Contributor License
Agreement (CLA). You (or your employer) retain the copyright to your contribution;
this simply gives us permission to use and redistribute your contributions as
part of the project. Head over to <https://cla.developers.google.com/> to see
your current agreements on file or to sign a new one.

You generally only need to submit a CLA once, so if you've already submitted one
(even if it was for a different project), you probably don't need to do it
again. If you're not certain whether you've signed a CLA, you can open your PR
and our friendly CI bot will check for you.

(single-change-commits)=

### Single-change commits and pull requests

A git commit ought to be a self-contained, single change with a descriptive
message. This helps with review and with identifying or reverting changes if
issues are uncovered later on.

**Pull requests typically comprise a single git commit.** (In some cases, for
instance for large refactors or internal rewrites, they may contain several.)
In preparing a pull request for review, you may need to squash together
multiple commits. We ask that you do this prior to sending the PR for review if
possible. The `git rebase -i` command might be useful to this end.

(linting-and-type-checking)=

### Linting and type-checking

JAX uses [mypy](https://mypy.readthedocs.io/) and
[ruff](https://docs.astral.sh/ruff/) to statically test code quality; the
easiest way to run these checks locally is via the
[pre-commit](https://pre-commit.com/) framework:

```bash
pip install pre-commit
pre-commit run --all-files
```

If your pull request touches documentation notebooks, this will also run some checks
on those (See {ref}`update-notebooks` for more details).

### Full GitHub test suite

Your PR will automatically be run through a full test suite on GitHub CI, which
covers a range of Python versions, dependency versions, and configuration options.
It's normal for these tests to turn up failures that you didn't catch locally; to
fix the issues you can push new commits to your branch.

### Restricted test suite

Once your PR has been reviewed, a JAX maintainer will mark it as `pull ready`. This
will trigger a larger set of tests, including tests on GPU and TPU backends that are
not available via standard GitHub CI. Detailed results of these tests are not publicly
viewable, but the JAX maintainer assigned to your PR will communicate with you regarding
any failures these might uncover; it's not uncommon, for example, that numerical tests
need different tolerances on TPU than on CPU.

### Wheel sources update

If a new python package or a new file is added to the wheel, one of the
following Bazel targets should be updated:

[jax wheel sources](https://github.com/jax-ml/jax/blob/0080c5c934d9e3668c93d88e2b94f66e05f9d8d8/BUILD.bazel#L29)

[jaxlib wheel sources](https://github.com/jax-ml/jax/blob/0080c5c934d9e3668c93d88e2b94f66e05f9d8d8/jaxlib/tools/BUILD.bazel#L151)

[jax CUDA plugin wheel sources](https://github.com/jax-ml/jax/blob/0080c5c934d9e3668c93d88e2b94f66e05f9d8d8/jaxlib/tools/BUILD.bazel#L210)

[jax CUDA pjrt wheel sources](https://github.com/jax-ml/jax/blob/0080c5c934d9e3668c93d88e2b94f66e05f9d8d8/jaxlib/tools/BUILD.bazel#L318)

1. A static source addition: add to `static_srcs` list.

   Example: add `//:file.txt` to `jax` wheel.

   ```
   wheel_sources(
      name = "jax_sources",
      data_srcs = [...],
      py_srcs = [...],
      static_srcs = [
         ...
         "//:file.txt"
      ],
   )
   ```

2. A platform-dependent source addition: add to `data_srcs` list.

   Example: add a `cc_library` target `//:cc_target` to `jax` wheel.

   ```
   wheel_sources(
      name = "jax_sources",
      data_srcs = [
         ...
         "//:cc_target"
      ],
      py_srcs = [...],
      static_srcs = [...],
   )
   ```

   If the existing targets in `data_srcs` already have a transitive
   dependency on `//:cc_target`, you don't need to add it explicitly.

3. A new python package addition: create `__init__.py` file and Bazel python
rule target with `__init__.py` in sources, add it to `py_srcs` list.

   Example: add a new package `jax.test_package` to `jax` wheel:

   The content of the file `jax/test_package/BUILD`:

   ```
   pytype_strict_library(
      name = "init",
      srcs = ["__init__.py"],
      visibility = ["//visibility:public"],
   )
   ```

   ```
   wheel_sources(
      name = "jax_sources",
      data_srcs = [...],
      py_srcs = [
         ...
         "//jax/test_package:init",
      ],
      static_srcs = [...],
   )
   ```

4. A new python source addition to existing package: create/update Bazel python
rule target with the new file in sources, add it to `py_srcs` list.

   Example: add a new file `jax/test_package/example.py` to `jax` wheel:

   The content of the file `jax/test_package/BUILD`:

   ```
   pytype_strict_library(
      name = "example",
      srcs = ["__init__.py",
               "example.py"],
      visibility = ["//visibility:public"],
   )
   ```

   ```
   wheel_sources(
      name = "jax_sources",
      data_srcs = [...],
      py_srcs = [
         ...
         "//jax/test_package:example",
      ],
      static_srcs = [...],
   )
   ```

   If the existing targets in `py_srcs` already have a transitive
   dependency on `example.py`, you don't need to add it explicitly.
