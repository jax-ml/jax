# Contributing to the JAX CI System

Our CI is a hybrid system using both GitHub Actions and an internal CI for
different tasks (presubmits, continuous, nightly, and release builds). The core
logic for building and testing resides in shell scripts within the [`ci/`](https://github.com/google/jax/tree/main/ci)
directory. This ensures that the same logic can be run locally, in GitHub
Actions, and in our internal CI. GitHub Actions workflows ([`.github/workflows/`](https://github.com/google/jax/tree/main/.github/workflows))
are primarily used to orchestrate and call these scripts with different
parameters. Configuration is managed through `JAXCI_` environment variables,
with defaults defined in [`ci/envs/default.env`](https://github.com/google/jax/blob/main/ci/envs/default.env).

## General Principles

*   **Keep it DRY (Don't Repeat Yourself):** For common test patterns (e.g.,
    running CPU tests), use reusable GitHub workflows (`on: workflow_call`). We
    have existing workflows like `pytest_cpu.yml` and `bazel_cpu.yml` for this.
*   **Isolate Logic from Orchestration:** Complex build and test logic should be
    in the [`ci/`](https://github.com/google/jax/tree/main/ci) scripts. The GitHub Actions YAML files should focus on
    orchestrating the calls to these scripts, not implementing the logic itself.
*   **Prioritize Presubmit Speed:** Presubmit checks, which run on every PR,
    should be fast (target < 10 minutes). Offload longer-running, more
    comprehensive tests to the continuous (every 3 hours) or nightly jobs.
*   **Run Locally First:** Before pushing changes to CI-related files, run the
    scripts locally to catch simple errors. See the "Running These Scripts
    Locally on Your Machine" section in [`ci/README.md`](https://github.com/google/jax/blob/main/ci/README.md).

## Modifying GitHub Actions Workflows

*   **Pin Actions to a Commit Hash:** All third-party GitHub Actions **must** be
    pinned to a specific commit hash, not a tag or branch. This ensures our
    workflows are deterministic. Use the **ratchet** tool as mentioned in
    [`.github/workflows/README.md`](https://github.com/google/jax/blob/main/.github/workflows/README.md) to manage this. Example: `uses:
    actions/checkout@08c6903cd8c0fde910a37f88322edcfb5dd907a8 # v5.0.0`
*   **Use Matrix Strategies:** To test across different configurations (e.g.,
    Python versions, runners, CUDA versions), use a `strategy: matrix`. Use
    `exclude` to prune unnecessary or redundant combinations from the matrix.
*   **Select Runners Carefully:** We use self-hosted runners for specific
    hardware (CPU, GPU, TPU). Runner names are descriptive (e.g.,
    `linux-x86-n4-16`, `linux-x86-g2-48-l4-4gpu`). See
    [here](https://github.com/jax-ml/jax/blob/main/.github/actionlint.yaml) for
    a list of available runners; choose the most appropriate one for the task.
*   **Set Permissions:** To enhance security, all workflows should explicitly
    define the permissions for the `GITHUB_TOKEN`. Default to the most
    restrictive permissions possible.
    *   For workflows that don't need to access repository contents, use
        `permissions: {}`.
    *   Only grant write permissions when absolutely necessary. Granting write
        permissions should be done with great care, and a discussion should be
        had before attempting to add an action with write permissions.
    *   More information about GitHub Token permissions can be found
    [here](https://docs.github.com/en/actions/security-guides/automatic-token-authentication#permissions-for-the-github_token).

## Working with CI Scripts

*   **Configuration via Environment Variables:** Control script behavior using
    `JAXCI_` environment variables. If you add a new variable, be sure to add a
    default in [`ci/envs/default.env`](https://github.com/google/jax/blob/main/ci/envs/default.env) and document it in [`ci/envs/README.md`](https://github.com/google/jax/blob/main/ci/envs/README.md).
*   **Local Execution with Docker:** For a consistent environment that mirrors
    CI, use the [`ci/utilities/run_docker_container.sh`](https://github.com/google/jax/blob/main/ci/utilities/run_docker_container.sh) script. This is the
    recommended way to test changes locally for Linux and Windows.

## Dependencies

*   **XLA:** Presubmit and continuous builds use XLA at HEAD to catch
    integration issues early. Nightly and release builds use a pinned XLA
    version for stability. This is controlled by the `JAXCI_CLONE_MAIN_XLA`
    variable.
*   **Python:** Use `uv` for installing Python packages where possible, as it is
    much faster than `pip`.

## Linting and Code Style

For any changes to GitHub Actions workflow files, it is recommended to run
[actionlint](https://github.com/rhysd/actionlint) to verify correctness and best
practices.

## Further Reading

For a more detailed overview of the JAX CI system, including its architecture
and workflows, please see the [`ci/README.md`](https://github.com/google/jax/blob/main/ci/README.md) file.
