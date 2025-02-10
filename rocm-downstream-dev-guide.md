# ROCm CI Dev Guide

This guide lays out how to do some dev operations, what branches live in this repo, and what CI workflows live in this repo.

# Quick Tips

1. Always use "Squash and Merge" when merging PRs into `rocm-main` (unless you're merging the daily sync from upstream).
2. When submitting a PR to `rocm-main`, make sure that your feature branch started from `rocm-main`. When you started working on your feature, did you do `git checkout rocm-main && git checkout -b <my feature branch>`?
3. Always fill out your PR's description with an explanation of why we need this change to give context to your fellow devs (and your future self).
4. In the PR description, link to the story or GitHub issue that this PR solves.

# Processes

## Making a Change

1. Clone `rocm/jax` and check out the `rocm-main` branch.
2. Create a new feature branch with `git checkout -b <my feature name>`.
3. Make your changes on the feature branch and test them locally.
4. Push your changes to a new feature branch in `rocm/jax` by running
   `git push orgin HEAD`.
5. Open a PR from your new feature branch into `rocm-main` with a nice description telling your
   team members what the change is for. Bonus points if you can link it to an issue or story.
6. Add reviewers, wait for approval, and make sure CI passes.
7. Depending on if your specific change, either:
  a. If this is a normal, run-of-the-mill change that we want to put upstream, add the
     `open-upstream` label to your PR and close your PR. In a few minutes, Actions will
     comment on your PR with a link that lets you open a new PR into upstream. The link will
     autofill some PR info, and the new PR be created on a new branch that has the same name
     as your old feature branch, but with the `-upstream` suffix appended to the end of it.
     If upstream reviewers request some changes to the new PR before merging, you can add
     or modify commits on the new `-upstream` feature branch.
  b. If this is an urgent change that we want in `rocm-main` right now but also want upstream,
     add the `open-upstream` label, merge your PR, and then follow the link that 
  c. If this is a change that we only want to keep in `rocm/jax` and not push into upstream,
     squash and merge your PR.

If you submitted your PR upstream with `open-upstream`, you should see your change in `rocm-main`
the next time the `ROCm Nightly Upstream Sync` workflow is run and the PR that it creates is
merged.

When using the `open-upstream` label to move changes to upstream, it's best to put the label on the PR when you either close or merge the PR. The GitHub Actions workflow that handles the `open-upstream` label uses `git rebase --onto` to set up the changes destined for upstream. Adding the label and creating this branch long after the PR has been merged or closed can cause merge conflicts with new upstream code and cause the workflow to fail. Adding the label right after creating your PR means that 1) any changes you make to your downstream PR while it is in review won't make it to upstream, and it is up to you to cherry-pick those changes into the upstream branch or remove and re-add the `open-upstream` label to get the Actions workflow to do it for you, and 2) that you're proposing changes to upstream that the rest of the AMD team might still have comments on.

## Daily Upstream Sync

Every day, GitHub Actions will attempt to run the `ROCm Nightly Upstream Sync` workflow. This job
normally does this on its own, but requires a developer to intervene if there's a merge conflict
or if the PR fails CI. Devs should fix or resolve issues with the merge by adding commits to the
PR's branch.

# Branches

 * `rocm-main` - the default "trunk" branch for this repo.  Should only be changed submitting PRs to it from feature branches created by devs.
 * `main` - a copy of `jax-ml/jax:main`. This branch is "read-only" and should only be changed by GitHub Actions.

# CI Workflows

We use GitHub Actions to run tests on PRs and to automate some of our
development tasks. These all live in `.github/workflows`.

| Name                       | File                             | Trigger                                              | Description                                                                            |
|----------------------------|----------------------------------|------------------------------------------------------|----------------------------------------------------------------------------------------|
| ROCm GPU CI                | `rocm-ci.yml`                    | Open or commit changes to a PR targeting `rocm-main` | Builds and runs JAX on ROCm for PRs going into `rocm-main`                             |
| ROCm Open Upstream PR      | `rocm-open-upstream-pr.yml`      | Add the `open-upstream` label to a PR                | Copies changes from a PR aimed at `rocm-main` into a new PR aimed at upstream's `main` |
| ROCm Nightly Upstream Sync | `rocm-nightly-upstream-sync.yml` | Runs nightly, can be triggered manually via Actions  | Opens a PR that merges changes from upstream `main` into our `rocm-main` branch        |

