(jep-effver)=
# JEP 25516: Effort-based versioning for JAX

This document proposes that the JAX core library should explicitly adopt
**Effort-based versioning (EffVer)** for past and future releases.
This versioning scheme is more fully described in
[EffVer: Version your code by the effort required to upgrade](https://jacobtomlinson.dev/effver/). 

The sections below discuss some of the considerations that favor this approach.

## Key features of EffVer
Effort-based versioning is a three-number versioning system, similar to the better
known semantic versioning ([SemVer](https://semver.org/)).
It uses a three-number version of the form **MACRO** . **MESO** . **MICRO**,
where version numbers are incremented based on the *expected effort*
required to adapt to the change.

As an example, consider software with current version `2.3.4`:

- Increasing the **micro** version (i.e. releasing `2.3.5`) signals to users that
  little to no effort is necessary on their part to adapt to the changes.
- Increasing the **meso** version (i.e. releasing `2.4.0`) signals to users that
  some small effort will be required for existing code to work with the changes.
- Increasing the **macro** version (i.e. releasing `3.0.0`) signals to users that
  significant effort may be required to update to the changes.

In some ways, this captures the essence of more commonly-used semantic versioning,
but avoids phrasing in terms of compatibility guarantees that are hard to meet
in practice.

## Zero version
In addition, EffVer gives special meaning to the Zero version. Early releases of
software often are versioned `0.X.Y`, and in this case `X` has the characteristics
of the macro version, and `Y` has the characteristics of the meso version.
JAX has been in a zero-version state since its initial release (the version as of
this writing is `0.4.37`), and EffVer’s zero-version case is a good *post-facto*
description of the implicit intent behind JAX’s releases to date.

In EffVer, bumping from `0.X.Y` to version `1.0.0` is recommended when a certain
level of stability has been reached in practice:

> If you end up on a version like `0.9.x` for many months it is a good signal that
> things are pretty stable and that it’s time to switch to a `1.0.0` release.

- **Pros:**
  - EffVer concisely communicates the intent of a change, without making
    compatibility guarantees that are difficult to adhere to in practice.
  - EffVer, via its special casing of zero versions, correctly describes JAX’s
    release strategy prior to this proposal.
  - EffVer provides a concrete recommendation for how to think about JAX 1.0.
- **Cons:**
  - EffVer is not as well-known as SemVer, and is not as immediately recognizable
    as CalVer, so may lead to some confusion among users.

## Alternatives considered

We considered a few alternatives to EffVer, outlined below. In each case, the
Cons were judged to outweigh the Pros when evaluated against EffVer.

## Non-semantic versioning (status quo)
JAX's current versioning uses three numbers with no formal semantic meaning
beyond simple orderabilty (i.e. versions increase over time).
In practice, JAX's version numbers up until now have been semantically quite
similar to the EffVer zero-version case.

One option would be to explicitly formalize this non-semantic versioning.

- **Pros:**
  - The status quo requires no action from the development team.
- **Cons:**
  - The status quo has led to confusion among users who expect the
    guarantees of SemVer to apply.
  - The status quo is unfriendly to users who would like some clear
    signal of the meaning of JAX releases.

### Semantic versioning
One common alternative is Semantic versioning ([SemVer](https://semver.org/)).
SemVer also encodes versions with three numbers, i.e.: **MAJOR** . **MINOR** . **MICRO**.
Consider software with current version `2.3.4`:

- Increasing the **micro** version (i.e. releasing `2.3.5`) indicates the release
  includes only bug fixes.
- Increasing the **minor** version (i.e. releasing `2.4.0`) indicates the release
  includes bug fixes as well as new features.
- Increasing the **major** version (i.e. releasing `3.0.0`) indicates the release
  includes bug fixes, new features, as well as breaking changes.

SemVer makes no special accommodation for zero version, meaning that JAX’s existing
releases violate the guarantees of the versioning scheme (up until this point, JAX
has generally used the micro version for feature releases, and the minor version
for significant backward-incompatible changes).

- **Pros:**
  - SemVer is well-known, and generally is the assumed model in the case of
    three-number versioning.
  - SemVer concisely describes the intent of each release.
- **Cons**:
  - The compatibility guarantees of SemVer are difficult to achieve in practice.
  - SemVer has no special-casing for the zero version, and as such is not a good
    description of JAX’s release processes up until this point.

### Calendar versioning
Another common versioning scheme is calendar-based versioning (CalVer), also typically
represented by three numbers **YEAR** . **MONTH** . **DAY**. By design, these numbers
contain no semantic meaning regarding the included changes, but rather encode the
calendar data on which the software was released. For example, the `2024.12.16` release
indicates that it reflects the state of the main branch on December 16, 2024.

- **Pros:**
  - CalVer immediately communicates the timestamp of the particular release, which
    may be difficult to determine in other versioning schemes.
- **Cons:**
  - CalVer version numbers do not provide any signal to users regarding the degree
    of severity of the changes it includes.
