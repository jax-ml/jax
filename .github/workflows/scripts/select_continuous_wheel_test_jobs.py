#!/usr/bin/env python3

# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Select jobs for wheel_tests_continuous.yml from workflow inputs."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobSpec:
    job_id: str
    platform: str
    suite: str
    requires_builds: tuple[str, ...]


BUILD_JOBS = (
    'build_jax_artifact',
    'build_jaxlib_artifact',
    'build_cuda_artifacts',
    'build_rocm_artifacts',
)

# requires_builds lists artifact jobs whose uploaded wheels are consumed by
# the selected job. Bazel py_import jobs with build_jaxlib='wheel' build and
# import wheels within Bazel, so they do not require GCS artifact builds.
JOB_SPECS = (
    # LINT.IfChange(run_pytest_cpu)
    JobSpec(
        job_id='run_pytest_cpu',
        platform='cpu',
        suite='pytest',
        requires_builds=('build_jax_artifact', 'build_jaxlib_artifact'),
    ),
    # LINT.ThenChange(.github/workflows/wheel_tests_continuous.yml:run_pytest_cpu)
    # LINT.IfChange(run_pytest_cuda)
    JobSpec(
        job_id='run_pytest_cuda',
        platform='cuda',
        suite='pytest',
        requires_builds=(
            'build_jax_artifact',
            'build_jaxlib_artifact',
            'build_cuda_artifacts',
        ),
    ),
    # LINT.ThenChange(.github/workflows/wheel_tests_continuous.yml:run_pytest_cuda)
    # LINT.IfChange(run_bazel_test_cpu_py_import)
    JobSpec(
        job_id='run_bazel_test_cpu_py_import',
        platform='cpu',
        suite='bazel',
        requires_builds=(),
    ),
    # LINT.ThenChange(.github/workflows/wheel_tests_continuous.yml:run_bazel_test_cpu_py_import)
    # LINT.IfChange(run_bazel_test_cuda)
    JobSpec(
        job_id='run_bazel_test_cuda',
        platform='cuda',
        suite='bazel',
        requires_builds=(
            'build_jax_artifact',
            'build_jaxlib_artifact',
            'build_cuda_artifacts',
        ),
    ),
    # LINT.ThenChange(.github/workflows/wheel_tests_continuous.yml:run_bazel_test_cuda)
    # LINT.IfChange(run_bazel_test_cuda_py_import)
    JobSpec(
        job_id='run_bazel_test_cuda_py_import',
        platform='cuda',
        suite='bazel',
        requires_builds=(),
    ),
    # LINT.ThenChange(.github/workflows/wheel_tests_continuous.yml:run_bazel_test_cuda_py_import)
    # LINT.IfChange(run_pytest_tpu)
    JobSpec(
        job_id='run_pytest_tpu',
        platform='tpu',
        suite='pytest',
        requires_builds=('build_jax_artifact', 'build_jaxlib_artifact'),
    ),
    # LINT.ThenChange(.github/workflows/wheel_tests_continuous.yml:run_pytest_tpu)
    # LINT.IfChange(run_bazel_test_tpu)
    JobSpec(
        job_id='run_bazel_test_tpu',
        platform='tpu',
        suite='bazel',
        requires_builds=(),
    ),
    # LINT.ThenChange(.github/workflows/wheel_tests_continuous.yml:run_bazel_test_tpu)
    # LINT.IfChange(run_pytest_rocm)
    JobSpec(
        job_id='run_pytest_rocm',
        platform='rocm',
        suite='pytest',
        requires_builds=(
            'build_jax_artifact',
            'build_jaxlib_artifact',
            'build_rocm_artifacts',
        ),
    ),
    # LINT.ThenChange(.github/workflows/wheel_tests_continuous.yml:run_pytest_rocm)
    # LINT.IfChange(run_bazel_test_rocm)
    JobSpec(
        job_id='run_bazel_test_rocm',
        platform='rocm',
        suite='bazel',
        requires_builds=(
            'build_jax_artifact',
            'build_jaxlib_artifact',
            'build_rocm_artifacts',
        ),
    ),
    # LINT.ThenChange(.github/workflows/wheel_tests_continuous.yml:run_bazel_test_rocm)
)

VALID_XLA_TRACKS = frozenset(('pinned', 'head', 'commit'))
ROCM_PLATFORM = 'rocm'
# 'all' is the literal full platform set. 'all-non-rocm' exists so dispatches
# can select custom XLA while excluding ROCm, which only supports pinned XLA.
ALL_NON_ROCM_PLATFORMS = 'all-non-rocm'
ALL_JOB_IDS = BUILD_JOBS + tuple(job.job_id for job in JOB_SPECS)
# ROCm jobs are intentionally excluded from per-job connection halts. Their
# callers keep halt-for-connection pinned to 'no'.
ROCM_JOB_IDS = frozenset((
    'build_rocm_artifacts',
    'run_pytest_rocm',
    'run_bazel_test_rocm',
))
HALT_FOR_CONNECTION_JOB_IDS = tuple(
    job_id for job_id in ALL_JOB_IDS if job_id not in ROCM_JOB_IDS
)
XLA_COMMIT_RE = re.compile(r'^[0-9a-f]{40}$')


def normalize_csv(value: str) -> str:
    return ''.join(value.lower().split())


def parse_selection(name: str, value: str, allowed: set[str]) -> set[str]:
    value = normalize_csv(value or 'all')
    if not value or value == 'all':
        return set(allowed)

    selected = set()
    for token in value.split(','):
        if not token:
            raise ValueError(f'{name} contains an empty value.')
        if token not in allowed:
            raise ValueError(f'Invalid {name} value: {token}')
        selected.add(token)
    return selected


def parse_platforms(value: str, allowed: set[str]) -> set[str]:
    value = normalize_csv(value or 'all')
    if value == ALL_NON_ROCM_PLATFORMS:
        return set(allowed) - {ROCM_PLATFORM}
    return parse_selection('platforms', value, allowed)


def parse_halt_for_connection(value: str, allowed: set[str]) -> set[str]:
    value = normalize_csv(value or 'none')
    if not value or value == 'none':
        return set()

    selected = set()
    for token in value.split(','):
        if not token:
            raise ValueError('halt-for-connection contains an empty value.')
        canonical_token = token.replace('-', '_')
        if canonical_token not in allowed:
            raise ValueError(f'Invalid halt-for-connection value: {token}')
        selected.add(canonical_token)
    return selected


def parse_bool(name: str, value: str) -> bool:
    normalized = (value or 'true').strip().lower()
    if normalized == 'true':
        return True
    if normalized == 'false':
        return False
    raise ValueError(f'Invalid {name} value: {value}')


def emit(name: str, value: str) -> None:
    output_path = os.environ.get('GITHUB_OUTPUT')
    if output_path:
        with Path(output_path).open('a', encoding='utf-8') as output_file:
            output_file.write(f'{name}={value}\n')
    else:
        print(f'{name}={value}')


def emit_json(name: str, value: dict[str, bool | str]) -> None:
    emit(name, json.dumps(value, separators=(',', ':')))


def selected_xla_values(xla_track: str, xla_commit: str) -> tuple[str, str]:
    if xla_track == 'pinned':
        return '0', ''
    if xla_track == 'commit':
        # CI setup clones XLA first, then checks out the requested commit.
        return '1', xla_commit
    return '1', ''


def write_summary(
    *,
    platforms: set[str],
    suites: set[str],
    xla_track: str,
    xla_commit: str,
    halt_for_connection_jobs: dict[str, str],
    run_build_bazel: bool,
    run_pytest: bool,
    selected_jobs: dict[str, bool],
    required_builds: dict[str, bool],
) -> None:
    summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
    if not summary_path:
        return

    selected_job_names = [
        job.job_id for job in JOB_SPECS if selected_jobs[job.job_id]
    ]
    required_build_names = [
        build for build in BUILD_JOBS if required_builds[build]
    ]
    halt_for_connection_names = [
        job_id for job_id in ALL_JOB_IDS
        if halt_for_connection_jobs[job_id] == 'yes'
    ]

    with Path(summary_path).open('a', encoding='utf-8') as summary_file:
        summary_file.write('### Wheel test selection\n\n')
        summary_file.write('| Setting | Value |\n')
        summary_file.write('| --- | --- |\n')
        summary_file.write(f'| platforms | `{",".join(sorted(platforms))}` |\n')
        summary_file.write(f'| suites | `{",".join(sorted(suites))}` |\n')
        summary_file.write(f'| xla_track | `{xla_track}` |\n')
        summary_file.write(f'| xla_commit | `{xla_commit or "none"}` |\n')
        summary_file.write('| halt-for-connection | `')
        summary_file.write(
            ','.join(halt_for_connection_names) or 'none'
        )
        summary_file.write('` |\n')
        summary_file.write(
            f'| run-build-bazel | `{str(run_build_bazel).lower()}` |\n'
        )
        summary_file.write(f'| run-pytest | `{str(run_pytest).lower()}` |\n')
        summary_file.write('\n')
        if xla_track != 'pinned' and selected_jobs['run_bazel_test_cuda']:
            summary_file.write('### pypi_latest note\n\n')
            summary_file.write(
                'Matrix legs with `jaxlib-version=head` consume wheels built '
                'from the selected XLA. Matrix legs with '
                '`jaxlib-version=pypi_latest` download jaxlib and, where '
                'applicable, PJRT and plugin wheels from PyPI.\n\n'
            )
        summary_file.write('### Selected jobs\n\n')
        summary_file.write(
            '\n'.join(f'- `{name}`' for name in selected_job_names)
            or 'None'
        )
        summary_file.write('\n\n')
        summary_file.write('### Required artifact builds\n\n')
        summary_file.write(
            '\n'.join(f'- `{name}`' for name in required_build_names)
            or 'None'
        )
        summary_file.write('\n')


def main() -> int:
    allowed_platforms = {job.platform for job in JOB_SPECS}
    allowed_suites = {job.suite for job in JOB_SPECS}
    allowed_halt_jobs = set(HALT_FOR_CONNECTION_JOB_IDS)

    platforms = parse_platforms(
        os.environ.get('INPUT_PLATFORMS', 'all'),
        allowed_platforms,
    )
    suites = parse_selection(
        'suites',
        os.environ.get('INPUT_SUITES', 'all'),
        allowed_suites,
    )

    xla_track = (
        os.environ.get('INPUT_XLA_TRACK', '').strip().lower() or 'pinned'
    )
    xla_commit = os.environ.get('INPUT_XLA_COMMIT', '').strip().lower()
    halt_for_connection = parse_halt_for_connection(
        os.environ.get('INPUT_HALT_FOR_CONNECTION', 'none'),
        allowed_halt_jobs,
    )
    run_build_bazel = parse_bool(
        'run-build-bazel',
        os.environ.get('INPUT_RUN_BUILD_BAZEL', 'true'),
    )
    run_pytest = parse_bool(
        'run-pytest',
        os.environ.get('INPUT_RUN_PYTEST', 'true'),
    )

    if xla_track not in VALID_XLA_TRACKS:
        raise ValueError(f'Invalid xla_track value: {xla_track}')
    if xla_track == 'commit' and not xla_commit:
        raise ValueError('xla_commit must be set when xla_track is "commit".')
    if xla_track != 'commit' and xla_commit:
        raise ValueError(
            'xla_commit must only be set when xla_track is "commit".'
        )
    if xla_track == 'commit' and not XLA_COMMIT_RE.fullmatch(xla_commit):
        raise ValueError(
            'xla_commit must be a full 40-character hexadecimal commit SHA.'
        )
    if xla_track != 'pinned' and ROCM_PLATFORM in platforms:
        raise ValueError(
            'ROCm jobs do not support custom XLA selection. Use '
            'xla_track="pinned" or exclude ROCm from platforms.'
        )
    selected_jobs = {}
    required_builds = {build: False for build in BUILD_JOBS}

    for job in JOB_SPECS:
        suite_allowed = run_pytest if job.suite == 'pytest' else run_build_bazel
        selected = (
            job.platform in platforms
            and job.suite in suites
            and suite_allowed
        )
        selected_jobs[job.job_id] = selected

        if selected:
            for build in job.requires_builds:
                required_builds[build] = True

    active_jobs = {
        job_id for job_id, selected in selected_jobs.items() if selected
    }
    active_jobs.update(
        build for build, required in required_builds.items() if required
    )
    inactive_halt_jobs = sorted(halt_for_connection - active_jobs)
    if inactive_halt_jobs:
        raise ValueError(
            'halt-for-connection targets are not selected or required: '
            + ','.join(inactive_halt_jobs)
        )

    halt_for_connection_jobs = {
        job_id: 'yes' if job_id in halt_for_connection else 'no'
        for job_id in ALL_JOB_IDS
    }

    clone_main_xla, selected_xla_commit = selected_xla_values(
        xla_track,
        xla_commit,
    )

    emit_json('selected_jobs', selected_jobs)
    emit_json('required_builds', required_builds)
    emit_json('halt_for_connection_jobs', halt_for_connection_jobs)
    emit('clone_main_xla', clone_main_xla)
    emit('xla_commit', selected_xla_commit)

    write_summary(
        platforms=platforms,
        suites=suites,
        xla_track=xla_track,
        xla_commit=selected_xla_commit,
        halt_for_connection_jobs=halt_for_connection_jobs,
        run_build_bazel=run_build_bazel,
        run_pytest=run_pytest,
        selected_jobs=selected_jobs,
        required_builds=required_builds,
    )
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except ValueError as err:
        print(err, file=sys.stderr)
        raise SystemExit(1)
