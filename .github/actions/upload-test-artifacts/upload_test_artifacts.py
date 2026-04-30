#!/usr/bin/env python3
# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utilities for the upload-test-artifacts composite action."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


_DEFAULT_LOG_LIMIT = 200


def _artifact_files(root: Path, exclude: Path | None = None) -> list[Path]:
    if not root.is_dir():
        return []
    files = []
    for path in root.rglob('*'):
        if not path.is_file():
            continue
        if exclude is not None and path == exclude:
            continue
        files.append(path)
    return sorted(files)


def _append_github_output(name: str, value: str) -> None:
    output_path = os.environ.get('GITHUB_OUTPUT')
    if not output_path:
        return
    # Composite action step outputs are passed by appending name=value lines to
    # the file path GitHub exposes in GITHUB_OUTPUT.
    with open(output_path, 'a', encoding='utf-8') as output:
        output.write(f'{name}={value}\n')


def _json_from_env(name: str) -> dict:
    value = os.environ.get(name, '{}') or '{}'
    parsed = json.loads(value) or {}
    if not isinstance(parsed, dict):
        raise ValueError(f'{name} must contain a JSON object')
    return parsed


def check_artifacts(args: argparse.Namespace) -> None:
    root = Path(args.test_artifacts_dir)
    padded_attempt = f'attempt{int(args.run_attempt):03d}'
    _append_github_output('padded_attempt', padded_attempt)

    if not root.is_dir():
        _append_github_output('has_artifacts', 'false')
        print(f'No test artifacts directory found at {root}/')
        return

    files = _artifact_files(root)
    if not files:
        _append_github_output('has_artifacts', 'false')
        print(f'No test artifacts found in {root}/')
        return

    artifact_bytes = sum(path.stat().st_size for path in files)
    _append_github_output('has_artifacts', 'true')
    print(
        f'Found {len(files)} test artifact file(s) in {root}/ '
        f'({artifact_bytes} bytes).'
    )


def write_metadata(args: argparse.Namespace) -> None:
    root = Path(args.test_artifacts_dir)
    root.mkdir(parents=True, exist_ok=True)

    inputs = _json_from_env('INPUTS_JSON')
    matrix = _json_from_env('MATRIX_JSON')
    env_vars = _json_from_env('ENV_JSON')
    custom = _json_from_env('CUSTOM_JSON')
    explicit_str = os.environ.get('EXPLICIT_METADATA', '')

    merged = {**inputs, **matrix, **env_vars, **custom}

    job_id = os.environ.get('GITHUB_JOB_ID')
    if job_id:
        merged['job_id'] = int(job_id)

    if explicit_str:
        try:
            explicit = json.loads(explicit_str)
            merged = {**merged, **explicit}
        except json.JSONDecodeError:
            print(
                'Warning: could not parse explicit metadata as JSON: '
                f'{explicit_str}'
            )
            merged['explicit_metadata_raw'] = explicit_str

    metadata_path = root / 'metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as metadata_file:
        json.dump(merged, metadata_file)
    print(f'Wrote metadata to {metadata_path}')


def prepare_artifacts(args: argparse.Namespace) -> None:
    root = Path(args.test_artifacts_dir)
    log_limit = (
      int(args.log_limit) if args.log_limit.isdecimal() else _DEFAULT_LOG_LIMIT
    )
    manifest_path = root / 'test-artifacts-manifest.tsv'
    # Rebuild the manifest on every upload, but do not list an older copy of
    # the manifest as an input artifact.
    files = _artifact_files(root, exclude=manifest_path)
    total_bytes = sum(path.stat().st_size for path in files)

    with open(manifest_path, 'w', encoding='utf-8', newline='\n') as manifest:
        manifest.write('path\tsize_bytes\n')
        for path in files:
            rel_path = path.relative_to(root).as_posix()
            manifest.write(f'{rel_path}\t{path.stat().st_size}\n')

    print(f'Preparing {len(files)} artifact file(s) from {root} ({total_bytes} bytes).')
    print(f'Wrote manifest to {manifest_path}')
    if files and log_limit > 0:
        print('Artifact files:')
        for path in files[:log_limit]:
            rel_path = path.relative_to(root).as_posix()
            print(f'  {rel_path} ({path.stat().st_size} bytes)')
    if len(files) > log_limit:
        print(f'  ... and {len(files) - log_limit} more file(s); see {manifest_path}')

    summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary_path:
        _write_upload_summary(
            summary_path=Path(summary_path),
            artifact_name=args.artifact_name,
            destination_uri=args.destination_uri,
            files=files,
            root=root,
            log_limit=log_limit,
            manifest_path=manifest_path,
            total_bytes=total_bytes,
        )


def _write_upload_summary(
    *,
    summary_path: Path,
    artifact_name: str,
    destination_uri: str,
    files: list[Path],
    root: Path,
    log_limit: int,
    manifest_path: Path,
    total_bytes: int,
) -> None:
    with open(summary_path, 'a', encoding='utf-8') as summary:
        summary.write('### Uploaded test artifacts\n\n')
        summary.write(f'- artifact: `{artifact_name}`\n')
        summary.write(f'- destination: `{destination_uri}`\n')
        summary.write(f'- files: {len(files)}\n')
        summary.write(f'- uncompressed bytes: {total_bytes}\n')
        summary.write(f'- manifest: `{manifest_path}`\n\n')
        if not files:
            return

        summary_label = 'Files included in artifact'
        if len(files) > log_limit:
            summary_label = f'{summary_label} (first {log_limit})'
        summary.write(f'<details><summary>{summary_label}</summary>\n\n')
        summary.write('```text\n')
        for path in files[:log_limit]:
            rel_path = path.relative_to(root).as_posix()
            summary.write(f'{path.stat().st_size}\t{rel_path}\n')
        if len(files) > log_limit:
            summary.write(f'... and {len(files) - log_limit} more file(s)\n')
        summary.write('```\n\n</details>\n\n')


def print_zip_size(args: argparse.Namespace) -> None:
    print(Path(args.zip_path).stat().st_size)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(required=True)

    check_parser = subparsers.add_parser('check')
    check_parser.add_argument('--test-artifacts-dir', required=True)
    check_parser.add_argument('--run-attempt', required=True)
    check_parser.set_defaults(func=check_artifacts)

    metadata_parser = subparsers.add_parser('metadata')
    metadata_parser.add_argument('--test-artifacts-dir', required=True)
    metadata_parser.set_defaults(func=write_metadata)

    prepare_parser = subparsers.add_parser('prepare')
    prepare_parser.add_argument('--test-artifacts-dir', required=True)
    prepare_parser.add_argument('--artifact-name', required=True)
    prepare_parser.add_argument('--destination-uri', required=True)
    prepare_parser.add_argument('--log-limit', required=True)
    prepare_parser.set_defaults(func=prepare_artifacts)

    zip_size_parser = subparsers.add_parser('zip-size')
    zip_size_parser.add_argument('--zip-path', required=True)
    zip_size_parser.set_defaults(func=print_zip_size)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
