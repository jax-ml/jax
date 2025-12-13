# Copyright 2022 The JAX Authors.
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

from __future__ import annotations

from typing import Any
import argparse
import gzip
import os
import pathlib
import tempfile

# pytype: disable=import-error
from jax._src import profiler as jax_profiler
try:
  from xprof.convert import _pywrap_profiler_plugin
  from xprof.convert import raw_to_tool_data as convert
except ImportError:
  raise ImportError(
      "This script requires `xprof` to be installed.")
# pytype: enable=import-error


_DESCRIPTION = """
To profile running JAX programs, you first need to start the profiler server
in the program of interest. You can do this via
`jax.profiler.start_server(<port>)`. Once the program is running and the
profiler server has started, you can run `collect_profile` to trace the execution
for a provided duration. The trace file will be dumped into a directory
(determined by `--log_dir`) and by default, a Perfetto UI link will be generated
to view the resulting trace.

Common tracer options (with defaults):
  --host_tracer_level=2         Profiler host tracer level.
  --device_tracer_level=1       Profiler device tracer level.
  --python_tracer_level=1       Profiler Python tracer level.
"""
_GRPC_PREFIX = 'grpc://'
DEFAULT_NUM_TRACING_ATTEMPTS = 3
parser = argparse.ArgumentParser(description=_DESCRIPTION,
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--log_dir", default=None,
                    help=("Directory to store log files. "
                          "Uses a temporary directory if none provided."),
                    type=str)
parser.add_argument("port", help="Port to collect trace", type=int)
parser.add_argument("duration_in_ms",
                    help="Duration to collect trace in milliseconds", type=int)
parser.add_argument("--no_perfetto_link",
                    help="Disable creating a perfetto link",
                    action="store_true")
parser.add_argument("--host", default="127.0.0.1",
                    help="Host to collect trace. Defaults to 127.0.0.1",
                    type=str)

def collect_profile(
    port: int,
    duration_in_ms: int,
    host: str,
    log_dir: os.PathLike | str | None,
    no_perfetto_link: bool,
    xprof_options: dict[str, Any] | None = None,):
  options: dict[str, Any] = {
      "host_tracer_level": 2,
      "device_tracer_level": 1,
      "python_tracer_level": 1,
  }
  if xprof_options:
    options.update(xprof_options)

  IS_GCS_PATH = str(log_dir).startswith("gs://")
  log_dir_ = pathlib.Path(log_dir if log_dir is not None else tempfile.mkdtemp())
  str_log_dir = log_dir if IS_GCS_PATH else str(log_dir_)
  _pywrap_profiler_plugin.trace(
      _strip_addresses(f"{host}:{port}", _GRPC_PREFIX),
      str_log_dir,
      '',
      True,
      duration_in_ms,
      DEFAULT_NUM_TRACING_ATTEMPTS,
      options,
  )
  print(f"Dumped profiling information in: {str_log_dir}")
  # Traces stored on GCS cannot be converted to a Perfetto trace, as JAX doesn't
  # directly support GCS paths.
  if IS_GCS_PATH:
    if not no_perfetto_link:
      print("Perfetto link is not supported for GCS paths, skipping creation.")
    return
  # The profiler dumps `xplane.pb` to the logging directory. To upload it to
  # the Perfetto trace viewer, we need to convert it to a `trace.json` file.
  # We do this by first finding the `xplane.pb` file, then passing it into
  # tensorflow_profile_plugin's `xplane` conversion function.
  curr_path = log_dir_.resolve()
  root_trace_folder = curr_path / "plugins" / "profile"
  trace_folders = [root_trace_folder / trace_folder for trace_folder
                   in root_trace_folder.iterdir()]
  latest_folder = max(trace_folders, key=os.path.getmtime)
  xplane = next(latest_folder.glob("*.xplane.pb"))
  result, _ = convert.xspace_to_tool_data([xplane], "trace_viewer", {})

  with gzip.open(str(latest_folder / "remote.trace.json.gz"), "wb") as fp:
    fp.write(result.encode("utf-8"))

  if not no_perfetto_link:
    path = jax_profiler._write_perfetto_trace_file(log_dir_)
    jax_profiler._host_perfetto_trace_file(path)

def _strip_prefix(s, prefix):
  return s[len(prefix):] if s.startswith(prefix) else s

def _strip_addresses(addresses, prefix):
  return ','.join([_strip_prefix(s, prefix) for s in addresses.split(',')])

def _parse_xprof_flags(unknown_flags: list[str]) -> dict[str, Any]:
  parsed: dict[str, Any] = {}
  i = 0
  while i < len(unknown_flags):
    arg = unknown_flags[i]
    if not arg.startswith('--'):
      raise ValueError(f"Unknown positional argument encountered: {arg}")

    key = arg[2:]
    if "=" in key:
      key, value_str = key.split("=", 1)
      i += 1
    elif i + 1 < len(unknown_flags) and not unknown_flags[i + 1].startswith('--'):
      value_str = unknown_flags[i + 1]
      i += 2
    else:
      parsed[key] = True
      i += 1
      continue

    value_lower = value_str.lower()
    if value_lower in {'true', 't', 'yes', 'y'}:
      parsed[key] = True
    elif value_lower in {'false', 'f', 'no', 'n'}:
      parsed[key] = False
    else:
      try:
        parsed[key] = int(value_str, 0)
      except ValueError:
        try:
          parsed[key] = float(value_str)
        except ValueError:
          parsed[key] = value_str  # Keep as string
  return parsed


def main(known_args, unknown_flags):
  xprof_options = _parse_xprof_flags(unknown_flags)
  collect_profile(
      known_args.port,
      known_args.duration_in_ms,
      known_args.host,
      known_args.log_dir,
      known_args.no_perfetto_link,
      xprof_options,
  )

if __name__ == "__main__":
  known_args, unknown_flags = parser.parse_known_args()
  main(known_args, unknown_flags)
