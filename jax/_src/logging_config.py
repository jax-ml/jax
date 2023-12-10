# Copyright 2023 The JAX Authors.
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

import logging
import sys

_debug_handler = logging.StreamHandler(sys.stderr)
_debug_handler.setLevel(logging.DEBUG)
# Example log message:
# DEBUG:2023-06-07 00:14:40,280:jax._src.xla_bridge:590: Initializing backend 'cpu'
_debug_handler.setFormatter(logging.Formatter(
    "{levelname}:{asctime}:{name}:{lineno}: {message}", style='{'))

_debug_enabled_loggers = []


def enable_debug_logging(logger_name):
  """Makes the specified logger log everything to stderr.

  Also adds more useful debug information to the log messages, e.g. the time.

  Args:
    logger_name: the name of the logger, e.g. "jax._src.xla_bridge".
  """
  logger = logging.getLogger(logger_name)
  logger.addHandler(_debug_handler)
  logger.setLevel(logging.DEBUG)
  _debug_enabled_loggers.append(logger)


def disable_all_debug_logging():
  """Disables all debug logging enabled via `enable_debug_logging`.

  The default logging behavior will still be in effect, i.e. WARNING and above
  will be logged to stderr without extra message formatting.
  """
  for logger in _debug_enabled_loggers:
    logger.removeHandler(_debug_handler)
    # Assume that the default non-debug log level is always WARNING. In theory
    # we could keep track of what it was set to before. This shouldn't make a
    # difference if not other handlers are attached, but set it back in case
    # something else gets attached (e.g. absl logger) and for consistency.
    logger.setLevel(logging.WARNING)
