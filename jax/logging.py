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
"""A module for interoperable logging between ABSL and Python in JAX."""
import importlib
from typing import Optional

from jax._src.config import FLAGS

# flag indicating whether or not to use absl instead of standard logging.
use_absl: bool = FLAGS.jax_use_absl_logging

# Step 0: Import standard Python logging APIs that have no counterpart
# in absl.logging, and can be used regardless of the chosen logging setup.
from logging import (
  addLevelName,
  captureWarnings,
  FileHandler,
  Filter,
  Filterer,
  Formatter,
  Handler,
  Logger,
  LogRecord,
  makeLogRecord,
  StreamHandler,
)

# Step 1: Conditionally import APIs and attributes that are
# _effectively_ identical in absl and standard logging.
# The exact values of the logging verbosity enum differ between
# absl and standard logging, but the approach below ensures that
# absl and standard APIs are never mixed, so this should be fine.
if use_absl:
  from absl.logging import (
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    FATAL,
    debug,
    info,
    warning,
    error,
    fatal,
    log,
  )
  # ABSL C++ vlog levels.
  CPP_DEBUG = 0
  CPP_INFO = 0
  CPP_WARNING = 1
  CPP_ERROR = 2
  CPP_CRITICAL = 3
  CPP_FATAL = 3

else:
  from logging import (
    DEBUG,
    INFO,
    WARNING,
    ERROR,
    FATAL,
    debug,
    info,
    warning,
    error,
    fatal,
    log,
  )
  # standard values of C++ vlog levels. This representation
  # is not as rich, because C++ warning calls are aliased
  # to absl.DEBUG level when using logging.vlog.
  CPP_DEBUG = 10
  CPP_INFO = 10
  CPP_WARNING = 10
  CPP_ERROR = 9
  CPP_CRITICAL = 8
  CPP_FATAL = 8

# set additional names in standard logging for the last two
# C++ log levels to improve display, namely
# error (vlog level 2) and critical/fatal (vlog level 3).
# These level values _need_ to be in standard logging, since absl's
# `vlog` translates all C++ log levels to standard logging internally.
addLevelName(9, "CPP_ERROR")
addLevelName(8, "CPP_CRITICAL")


# Step 2: Implement APIs with differences between absl and standard
# logging in a unified, as lightweight as possible interface.
def set_warn_preinit_stderr(val: bool):
  """
  Toggle the `_warn_preinit_stderr` state variable in ABSL logging.
  Can be used to suppress the
  'WARNING: Logging before flag parsing goes to stderr.' message.
  """
  if use_absl:
    absl_logging = importlib.import_module("absl.logging")
    setattr(absl_logging, "_warn_preinit_stderr", val)


def getLogger(name: Optional[str] = None):
  if use_absl:
    del name
    # logger is a module-wide singleton with name `absl`.
    from absl.logging import get_absl_logger
    return get_absl_logger()
  else:
    from logging import getLogger
    return getLogger(name=name)


def vlog_is_on(level: int, name: Optional[str] = None):
  """
  Check whether a specific logger is enabled for a given C++ log level.

  Input level should be a sensible integer input given the current
  logging facility's level hierarchy.

  When using absl C++ logging, the level should be an integer between
  0 (debug/info) and 3 (critical/fatal), while for builtin Python logging,
  it should be between 10 (debug) and 50 (critical/fatal).

  The name argument is used to acquire the logger whose enabled level
  should be checked. If unspecified, the level of the absl logger
  (when using absl) or the builtin root logger (when using builtin logging)
  will be checked.
  """
  if use_absl:
    from absl.logging import vlog_is_on
    return vlog_is_on(level=level)
  else:
    logger = getLogger(name=name)
    return logger.isEnabledFor(level=level)


def vlog(level, msg, *args, **kwargs):
  """Log a message at C++ log level `level`."""
  if use_absl:
    from absl.logging import vlog
    vlog(level, msg, *args, **kwargs)
  else:
    log(level, msg, *args, **kwargs)
