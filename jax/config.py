# Copyright 2018 Google LLC
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

import contextlib
import functools
import os
import sys
import threading

from jax import lib
from typing import Callable, Optional

def bool_env(varname: str, default: bool) -> bool:
  """Read an environment variable and interpret it as a boolean.

  True values are (case insensitive): 'y', 'yes', 't', 'true', 'on', and '1';
  false values are 'n', 'no', 'f', 'false', 'off', and '0'.

  Args:
    varname: the name of the variable
    default: the default boolean value
  Raises: ValueError if the environment variable is anything else.
  """
  val = os.getenv(varname, str(default))
  val = val.lower()
  if val in ('y', 'yes', 't', 'true', 'on', '1'):
    return True
  elif val in ('n', 'no', 'f', 'false', 'off', '0'):
    return False
  else:
    raise ValueError("invalid truth value %r for environment %r" % (val, varname))

def int_env(varname: str, default: int) -> int:
  """Read an environment variable and interpret it as an integer."""
  return int(os.getenv(varname, default))


class Config:
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self):
    self.values = {}
    self.meta = {}
    self.FLAGS = NameSpace(self.read)
    self.use_absl = False
    self._contextmanager_flags = set()

    # TODO(mattjj): delete these when only omnistaging is available
    self.omnistaging_enabled = bool_env('JAX_OMNISTAGING', True)
    self._omnistaging_disablers = []
    self._update_hooks = {}

  def update(self, name, val):
    if self.use_absl:
      setattr(self.absl_flags.FLAGS, name, val)
    else:
      self.check_exists(name)
      if name not in self.values:
        raise Exception("Unrecognized config option: {}".format(name))
      self.values[name] = val

    hook = self._update_hooks.get(name, None)
    if hook:
      hook(val)

    if name == "jax_disable_jit":
      lib.jax_jit.global_state().disable_jit = val

  def read(self, name):
    if name in self._contextmanager_flags:
      raise AttributeError(
          "For flags with a corresponding contextmanager, read their value "
          f"via e.g. `config.{name}` rather than `config.FLAGS.{name}`.")
    return self._read(name)

  def _read(self, name):
    if self.use_absl:
      return getattr(self.absl_flags.FLAGS, name)
    else:
      self.check_exists(name)
      return self.values[name]

  def add_option(self, name, default, opt_type, meta_args, meta_kwargs,
                 update_hook=None):
    if name in self.values:
      raise Exception("Config option {} already defined".format(name))
    self.values[name] = default
    self.meta[name] = (opt_type, meta_args, meta_kwargs)
    if update_hook:
      self._update_hooks[name] = update_hook
      update_hook(default)

  def check_exists(self, name):
    if name not in self.values:
      raise AttributeError("Unrecognized config option: {}".format(name))

  def DEFINE_bool(self, name, default, *args, **kwargs):
    update_hook = kwargs.pop("update_hook", None)
    self.add_option(name, default, bool, args, kwargs, update_hook=update_hook)

  def DEFINE_integer(self, name, default, *args, **kwargs):
    update_hook = kwargs.pop("update_hook", None)
    self.add_option(name, default, int, args, kwargs, update_hook=update_hook)

  def DEFINE_string(self, name, default, *args, **kwargs):
    update_hook = kwargs.pop("update_hook", None)
    self.add_option(name, default, str, args, kwargs, update_hook=update_hook)

  def DEFINE_enum(self, name, default, *args, **kwargs):
    update_hook = kwargs.pop("update_hook", None)
    self.add_option(name, default, 'enum', args, kwargs,
                    update_hook=update_hook)

  def config_with_absl(self):
    # Run this before calling `app.run(main)` etc
    import absl.flags as absl_FLAGS  # noqa: F401
    from absl import app, flags as absl_flags

    self.use_absl = True
    self.absl_flags = absl_flags
    absl_defs = { bool: absl_flags.DEFINE_bool,
                  int:  absl_flags.DEFINE_integer,
                  str:  absl_flags.DEFINE_string,
                  'enum': absl_flags.DEFINE_enum }

    for name, val in self.values.items():
      flag_type, meta_args, meta_kwargs = self.meta[name]
      absl_defs[flag_type](name, val, *meta_args, **meta_kwargs)

    app.call_after_init(lambda: self.complete_absl_config(absl_flags))

  def complete_absl_config(self, absl_flags):
    for name, _ in self.values.items():
      self.update(name, getattr(absl_flags.FLAGS, name))

  def parse_flags_with_absl(self):
    global already_configured_with_absl
    if not already_configured_with_absl:
      import absl.flags
      self.config_with_absl()
      absl.flags.FLAGS(sys.argv, known_only=True)
      self.complete_absl_config(absl.flags)
      already_configured_with_absl = True

      if not FLAGS.jax_omnistaging:
        self.disable_omnistaging()

  def register_omnistaging_disabler(self, disabler):
    if self.omnistaging_enabled:
      self._omnistaging_disablers.append(disabler)
    else:
      disabler()

  def enable_omnistaging(self):
    if not self.omnistaging_enabled:
      raise Exception("can't re-enable omnistaging after it's been disabled")

  def disable_omnistaging(self):
    return

  def temporary_hack_do_not_call_me(self):
    if self.omnistaging_enabled:
      for disabler in self._omnistaging_disablers:
        disabler()
      self.omnistaging_enabled = False

  def define_bool_state(
    self, name: str, default: bool, help: str, *,
    update_global_hook: Optional[Callable[[bool], None]] = None,
    update_thread_local_hook: Optional[Callable[[Optional[bool]], None]] = None):
    """Set up thread-local state and return a contextmanager for managing it.

    This function is a convenience wrapper. It defines a flag and corresponding
    thread-local state, which can be managed via the contextmanager it returns.

    The thread-local state value can be read via the ``config.<option_name>``
    attribute, where ``config`` is the singleton ``Config`` instance.

    Args:
      name: string, converted to lowercase to define the name of the config
        option (and absl flag). It is converted to uppercase to define the
        corresponding shell environment variable.
      default: boolean, a default value for the option.
      help: string, used to populate the flag help information as well as the
        docstring of the returned context manager.
      update_global_hook: a optional callback that is called with the updated
        value of the global state when it is altered or set initially.
      update_thread_local_hook: a optional callback that is called with the
        updated value of the thread-local state when it is altered or set
        initially.

    Returns:
      A contextmanager to control the thread-local state value.

    Example:

      enable_foo = config.define_bool_state(
          name='jax_enable_foo',
          default=False,
          help='Enable foo.')

      # Now the JAX_ENABLE_FOO shell environment variable and --jax_enable_foo
      # command-line flag can be used to control the process-level value of
      # the configuration option, in addition to using e.g.
      # ``config.update("jax_enable_foo", True)`` directly. We can also use a
      # context manager:

      with enable_foo(True):
        ...

    The value of the thread-local state or flag can be accessed via
    ``config.jax_enable_foo``. Reading it via ``config.FLAGS.jax_enable_foo`` is
    an error.
    """
    name = name.lower()
    self.DEFINE_bool(name, bool_env(name.upper(), default), help,
                     update_hook=update_global_hook)
    self._contextmanager_flags.add(name)

    def get_state(self):
      val = getattr(_thread_local_state, name, unset)
      return val if val is not unset else self._read(name)
    setattr(Config, name, property(get_state))

    @contextlib.contextmanager
    def set_state(new_val: bool):
      prev_val = getattr(_thread_local_state, name, unset)
      setattr(_thread_local_state, name, new_val)
      if update_thread_local_hook:
        update_thread_local_hook(new_val)
      try:
        yield
      finally:
        if prev_val is unset:
          delattr(_thread_local_state, name)
          if update_thread_local_hook:
            update_thread_local_hook(None)
        else:
          setattr(_thread_local_state, name, prev_val)
          if update_thread_local_hook:
            update_thread_local_hook(prev_val)

    set_state.__name__ = name[4:] if name.startswith('jax_') else name
    set_state.__doc__ = f"Context manager for `{name}` config option.\n\n{help}"
    return set_state

_thread_local_state = threading.local()

class Unset: pass
unset = Unset()

class NameSpace(object):
  def __init__(self, getter):
    self._getter = getter

  def __getattr__(self, name):
    return self._getter(name)


config = Config()
flags = config
FLAGS = flags.FLAGS

already_configured_with_absl = False


flags.DEFINE_bool(
    'jax_omnistaging',
    bool_env('JAX_OMNISTAGING', True),
    help='Enable staging based on dynamic context rather than data dependence.'
)

flags.DEFINE_integer(
    'jax_tracer_error_num_traceback_frames',
    int_env('JAX_TRACER_ERROR_NUM_TRACEBACK_FRAMES', 5),
    help='Set the number of stack frames in JAX tracer error messages.'
)

flags.DEFINE_bool(
    'jax_host_callback_inline',
    bool_env('JAX_HOST_CALLBACK_INLINE', False),
    help='Inline the host_callback, if not in a staged context.'
)
flags.DEFINE_integer(
    'jax_host_callback_max_queue_byte_size',
    int_env('JAX_HOST_CALLBACK_MAX_QUEUE_BYTE_SIZE', int(256 * 1e6)),
    help=('The size in bytes of the buffer used to hold outfeeds from each '
          'device. When this capacity is reached consuming outfeeds from the '
          'device is paused, thus potentially pausing the device computation, '
          'until the Python callback consume more outfeeds.'),
    lower_bound=int(16 * 1e6)
)


enable_checks = config.define_bool_state(
    name='jax_enable_checks',
    default=False,
    help='Turn on invariant checking for JAX internals. Makes things slower.')

check_tracer_leaks = config.define_bool_state(
    name='jax_check_tracer_leaks',
    default=False,
    help=('Turn on checking for leaked tracers as soon as a trace completes. '
          'Enabling leak checking may have performance impacts: some caching '
          'is disabled, and other overheads may be added.'))
checking_leaks = functools.partial(check_tracer_leaks, True)

debug_nans = config.define_bool_state(
    name='jax_debug_nans',
    default=False,
    help=('Add nan checks to every operation. When a nan is detected on the '
          'output of a jit-compiled computation, call into the un-compiled '
          'version in an attempt to more precisely identify the operation '
          'which produced the nan.'))

debug_infs = config.define_bool_state(
    name='jax_debug_infs',
    default=False,
    help=('Add inf checks to every operation. When an inf is detected on the '
          'output of a jit-compiled computation, call into the un-compiled '
          'version in an attempt to more precisely identify the operation '
          'which produced the inf.'))

log_compiles = config.define_bool_state(
    name='jax_log_compiles',
    default=False,
    help=('Log a message each time every time `jit` or `pmap` compiles an XLA '
          'computation. Logging is performed with `absl.logging`. When this '
          'option is set, the log level is WARNING; otherwise the level is '
          'DEBUG.'))

def _update_x64_global(val):
  lib.jax_jit.global_state().enable_x64 = val

def _update_x64_thread_local(val):
  lib.jax_jit.thread_local_state().enable_x64 = val

enable_x64 = config.define_bool_state(
    name='jax_enable_x64',
    default=False,
    help='Enable 64-bit types to be used',
    update_global_hook=_update_x64_global,
    update_thread_local_hook=_update_x64_thread_local)

# TODO(phawkins): remove after fixing users of FLAGS.x64_enabled.
config._contextmanager_flags.remove("jax_enable_x64")

Config.x64_enabled = Config.jax_enable_x64  # type: ignore
