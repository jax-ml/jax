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

# TODO(phawkins): this file triggers a pytype bug.
# pytype: skip-file

import contextlib
import functools
import itertools
import os
import sys
import threading
from typing import Any, List, Callable, NamedTuple, Iterator, Optional
import warnings

from absl import logging

from jax._src import lib
from jax._src.lib import jax_jit
from jax._src.lib import transfer_guard_lib
from jax._src.lib import xla_client

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
    raise ValueError(f"invalid truth value {val!r} for environment {varname!r}")

def int_env(varname: str, default: int) -> int:
  """Read an environment variable and interpret it as an integer."""
  return int(os.getenv(varname, str(default)))


UPGRADE_BOOL_HELP = (
    " This will be enabled by default in future versions of JAX, at which "
    "point all uses of the flag will be considered deprecated (following "
    "the `API compatibility policy "
    "<https://jax.readthedocs.io/en/latest/api_compatibility.html>`_).")

UPGRADE_BOOL_EXTRA_DESC = " (transient)"


class Config:
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self):
    self.values = {}
    self.meta = {}
    self.FLAGS = NameSpace(self.read, self.update)
    self.use_absl = False
    self._contextmanager_flags = set()
    self._update_hooks = {}

    self.omnistaging_enabled = True  # TODO(mattjj): remove this

  def update(self, name, val):
    if self.use_absl:
      setattr(self.absl_flags.FLAGS, name, val)
    else:
      self.check_exists(name)
      if name not in self.values:
        raise Exception(f"Unrecognized config option: {name}")
      self.values[name] = val

    hook = self._update_hooks.get(name, None)
    if hook:
      hook(val)

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
      raise Exception(f"Config option {name} already defined")
    self.values[name] = default
    self.meta[name] = (opt_type, meta_args, meta_kwargs)
    if update_hook:
      self._update_hooks[name] = update_hook
      update_hook(default)

  def check_exists(self, name):
    if name not in self.values:
      raise AttributeError(f"Unrecognized config option: {name}")

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
      # Extract just the --jax... flags (before the first --) from argv. In some
      # environments (e.g. ipython/colab) argv might be a mess of things
      # parseable by absl and other junk.
      jax_argv = itertools.takewhile(lambda a: a != '--', sys.argv)
      jax_argv = ['', *(a for a in jax_argv if a.startswith('--jax'))]

      import absl.flags
      self.config_with_absl()
      absl.flags.FLAGS(jax_argv, known_only=True)
      self.complete_absl_config(absl.flags)
      already_configured_with_absl = True

      if not FLAGS.jax_omnistaging:
        raise Exception(
          "Disabling of omnistaging is no longer supported in JAX version 0.2.12 and higher: "
          "see https://github.com/google/jax/blob/main/design_notes/omnistaging.md.\n"
          "To remove this warning, unset the JAX_OMNISTAGING environment variable.")

  def enable_omnistaging(self):
    warnings.warn(
        "enable_omnistaging() is a no-op in JAX versions 0.2.12 and higher;\n"
        "see https://github.com/google/jax/blob/main/design_notes/omnistaging.md")

  def disable_omnistaging(self):
    raise Exception(
      "Disabling of omnistaging is no longer supported in JAX version 0.2.12 and higher: "
      "see https://github.com/google/jax/blob/main/design_notes/omnistaging.md.")

  def define_bool_state(
    self, name: str, default: bool, help: str, *,
    update_global_hook: Optional[Callable[[bool], None]] = None,
    update_thread_local_hook: Optional[Callable[[Optional[bool]], None]] = None,
    upgrade: bool = False,
    extra_description: str = ""):
    """Set up thread-local state and return a contextmanager for managing it.

    This function is a convenience wrapper. It defines a flag, environment
    variable, and corresponding thread-local state, which can be managed via the
    contextmanager it returns.

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
      upgrade: optional indicator that this flag controls a canonical feature
        upgrade, so that it is `True` for the incoming functionality, `False`
        for the outgoing functionality to be deprecated.
      extra_description: string, optional: extra information to add to the
        summary description.

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
    if upgrade:
      help += ' ' + UPGRADE_BOOL_HELP
      extra_description += UPGRADE_BOOL_EXTRA_DESC
    self.DEFINE_bool(name, bool_env(name.upper(), default), help,
                     update_hook=update_global_hook)
    self._contextmanager_flags.add(name)

    def get_state(self):
      val = getattr(_thread_local_state, name, unset)
      return val if val is not unset else self._read(name)
    setattr(Config, name, property(get_state))

    return _StateContextManager(name, help, update_thread_local_hook,
                                extra_description=extra_description,
                                default_value=True)

  def define_enum_state(
      self, name: str, enum_values: List[str], default: Optional[str],
      help: str, update_global_hook: Optional[Callable[[str], None]] = None,
      update_thread_local_hook: Optional[Callable[[Optional[str]], None]] \
        = None):
    """Set up thread-local state and return a contextmanager for managing it.
    Args:
      name: string, converted to lowercase to define the name of the config
        option (and absl flag). It is converted to uppercase to define the
        corresponding shell environment variable.
      enum_values: list of strings representing the possible values for the
        option.
      default: optional string, default value.
      help: string, used to populate the flag help information as well as the
        docstring of the returned context manager.
    Returns:
      A contextmanager to control the thread-local state value.
    See docstring for ``define_bool_state``.
    """
    name = name.lower()
    default = os.getenv(name.upper(), default)
    if default is not None and default not in enum_values:
      raise ValueError(f"Invalid value \"{default}\" for JAX flag {name}")
    self.DEFINE_enum(name, default,
                     enum_values=enum_values, help=help,
                     update_hook=update_global_hook)
    self._contextmanager_flags.add(name)

    def get_state(self):
      val = getattr(_thread_local_state, name, unset)
      return val if val is not unset else self._read(name)
    setattr(Config, name, property(get_state))

    def validate(new_val):
      if (new_val is not None and
          (type(new_val) is not str or new_val not in enum_values)):
        raise ValueError(f"new enum value must be None or in {enum_values}, "
                         f"got {new_val} of type {type(new_val)}.")

    return _StateContextManager(name, help, update_thread_local_hook, validate)

  def define_string_state(
      self, name: str, default: Optional[str], help: str,
      update_global_hook: Optional[Callable[[str], None]] = None,
      update_thread_local_hook: Optional[Callable[[Optional[str]], None]] = None):
    """Set up thread-local state and return a contextmanager for managing it.

    See docstring for ``define_bool_state``.

    Args:
      name: string, converted to lowercase to define the name of the config
        option (and absl flag). It is converted to uppercase to define the
        corresponding shell environment variable.
      default: string, a default value for the option.
      help: string, used to populate the flag help information as well as the
        docstring of the returned context manager.
      update_global_hook: an optional callback that is called with the updated
        value of the global state when it is altered or set initially.
      update_thread_local_hook: an optional callback that is called with the
        updated value of the thread-local state when it is altered or set
        initially.

    Returns:
      A contextmanager to control the thread-local state value.
    """

    def validate(new_val):
      if new_val is not None and not isinstance(new_val, str):
        raise ValueError(f'new string config value must be None or of type str,'
                         f' got {new_val} of type {type(new_val)}.')

    return self.define_string_or_object_state(name, default, help,
                                              update_global_hook,
                                              update_thread_local_hook,
                                              validate)

  def define_string_or_object_state(
      self,
      name: str,
      default: Any,
      help: str,
      update_global_hook: Optional[Callable[[Any], None]] = None,
      update_thread_local_hook: Optional[Callable[[Any], None]] = None,
      validate_new_val_hook: Optional[Callable[[Any], None]] = None):
    """Set up thread-local state and return a contextmanager for managing it.

    Similar to ``define_string_state``, except the context manager will accept
    any object, not just a string. Any value passed via commandline flag or
    environment variable will be treated as a string.

    Args:
      name: string, converted to lowercase to define the name of the config
        option (and absl flag). It is converted to uppercase to define the
        corresponding shell environment variable.
      default: string, a default value for the option.
      help: string, used to populate the flag help information as well as the
        docstring of the returned context manager.
      update_global_hook: an optional callback that is called with the updated
        value of the global state when it is altered or set initially.
      update_thread_local_hook: an optional callback that is called with the
        updated value of the thread-local state when it is altered or set
        initially.
      validate_new_val_hook: an optional callback that is called with the new
        value on any update, and should raise an error if the new value is
        invalid.

    Returns:
      A contextmanager to control the thread-local state value.
    """
    name = name.lower()
    default = os.getenv(name.upper(), default)
    self.DEFINE_string(name, default, help=help,
                       update_hook=update_global_hook)
    self._contextmanager_flags.add(name)

    def get_state(self):
      val = getattr(_thread_local_state, name, unset)
      return val if val is not unset else self._read(name)
    setattr(Config, name, property(get_state))

    return _StateContextManager(name, help, update_thread_local_hook,
                                validate_new_val_hook)

  def _trace_context(self):
    """Returns a tuple of configuration values that affect tracing.

    These values are included in the cache key for linear_util.cache.

    Values included in this set should also most likely be included in
    the C++ JIT state, which is handled separately."""
    return (self.x64_enabled, self.jax_numpy_rank_promotion,
            self.jax_default_matmul_precision, self.jax_dynamic_shapes,
            self.jax_numpy_dtype_promotion)

class NoDefault: pass
no_default = NoDefault()

class _StateContextManager:
  def __init__(self, name, help, update_thread_local_hook,
               validate_new_val_hook: Optional[Callable[[Any], None]] = None,
               extra_description: str = "", default_value: Any = no_default):
    self._name = name
    self.__name__ = name[4:] if name.startswith('jax_') else name
    self.__doc__ = (f"Context manager for `{name}` config option"
                    f"{extra_description}.\n\n{help}")
    self._update_thread_local_hook = update_thread_local_hook
    self._validate_new_val_hook = validate_new_val_hook
    self._default_value = default_value

  @contextlib.contextmanager
  def __call__(self, new_val=no_default):
    if new_val is no_default:
      if self._default_value is not no_default:
        new_val = self._default_value  # default_value provided to constructor
      else:
        # no default_value provided to constructor and no value provided as an
        # argument, so we raise an error
        raise TypeError(f"Context manager for {self.__name__} config option "
                        "requires an argument representing the new value for "
                        "the config option.")
    if self._validate_new_val_hook:
      self._validate_new_val_hook(new_val)
    prev_val = getattr(_thread_local_state, self._name, unset)
    setattr(_thread_local_state, self._name, new_val)
    if self._update_thread_local_hook:
      self._update_thread_local_hook(new_val)
    try:
      yield
    finally:
      if prev_val is unset:
        delattr(_thread_local_state, self._name)
        if self._update_thread_local_hook:
          self._update_thread_local_hook(None)
      else:
        setattr(_thread_local_state, self._name, prev_val)
        if self._update_thread_local_hook:
          self._update_thread_local_hook(prev_val)

  def _add_hooks(self, update_global_hook, update_thread_local_hook):
    """Private method that adds hooks to an existing context-manager.

    Used to avoid cyclic import dependencies."""
    self._update_thread_local_hook = update_thread_local_hook
    config._update_hooks[self._name] = update_global_hook
    update_global_hook(config._read(self._name))


_thread_local_state = threading.local()

class _Unset: pass
unset = _Unset()

class NameSpace:
  def __init__(self, getter, setter):
    # must use super because we override this class's __setattr__, see
    # https://docs.python.org/3/reference/datamodel.html#object.__setattr__
    super().__setattr__('_getter', getter)
    super().__setattr__('_setter', setter)

  def __getattr__(self, name):
    return self._getter(name)

  def __setattr__(self, name, val):
    self._setter(name, val)


config = Config()
flags = config
FLAGS = flags.FLAGS

already_configured_with_absl = False


# The C++ JIT maintains its own copy of several configuration items as
# a global/thread-local state. These methods allow updates to part of the
# state when a configuration value changes.
class _GlobalExtraJitContext(NamedTuple):
  numpy_rank_promotion: Optional[str] = None
  numpy_dtype_promotion: Optional[str] = None
  default_matmul_precision: Optional[Any] = None
  dynamic_shapes: bool = False


def _update_global_jit_state(**kw):
  gs = jax_jit.global_state()
  context = gs.extra_jit_context or _GlobalExtraJitContext()
  gs.extra_jit_context = context._replace(**kw)


class _ThreadLocalExtraJitContext(NamedTuple):
  """"A namedtuple containing states to add to the cache key.

  Just in time compilation (for jit, pmap, etc) behavior is configurable through
  global and thread-local options, used in the cache key.

  The initialization, which uses both config.py and core.py is done using
  `_update_thread_local_jit_state` in core.py to prevent circular imports.
  """
  dynamic_trace_state: Optional[Any] = None
  numpy_rank_promotion: Optional[str] = None
  numpy_dtype_promotion: Optional[str] = None
  default_matmul_precision: Optional[Any] = None
  dynamic_shapes: bool = False


def update_thread_local_jit_state(**kw):
  tls = jax_jit.thread_local_state()
  # After xla_client._version >= 70, the thread_local object will necessarily
  # be initialized when accessed. The following line can be removed when the
  # minimum  jaxlib version is past version 70
  context = tls.extra_jit_context or _ThreadLocalExtraJitContext()
  tls.extra_jit_context = context._replace(**kw)


# TODO(mattjj): remove all uses of this flag
flags.DEFINE_bool(
    'jax_omnistaging',
    bool_env('JAX_OMNISTAGING', True),
    help=('Deprecated. Setting this flag to False raises an error. Setting it '
          'to True has no effect.'),
)

flags.DEFINE_integer(
    'jax_tracer_error_num_traceback_frames',
    int_env('JAX_TRACER_ERROR_NUM_TRACEBACK_FRAMES', 5),
    help='Set the number of stack frames in JAX tracer error messages.'
)

flags.DEFINE_bool(
    'jax_pprint_use_color',
    bool_env('JAX_PPRINT_USE_COLOR', True),
    help='Enable jaxpr pretty-printing with colorful syntax highlighting.'
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
flags.DEFINE_bool(
    'jax_host_callback_outfeed',
    bool_env('JAX_HOST_CALLBACK_OUTFEED', False),
    help=(
        'Use outfeed implementation for host_callback, even on CPU and GPU. '
        'If false, use the CustomCall implementation. '
        'Has no effect on TPU, since only the outfeed mechanism is implemented.'
    )
)
flags.DEFINE_bool(
    'jax_host_callback_ad_transforms',
    bool_env('JAX_HOST_CALLBACK_AD_TRANSFORMS', False),
    help=(
        'Enable support for jvp/vjp for the host_callback primitives. Default is '
        'False, which means that host_callback operates only on primals. '
        'The flag exists only temporarily, for backward compatibility.'
    )
)

# TODO(b/214340779): remove flag when XLA:CPU is improved.
jax2tf_associative_scan_reductions = config.define_bool_state(
    name='jax2tf_associative_scan_reductions',
    default=False,
    help=(
        'JAX has two separate lowering rules for the cumulative reduction '
        'primitives (cumsum, cumprod, cummax, cummin). On CPUs and GPUs it uses '
        'a lax.associative_scan, while for TPUs it uses the HLO ReduceWindow. '
        'The latter has a slow implementation on CPUs and GPUs. '
        'By default, jax2tf uses the TPU lowering. Set this flag to True to '
        'use the associative scan lowering usage, and only if it makes a difference '
        'for your application. '
        'See the jax2tf README.md for more details.'
    )
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
          'is disabled, and other overheads may be added. Additionally, be aware '
          'that some Python debuggers can cause false positives, so it is recommended '
          'to disable any debuggers while leak checking is enabled.'))
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

parallel_functions_output_gda = config.define_bool_state(
    name='jax_parallel_functions_output_gda',
    default=False,
    help='If True, pjit will output GDAs.')

jax_array = config.define_bool_state(
    name='jax_array',
    default=False,
    help=('If True, new pjit behavior will be enabled and `jax.Array` will be '
          'used.'))


distributed_debug = config.define_bool_state(
    name='jax_distributed_debug',
    default=False,
    help=('Enable logging useful for debugging multi-process distributed '
          'computations. Logging is performed with `absl.logging` at WARNING '
          'level.'))


enable_custom_prng = config.define_bool_state(
    name='jax_enable_custom_prng',
    default=False,
    upgrade=True,
    help=('Enables an internal upgrade that allows one to define custom '
          'pseudo-random number generator implementations.'))

default_prng_impl = config.define_enum_state(
    name='jax_default_prng_impl',
    enum_values=['threefry2x32', 'rbg', 'unsafe_rbg'],
    default='threefry2x32',
    help=('Select the default PRNG implementation, used when one is not '
          'explicitly provided at seeding time.'))

enable_custom_vjp_by_custom_transpose = config.define_bool_state(
    name='jax_enable_custom_vjp_by_custom_transpose',
    default=False,
    upgrade=True,
    help=('Enables an internal upgrade that implements `jax.custom_vjp` by '
          'reduction to `jax.custom_jvp` and `jax.custom_transpose`.'))

hlo_source_file_canonicalization_regex = config.define_string_state(
    name='jax_hlo_source_file_canonicalization_regex',
    default=None,
    help=('Used to canonicalize the source_path metadata of HLO instructions '
          'by removing the given regex. If set, re.sub() is called on each '
          'source_file with the given regex, and all matches are removed. '
          'This can be used to avoid spurious cache misses when using the '
          'persistent compilation cache, which includes HLO metadata in the '
          'cache key.'))

config.define_enum_state(
    name='jax_default_dtype_bits',
    enum_values=['32', '64'],
    default='64',
    help=('Specify bit width of default dtypes, either 32-bit or 64-bit. '
          'This is a temporary flag that will be used during the process '
          'of deprecating the ``jax_enable_x64`` flag.'))

numpy_dtype_promotion = config.define_enum_state(
    name='jax_numpy_dtype_promotion',
    enum_values=['standard', 'strict'],
    default='standard',
    help=('Specify the rules used for implicit type promotion in operations '
          'between arrays. Options are "standard" or "strict"; in strict-mode, '
          'binary operations between arrays of differing strongly-specified '
          'dtypes will result in an error.'),
    update_global_hook=lambda val: \
      _update_global_jit_state(numpy_dtype_promotion=val),
    update_thread_local_hook=lambda val: \
      update_thread_local_jit_state(numpy_dtype_promotion=val))

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
config._contextmanager_flags.remove('jax_enable_x64')

Config.x64_enabled = Config.jax_enable_x64  # type: ignore


def _update_default_device_global(val):
  lib.jax_jit.global_state().default_device = val


def _update_default_device_thread_local(val):
  lib.jax_jit.thread_local_state().default_device = val


def _validate_default_device(val):
  if val is not None and not isinstance(val, xla_client.Device):
    # TODO(skyewm): this is a workaround for non-PJRT Device types. Remove when
    # all JAX backends use a single C++ device interface.
    if 'Device' in str(type(val)):
      logging.info(
          'Allowing non-`xla_client.Device` default device: %s, type: %s',
          repr(val), type(val))
      return
    raise ValueError('jax.default_device must be passed a Device object (e.g. '
                     f"`jax.devices('cpu')[0]`), got: {repr(val)}")


# TODO(skye): default_device only accepts devices for now. Make it work with
# platform names as well (e.g. "cpu" to mean the same as jax.devices("cpu")[0]).
default_device = config.define_string_or_object_state(
    name='jax_default_device',
    default=None,
    help=(
        'Configure the default device for JAX operations. Set to a Device '
        'object (e.g. ``jax.devices("cpu")[0]``) to use that Device as the '
        'default device for JAX operations and jit\'d function calls (there is '
        'no effect on multi-device computations, e.g. pmapped function calls). '
        'Set to None to use the system default device. See '
        ':ref:`faq-data-placement` for more information on device placement.'),
    update_global_hook=_update_default_device_global,
    update_thread_local_hook=_update_default_device_thread_local,
    validate_new_val_hook=_validate_default_device)

def _update_disable_jit_global(val):
  lib.jax_jit.global_state().disable_jit = val

def _update_disable_jit_thread_local(val):
  lib.jax_jit.thread_local_state().disable_jit = val

disable_jit = config.define_bool_state(
    name='jax_disable_jit',
    default=False,
    help=('Disable JIT compilation and just call original Python.'),
    update_global_hook=_update_disable_jit_global,
    update_thread_local_hook=_update_disable_jit_thread_local)


numpy_rank_promotion = config.define_enum_state(
    name='jax_numpy_rank_promotion',
    enum_values=['allow', 'warn', 'raise'],
    default='allow',
    help=('Control NumPy-style automatic rank promotion broadcasting '
          '("allow", "warn", or "raise").'),
    update_global_hook=lambda val: \
      _update_global_jit_state(numpy_rank_promotion=val),
    update_thread_local_hook=lambda val: \
      update_thread_local_jit_state(numpy_rank_promotion=val))

default_matmul_precision = config.define_enum_state(
    name='jax_default_matmul_precision',
    enum_values=['bfloat16', 'tensorfloat32', 'float32'],
    default=None,
    help=('Control the default matmul and conv precision for 32bit inputs.\n\n'

          'Some platforms, like TPU, offer configurable precision levels for '
          'matrix multiplication and convolution computations, trading off '
          'accuracy for speed. The precision can be controlled for each '
          'operation; for example, see the :func:`jax.lax.conv_general_dilated` '
          'and :func:`jax.lax.dot` docstrings. But it can be useful to control '
          'the default behavior obtained when an operation is not given a '
          'specific precision.\n\n'

          'This option can be used to control the default precision '
          'level for computations involved in matrix multiplication and '
          'convolution on 32bit inputs. The levels roughly describe the '
          "precision at which scalar products are computed. The 'bfloat16' "
          "option is the fastest and least precise; 'float32' is similar to "
          "full float32 precision; 'tensorfloat32' is intermediate.\n\n"),
    update_global_hook=lambda val: \
      _update_global_jit_state(default_matmul_precision=val),
    update_thread_local_hook=lambda val: \
      update_thread_local_jit_state(default_matmul_precision=val))

traceback_filtering = config.define_enum_state(
    name = 'jax_traceback_filtering',
    enum_values=["off", "tracebackhide", "remove_frames", "auto"],
    default="auto",
    help="Controls how JAX filters internal frames out of tracebacks.\n\n"
         "Valid values are:\n"
         " * \"off\": disables traceback filtering.\n"
         " * \"auto\": use \"tracebackhide\" if running under a sufficiently "
         "new IPython, or \"remove_frames\" otherwise.\n"
         " * \"tracebackhide\": adds \"__tracebackhide__\" annotations to "
         " hidden stack frames, which some traceback printers support.\n"
         " * \"remove_frames\": removes hidden frames from tracebacks, and adds "
         " the unfiltered traceback as a __cause__ of the exception.\n")

# This flag is for internal use.
# TODO(tianjianlu): Removes once we always enable cusparse lowering.
bcoo_cusparse_lowering = config.define_bool_state(
    name='jax_bcoo_cusparse_lowering',
    default=True,
    help=('Enables lowering BCOO ops to cuSparse.'))

# TODO(mattjj): remove this flag when we ensure we only succeed at trace-staging
# if the intended backend can handle lowering the result
config.define_bool_state(
    name='jax_dynamic_shapes',
    default=False,
    help=('Enables experimental features for staging out computations with '
          'dynamic shapes.'),
    update_global_hook=lambda val: \
      _update_global_jit_state(dynamic_shapes=val),
    update_thread_local_hook=lambda val: \
      update_thread_local_jit_state(dynamic_shapes=val))

config.define_bool_state(
    name='jax_experimental_name_stack',
    default=True,
    help='Enable using the context manager-based name stack.')

# This flag is temporary during rollout of the remat barrier.
# TODO(parkers): Remove if there are no complaints.
config.define_bool_state(
    name='jax_remat_opt_barrier',
    default=(lib.version >= (0, 3, 6)),
    help=('Enables using optimization-barrier op for lowering remat.'))

# TODO(mattjj): remove after May 19 2022, NeurIPS submission deadline
config.define_bool_state(
    name='after_neurips',
    default=False,
    upgrade=True,
    help='Gate changes until after NeurIPS 2022 deadline.')

# TODO(b/205307544): Remove flag once coordination service has rolled out.
config.define_bool_state(
    name='jax_coordination_service',
    default=False,
    help=(
         'Use coordination service (experimental) instead of the default PjRT '
         'distributed runtime.'
    )
)

@contextlib.contextmanager
def explicit_device_put_scope() -> Iterator[None]:
  """Indicates that the current context is an explicit device_put*() call."""
  state = transfer_guard_lib.thread_local_state()
  prev = state.explicit_device_put
  state.explicit_device_put = True
  try:
    yield
  finally:
    state.explicit_device_put = prev

@contextlib.contextmanager
def explicit_device_get_scope() -> Iterator[None]:
  """Indicates that the current context is an explicit device_get() call."""
  state = transfer_guard_lib.thread_local_state()
  prev = state.explicit_device_get
  state.explicit_device_get = True
  try:
    yield
  finally:
    state.explicit_device_get = prev

def _update_transfer_guard(state, key, val):
  """Applies the transfer guard level within transfer_guard_lib."""
  if val is None:
    setattr(state, key, None)
  elif val == 'allow':
    setattr(state, key, transfer_guard_lib.TransferGuardLevel.ALLOW)
  elif val == 'log':
    setattr(state, key, transfer_guard_lib.TransferGuardLevel.LOG)
  elif val == 'disallow':
    setattr(state, key, transfer_guard_lib.TransferGuardLevel.DISALLOW)
  elif val == 'log_explicit':
    setattr(state, key, transfer_guard_lib.TransferGuardLevel.LOG_EXPLICIT)
  elif val == 'disallow_explicit':
    setattr(state, key, transfer_guard_lib.TransferGuardLevel.DISALLOW_EXPLICIT)
  else:
    assert False, f'Invalid transfer guard level {val}'

transfer_guard_host_to_device = config.define_enum_state(
    name='jax_transfer_guard_host_to_device',
    enum_values=[
        'allow', 'log', 'disallow', 'log_explicit', 'disallow_explicit'
    ],
    # The default is applied by transfer_guard_lib. Use None here to avoid
    # accidentally overriding --jax_transfer_guard.
    default=None,
    help=('Select the transfer guard level for host-to-device transfers. '
          'Default is "allow".'),
    update_global_hook=lambda val: _update_transfer_guard(
        transfer_guard_lib.global_state(), 'host_to_device', val),
    update_thread_local_hook=lambda val: _update_transfer_guard(
        transfer_guard_lib.thread_local_state(), 'host_to_device', val))

transfer_guard_device_to_device = config.define_enum_state(
    name='jax_transfer_guard_device_to_device',
    enum_values=[
        'allow', 'log', 'disallow', 'log_explicit', 'disallow_explicit'
    ],
    # The default is applied by transfer_guard_lib. Use None here to avoid
    # accidentally overriding --jax_transfer_guard.
    default=None,
    help=('Select the transfer guard level for device-to-device transfers. '
          'Default is "allow".'),
    update_global_hook=lambda val: _update_transfer_guard(
        transfer_guard_lib.global_state(), 'device_to_device', val),
    update_thread_local_hook=lambda val: _update_transfer_guard(
        transfer_guard_lib.thread_local_state(), 'device_to_device', val))

transfer_guard_device_to_host = config.define_enum_state(
    name='jax_transfer_guard_device_to_host',
    enum_values=[
        'allow', 'log', 'disallow', 'log_explicit', 'disallow_explicit'
    ],
    # The default is applied by transfer_guard_lib. Use None here to avoid
    # accidentally overriding --jax_transfer_guard.
    default=None,
    help=('Select the transfer guard level for device-to-host transfers. '
          'Default is "allow".'),
    update_global_hook=lambda val: _update_transfer_guard(
        transfer_guard_lib.global_state(), 'device_to_host', val),
    update_thread_local_hook=lambda val: _update_transfer_guard(
        transfer_guard_lib.thread_local_state(), 'device_to_host', val))

def _update_all_transfer_guard_global(val):
  for name in ('jax_transfer_guard_host_to_device',
               'jax_transfer_guard_device_to_device',
               'jax_transfer_guard_device_to_host'):
    config.update(name, val)

_transfer_guard = config.define_enum_state(
    name='jax_transfer_guard',
    enum_values=[
        'allow', 'log', 'disallow', 'log_explicit', 'disallow_explicit'
    ],
    # The default is applied by transfer_guard_lib. Use None here to avoid
    # accidentally overriding --jax_transfer_guard_*.
    default=None,
    help=('Select the transfer guard level for all transfers. This option is '
          'set-only; the transfer guard level for a specific direction should '
          'be read using the per-transfer direction option. '
          'Default is "allow".'),
    update_global_hook=_update_all_transfer_guard_global)

@contextlib.contextmanager
def transfer_guard(new_val: str) -> Iterator[None]:
  """Set up thread-local state and return a contextmanager for managing it."""
  with contextlib.ExitStack() as stack:
    stack.enter_context(transfer_guard_host_to_device(new_val))
    stack.enter_context(transfer_guard_device_to_device(new_val))
    stack.enter_context(transfer_guard_device_to_host(new_val))
    stack.enter_context(_transfer_guard(new_val))
    yield
