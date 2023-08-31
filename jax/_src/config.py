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

from __future__ import annotations

from collections.abc import Hashable, Iterator
import contextlib
import functools
import itertools
import logging
import os
import sys
import threading
from typing import Any, Callable, Generic, NamedTuple, Optional, TypeVar

from jax._src import lib
from jax._src.lib import jax_jit
from jax._src.lib import transfer_guard_lib
from jax._src.lib import xla_client
from jax._src import logging_config
from jax._src.lib import xla_extension_version

logger = logging.getLogger(__name__)

_T = TypeVar('_T')


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


class FlagHolder(Generic[_T]):
  def __init__(self, flags: NameSpace, name: str):
    self._flags = flags
    self._name = name

  @property
  def value(self) -> _T:
    return getattr(self._flags, self._name)


class Config:
  _HAS_DYNAMIC_ATTRIBUTES = True

  def __init__(self):
    self.values = {}
    self.meta = {}
    self.FLAGS = NameSpace(self.read, self.update)
    self.use_absl = False
    self._contextmanager_flags = set()
    self._update_hooks = {}

  def update(self, name, val):
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
    try:
      return self.values[name]
    except KeyError:
      raise AttributeError(f"Unrecognized config option: {name}")

  def add_option(self, name, default, opt_type, meta_args, meta_kwargs,
                 update_hook: Optional[Callable[[Any], None]] = None):
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

  def DEFINE_bool(self, name, default, *args, **kwargs) -> FlagHolder[bool]:
    update_hook = kwargs.pop("update_hook", None)
    self.add_option(name, default, bool, args, kwargs, update_hook=update_hook)
    return FlagHolder(self.FLAGS, name)

  def DEFINE_integer(self, name, default, *args, **kwargs) -> FlagHolder[int]:
    update_hook = kwargs.pop("update_hook", None)
    self.add_option(name, default, int, args, kwargs, update_hook=update_hook)
    return FlagHolder(self.FLAGS, name)

  def DEFINE_float(self, name, default, *args, **kwargs) -> FlagHolder[float]:
    update_hook = kwargs.pop("update_hook", None)
    self.add_option(name, default, float, args, kwargs, update_hook=update_hook)
    return FlagHolder(self.FLAGS, name)

  def DEFINE_string(self, name, default, *args, **kwargs) -> FlagHolder[str]:
    update_hook = kwargs.pop("update_hook", None)
    self.add_option(name, default, str, args, kwargs, update_hook=update_hook)
    return FlagHolder(self.FLAGS, name)

  def DEFINE_enum(self, name, default, *args, **kwargs) -> FlagHolder[str]:
    update_hook = kwargs.pop("update_hook", None)
    self.add_option(name, default, 'enum', args, kwargs,
                    update_hook=update_hook)
    return FlagHolder(self.FLAGS, name)

  def config_with_absl(self):
    # Run this before calling `app.run(main)` etc
    import absl.flags as absl_FLAGS  # noqa: F401  # pytype: disable=import-error
    from absl import app, flags as absl_flags  # pytype: disable=import-error

    self.use_absl = True
    self.absl_flags = absl_flags
    absl_defs = { bool: absl_flags.DEFINE_bool,
                  int:  absl_flags.DEFINE_integer,
                  float: absl_flags.DEFINE_float,
                  str:  absl_flags.DEFINE_string,
                  'enum': absl_flags.DEFINE_enum }

    for name, val in self.values.items():
      flag_type, meta_args, meta_kwargs = self.meta[name]
      absl_defs[flag_type](name, val, *meta_args, **meta_kwargs)
    app.call_after_init(lambda: self.complete_absl_config(absl_flags))

  def complete_absl_config(self, absl_flags):
    for name, _ in self.values.items():
      try:
        flag = absl_flags.FLAGS[name]
      except KeyError:
        # This can happen if a new flag was added after config_with_absl() was
        # called, but before complete_absl_config was run. We could in principle
        # add code to DEFINE_... to register any newly added flags with ABSL
        # if config_with_absl() has already been called, but arguably the user
        # should have called config_with_absl() later.
        continue
      if flag.present:
        self.update(name, flag.value)

  def parse_flags_with_absl(self):
    global already_configured_with_absl
    if not already_configured_with_absl:
      # Extract just the --jax... flags (before the first --) from argv. In some
      # environments (e.g. ipython/colab) argv might be a mess of things
      # parseable by absl and other junk.
      jax_argv = itertools.takewhile(lambda a: a != '--', sys.argv)
      jax_argv = ['', *(a for a in jax_argv if a.startswith('--jax'))]

      import absl.flags  # pytype: disable=import-error
      self.config_with_absl()
      absl.flags.FLAGS(jax_argv, known_only=True)
      self.complete_absl_config(absl.flags)
      already_configured_with_absl = True

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
      val = _thread_local_state.__dict__.get(name, unset)
      return val if val is not unset else self._read(name)
    setattr(Config, name, property(get_state))

    return _StateContextManager(name, help, update_thread_local_hook,
                                extra_description=extra_description,
                                default_value=True)

  def define_enum_state(
      self, name: str, enum_values: list[str], default: Optional[str],
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
      val = _thread_local_state.__dict__.get(name, unset)
      return val if val is not unset else self._read(name)
    setattr(Config, name, property(get_state))

    def validate(new_val):
      if (new_val is not None and
          (type(new_val) is not str or new_val not in enum_values)):
        raise ValueError(f"new enum value must be None or in {enum_values}, "
                         f"got {new_val} of type {type(new_val)}.")

    return _StateContextManager(name, help, update_thread_local_hook, validate)

  def define_int_state(
      self, name: str, default: Optional[int],
      help: str, update_global_hook: Optional[Callable[[str], None]] = None,
      update_thread_local_hook: Optional[Callable[[Optional[str]], None]] \
        = None):
    """Set up thread-local state and return a contextmanager for managing it.
    Args:
      name: string, converted to lowercase to define the name of the config
        option (and absl flag). It is converted to uppercase to define the
        corresponding shell environment variable.
      default: optional int, default value.
      help: string, used to populate the flag help information as well as the
        docstring of the returned context manager.
    Returns:
      A contextmanager to control the thread-local state value.
    See docstring for ``define_bool_state``.
    """
    name = name.lower()
    default_env = os.getenv(name.upper(), default)
    if default_env is not None:
      try:
        default = int(default_env)
      except ValueError:
        raise ValueError(f"Invalid value \"{default_env}\" for JAX flag {name}")
    self.DEFINE_integer(name, default, help=help, update_hook=update_global_hook)
    self._contextmanager_flags.add(name)

    def get_state(self):
      val = _thread_local_state.__dict__.get(name, unset)
      return val if val is not unset else self._read(name)
    setattr(Config, name, property(get_state))

    def validate(new_val):
      if new_val is not None and not isinstance(new_val, int):
        raise ValueError(f'new int config value must be None or of type int, '
                         f'got {new_val} of type {type(new_val)}')

    return _StateContextManager(name, help, update_thread_local_hook, validate)

  def define_float_state(
      self, name: str, default: Optional[float],
      help: str, update_global_hook: Optional[Callable[[str], None]] = None,
      update_thread_local_hook: Optional[Callable[[Optional[str]], None]] \
        = None):
    """Set up thread-local state and return a contextmanager for managing it.
    Args:
      name: string, converted to lowercase to define the name of the config
        option (and absl flag). It is converted to uppercase to define the
        corresponding shell environment variable.
      default: optional float, default value.
      help: string, used to populate the flag help information as well as the
        docstring of the returned context manager.
    Returns:
      A contextmanager to control the thread-local state value.
    See docstring for ``define_bool_state``.
    """
    name = name.lower()
    default_env = os.getenv(name.upper(), default)
    if default_env is not None:
      try:
        default = float(default_env)
      except ValueError:
        raise ValueError(f"Invalid value \"{default_env}\" for JAX flag {name}")
    self.DEFINE_float(name, default, help=help, update_hook=update_global_hook)
    self._contextmanager_flags.add(name)

    def get_state(self):
      val = _thread_local_state.__dict__.get(name, unset)
      return val if val is not unset else self._read(name)
    setattr(Config, name, property(get_state))

    def validate(new_val):
      if new_val is not None and not isinstance(new_val, (float, int)):
        raise ValueError(f'new float config value must be None or of type float, '
                         f'got {new_val} of type {type(new_val)}')

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
      val = _thread_local_state.__dict__.get(name, unset)
      return val if val is not unset else self._read(name)
    setattr(Config, name, property(get_state))

    return _StateContextManager(name, help, update_thread_local_hook,
                                validate_new_val_hook)

  def _trace_context(self):
    """Returns a tuple of configuration values that affect tracing.

    These values are included in the cache key for linear_util.cache.

    Values included in this set should also most likely be included in
    the C++ JIT state, which is handled separately."""
    tls = jax_jit.thread_local_state()
    axis_env_state = ()
    mesh_context_manager = ()
    context = tls.extra_jit_context
    if context and context.axis_env_state is not None:
      axis_env_state = context.axis_env_state
    if context and context.mesh_context_manager:
      mesh_context_manager = context.mesh_context_manager
    return (axis_env_state, mesh_context_manager, self.x64_enabled,
            self.jax_numpy_rank_promotion, self.jax_default_matmul_precision,
            self.jax_dynamic_shapes, self.jax_numpy_dtype_promotion,
            self.jax_default_device,
            self.jax_threefry_partitionable,
            self.jax_softmax_custom_jvp,
            self.jax_enable_memories,
            self.jax_disable_jit,
            # Technically this affects jaxpr->MHLO lowering, not tracing.
            self.jax_hlo_source_file_canonicalization_regex)

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
  def __call__(self, new_val: Any = no_default):
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

def DEFINE_bool(name, default, *args, **kwargs):
  return flags.DEFINE_bool(name, default, *args, **kwargs)

def DEFINE_integer(name, default, *args, **kwargs):
  return flags.DEFINE_integer(name, default, *args, **kwargs)

def DEFINE_float(name, default, *args, **kwargs):
  return flags.DEFINE_float(name, default, *args, **kwargs)

def DEFINE_string(name, default, *args, **kwargs):
  return flags.DEFINE_string(name, default, *args, **kwargs)

def DEFINE_enum(name, default, *args, **kwargs):
  return flags.DEFINE_enum(name, default, *args, **kwargs)


already_configured_with_absl = False


# The C++ JIT maintains its own copy of several configuration items as
# a global/thread-local state. These methods allow updates to part of the
# state when a configuration value changes.
class _GlobalExtraJitContext(NamedTuple):
  numpy_rank_promotion: Optional[str] = None
  numpy_dtype_promotion: Optional[str] = None
  default_matmul_precision: Optional[Any] = None
  dynamic_shapes: bool = False
  threefry_partitionable: bool = False
  softmax_custom_jvp: bool = False


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
  axis_env_state: Hashable = ()
  mesh_context_manager: Hashable = ()
  numpy_rank_promotion: Optional[str] = None
  numpy_dtype_promotion: Optional[str] = None
  default_matmul_precision: Optional[Any] = None
  dynamic_shapes: bool = False
  softmax_custom_jvp: bool = False


class _ThreadLocalStateCache(threading.local):
  """"A thread local cache for _ThreadLocalExtraJitContext

  The extra_jit_context in jax_jit.thread_local_state() may get updated and thus
  incurring dispatch overhead for comparing this python object during jit calls.
  We want to duduplicate the objects that have the same hash/equality to also
  have the same object ID, since the equality check is much faster if the object
  IDs match.
  """
  def __init__(self):
    self.canonicalize = functools.lru_cache(128)(lambda x: x)


_thread_local_state_cache = _ThreadLocalStateCache()


def update_thread_local_jit_state(**kw):
  tls = jax_jit.thread_local_state()
  # After xla_client._version >= 70, the thread_local object will necessarily
  # be initialized when accessed. The following line can be removed when the
  # minimum  jaxlib version is past version 70
  context = tls.extra_jit_context or _ThreadLocalExtraJitContext()
  tmp = context._replace(**kw)
  tls.extra_jit_context = _thread_local_state_cache.canonicalize(tmp)


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

jax2tf_default_native_serialization = config.define_bool_state(
    name='jax2tf_default_native_serialization',
    default=bool_env('JAX2TF_DEFAULT_NATIVE_SERIALIZATION', True),
    help=(
        'Sets the default value of the native_serialization parameter to '
        'jax2tf.convert. Prefer using the parameter instead of the flag, '
        'the flag may be removed in the future.'
    )
)

jax_serialization_version = config.define_int_state(
    name='jax_serialization_version',
    # Note: bump the default serialization version at least one month after
    # we update XlaCallModule to support the new version, so that serialized
    # modules are forward compatible with deployed versions of XlaCallModule.
    # Version 7 of XlaCallModule is supported since July 12th, 2023.
    default=int_env('JAX_SERIALIZATION_VERSION', 7),
    help=(
        'The version number to use for native serialization. This must be '
        'within the range of versions supported by the tf.XlaCallModule '
        'used in your deployment environment. '
        'See https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#native-serialization-versions.'
    )
)

jax_platforms = config.define_string_state(
    name='jax_platforms',
    default=None,
    help=(
        'Comma-separated list of platform names specifying which platforms jax '
        'should initialize. If any of the platforms in this list are not successfully '
        'initialized, an exception will be raised and the program will be aborted. '
        'The first platform in the list will be the default platform. '
        'For example, config.jax_platforms=cpu,tpu means that CPU and TPU backends '
        'will be initialized, and the CPU backend will be used unless otherwise '
        'specified. If TPU initialization fails, it will raise an exception. '
        'By default, jax will try to initialize all available '
        'platforms and will default to GPU or TPU if available, and fallback to CPU '
        'otherwise.'
        ))

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
          'computation. Logging is performed with `logging`. When this '
          'option is set, the log level is WARNING; otherwise the level is '
          'DEBUG.'))

log_checkpoint_residuals = config.define_bool_state(
    name='jax_log_checkpoint_residuals',
    default=False,
    help=('Log a message every time jax.checkpoint (aka jax.remat) is '
          'partially evaluated (e.g. for autodiff), printing what residuals '
          'are saved.'))

parallel_functions_output_gda = config.define_bool_state(
    name='jax_parallel_functions_output_gda',
    default=False,
    help='If True, pjit will output GDAs.')

pmap_shmap_merge = config.define_bool_state(
    name='jax_pmap_shmap_merge',
    default=False,
    upgrade=True,
    help='If True, pmap and shard_map API will be merged.')

def _update_jax_memories_global(val):
  if xla_extension_version >= 190:
    lib.jax_jit.global_state().enable_memories = val

def _update_jax_memories_thread_local(val):
  if xla_extension_version >= 190:
    lib.jax_jit.thread_local_state().enable_memories = val

enable_memories = config.define_bool_state(
    'jax_enable_memories',
    default=False,
    upgrade=True,
    update_global_hook=_update_jax_memories_global,
    update_thread_local_hook=_update_jax_memories_thread_local,
    help=("If True, will allow fetching memory kinds available on executable "
          "and annotate Shardings with it."))

spmd_mode = config.define_enum_state(
    name='jax_spmd_mode',
    enum_values=['allow_all', 'allow_jit', 'allow_pjit'],
    default='allow_jit',
    help=("Decides whether Math on `jax.Array`'s that are not fully addressable "
          "(i.e. spans across multiple processes) is allowed. The options are: "
          "* allow_pjit: Default, only `pjit` computations are allowed to "
          "    execute on non-fully addressable `jax.Array`s\n"
          "* allow_jit: `pjit` and `jax.jit` computations are allowed to "
          "    execute on non-fully addressable `jax.Array`s\n"
          "* allow_all: `jnp`, normal math (like `a + b`, etc), `pjit`, "
          "     `jax.jit` and all other operations are allowed to "
          "     execute on non-fully addresable `jax.Array`s."))


distributed_debug = config.define_bool_state(
    name='jax_distributed_debug',
    default=False,
    help=('Enable logging useful for debugging multi-process distributed '
          'computations. Logging is performed with `logging` at WARNING '
          'level.'))

legacy_prng_key = config.define_enum_state(
    name='jax_legacy_prng_key',
    enum_values=['allow', 'warn', 'error'],
    default='allow',
    help=('Specify the behavior when raw PRNG keys are passed to '
          'jax.random APIs.')
)

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

threefry_partitionable = config.define_bool_state(
    name='jax_threefry_partitionable',
    default=False,
    upgrade=True,
    help=('Enables internal threefry PRNG implementation changes that '
          'render it automatically partitionable in some cases. Without this '
          'flag, using the standard jax.random pseudo-random number generation '
          'may result in extraneous communication and/or redundant distributed '
          'computation. With this flag, the communication overheads disappear '
          'in some cases.'),
    update_global_hook=lambda val: _update_global_jit_state(
        threefry_partitionable=val),
    update_thread_local_hook=lambda val: update_thread_local_jit_state(
        threefry_partitionable=val))


softmax_custom_jvp = config.define_bool_state(
    name='jax_softmax_custom_jvp',
    default=False,
    upgrade=True,
    help=('Use a new custom_jvp rule for jax.nn.softmax. The new rule should '
          'improve memory usage and stability. Set True to use new '
          'behavior. See https://github.com/google/jax/pull/15677'),
    update_global_hook=lambda val: _update_global_jit_state(
        softmax_custom_jvp=val),
    update_thread_local_hook=lambda val: update_thread_local_jit_state(
        softmax_custom_jvp=val))


enable_custom_vjp_by_custom_transpose = config.define_bool_state(
    name='jax_enable_custom_vjp_by_custom_transpose',
    default=False,
    upgrade=True,
    help=('Enables an internal upgrade that implements `jax.custom_vjp` by '
          'reduction to `jax.custom_jvp` and `jax.custom_transpose`.'))

raise_persistent_cache_errors = config.define_bool_state(
    name='jax_raise_persistent_cache_errors',
    default=False,
    help=('If true, exceptions raised when reading or writing to the '
          'persistent compilation cache will be allowed through, halting '
          'program execution if not manually caught. If false, exceptions are '
          'caught and raised as warnings, allowing program execution to '
          'continue. Defaults to false so cache bugs or intermittent issues '
          'are non-fatal.'))

persistent_cache_min_compile_time_secs = config.define_float_state(
    name='jax_persistent_cache_min_compile_time_secs',
    default=1,
    help=('The minimum compile time of a computation to be written to the '
          'persistent compilation cache. This threshold can be raised to '
          'decrease the number of entries written to the cache.'))

compilation_cache_include_metadata_in_key = config.define_bool_state(
    name='jax_compilation_cache_include_metadata_in_key',
    default=False,
    help=(
        'Include metadata, such as file names and line numbers, in the'
        ' compilation cache key. If false, the cache will still get hits even'
        ' if functions or files are moved, etc. However, it means that'
        ' executables loaded from the cache may have stale metadata, which'
        ' may show up in, e.g., profiles.'
    ),
)

hlo_source_file_canonicalization_regex = config.define_string_state(
    name='jax_hlo_source_file_canonicalization_regex',
    default=None,
    help=('Used to canonicalize the source_path metadata of HLO instructions '
          'by removing the given regex. If set, re.sub() is called on each '
          'source_file with the given regex, and all matches are removed. '
          'This can be used to avoid spurious cache misses when using the '
          'persistent compilation cache, which includes HLO metadata in the '
          'cache key.'))

include_full_tracebacks_in_locations = config.define_bool_state(
    name='jax_include_full_tracebacks_in_locations',
    default=False,
    help=(
        'Include full Python tracebacks in MLIR locations in IR emitted by JAX.'
    ),
)

use_original_compilation_cache_key_generation = config.define_bool_state(
    name='jax_use_original_compilation_cache_key_generation',
    default=True,
    help="If true, use the original cache-key generation algorithm. This is "
         "a transient flag; once the new cache-key generation algorithm is "
         "deployed, this flag and the original cache-key generation algorithm "
         "will be removed.")

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
      logger.info(
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
    enum_values=["off", "tracebackhide", "remove_frames", "quiet_remove_frames",
                 "auto"],
    default="auto",
    help="Controls how JAX filters internal frames out of tracebacks.\n\n"
         "Valid values are:\n"
         " * \"off\": disables traceback filtering.\n"
         " * \"auto\": use \"tracebackhide\" if running under a sufficiently"
         " new IPython, or \"remove_frames\" otherwise.\n"
         " * \"tracebackhide\": adds \"__tracebackhide__\" annotations to"
         " hidden stack frames, which some traceback printers support.\n"
         " * \"remove_frames\": removes hidden frames from tracebacks, and adds"
         " the unfiltered traceback as a __cause__ of the exception.\n"
         " * \"quiet_remove_frames\": removes hidden frames from tracebacks, and adds"
         " a brief message (to the __cause__ of the exception) describing that this has"
         " happened.\n")

# This flag is for internal use.
# TODO(tianjianlu): Removes once we always enable cusparse lowering.
# TODO(b/262050896): Set to true after bug is fixed
bcoo_cusparse_lowering = config.define_bool_state(
    name='jax_bcoo_cusparse_lowering',
    default=False,
    help=('Enables lowering BCOO ops to cuSparse.'))

# TODO(mattjj): remove this flag when we ensure we only succeed at trace-staging
# if the intended backend can handle lowering the result
config.define_bool_state(
    name='jax_dynamic_shapes',
    default=bool(os.getenv('JAX_DYNAMIC_SHAPES', '')),
    help=('Enables experimental features for staging out computations with '
          'dynamic shapes.'),
    update_global_hook=lambda val: \
      _update_global_jit_state(dynamic_shapes=val),
    update_thread_local_hook=lambda val: \
      update_thread_local_jit_state(dynamic_shapes=val))

# This flag is temporary during rollout of the remat barrier.
# TODO(parkers): Remove if there are no complaints.
config.define_bool_state(
    name='jax_remat_opt_barrier',
    default=(lib.version >= (0, 3, 6)),
    help=('Enables using optimization-barrier op for lowering remat.'))

# TODO(sharadmv,mattjj): set default to True, then remove
config.define_bool_state(
    name='jax_eager_pmap',
    default=True,
    upgrade=True,
    help='Enable eager-mode pmap when jax_disable_jit is activated.')

config.define_bool_state(
    name='jax_experimental_unsafe_xla_runtime_errors',
    default=False,
    help=('Enable XLA runtime errors for jax.experimental.checkify.checks '
          'on CPU and GPU. These errors are async, might get lost and are not '
          'very readable. But, they crash the computation and enable you '
          'to write jittable checks without needing to checkify. Does not '
          'work under pmap/pjit.')
)

jax_xla_profile_version = config.define_int_state(
    name='jax_xla_profile_version',
    default=0,
    help=('Optional profile version for XLA compilation. This is meaningful '
          'only when XLA is configured to support the remote compilation '
          'profile feature.')
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
  """A contextmanager to control the transfer guard level for all transfers.

  For more information, see
  https://jax.readthedocs.io/en/latest/transfer_guard.html

  Args:
    new_val: The new thread-local transfer guard level for all transfers.

  Yields:
    None.
  """
  with contextlib.ExitStack() as stack:
    stack.enter_context(transfer_guard_host_to_device(new_val))
    stack.enter_context(transfer_guard_device_to_device(new_val))
    stack.enter_context(transfer_guard_device_to_host(new_val))
    stack.enter_context(_transfer_guard(new_val))
    yield


def _update_debug_log_modules(module_names_str: Optional[str]):
  logging_config.disable_all_debug_logging()
  if not module_names_str:
    return
  module_names = module_names_str.split(',')
  for module_name in module_names:
    logging_config.enable_debug_logging(module_name)

# Don't define a context manager since this isn't threadsafe.
config.define_string_state(
    name='jax_debug_log_modules',
    default='',
    help=('Comma-separated list of module names (e.g. "jax" or '
          '"jax._src.xla_bridge,jax._src.dispatch") to enable debug logging '
          'for.'),
    update_global_hook=_update_debug_log_modules)
