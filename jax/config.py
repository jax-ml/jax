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

import sys

class Config(object):
  def __init__(self):
    self.values = {}
    self.meta = {}
    self.FLAGS = NameSpace(self.read)
    self.use_absl = False

  def update(self, name, val):
    if self.use_absl:
      setattr(self.absl_flags.FLAGS, name, val)
    else:
      self.check_exists(name)
      if name not in self.values:
        raise Exception("Unrecognized config option: {}".format(name))
      self.values[name] = val

  def read(self, name):
    if self.use_absl:
      return getattr(self.absl_flags.FLAGS, name)
    else:
      self.check_exists(name)
      return self.values[name]

  def add_option(self, name, default, opt_type, meta_args, meta_kwargs):
    if name in self.values:
      raise Exception("Config option {} already defined".format(name))
    self.values[name] = default
    self.meta[name] = (opt_type, meta_args, meta_kwargs)

  def check_exists(self, name):
    if name not in self.values:
      raise Exception("Unrecognized config option: {}".format(name))

  def DEFINE_bool(self, name, default, *args, **kwargs):
    self.add_option(name, default, bool, args, kwargs)

  def DEFINE_integer(self, name, default, *args, **kwargs):
    self.add_option(name, default, int, args, kwargs)

  def DEFINE_string(self, name, default, *args, **kwargs):
    self.add_option(name, default, str, args, kwargs)

  def DEFINE_enum(self, name, default, *args, **kwargs):
    self.add_option(name, default, 'enum', args, kwargs)

  def config_with_absl(self):
    # Run this before calling `app.run(main)` etc
    import absl.flags as absl_FLAGS
    from absl import app, flags as absl_flags

    self.use_absl = True
    self.absl_flags = absl_flags
    absl_defs = {
        bool: absl_flags.DEFINE_bool,
        int: absl_flags.DEFINE_integer,
        str: absl_flags.DEFINE_string,
        'enum': absl_flags.DEFINE_enum
    }

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

class NameSpace(object):
  def __init__(self, getter):
    self._getter = getter

  def __getattr__(self, name):
    return self._getter(name)

config = Config()
flags = config
already_configured_with_absl = False
