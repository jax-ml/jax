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

"""Utilities for instrumenting code.

Code points can be marked as a named event. Every time an event is reached
during program execution, the registered listeners will be invoked.

A typical listener callback is to send an event to a metrics collector for
aggregation/exporting.
"""
from typing import Callable, List

_event_listeners: List[Callable[[str], None]] = []
_event_duration_secs_listeners: List[Callable[[str, float], None]] = []

def record_event(event: str):
  """Record an event."""
  for callback in _event_listeners:
    callback(event)

def record_event_duration_secs(event: str, duration: float):
  """Record an event duration in seconds (float)."""
  for callback in _event_duration_secs_listeners:
    callback(event, duration)

def register_event_listener(callback: Callable[[str], None]):
  """Register a callback to be invoked during record_event()."""
  _event_listeners.append(callback)

def register_event_duration_secs_listener(callback : Callable[[str, float], None]):
  """Register a callback to be invoked during record_event_duration_secs()."""
  _event_duration_secs_listeners.append(callback)

def _clear_event_listeners():
  """Clear event listeners."""
  global _event_listeners, _event_duration_secs_listeners
  _event_listeners = []
  _event_duration_secs_listeners = []
