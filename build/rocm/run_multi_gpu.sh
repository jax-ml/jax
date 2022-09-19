#!/usr/bin/env bash
# Copyright 2022 Google LLC
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

set -eux
# run test module with multi-gpu requirements. We currently do not have a way to filter tests.
# this issue is also tracked in https://github.com/google/jax/issues/7323
python3 -m pytest --reruns 3 -x tests/pmap_test.py
python3 -m pytest --reruns 3 -x tests/multi_device_test.py
