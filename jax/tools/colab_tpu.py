# Copyright 2020 The JAX Authors.
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

"""Utilities for running JAX on Cloud TPUs via Colab."""

import textwrap

message = """
As of JAX 0.4.0, JAX only supports TPU VMs, not the older Colab TPUs.

We recommend trying Kaggle Notebooks
(https://www.kaggle.com/code, click on "New Notebook" near the top) which offer
TPU VMs. You have to create an account, log in, and verify your account to get
accelerator support.
Once you do that, there's a new "TPU 1VM v3-8" accelerator option. This gives
you a TPU notebook environment similar to Colab, but using the newer TPU VM
architecture. This should be a less buggy, more performant, and overall better
experience than the older TPU node architecture.

It is also possible to use Colab together with a self-hosted Jupyter kernel
running on a Cloud TPU VM. See
https://research.google.com/colaboratory/local-runtimes.html
for details.
"""

def setup_tpu(tpu_driver_version=None):
  """Returns an error. Do not use."""
  raise RuntimeError(textwrap.dedent(message))
