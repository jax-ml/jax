# Copyright 2026 The JAX Authors.
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

import portpicker
from jupyter_client.provisioning.local_provisioner import LocalProvisioner

# The default LocalProvisioner selects ports in a racy way. By using portpicker
# we can have confidence that we will not reuse ports.
class PortpickerProvisioner(LocalProvisioner):
    async def pre_launch(self, **kwargs):
        self.ports_cached = False
        km = self.parent
        km.shell_port = portpicker.pick_unused_port()
        km.iopub_port = portpicker.pick_unused_port()
        km.stdin_port = portpicker.pick_unused_port()
        km.hb_port = portpicker.pick_unused_port()
        km.control_port = portpicker.pick_unused_port()
        self.ports_cached = True
        return await super().pre_launch(**kwargs)
