# Copyright 2020 Google LLC
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

from . import core


def get_var_partition_ids(jaxpr, partition_ids):
  """Returns a map from each `Var` to a partition ID."""
  var_partition_ids = {core.unitvar: partition_ids[0]}
  # FIXME(cjfj): For now, all inputs are assumed to be on inner partition 0.
  var_partition_ids.update((v, partition_ids[0]) for v in jaxpr.invars)
  var_partition_ids.update((v, partition_ids[0]) for v in jaxpr.constvars)

  for eqn in jaxpr.eqns:
    if eqn.primitive.name == "partition":
      inner_jaxpr = eqn.params["jaxpr"].jaxpr
      inner_partition_ids = eqn.params["partition_ids"]
      inner_var_partition_ids = get_var_partition_ids(inner_jaxpr, inner_partition_ids)
      var_partition_ids.update(
          (v1, inner_var_partition_ids[v]) for v, v1 in zip(inner_jaxpr.outvars, eqn.outvars))
    else:
      if eqn.primitive.name == "partition_put":
        # Map from 'inner' partition ID to the global partition ID.
        partition_id = partition_ids[eqn.params["partition_id"]]
      else:
        invars = [v for v in eqn.invars if not isinstance(v, core.Literal)]
        if invars:
          input_pids = [var_partition_ids[v] for v in invars]
          partition_id = input_pids[0]
          if any(pid != partition_id for pid in input_pids[1:]):
            raise ValueError(
                "mismatched input partition IDs {} for eqn: {}".format(
                    input_pids, core.pp_eqn(eqn)))
        else:
          # FIXME(cjfj): Do something cleverer here?
          partition_id = partition_ids[0]

      var_partition_ids.update((v, partition_id) for v in eqn.outvars)
  return var_partition_ids
