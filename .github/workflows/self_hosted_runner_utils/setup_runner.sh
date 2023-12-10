#!/bin/bash

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

# Heavily influenced by
# https://github.com/openxla/iree/tree/main/build_tools/github_actions/runner/config

set -eux

if [ "$#" -ne 3 ]; then
  echo "Usage: setup_runner.sh <runner name> <tags> <github token>"
fi

runner_name="$1"
runner_tags="$2"
runner_token="$3"

# Secret fourth argument for setting the repo URL. Useful for testing with forks.
# - sets empty string as default to avoid unbound variable error from set -u
jax_repo_url="${4-}"
if [ -z "${jax_repo_url}" ]; then
  jax_repo_url="https://github.com/google/jax"
fi

# Create `runner` user. This user won't have sudo access unless you ssh into the
# GCP VM as `runner` using gcloud. Don't do that!
sudo useradd runner -m

# Find the latest actions-runner download. The runner will automatically update
# itself when new versions are released. Github requires that all self-hosted
# runners are updated to the latest version within 30 days of release
# (https://docs.github.com/en/actions/hosting-your-own-runners/autoscaling-with-self-hosted-runners#controlling-runner-software-updates-on-self-hosted-runners).
# Example URL:
# https://github.com/actions/runner/releases/download/v2.298.2/actions-runner-linux-x64-2.298.2.tar.gz
actions_runner_download_regexp='https://github.com/actions/runner/releases/'\
'download/v[0-9.]\+/actions-runner-linux-x64-[0-9.]\+\.tar\.gz'
# Use `head -n 1` because there are multiple instances of the same URL
actions_runner_download=$(
  curl -s -X GET 'https://api.github.com/repos/actions/runner/releases/latest' |
    grep -o "${actions_runner_download_regexp}" |
    head -n 1)
echo "actions_runner_download: ${actions_runner_download}"

# Run the rest of the setup as `runner`.
#
# Note that env vars in the heredoc will be expanded according to the _calling_
# environment, not the `runner` environment we're creating -- it acts like a
# double-quoted string. This is how variables like $runner_name are inserted
# without using sudo -E (which would cause the current environment to be
# inherited). This also means we must be careful to escape any variables that
# we'd like to evaluate in the `runner` environment, e.g. $HOME.
sudo -i -u runner bash -ex <<EOF

cd ~/

git clone ${jax_repo_url}

# Based on https://github.com/google/jax/settings/actions/runners/new
# (will be 404 for github users with insufficient repo permissions)
mkdir actions-runner && cd actions-runner
curl -o actions-runner-linux-x64.tar.gz -L ${actions_runner_download}
tar xzf ./actions-runner-linux-x64.tar.gz

# Register the runner with Github
./config.sh --unattended \
--url ${jax_repo_url} \
--labels ${runner_tags} \
--token ${runner_token} \
--name ${runner_name}

# Setup pre-job hook
cat ~/jax/.github/workflows/self_hosted_runner_utils/runner.env | envsubst >> ~/actions-runner/.env

# Setup Github Actions Runner to automatically start on reboot (e.g. due to TPU
# VM maintenance events)
echo "@reboot \${HOME}/jax/.github/workflows/self_hosted_runner_utils/start_github_runner.sh" | crontab -

EOF
