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

"""A tool to create and destroy clusters on demand.

To create a cluster:
    export WEB_API_URL=...   # URL of the API endpoint
    export WEB_API_TOKEN=... # Authentication token
    python3 oci_cluster_manager.py create_cluster --pubkey "$(cat ~.ssh/id_rsa.pub)"
This will first try to find an existing running cluster, and otherwise attempt to create one in any region.
When succesfull, the output will contains the headnode hostkeys, the username and ip address of the cluster, or FAILED.

To create all previously created clusters
    export WEB_API_URL=...   # URL of the API endpoint
    export WEB_API_TOKEN=... # Authentication token
    python3 oci_cluster_manager.py destroy_clusters

This function is used to create and destroy clusters on demand.
A few caveats should be noted:
- Depending on resource availability, it might not be possible to create a cluster.
  In that case, the script will eventually fail.
- Creating a cluster takes time (30 to 60 mins).
  Similarly, destroying a cluster also takes time.
  User should not attempt to concurrently create clusters.
  As a rule of thumb, this script should only be used at most once every ~12 hours.
- In case a pull-request indirectly calls this script, users should take care to ensure
  no other pipeline is attempting to create a cluster at the same time and within a ~3h time window.
- Clusters are automatically destroyed 2h after being created, regardless of whether
  `destroy_all` is called or not.

"""

import os
import argparse
import time
import logging
import sys
import requests

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

_API_URL = os.environ['WEB_API_URL']
_API_AUTH = requests.auth.HTTPBasicAuth('login', os.environ['WEB_API_TOKEN'])

_SLEEP_TIME_SECONDS = int(os.environ.get("SLEEP_TIME_SECONDS", default="30"))
_RETRY_PER_REGION = int(os.environ.get("RETRY_PER_REGION", default="3"))
_REQUEST_TIMEOUT_SECONDS = int(os.environ.get("REQUEST_TIMEOUT_SECONDS", default=120))

def get_regions():
    return requests.post(_API_URL, json={'name':'list_regions'},
                         auth=_API_AUTH, timeout=_REQUEST_TIMEOUT_SECONDS).json()['regions']

def find_existing_cluster():
    return requests.post(_API_URL, json={'name':'find_cluster'},
                         auth=_API_AUTH, timeout=_REQUEST_TIMEOUT_SECONDS).json()['region']

def get_cluster_ip(region):
    return requests.post(_API_URL, json={'name':'get_cluster_ip', 'region':region},
                         auth=_API_AUTH, timeout=_REQUEST_TIMEOUT_SECONDS).json()['cluster_ip']

def get_cluster_username(region):
    return requests.post(_API_URL, json={'name':'get_cluster_username', 'region':region},
                         auth=_API_AUTH, timeout=_REQUEST_TIMEOUT_SECONDS).json()['cluster_username']

def create_cluster(region):
    logging.debug(requests.post(_API_URL, json={'name':'create_cluster', 'region':region},
                                auth=_API_AUTH, timeout=_REQUEST_TIMEOUT_SECONDS))

def add_pubkey(region, pubkey):
    logging.debug(requests.post(_API_URL, json={'name':'add_pubkey', 'region':region, 'pubkey':pubkey},
                                auth=_API_AUTH, timeout=_REQUEST_TIMEOUT_SECONDS))

def get_cluster_hostkeys(region):
    return requests.post(_API_URL, json={'name':'get_cluster_hostkeys', 'region':region},
                         auth=_API_AUTH, timeout=_REQUEST_TIMEOUT_SECONDS).json()['cluster_hostkeys']

def get_status(region):
    return requests.post(_API_URL, json={'name':'get_status', 'region': region},
                         auth=_API_AUTH, timeout=_REQUEST_TIMEOUT_SECONDS).json()['status']

def destroy_all_clusters():
    logging.debug(requests.post(_API_URL, json={'name':'destroy_all_clusters'},
                                auth=_API_AUTH, timeout=_REQUEST_TIMEOUT_SECONDS))

def main():

    command_choices = ["create_cluster", "destroy_clusters"]
    parser = argparse.ArgumentParser(description="Creates and destroy compute clusters on-demand.")
    parser.add_argument("command", choices=command_choices, \
        help="""create_cluster will first try to find an existing running cluster,
                and otherwise attempt to create one in any region.
                When succesfull, the output will contains the headnode hostkeys, the username and ip address of the cluster, or FAILED.
                destroy_clusters will destroy all existing clusters.""")
    parser.add_argument("--pubkey", default=None, \
                        help='public key to upload to the cluster (relevant only with `create_cluster`).')

    args = parser.parse_args()

    if args.command == "create_cluster":
        found_region = find_existing_cluster()
        logging.info(f"Found cluster {found_region}")

        if found_region is None:
            logging.info("Could not find existing cluster. Trying to create one.")
            regions = get_regions()
            logging.info(f"Regions considered: {regions}")

            for region in regions * _RETRY_PER_REGION:
                logging.info(f"Trying region {region}")
                create_cluster(region)
                status = get_status(region)

                while status == 'WAIT':
                    logging.info(f"Waiting {_SLEEP_TIME_SECONDS} seconds...")
                    time.sleep(_SLEEP_TIME_SECONDS)
                    status = get_status(region)

                if status == 'SUCCEEDED':
                    logging.info("Successfully allocated cluster")
                    found_region = region
                    break

                else:
                    logging.info("Moving to next region")
                    continue

        else:
            logging.info(f"Found existing cluster in {found_region}")

        if found_region is not None:
            logging.info(f"Found cluster in {found_region}")
            logging.info(f"Adding pubkey {args.pubkey} to cluster")
            add_pubkey(found_region, args.pubkey)
            logging.info("Fetching host keys, username and IP address")
            ip = get_cluster_ip(found_region)
            username = get_cluster_username(found_region)
            hostkeys = get_cluster_hostkeys(found_region)
            print(hostkeys)
            print(username)
            print(ip)

        else:
            logging.info("Failed to allocate cluster")
            sys.exit(1)

    elif args.command == "destroy_clusters":
        logging.info("Destroying all")
        destroy_all_clusters()

    else:
        raise ValueError(f"Wrong `command` argument. Got {args.command}. Valid choices are {command_choices}")

if __name__ == "__main__":
    main()
