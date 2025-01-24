#!/usr/bin/env python3
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

import os
import csv
import json
import argparse
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor

GPU_LOCK = threading.Lock()
LAST_CODE = 0
base_dir = "./logs"


def extract_filename(path):
    base_name = os.path.basename(path)
    file_name, _ = os.path.splitext(base_name)
    return file_name


def combine_json_reports():
    all_json_files = [f for f in os.listdir(base_dir) if f.endswith("_log.json")]
    combined_data = []
    for json_file in all_json_files:
        with open(os.path.join(base_dir, json_file), "r") as infile:
            data = json.load(infile)
            combined_data.append(data)
    combined_json_file = f"{base_dir}/final_compiled_report.json"
    with open(combined_json_file, "w") as outfile:
        json.dump(combined_data, outfile, indent=4)


def combine_csv_reports():
    all_csv_files = [f for f in os.listdir(base_dir) if f.endswith("_log.csv")]
    combined_csv_file = f"{base_dir}/final_compiled_report.csv"
    with open(combined_csv_file, mode="w", newline="") as outfile:
        csv_writer = csv.writer(outfile)
        for i, csv_file in enumerate(all_csv_files):
            with open(os.path.join(base_dir, csv_file), mode="r") as infile:
                csv_reader = csv.reader(infile)
                if i == 0:
                    # write headers only once
                    csv_writer.writerow(next(csv_reader))
                for row in csv_reader:
                    csv_writer.writerow(row)


def generate_final_report(shell=False, env_vars={}):
    env = os.environ
    env = {**env, **env_vars}
    cmd = [
        "pytest_html_merger",
        "-i",
        f"{base_dir}",
        "-o",
        f"{base_dir}/final_compiled_report.html",
    ]
    result = subprocess.run(cmd, shell=shell, capture_output=True, env=env)
    if result.returncode != 0:
        print("FAILED - {}".format(" ".join(cmd)))
        print(result.stderr.decode())

    # Generate json reports.
    combine_json_reports()
    # Generate csv reports.
    combine_csv_reports()


def run_shell_command(cmd, shell=False, env_vars={}):
    env = os.environ
    env = {**env, **env_vars}
    result = subprocess.run(cmd, shell=shell, capture_output=True, env=env)
    if result.returncode != 0:
        print("FAILED - {}".format(" ".join(cmd)))
        print(result.stderr.decode())

    return result.returncode, result.stderr.decode(), result.stdout.decode()


def parse_test_log(log_file):
    """Parses the test module log file to extract test modules and functions."""
    test_files = set()
    with open(log_file, "r") as f:
        for line in f:
            report = json.loads(line)
            if "nodeid" in report:
                module = report["nodeid"].split("::")[0]
                if module and ".py" in module:
                    test_files.add(os.path.abspath(module))
    return test_files


def collect_testmodules():
    log_file = f"{base_dir}/collect_module_log.jsonl"
    return_code, stderr, stdout = run_shell_command(
        [
            "python3",
            "-m",
            "pytest",
            "--collect-only",
            "tests",
            f"--report-log={log_file}",
        ]
    )
    if return_code != 0:
        print("Test module discovery failed.")
        print("STDOUT:", stdout)
        print("STDERR:", stderr)
        exit(return_code)
    print("---------- collected test modules ----------")
    test_files = parse_test_log(log_file)
    print("Found %d test modules." % (len(test_files)))
    print("--------------------------------------------")
    print("\n".join(test_files))
    return test_files


def run_test(testmodule, gpu_tokens, continue_on_fail):
    global LAST_CODE
    with GPU_LOCK:
        if LAST_CODE != 0:
            return
        target_gpu = gpu_tokens.pop()
    env_vars = {
        "HIP_VISIBLE_DEVICES": str(target_gpu),
        "XLA_PYTHON_CLIENT_ALLOCATOR": "default",
    }
    testfile = extract_filename(testmodule)
    if continue_on_fail:
        cmd = [
            "python3",
            "-m",
            "pytest",
            "--json-report",
            f"--json-report-file={base_dir}/{testfile}_log.json",
            f"--csv={base_dir}/{testfile}_log.csv",
            "--csv-columns",
            "id,module,name,file,status,duration",
            f"--html={base_dir}/{testfile}_log.html",
            "--reruns",
            "3",
            "-v",
            testmodule,
        ]
    else:
        cmd = [
            "python3",
            "-m",
            "pytest",
            "--json-report",
            f"--json-report-file={base_dir}/{testfile}_log.json",
            f"--csv={base_dir}/{testfile}_log.csv",
            "--csv-columns",
            "id,module,name,file,status,duration",
            f"--html={base_dir}/{testfile}_log.html",
            "--reruns",
            "3",
            "-x",
            "-v",
            testmodule,
        ]

    return_code, stderr, stdout = run_shell_command(cmd, env_vars=env_vars)
    with GPU_LOCK:
        gpu_tokens.append(target_gpu)
        if LAST_CODE == 0:
            print("Running tests in module %s on GPU %d:" % (testmodule, target_gpu))
            print(stdout)
            print(stderr)
            if continue_on_fail == False:
                LAST_CODE = return_code


def run_parallel(all_testmodules, p, c):
    print(f"Running tests with parallelism = {p}")
    available_gpu_tokens = list(range(p))
    executor = ThreadPoolExecutor(max_workers=p)
    # walking through test modules.
    for testmodule in all_testmodules:
        executor.submit(run_test, testmodule, available_gpu_tokens, c)
    # waiting for all modules to finish.
    executor.shutdown(wait=True)


def find_num_gpus():
    cmd = [r"lspci|grep 'controller\|accel'|grep 'AMD/ATI'|wc -l"]
    _, _, stdout = run_shell_command(cmd, shell=True)
    return int(stdout)


def main(args):
    all_testmodules = collect_testmodules()
    run_parallel(all_testmodules, args.parallel, args.continue_on_fail)
    generate_final_report()
    exit(LAST_CODE)


if __name__ == "__main__":
    os.environ["HSA_TOOLS_LIB"] = "libroctracer64.so"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p", "--parallel", type=int, help="number of tests to run in parallel"
    )
    parser.add_argument(
        "-c", "--continue_on_fail", action="store_true", help="continue on failure"
    )
    args = parser.parse_args()
    if args.continue_on_fail:
        print("continue on fail is set")
    if args.parallel is None:
        sys_gpu_count = find_num_gpus()
        args.parallel = sys_gpu_count
        print("%d GPUs detected." % sys_gpu_count)

    main(args)
