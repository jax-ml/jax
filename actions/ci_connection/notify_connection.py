# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import threading
import subprocess
from multiprocessing.connection import Client

lock = threading.Lock()

# Configuration (same as wait_for_connection.py)
address = ("localhost", 12455)
keep_alive_interval = 30  # 30 seconds


def timer(conn):
  while True:
    # We lock because closed and keep_alive could technically arrive at the same time
    with lock:
      conn.send("keep_alive")
    time.sleep(keep_alive_interval)


if __name__ == "__main__":
  with Client(address) as conn:
    conn.send("connected")

    # Thread is running as a daemon so it will quit when the
    # main thread terminates.
    timer_thread = threading.Thread(target=timer, daemon=True, args=(conn,))
    timer_thread.start()

    print("Entering interactive bash session")
    # Enter interactive bash session
    subprocess.run(["/bin/bash", "-i"])

    print("Exiting interactive bash session")
    with lock:
      conn.send("closed")
    conn.close()
