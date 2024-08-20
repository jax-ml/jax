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

import os
from multiprocessing.connection import Listener
import time
import signal
import threading
import sys

last_time = time.time()
timeout = 600  # 10 minutes for initial connection
keep_alive_timeout = (
  900  # 30 minutes for keep-alive if no closed message (allow for reconnects)
)


def wait_for_notification(address):
  """Waits for connection notification from the listener."""
  global last_time
  while True:
    with Listener(address) as listener:
      print("Waiting for connection")
      with listener.accept() as conn:
        while True:
          try:
            message = conn.recv()
          except EOFError as e:
            print("EOFError occurred:", e)
            break
          print("Received message")
          if message == "keep_alive":
            print("Keep alive received")
            last_time = time.time()
            continue  # Keep-alive received, continue waiting
          elif message == "closed":
            print("Connection closed by the other process.")
            return  # Graceful exit
          elif message == "connected":
            last_time = time.time()
            timeout = keep_alive_timeout
            print("Connected")
          else:
            print("Unknown message received:", message)
            continue


def timer():
  while True:
    print("Checking status")
    time_elapsed = time.time() - last_time
    if time_elapsed < timeout:
      print(f"Time since last keepalive {int(time_elapsed)}s")
    else:
      print("Timeout reached, exiting")
      os.kill(os.getpid(), signal.SIGTERM)
    time.sleep(60)


if __name__ == "__main__":
  address = ("localhost", 12455)  # Address and port to listen on
  # Check if we should wait for the connection
  wait_for_connection = False
  # if os.environ.get("WAIT_ON_ERROR") == "1":
  #   print("WAIT_ON_ERROR is set")
  #   if os.getppid() != 1:
  #     print("Previous command did not exit with success, waiting for connection")
  #     wait_for_connection = True
  #   else:
  #     print("Previous command exited with success")
  # else:
  #   print("WAIT_ON_ERROR is not set")

  if os.environ.get("INTERACTIVE_CI") == "1":
    print("INTERACTIVE_CI is set, waiting for connection")
    wait_for_connection = True
  else:
    print("INTERACTIVE_CI is not set")

  if not wait_for_connection:
    print("No condition was met to wait for connection. Continuing Job")
    exit(0)

  # Grab and print the data required to connect to this vm
  host = os.environ.get("HOSTNAME")
  repo = os.environ.get("REPOSITORY")
  cluster = os.environ.get("CONNECTION_CLUSTER")
  location = os.environ.get("CONNECTION_LOCATION")
  ns = os.environ.get("CONNECTION_NS")

  print("Googler connection only\nSee go/<insert final golink> for details")
  print(
    f"Connection string: ml_actions_connect  '{host}' '{ns}' '{location}' '{cluster}' '{repo}'"
  )

  # Thread is running as a daemon so it will quit when the
  # main thread terminates.
  timer_thread = threading.Thread(target=timer, daemon=True)
  timer_thread.start()

  wait_for_notification(address)  # Wait for connection and get the connection object

  print("Exiting connection wait loop.")
  # Force a flush so we don't miss messages
  sys.stdout.flush()
