#!/bin/bash

# Set stack size to 64MB (KB value)
ulimit -s 65536

# Run the actual command passed by Bazel
exec "$@"

