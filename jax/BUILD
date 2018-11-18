# JAX is Autograd and XLA
package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2

exports_files(["LICENSE"])

load(":build_defs.bzl", "jax_test")

package_group(name = "jax")

py_library(
    name = "libjax",
    srcs = glob(
        [
            "*.py",
            "lib/*.py",
            "interpreters/*.py",
            "numpy/*.py",
            "scipy/*.py",
        ],
        exclude = [
            "*_test.py",
            "**/*_test.py",
        ],
    ),
    deps = [
        "@org_tensorflow//tensorflow/compiler/xla/python:xla_client",
    ],
)

jax_test(
    name = "core_test",
    srcs = ["tests/core_test.py"],
    shard_count = {
        "cpu": 5,
    },
)

jax_test(
    name = "lax_test",
    srcs = ["tests/lax_test.py"],
    shard_count = {
        "cpu": 40,
        "gpu": 20,
    },
)

jax_test(
    name = "lax_numpy_test",
    srcs = ["tests/lax_numpy_test.py"],
    shard_count = {
        "cpu": 40,
        "gpu": 20,
    },
)

jax_test(
    name = "lax_numpy_indexing_test",
    srcs = ["tests/lax_numpy_indexing_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 2,
    },
)

jax_test(
    name = "lax_scipy_test",
    srcs = ["tests/lax_scipy_test.py"],
    shard_count = {
        "cpu": 10,
        "gpu": 2,
    },
)

jax_test(
    name = "random_test",
    srcs = ["tests/random_test.py"],
)

jax_test(
    name = "api_test",
    srcs = ["tests/api_test.py"],
)

jax_test(
    name = "batching_test",
    srcs = ["tests/batching_test.py"],
)

py_binary(
    name = "interactive",
    srcs = ["examples/interactive.py"],
    deps = [":libjax"],
)

py_library(
    name = "stax",
    srcs = ["experimental/stax.py"],
    deps = [":libjax"],
)

jax_test(
    name = "stax_test",
    srcs = ["tests/stax_test.py"],
    deps = [":stax"],
)

py_library(
    name = "minmax",
    srcs = ["experimental/minmax.py"],
    deps = [":libjax"],
)

jax_test(
    name = "minmax_test",
    srcs = ["tests/minmax_test.py"],
    deps = [":minmax"],
)

py_library(
    name = "lapax",
    srcs = ["experimental/lapax.py"],
    deps = [":libjax"],
)

jax_test(
    name = "lapax_test",
    srcs = ["tests/lapax_test.py"],
    deps = [":lapax"],
)

py_library(
    name = "datasets",
    srcs = ["examples/datasets.py"],
)

py_binary(
    name = "mnist_classifier",
    srcs = ["examples/mnist_classifier.py"],
    deps = [
        ":datasets",
        ":libjax",
    ],
)

py_binary(
    name = "mnist_vae",
    srcs = ["examples/mnist_vae.py"],
    deps = [
        ":datasets",
        ":libjax",
        ":minmax",
        ":stax",
    ],
)
