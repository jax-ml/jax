# JAX is Autograd and XLA
package(default_visibility = ["//visibility:public"])

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

py_library(
    name = "stax",
    srcs = ["experimental/stax.py"],
    deps = [":libjax"],
)

py_library(
    name = "minmax",
    srcs = ["experimental/minmax.py"],
    deps = [":libjax"],
)

py_library(
    name = "lapax",
    srcs = ["experimental/lapax.py"],
    deps = [":libjax"],
)

# this is a dummy target for building purposes
py_binary(
    name = "build_jax",
    srcs = ["core.py"],
    deps = [":libjax"],
)
