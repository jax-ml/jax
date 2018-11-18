# JAX is Autograd and XLA
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
