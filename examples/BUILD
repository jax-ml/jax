py_binary(
    name = "interactive",
    srcs = ["interactive.py"],
    deps = ["//jax:libjax"],
)

py_library(
    name = "datasets",
    srcs = ["datasets.py"],
)

py_binary(
    name = "mnist_classifier",
    srcs = ["mnist_classifier.py"],
    deps = [
        ":datasets",
        "//jax:libjax",
    ],
)

py_binary(
    name = "mnist_vae",
    srcs = ["mnist_vae.py"],
    deps = [
        ":datasets",
        ":minmax",
        ":stax",
        "//jax:libjax",
    ],
)
