load("@rules_python//python:defs.bzl", "py_test")
load(
    "//jaxlib:jax.bzl",
    "cuda_library",
)

py_test(
    name = "cuda_custom_call_test",
    srcs = ["cuda_custom_call_test.py"],
    data = [":foo"],
    deps = [
        "//jax",
        "//jax:extend",
    ]
)

cuda_library(
    name = "foo",
    srcs = ["foo.cu.cc"],
    deps = [
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/ffi/api:api",
        "@xla//xla/ffi/api:c_api",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)
