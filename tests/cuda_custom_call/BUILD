load("@rules_python//python:defs.bzl", "py_test")
load(
    "//jaxlib:jax.bzl",
    "cuda_library",
    "jax_test",
)

jax_test(
    name = "cuda_custom_call_test",
    srcs = ["cuda_custom_call_test.py"],
    disable_backends = [
        "cpu",
        "tpu",
    ],
    # libfoo.so is a runtime dependency for this test
    data = [":foo"],
    deps = [
        "//jax:extend",
    ]
)

# this second target is needed to properly link in CUDA runtime symbols
# such as cudaLaunchKernel, even though we are only building one library.
cc_shared_library(
    name = "foo",
    deps = [
        ":foo_",
        "@xla//xla/tsl/cuda:cudart",
    ],
)

cuda_library(
    name = "foo_",
    srcs = ["foo.cu.cc"],
    deps = [
        "@xla//xla/ffi/api:ffi",
        "@xla//xla/ffi/api:api",
        "@xla//xla/ffi/api:c_api",
        "@local_config_cuda//cuda:cuda_headers",
    ],
)
