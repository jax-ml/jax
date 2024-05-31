load("//jaxlib:jax.bzl", "cuda_library")

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
