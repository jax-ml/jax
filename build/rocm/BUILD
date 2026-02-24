load("@rules_shell//shell:sh_binary.bzl", "sh_binary")

package(
    default_visibility = ["//visibility:public"],
)

sh_binary(
    name = "parallel_gpu_pytest",
    srcs = ["parallel_gpu_pytest.sh"],
    data = [
        "@local_config_rocm//rocm:rocminfo",
        "@pypi//pytest",
    ],
)
