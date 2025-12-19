load("//third_party:repo.bzl", "tf_http_archive", "tf_mirror_urls")

# The XLA commit is determined by third_party/xla/revision.bzl.
load("//third_party/xla:workspace.bzl", jax_xla_workspace = "repo")

jax_xla_workspace()

load("@xla//:workspace4.bzl", "xla_workspace4")

xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")

xla_workspace3()

# Initialize Hermetic toolchains
# Details: https://github.com/google-ml-infra/rules_ml_toolchain
tf_http_archive(
    name = "rules_ml_toolchain",
    sha256 = "bdee5b88bfb1fc6e79824e8b01f9c77a613fd969c5cb8eb7ab8ccdafac3499a5",
    strip_prefix = "rules_ml_toolchain-c01ab2847468126c2a2db2aa6d9d91a99e91da96",
    urls = tf_mirror_urls(
        "https://github.com/yuriivcs/rules_ml_toolchain/archive/c01ab2847468126c2a2db2aa6d9d91a99e91da96.tar.gz",
    ),
)

load(
    "@rules_ml_toolchain//cc/deps:cc_toolchain_deps.bzl",
    "cc_toolchain_deps",
)

cc_toolchain_deps()

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64")

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64_with_sanitizers")

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64_cuda")

register_toolchains("@rules_ml_toolchain//cc:linux_x86_64_linux_x86_64_cuda_with_sanitizers")

register_toolchains("@rules_ml_toolchain//cc:linux_aarch64_linux_aarch64")

register_toolchains("@rules_ml_toolchain//cc:linux_aarch64_linux_aarch64_cuda")

# Initialize hermetic Python
load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")

python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")

python_init_repositories(
    default_python_version = "system",
    local_wheel_dist_folder = "../dist",
    local_wheel_inclusion_list = [
        "libtpu*",
        "ml_dtypes*",
        "ml-dtypes*",
        "numpy*",
        "scipy*",
        "jax-*",
        "jaxlib*",
        "jax_cuda*",
        "jax-cuda*",
    ],
    local_wheel_workspaces = ["//jaxlib:jax.bzl"],
    requirements = {
        "3.11": "//build:requirements_lock_3_11.txt",
        "3.12": "//build:requirements_lock_3_12.txt",
        "3.13": "//build:requirements_lock_3_13.txt",
        "3.14": "//build:requirements_lock_3_14.txt",
        "3.13-ft": "//build:requirements_lock_3_13_ft.txt",
        "3.14-ft": "//build:requirements_lock_3_14_ft.txt",
    },
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")

python_init_toolchains()

load("@xla//third_party/py:python_init_pip.bzl", "python_init_pip")

python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")

install_deps()

load("@xla//:workspace2.bzl", "xla_workspace2")

xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")

xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")

xla_workspace0()

load("//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")

flatbuffers()

load("//:test_shard_count.bzl", "test_shard_count_repository")

test_shard_count_repository(
    name = "test_shard_count",
)

load("//jaxlib:jax_python_wheel.bzl", "jax_python_wheel_repository")

jax_python_wheel_repository(
    name = "jax_wheel",
    version_key = "_version",
    version_source = "//jax:version.py",
)

load(
    "@xla//third_party/py:python_wheel.bzl",
    "nvidia_wheel_versions_repository",
    "python_wheel_version_suffix_repository",
)

nvidia_wheel_versions_repository(
    name = "nvidia_wheel_versions",
    versions_source = "//build:nvidia-requirements.txt",
)

python_wheel_version_suffix_repository(
    name = "jax_wheel_version_suffix",
)

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/cuda:cuda_redist_init_repositories.bzl",
    "cuda_redist_init_repositories",
    "cudnn_redist_init_repository",
)

cuda_redist_init_repositories(
    cuda_redistributions = CUDA_REDISTRIBUTIONS,
)

cudnn_redist_init_repository(
    cudnn_redistributions = CUDNN_REDISTRIBUTIONS,
)

load(
    "@rules_ml_toolchain//gpu/cuda:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@rules_ml_toolchain//gpu/nccl:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")

load(
    "@rules_ml_toolchain//gpu/nvshmem:nvshmem_json_init_repository.bzl",
    "nvshmem_json_init_repository",
)

nvshmem_json_init_repository()

load(
    "@nvshmem_redist_json//:distributions.bzl",
    "NVSHMEM_REDISTRIBUTIONS",
)
load(
    "@rules_ml_toolchain//gpu/nvshmem:nvshmem_redist_init_repository.bzl",
    "nvshmem_redist_init_repository",
)

nvshmem_redist_init_repository(
    nvshmem_redistributions = NVSHMEM_REDISTRIBUTIONS,
)
