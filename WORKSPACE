# The XLA commit is determined by third_party/xla/workspace.bzl.
load("//third_party/xla:workspace.bzl", jax_xla_workspace = "repo")
jax_xla_workspace()

# Initialize hermetic Python
load("@xla//third_party/py:python_init_rules.bzl", "python_init_rules")
python_init_rules()

load("@xla//third_party/py:python_init_repositories.bzl", "python_init_repositories")
python_init_repositories(
    requirements = {
        "3.10": "//build:requirements_lock_3_10.txt",
        "3.11": "//build:requirements_lock_3_11.txt",
        "3.12": "//build:requirements_lock_3_12.txt",
        "3.13": "//build:requirements_lock_3_13.txt",
        "3.13-ft": "//build:requirements_lock_3_13_ft.txt",
    },
    local_wheel_inclusion_list = [
        "jaxlib*",
        "jax_cuda*",
        "jax-cuda*",
    ],
    local_wheel_workspaces = ["//jaxlib:jax.bzl"],
    local_wheel_dist_folder = "../dist",
    default_python_version = "system",
)

load("@xla//third_party/py:python_init_toolchains.bzl", "python_init_toolchains")
python_init_toolchains()

load("@xla//third_party/py:python_init_pip.bzl", "python_init_pip")
python_init_pip()

load("@pypi//:requirements.bzl", "install_deps")
install_deps()

# Optional, to facilitate testing against newest versions of Python
load("@xla//third_party/py:python_repo.bzl", "custom_python_interpreter")
custom_python_interpreter(
    name = "python_dev",
    urls = ["https://www.python.org/ftp/python/{version}/Python-{version_variant}.tgz"],
    strip_prefix = "Python-{version_variant}",
    version = "3.13.0",
    version_variant = "3.13.0rc2",
)

load("@xla//:workspace4.bzl", "xla_workspace4")
xla_workspace4()

load("@xla//:workspace3.bzl", "xla_workspace3")
xla_workspace3()

load("@xla//:workspace2.bzl", "xla_workspace2")
xla_workspace2()

load("@xla//:workspace1.bzl", "xla_workspace1")
xla_workspace1()

load("@xla//:workspace0.bzl", "xla_workspace0")
xla_workspace0()

load("//third_party/flatbuffers:workspace.bzl", flatbuffers = "repo")
flatbuffers()

load("//jaxlib:jax_python_wheel.bzl", "jax_python_wheel_repository")
jax_python_wheel_repository(
    name = "jax_wheel",
    version_key = "_version",
    version_source = "//jax:version.py",
)

load(
    "@tsl//third_party/py:python_wheel.bzl",
    "python_wheel_version_suffix_repository",
)
python_wheel_version_suffix_repository(
    name = "jax_wheel_version_suffix",
)

load(
    "@tsl//third_party/gpus/cuda/hermetic:cuda_json_init_repository.bzl",
    "cuda_json_init_repository",
)

cuda_json_init_repository()

load(
    "@cuda_redist_json//:distributions.bzl",
    "CUDA_REDISTRIBUTIONS",
    "CUDNN_REDISTRIBUTIONS",
)
load(
    "@tsl//third_party/gpus/cuda/hermetic:cuda_redist_init_repositories.bzl",
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
    "@tsl//third_party/gpus/cuda/hermetic:cuda_configure.bzl",
    "cuda_configure",
)

cuda_configure(name = "local_config_cuda")

load(
    "@tsl//third_party/nccl/hermetic:nccl_redist_init_repository.bzl",
    "nccl_redist_init_repository",
)

nccl_redist_init_repository()

load(
    "@tsl//third_party/nccl/hermetic:nccl_configure.bzl",
    "nccl_configure",
)

nccl_configure(name = "local_config_nccl")
