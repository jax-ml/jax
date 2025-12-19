# Copyright 2021 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bazel macros used by the JAX build."""

load("@bazel_skylib//rules:common_settings.bzl", "BuildSettingInfo")
load("@com_github_google_flatbuffers//:build_defs.bzl", _flatbuffer_cc_library = "flatbuffer_cc_library")
load("@jax_wheel//:wheel.bzl", "WHEEL_VERSION")
load("@jax_wheel_version_suffix//:wheel_version_suffix.bzl", "WHEEL_VERSION_SUFFIX")
load("@local_config_cuda//cuda:build_defs.bzl", _cuda_library = "cuda_library", _if_cuda_is_configured = "if_cuda_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", _if_rocm_is_configured = "if_rocm_is_configured", _rocm_library = "rocm_library")
load("@nvidia_wheel_versions//:versions.bzl", "NVIDIA_WHEEL_VERSIONS")
load("@python_version_repo//:py_version.bzl", "HERMETIC_PYTHON_VERSION", "HERMETIC_PYTHON_VERSION_KIND")
load("@rules_cc//cc:defs.bzl", _cc_proto_library = "cc_proto_library")
load("@rules_python//python:defs.bzl", "py_library", "py_test")
load("@test_shard_count//:test_shard_count.bzl", "USE_MINIMAL_SHARD_COUNT")
load("@xla//third_party/py:python_wheel.bzl", "collect_data_files", "transitive_py_deps")
load("@xla//xla/tsl:tsl.bzl", "transitive_hdrs", _if_windows = "if_windows", _pybind_extension = "tsl_pybind_extension_opensource")
load("@xla//xla/tsl/platform:build_config_root.bzl", _tf_cuda_tests_tags = "tf_cuda_tests_tags", _tf_exec_properties = "tf_exec_properties")

# Explicitly re-exports names to avoid "unused variable" warnings from .bzl
# lint tools.
cc_proto_library = _cc_proto_library
cuda_library = _cuda_library
rocm_library = _rocm_library
proto_library = native.proto_library
nanobind_extension = _pybind_extension
if_cuda_is_configured = _if_cuda_is_configured
if_rocm_is_configured = _if_rocm_is_configured
if_windows = _if_windows
flatbuffer_cc_library = _flatbuffer_cc_library
tf_exec_properties = _tf_exec_properties
tf_cuda_tests_tags = _tf_cuda_tests_tags

jax_internal_packages = []
jax_extend_internal_users = []
experimental_transfer_users = []
mosaic_gpu_internal_users = []
mosaic_internal_users = []
pallas_gpu_internal_users = []
pallas_sc_internal_users = []
pallas_fuser_users = []
serialize_executable_internal_users = []
buffer_callback_internal_users = []

jax_internal_export_back_compat_test_util_visibility = []
jax_internal_test_harnesses_visibility = []
jax_test_util_visibility = []
loops_visibility = []

PLATFORM_TAGS_DICT = {
    ("Linux", "x86_64"): ("manylinux_2_27", "x86_64"),
    ("Linux", "aarch64"): ("manylinux_2_27", "aarch64"),
    ("Linux", "ppc64le"): ("manylinux2014", "ppc64le"),
    ("Darwin", "x86_64"): ("macosx_11_0", "x86_64"),
    ("Darwin", "arm64"): ("macosx_11_0", "arm64"),
    ("Windows", "AMD64"): ("win", "amd64"),
}

def get_optional_dep(package, excluded_py_versions = ["3.14", "3.14-ft"]):
    py_ver = HERMETIC_PYTHON_VERSION
    if HERMETIC_PYTHON_VERSION_KIND == "ft":
        py_ver += "-ft"
    if py_ver in excluded_py_versions:
        return []
    return [package]

_py_deps = {
    "absl-all": ["@pypi//absl_py"],
    "absl/logging": ["@pypi//absl_py"],
    "absl/testing": ["@pypi//absl_py"],
    "absl/testing:flagsaver": ["@pypi//absl_py"],
    "absl/flags": ["@pypi//absl_py"],
    "cloudpickle": get_optional_dep("@pypi//cloudpickle"),
    "disable_pmap_shmap_merge": [],
    "epath": get_optional_dep("@pypi//etils"),  # etils.epath
    "filelock": get_optional_dep("@pypi//filelock"),
    "flatbuffers": ["@pypi//flatbuffers"],
    "hypothesis": ["@pypi//hypothesis"],
    "magma": [],
    "matplotlib": get_optional_dep("@pypi//matplotlib"),
    "mpmath": [],
    "opt_einsum": ["@pypi//opt_einsum"],
    "pil": get_optional_dep("@pypi//pillow"),
    "portpicker": ["@pypi//portpicker"],
    "ml_dtypes": ["@pypi//ml_dtypes"],
    "numpy": ["@pypi//numpy"],
    "scipy": ["@pypi//scipy"],
    "tensorflow_core": [],
    "tensorstore": get_optional_dep("@pypi//tensorstore"),
    "torch": [],
    "tensorflow": get_optional_dep("@pypi//tensorflow", ["3.13-ft", "3.14", "3.14-ft"]),
    "tpu_ops": [],
    # TODO(vam): remove this once zstandard builds against Python >3.13
    "zstandard": get_optional_dep("@pypi//zstandard", ["3.13", "3.13-ft", "3.14", "3.14-ft"]),
}

def all_py_deps(excluded = []):
    py_deps_copy = dict(_py_deps)
    for excl in excluded:
        py_deps_copy.pop(excl)
    return py_deps(py_deps_copy.keys())

def py_deps(_package):
    """Returns the Bazel deps for Python package `package`."""

    if type(_package) == type([]) or type(_package) == type(()):
        deduped_py_deps = {}
        for _pkg in _package:
            for py_dep in _py_deps[_pkg]:
                deduped_py_deps[py_dep] = _pkg

        return deduped_py_deps.keys()

    return _py_deps[_package]

def jax_visibility(_target):
    """Returns the additional Bazel visibilities for `target`."""
    return [
        "//jax:__subpackages__",
        "//jaxlib:__subpackages__",
    ]

jax_extra_deps = []
jax_gpu_support_deps = []
jax2tf_deps = []

def pytype_library(name, pytype_srcs = None, **kwargs):
    _ = pytype_srcs  # @unused
    kwargs.pop("lazy_imports", None)
    py_library(name = name, **kwargs)

def pytype_strict_library(name, pytype_srcs = [], **kwargs):
    data = pytype_srcs + (kwargs["data"] if "data" in kwargs else [])
    new_kwargs = {k: v for k, v in kwargs.items() if k != "data"}
    new_kwargs.pop("lazy_imports", None)
    py_library(name = name, data = data, **new_kwargs)

py_strict_library = py_library
py_strict_test = py_test

def py_library_providing_imports_info(*, name, lib_rule = py_library, pytype_srcs = [], **kwargs):
    data = pytype_srcs + (kwargs["data"] if "data" in kwargs else [])
    new_kwargs = {k: v for k, v in kwargs.items() if k != "data"}
    new_kwargs.pop("lazy_imports", None)
    lib_rule(name = name, data = data, **new_kwargs)

def py_extension(name, srcs, copts, deps, linkopts = []):
    nanobind_extension(name, srcs = srcs, copts = copts, linkopts = linkopts, deps = deps, module_name = name)

ALL_BACKENDS = ["cpu", "gpu", "tpu"]
TEST_SUITE_SUFFIX = "_tests"
BACKEND_INDEPENDENT_TESTS = "backend_independent_tests"

def if_building_jaxlib(
        if_building,
        if_not_building = [
            "@pypi//jaxlib",
        ]):
    """Adds jaxlib wheels as dependencies instead of depending on sources.

    This allows us to test prebuilt versions of jaxlib wheels against the rest of the JAX codebase.

    Args:
      if_building: the source code targets to depend on in case we don't depend on the jaxlib wheels
      if_not_building: the wheels to depend on if we are not depending directly on //jaxlib.
    """
    return select({
        "//jax:config_build_jaxlib_true": if_building,
        "//jax:config_build_jaxlib_false": if_not_building,
        "//jax:config_build_jaxlib_wheel": [],
    })

def _cpu_test_deps():
    """Returns the test dependencies needed for a CPU-only JAX test."""
    return select({
        "//jax:config_build_jaxlib_true": [],
        "//jax:config_build_jaxlib_false": ["@pypi//jaxlib"],
        "//jax:config_build_jaxlib_wheel": ["//jaxlib/tools:jaxlib_py_import"],
    })

def _gpu_test_deps():
    """Returns the additional dependencies needed for a GPU test."""
    return select({
        "//jax:config_build_jaxlib_true": [
            "//jaxlib/cuda:gpu_only_test_deps",
            "//jaxlib/rocm:gpu_only_test_deps",
            "//jax_plugins:gpu_plugin_only_test_deps",
        ],
        "//jax:config_build_jaxlib_false": [
            "//jaxlib/tools:pypi_jax_cuda_plugin_with_cuda_deps",
            "//jaxlib/tools:pypi_jax_cuda_pjrt_with_cuda_deps",
        ],
        "//jax:config_build_jaxlib_wheel": [
            "//jaxlib/tools:jax_cuda_plugin_py_import",
            "//jaxlib/tools:jax_cuda_pjrt_py_import",
        ],
    })

def _get_jax_test_deps(deps):
    """Returns the jax build deps, pypi jax wheel dep, or jax py_import dep for the given backend.

    Args:
      deps: the full list of test dependencies

    Returns:
      A list of jax test deps.

      If --//jax:build_jax=true, returns jax build deps.
      If --//jax:build_jax=false, returns jax pypi wheel dep and transitive pypi test deps.
      If --//jax:build_jax=wheel, returns jax py_import dep and transitive pypi test deps.
    """
    non_pypi_deps = [d for d in deps if not d.startswith("@pypi//")]

    # A lot of tests don't have explicit dependencies on scipy, ml_dtypes, etc. But the tests
    # transitively depends on them via //jax. So we need to make sure that these dependencies are
    # included in the test when JAX is built from source.
    pypi_deps = depset([d for d in deps if d.startswith("@pypi//")])
    pypi_deps = depset(py_deps([
        "ml_dtypes",
        "scipy",
        "opt_einsum",
        "flatbuffers",
    ]), transitive = [pypi_deps]).to_list()

    return pypi_deps + select({
        "//jax:config_build_jax_false": ["//:jax_wheel_with_internal_test_util"],
        "//jax:config_build_jax_wheel": ["//:jax_py_import"],
        "//jax:config_build_jax_true": non_pypi_deps,
    })

# buildifier: disable=function-docstring
def jax_multiplatform_test(
        name,
        srcs,
        args = [],
        env = {},
        shard_count = None,
        minimal_shard_count = None,
        deps = [],
        data = [],
        enable_backends = None,
        backend_variant_args = {},
        backend_tags = {},  # buildifier: disable=unused-variable
        disable_configs = None,  # buildifier: disable=unused-variable
        enable_configs = [],
        config_tags_overrides = None,  # buildifier: disable=unused-variable
        tags = [],
        main = None,
        size = None,  # buildifier: disable=unused-variable
        pjrt_c_api_bypass = False):  # buildifier: disable=unused-variable
    # enable_configs and disable_configs do not do anything in OSS, only in Google's CI.
    # The order in which `enable_backends`, `enable_configs`, and `disable_configs` are applied is
    # as follows:
    # 1. `enable_backends` is applied first, enabling all test configs for the given backends.
    # 2. `disable_configs` is applied second, disabling the named test configs.
    # 3. `enable_configs` is applied last, enabling the named test configs.

    if main == None:
        if len(srcs) == 1:
            main = srcs[0]
        else:
            fail("Must set a main file to test multiple source files.")

    env = dict(env)
    env.setdefault("PYTHONWARNINGS", "error")

    for backend in ALL_BACKENDS:
        test_shard_count = minimal_shard_count if USE_MINIMAL_SHARD_COUNT else shard_count
        if test_shard_count == None or type(test_shard_count) == type(0):
            test_shards = test_shard_count
        else:
            test_shards = test_shard_count.get(backend, 1)
        test_args = list(args) + [
            "--jax_test_dut=" + backend,
            "--jax_platform_name=" + backend,
        ]
        test_args += backend_variant_args.get(backend, [])
        test_tags = list(tags) + ["jax_test_%s" % backend] + backend_tags.get(backend, [])
        if enable_backends != None and backend not in enable_backends and not any([config.startswith(backend) for config in enable_configs]):
            test_tags.append("manual")
        test_deps = _cpu_test_deps() + _get_jax_test_deps([
            "//jax",
            "//jax/_src:test_util",
        ] + deps)
        if backend == "gpu":
            test_deps += _gpu_test_deps()
            test_tags += tf_cuda_tests_tags()
        elif backend == "tpu":
            test_deps += ["@pypi//libtpu"]
        py_test(
            name = name + "_" + backend,
            srcs = srcs,
            args = test_args,
            env = env,
            deps = test_deps,
            data = data,
            shard_count = test_shards,
            tags = test_tags,
            main = main,
            exec_properties = tf_exec_properties({"tags": test_tags}),
            visibility = jax_visibility(name),
        )

def get_test_suite_list(paths, backends = []):
    test_suite_list = []
    if not backends:
        backends = ALL_BACKENDS
    for path in paths:
        for backend in backends:
            test_suite_list.append("//{}:{}{}".format(path, backend, TEST_SUITE_SUFFIX))
        test_suite_list.append("//{}:{}".format(path, BACKEND_INDEPENDENT_TESTS))
    return test_suite_list

def jax_generate_backend_suites(backends = []):
    """Generates test suite targets named cpu_tests, gpu_tests, etc.

    Args:
      backends: the set of backends for which rules should be generated. Defaults to all backends.
    """
    if not backends:
        backends = ALL_BACKENDS
    for backend in backends:
        native.test_suite(
            name = backend + TEST_SUITE_SUFFIX,
            tags = ["jax_test_%s" % backend, "-manual"],
            visibility = jax_visibility(backend + TEST_SUITE_SUFFIX),
        )
    native.test_suite(
        name = BACKEND_INDEPENDENT_TESTS,
        tags = ["-jax_test_%s" % backend for backend in backends] + ["-manual"],
        visibility = jax_visibility(BACKEND_INDEPENDENT_TESTS),
    )

def _get_full_wheel_name(
        package_name,
        no_abi,
        platform_independent,
        platform_name,
        cpu_name,
        wheel_version,
        py_freethreaded):
    if no_abi or platform_independent:
        wheel_name_template = "{package_name}-{wheel_version}-py{major_python_version}-none-{wheel_platform_tag}.whl"
    else:
        wheel_name_template = "{package_name}-{wheel_version}-cp{python_version}-cp{python_version}{free_threaded_suffix}-{wheel_platform_tag}.whl"
    python_version = HERMETIC_PYTHON_VERSION.replace(".", "")
    return wheel_name_template.format(
        package_name = package_name,
        python_version = python_version,
        major_python_version = python_version[0],
        wheel_version = wheel_version,
        wheel_platform_tag = "any" if platform_independent else "_".join(
            PLATFORM_TAGS_DICT[platform_name, cpu_name],
        ),
        free_threaded_suffix = "t" if py_freethreaded.lower() == "yes" else "",
    )

def _get_source_package_name(package_name, wheel_version):
    return "{package_name}-{wheel_version}.tar.gz".format(
        package_name = package_name,
        wheel_version = wheel_version,
    )

def _jax_wheel_impl(ctx):
    include_cuda_libs = ctx.attr.include_cuda_libs[BuildSettingInfo].value
    override_include_cuda_libs = ctx.attr.override_include_cuda_libs[BuildSettingInfo].value
    output_path = ctx.attr.output_path[BuildSettingInfo].value
    git_hash = ctx.attr.git_hash[BuildSettingInfo].value
    py_freethreaded = ctx.attr.py_freethreaded[BuildSettingInfo].value
    executable = ctx.executable.wheel_binary

    if include_cuda_libs and not override_include_cuda_libs:
        fail("JAX wheel shouldn't be built directly against the CUDA libraries." +
             " Please provide `--config=cuda_libraries_from_stubs` for bazel build command." +
             " If you absolutely need to build links directly against the CUDA libraries, provide" +
             " `--@local_config_cuda//cuda:override_include_cuda_libs=true`.")

    env = {}
    args = ctx.actions.args()

    full_wheel_version = (WHEEL_VERSION + WHEEL_VERSION_SUFFIX)
    env["WHEEL_VERSION_SUFFIX"] = WHEEL_VERSION_SUFFIX
    if not WHEEL_VERSION_SUFFIX:
        env["JAX_RELEASE"] = "1"

    cpu = ctx.attr.cpu
    no_abi = ctx.attr.no_abi
    platform_independent = ctx.attr.platform_independent
    build_wheel_only = ctx.attr.build_wheel_only
    build_source_package_only = ctx.attr.build_source_package_only
    editable = ctx.attr.editable
    platform_name = ctx.attr.platform_name

    output_dir_path = ""
    outputs = []
    if editable:
        output_dir = ctx.actions.declare_directory(output_path + "/" + ctx.attr.wheel_name)
        output_dir_path = output_dir.path
        outputs = [output_dir]
        args.add("--editable")
    else:
        if build_wheel_only:
            wheel_name = _get_full_wheel_name(
                package_name = ctx.attr.wheel_name,
                no_abi = no_abi,
                platform_independent = platform_independent,
                platform_name = platform_name,
                cpu_name = cpu,
                wheel_version = full_wheel_version,
                py_freethreaded = py_freethreaded,
            )
            wheel_file = ctx.actions.declare_file(output_path +
                                                  "/" + wheel_name)
            output_dir_path = wheel_file.path[:wheel_file.path.rfind("/")]
            outputs = [wheel_file]
            if ctx.attr.wheel_name == "jax":
                args.add("--build-wheel-only", "True")
        if build_source_package_only:
            source_package_name = _get_source_package_name(
                package_name = ctx.attr.wheel_name,
                wheel_version = full_wheel_version,
            )
            source_package_file = ctx.actions.declare_file(output_path +
                                                           "/" + source_package_name)
            output_dir_path = source_package_file.path[:source_package_file.path.rfind("/")]
            outputs = [source_package_file]
            if ctx.attr.wheel_name == "jax":
                args.add("--build-source-package-only", "True")

    args.add("--output_path", output_dir_path)  # required argument
    if not platform_independent:
        args.add("--cpu", cpu)
    args.add("--jaxlib_git_hash", git_hash)  # required argument

    if ctx.attr.enable_cuda:
        args.add("--enable-cuda", "True")
        if ctx.attr.platform_version == "":
            fail("platform_version must be set to a valid cuda version for cuda wheels")
        args.add("--platform_version", ctx.attr.platform_version)  # required for gpu wheels
        args.add("--nvidia_wheel_versions_data", NVIDIA_WHEEL_VERSIONS)  # required for gpu wheels
    if ctx.attr.enable_rocm:
        args.add("--enable-rocm", "True")
        if ctx.attr.platform_version == "":
            fail("platform_version must be set to a valid rocm version for rocm wheels")
        args.add("--platform_version", ctx.attr.platform_version)  # required for gpu wheels
    if ctx.attr.skip_gpu_kernels:
        args.add("--skip_gpu_kernels")

    srcs = []
    for src in ctx.attr.source_files:
        for f in src.files.to_list():
            srcs.append(f)
            args.add("--srcs=%s" % (f.path))

    args.set_param_file_format("flag_per_line")
    args.use_param_file("@%s", use_always = False)
    ctx.actions.run(
        arguments = [args],
        inputs = srcs,
        outputs = outputs,
        executable = executable,
        env = env,
        mnemonic = "BuildJaxWheel",
    )

    return [DefaultInfo(files = depset(direct = outputs))]

_jax_wheel = rule(
    attrs = {
        "wheel_binary": attr.label(
            default = Label("//jaxlib/tools:build_wheel_tool"),
            executable = True,
            cfg = "exec",
        ),
        "wheel_name": attr.string(mandatory = True),
        "no_abi": attr.bool(default = False),
        "platform_independent": attr.bool(default = False),
        "build_wheel_only": attr.bool(mandatory = True, default = True),
        "build_source_package_only": attr.bool(mandatory = True, default = False),
        "editable": attr.bool(default = False),
        "cpu": attr.string(),
        "platform_name": attr.string(),
        "git_hash": attr.label(default = Label("//jaxlib/tools:jaxlib_git_hash")),
        "source_files": attr.label_list(allow_files = True),
        "output_path": attr.label(default = Label("//jaxlib/tools:output_path")),
        "enable_cuda": attr.bool(default = False),
        # A cuda/rocm version is required for gpu wheels; for cpu wheels, it can be an empty string.
        "platform_version": attr.string(),
        "skip_gpu_kernels": attr.bool(default = False),
        "enable_rocm": attr.bool(default = False),
        "include_cuda_libs": attr.label(default = Label("@local_config_cuda//cuda:include_cuda_libs")),
        "override_include_cuda_libs": attr.label(default = Label("@local_config_cuda//cuda:override_include_cuda_libs")),
        "py_freethreaded": attr.label(default = Label("@rules_python//python/config_settings:py_freethreaded")),
    },
    implementation = _jax_wheel_impl,
    executable = False,
)

def jax_wheel(
        name,
        wheel_binary,
        wheel_name,
        no_abi = False,
        platform_independent = False,
        editable = False,
        enable_cuda = False,
        enable_rocm = False,
        platform_version = "",
        source_files = []):
    """Create jax artifact wheels.

    Common artifact attributes are grouped within a single macro.

    Args:
      name: the target name
      wheel_binary: the binary to use to build the wheel
      wheel_name: the name of the wheel
      no_abi: whether to build a wheel without ABI
      editable: whether to build an editable wheel
      platform_independent: whether to build a wheel without platform tag
      enable_cuda: whether to build a cuda wheel
      enable_rocm: whether to build a rocm wheel
      platform_version: the cuda version to use for the wheel
      source_files: the source files to include in the wheel

    Returns:
      A wheel file or a wheel directory.
    """
    _jax_wheel(
        name = name,
        wheel_binary = wheel_binary,
        wheel_name = wheel_name,
        no_abi = no_abi,
        platform_independent = platform_independent,
        build_wheel_only = True,
        build_source_package_only = False,
        editable = editable,
        enable_cuda = enable_cuda,
        enable_rocm = enable_rocm,
        platform_version = platform_version,
        # git_hash is empty by default. Use `--//jaxlib/tools:jaxlib_git_hash=$(git rev-parse HEAD)`
        # flag in bazel command to pass the git hash for nightly or release builds.
        platform_name = select({
            "@platforms//os:osx": "Darwin",
            "@platforms//os:macos": "Darwin",
            "@platforms//os:windows": "Windows",
            "@platforms//os:linux": "Linux",
        }),
        # TODO(kanglan) Add @platforms//cpu:ppc64le once JAX Bazel is upgraded > 6.5.0.
        cpu = select({
            "//jaxlib/tools:macos_arm64": "arm64",
            "//jaxlib/tools:macos_x86_64": "x86_64",
            "//jaxlib/tools:win_amd64": "AMD64",
            "//jaxlib/tools:linux_aarch64": "aarch64",
            "//jaxlib/tools:linux_x86_64": "x86_64",
        }),
        source_files = source_files,
    )

def jax_source_package(
        name,
        source_package_binary,
        source_package_name,
        source_files = []):
    """Create jax source package.

    Common artifact attributes are grouped within a single macro.

    Args:
      name: the target name
      source_package_binary: the binary to use to build the package
      source_package_name: the name of the source package
      source_files: the source files to include in the package

    Returns:
      A jax source package file.
    """
    _jax_wheel(
        name = name,
        wheel_binary = source_package_binary,
        wheel_name = source_package_name,
        build_source_package_only = True,
        build_wheel_only = False,
        platform_independent = True,
        source_files = source_files,
    )

jax_test_file_visibility = []

jax_export_file_visibility = []

def xla_py_proto_library(*args, **kw):  # buildifier: disable=unused-variable
    pass

def jax_py_test(
        name,
        env = {},
        **kwargs):
    env = dict(env)
    env.setdefault("PYTHONWARNINGS", "error")
    deps = kwargs.get("deps", [])
    test_deps = _cpu_test_deps() + _get_jax_test_deps(deps)
    kwargs["deps"] = test_deps
    py_test(name = name, env = env, **kwargs)

def pytype_test(name, **kwargs):
    deps = kwargs.get("deps", [])
    test_deps = _cpu_test_deps() + _get_jax_test_deps(deps)
    kwargs["deps"] = test_deps
    py_test(name = name, **kwargs)

def if_oss(oss_value, google_value = []):
    """Returns one of the arguments based on the non-configurable build env.

    Specifically, it does not return a `select`, and can be used to e.g.
    compute elements of list attributes.
    """
    _ = (google_value, oss_value)  # buildifier: disable=unused-variable
    return oss_value

def wheel_sources(
        name,
        py_srcs = [],
        data_srcs = [],
        symlink_data_srcs = [],
        hdr_srcs = [],
        static_srcs = []):
    """Create a filegroup containing the list of source files for a wheel.

    The sources are collected from the static files and from the transitive dependencies of the
    given srcs.

    Args:
      name: the target name
      py_srcs: targets which transitive python dependencies should be included in the wheel
      data_srcs: targets which platform-dependent data dependencies should be included in the wheel
      symlink_data_srcs: targets which symlinked data dependencies should be included in the wheel
      hdr_srcs: targets which transitive header dependencies should be included in the wheel
      static_srcs: the platform-independent file dependencies of the wheel
    """
    transitive_py_deps(name = "{}_py".format(name), deps = py_srcs)
    collect_data_files(
        name = "{}_data".format(name),
        deps = data_srcs,
        symlink_deps = symlink_data_srcs,
    )
    transitive_hdrs(name = "{}_hdrs".format(name), deps = hdr_srcs)
    native.filegroup(
        name = name,
        srcs = [
            ":{}_py".format(name),
            ":{}_data".format(name),
            ":{}_hdrs".format(name),
        ] + static_srcs,
        visibility = jax_visibility(name),
    )

def if_pypi_cuda_wheel_deps(if_true, if_false = []):
    """ select() on whether we're adding pypi CUDA wheel deps. """
    return select({
        "//jaxlib/tools:pypi_cuda_wheel_deps": if_true,
        "//conditions:default": if_false,
    })

def jax_multiprocess_test(
        name,
        srcs,
        args = [],
        env = {},
        shard_count = None,
        minimal_shard_count = None,
        deps = [],
        data = [],
        enable_backends = None,
        backend_variant_args = {},
        backend_tags = {},
        disable_configs = None,
        enable_configs = [],
        config_tags_overrides = None,
        tags = [],
        main = None):
    # TODO(emilyaf): Avoid hard-coding the number of processes and chips/gpus per process.
    multiprocess_backend_args = {
        "cpu": backend_variant_args.get("cpu", []) + [
            "--num_processes=4",
        ],
        "gpu": backend_variant_args.get("gpu", []) + [
            "--num_processes=4",
            "--gpus_per_process=2",
        ],
        "tpu": backend_variant_args.get("tpu", []) + [
            "--num_processes=4",
            "--tpu_chips_per_process=1",
        ],
    }
    tags = tags + ["multiaccelerator"]
    deps = deps + py_deps(["absl-all", "portpicker"])
    return jax_multiplatform_test(
        name = name,
        srcs = srcs,
        args = args,
        env = env,
        shard_count = shard_count,
        minimal_shard_count = minimal_shard_count,
        deps = deps,
        data = data,
        enable_backends = enable_backends,
        backend_variant_args = multiprocess_backend_args,
        backend_tags = backend_tags,
        disable_configs = disable_configs,
        enable_configs = enable_configs,
        config_tags_overrides = config_tags_overrides,
        tags = tags,
        main = main,
    )

def jax_multiprocess_generate_backend_suites(name = None, backends = []):
    return jax_generate_backend_suites(backends = backends)

WheelAdditivesInfo = provider(
    "Provider to collect non-wheel files required for the tests",
    fields = {"wheel_additives": "depset of files"},
)

def _wheel_additives_depset(wheel_additives):
    return [WheelAdditivesInfo(wheel_additives = depset(transitive = wheel_additives))]

def _collect_wheel_additives_aspect_impl(_, ctx):
    wheel_additives = []
    if not hasattr(ctx.rule.attr, "srcs"):
        return _wheel_additives_depset(wheel_additives)

    attr_val = getattr(ctx.rule.attr, "srcs")
    if type(attr_val) != "list":
        return _wheel_additives_depset(wheel_additives)

    for dep in attr_val:
        transitive_sources = {}
        if not PyInfo in dep:
            continue
        for ts in dep[PyInfo].transitive_sources.to_list():
            # We don't need to collect third party dependencies and test files.
            if not ("site-packages/" in ts.path or ts.path.endswith("_test.py")):
                transitive_sources[ts] = True
        wheel_additives.append(depset(transitive_sources.keys()))
    return _wheel_additives_depset(wheel_additives)

collect_wheel_additives_aspect = aspect(
    implementation = _collect_wheel_additives_aspect_impl,
    attr_aspects = ["srcs"],
)

TestDepsInfo = provider(
    "Provider to collect files from test dependencies",
    fields = {"test_dependencies": "depset of files"},
)

def _collect_test_dependencies_aspect_impl(_, ctx):
    test_deps = []
    attrs_to_traverse = []
    if ctx.rule.kind == "test_suite":
        attrs_to_traverse.append("_implicit_tests")
    elif hasattr(ctx.rule.attr, "deps"):
        attrs_to_traverse.append("deps")

    for attr_name in attrs_to_traverse:
        if not hasattr(ctx.rule.attr, attr_name):
            continue
        attr_val = getattr(ctx.rule.attr, attr_name)
        if type(attr_val) != "list":
            continue
        for dep in attr_val:
            transitive_sources = {}
            if not PyInfo in dep:
                continue

            # The last file in the transitive sources is the actual test file.
            # We don't need to collect it.
            for ts in dep[PyInfo].transitive_sources.to_list()[:-1]:
                if not ("site-packages/" in ts.path):
                    transitive_sources[ts] = True
            test_deps.append(depset(transitive_sources.keys()))

    return [TestDepsInfo(test_dependencies = depset(transitive = test_deps))]

collect_test_dependencies_aspect = aspect(
    implementation = _collect_test_dependencies_aspect_impl,
    attr_aspects = ["tests"],
)

def _compare_srcs_and_test_deps_test_impl(ctx):
    build_jaxlib = ctx.attr.build_jaxlib[BuildSettingInfo].value
    build_jax = ctx.attr.build_jax[BuildSettingInfo].value
    message = "PASSED: All test dependencies are present in the wheel."
    test_result = 0
    doc_link = "https://github.com/jax-ml/jax/blob/main/docs/contributing.md#wheel-sources-update"

    if build_jax == "true" and build_jaxlib == "true":
        wheel_sources_map = {
            f.short_path: True
            for f in ctx.files.wheel_sources
            if not "site-packages/" in f.short_path
        }

        wheel_additives_list = []
        for additive in ctx.attr.wheel_additives:
            if WheelAdditivesInfo in additive:
                wheel_additives_list.append(additive[WheelAdditivesInfo].wheel_additives)

        wheel_additives_depset = depset(transitive = wheel_additives_list)
        wheel_sources_map = wheel_sources_map | {
            f.short_path: True
            for f in wheel_additives_depset.to_list()
        }

        test_dependencies_list = []
        for test in ctx.attr.tests:
            if TestDepsInfo in test:
                test_dependencies_list.append(test[TestDepsInfo].test_dependencies)

        test_dependencies_depset = depset(transitive = test_dependencies_list)
        test_dependencies_map = {}

        # We need to add __init__.py files for all python modules to make them available via API.
        for f in test_dependencies_depset.to_list():
            test_dependencies_map[f.short_path] = True
            init_py_path = f.short_path.replace(f.basename, "__init__.py")
            if f.short_path.startswith("jax") and init_py_path not in ctx.attr.ignored_init_py_files:
                test_dependencies_map[init_py_path] = True

        test_dependencies_paths = [k for k in test_dependencies_map.keys()]
        wheel_sources_paths = wheel_sources_map.keys()

        if wheel_sources_paths != test_dependencies_paths:
            missing_in_wheel_sources = sorted([
                p
                for p in test_dependencies_paths
                if p not in wheel_sources_map
            ])

            if missing_in_wheel_sources:
                message = ("FAILED: Files in test dependencies not found in wheel sources: %s" %
                           missing_in_wheel_sources + "\n" +
                           "See instructions in %s" % doc_link)
                test_result = 1

    else:
        message = "SKIPPED: The test will be executed only with //jax:build_jax=true and //jax:build_jaxlib=true."
        test_result = 0

    script_content = """#!/bin/bash
echo "%s"
exit %s""" % (message, test_result)
    test_runner_script = ctx.actions.declare_file(ctx.label.name + "_runner.sh")
    ctx.actions.write(
        output = test_runner_script,
        content = script_content,
        is_executable = True,
    )

    runfiles = ctx.runfiles(files = [])

    return [
        DefaultInfo(
            executable = test_runner_script,
            runfiles = runfiles,
        ),
    ]

compare_srcs_and_test_deps_test = rule(
    implementation = _compare_srcs_and_test_deps_test_impl,
    attrs = {
        "wheel_sources": attr.label_list(
            allow_files = True,
            mandatory = True,
        ),
        "wheel_additives": attr.label_list(
            allow_files = True,
            mandatory = True,
            aspects = [collect_wheel_additives_aspect],
        ),
        "tests": attr.label_list(
            allow_empty = False,
            mandatory = True,
            aspects = [collect_test_dependencies_aspect],
        ),
        "ignored_init_py_files": attr.string_list(
            mandatory = True,
        ),
        "build_jaxlib": attr.label(default = Label("//jax:build_jaxlib")),
        "build_jax": attr.label(default = Label("//jax:build_jax")),
    },
    test = True,
)
