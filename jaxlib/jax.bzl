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
load("@jax_wheel_version//:wheel_version.bzl", "WHEEL_VERSION")
load("@jax_wheel_version_suffix//:wheel_version_suffix.bzl", "BUILD_TAG", "WHEEL_VERSION_SUFFIX")
load("@local_config_cuda//cuda:build_defs.bzl", _cuda_library = "cuda_library", _if_cuda_is_configured = "if_cuda_is_configured")
load("@local_config_rocm//rocm:build_defs.bzl", _if_rocm_is_configured = "if_rocm_is_configured", _rocm_library = "rocm_library")
load("@python_version_repo//:py_version.bzl", "HERMETIC_PYTHON_VERSION")
load("@rules_cc//cc:defs.bzl", _cc_proto_library = "cc_proto_library")
load("@rules_python//python:defs.bzl", "py_test")
load("@xla//xla/tsl:tsl.bzl", _if_windows = "if_windows", _pybind_extension = "tsl_pybind_extension_opensource")
load("@xla//xla/tsl/platform:build_config_root.bzl", _tf_cuda_tests_tags = "tf_cuda_tests_tags", _tf_exec_properties = "tf_exec_properties")

# Explicitly re-exports names to avoid "unused variable" warnings from .bzl
# lint tools.
cc_proto_library = _cc_proto_library
cuda_library = _cuda_library
rocm_library = _rocm_library
pytype_test = native.py_test
nanobind_extension = _pybind_extension
if_cuda_is_configured = _if_cuda_is_configured
if_rocm_is_configured = _if_rocm_is_configured
if_windows = _if_windows
flatbuffer_cc_library = _flatbuffer_cc_library
tf_exec_properties = _tf_exec_properties
tf_cuda_tests_tags = _tf_cuda_tests_tags

jax_internal_packages = []
jax_extend_internal_users = []
mosaic_gpu_internal_users = []
mosaic_internal_users = []
pallas_gpu_internal_users = []
pallas_tpu_internal_users = []
pallas_extension_deps = []

jax_internal_export_back_compat_test_util_visibility = []
jax_internal_test_harnesses_visibility = []
jax_test_util_visibility = []
loops_visibility = []

# TODO(vam): remove this once zstandard builds against Python 3.13
def get_zstandard():
    if HERMETIC_PYTHON_VERSION == "3.13":
        return []
    return ["@pypi_zstandard//:pkg"]

_py_deps = {
    "absl/logging": ["@pypi_absl_py//:pkg"],
    "absl/testing": ["@pypi_absl_py//:pkg"],
    "absl/flags": ["@pypi_absl_py//:pkg"],
    "cloudpickle": ["@pypi_cloudpickle//:pkg"],
    "colorama": ["@pypi_colorama//:pkg"],
    "epath": ["@pypi_etils//:pkg"],  # etils.epath
    "filelock": ["@pypi_filelock//:pkg"],
    "flatbuffers": ["@pypi_flatbuffers//:pkg"],
    "hypothesis": ["@pypi_hypothesis//:pkg"],
    "magma": [],
    "matplotlib": ["@pypi_matplotlib//:pkg"],
    "mpmath": [],
    "opt_einsum": ["@pypi_opt_einsum//:pkg"],
    "pil": ["@pypi_pillow//:pkg"],
    "portpicker": ["@pypi_portpicker//:pkg"],
    "ml_dtypes": ["@pypi_ml_dtypes//:pkg"],
    "numpy": ["@pypi_numpy//:pkg"],
    "scipy": ["@pypi_scipy//:pkg"],
    "tensorflow_core": [],
    "torch": [],
    "zstandard": get_zstandard(),
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

    # This is only useful as part of a larger Bazel repository.
    return []

jax_extra_deps = []
jax2tf_deps = []

def pytype_library(name, pytype_srcs = None, **kwargs):
    _ = pytype_srcs  # @unused
    native.py_library(name = name, **kwargs)

def pytype_strict_library(name, pytype_srcs = None, **kwargs):
    _ = pytype_srcs  # @unused
    native.py_library(name = name, **kwargs)

def py_library_providing_imports_info(*, name, lib_rule = native.py_library, pytype_srcs = [], **kwargs):
    lib_rule(name = name, **kwargs)

def py_extension(name, srcs, copts, deps, linkopts = []):
    nanobind_extension(name, srcs = srcs, copts = copts, linkopts = linkopts, deps = deps, module_name = name)

def windows_cc_shared_mlir_library(name, out, deps = [], srcs = [], exported_symbol_prefixes = []):
    """Workaround DLL building issue.

    1. cc_binary with linkshared enabled cannot produce DLL with symbol
       correctly exported.
    2. Even if the DLL is correctly built, the resulting target cannot be
       correctly consumed by other targets.

    Args:
      name: the name of the output target
      out: the name of the output DLL filename
      deps: deps
      srcs: srcs
    """

    # create a dummy library to get the *.def file
    dummy_library_name = name + ".dummy.dll"
    native.cc_binary(
        name = dummy_library_name,
        linkshared = 1,
        linkstatic = 1,
        deps = deps,
        target_compatible_with = ["@platforms//os:windows"],
    )

    # .def file with all symbols, not usable
    full_def_name = name + ".full.def"
    native.filegroup(
        name = full_def_name,
        srcs = [dummy_library_name],
        output_group = "def_file",
        target_compatible_with = ["@platforms//os:windows"],
    )

    # say filtered_symbol_prefixes == ["mlir", "chlo"], then construct the regex
    # pattern as "^\\s*(mlir|clho)" to use grep
    pattern = "^\\s*(" + "|".join(exported_symbol_prefixes) + ")"

    # filtered def_file, only the needed symbols are included
    filtered_def_name = name + ".filtered.def"
    filtered_def_file = out + ".def"
    native.genrule(
        name = filtered_def_name,
        srcs = [full_def_name],
        outs = [filtered_def_file],
        cmd = """echo 'LIBRARY {}\nEXPORTS ' > $@ && grep -E '{}' $(location :{}) >> $@""".format(out, pattern, full_def_name),
        target_compatible_with = ["@platforms//os:windows"],
    )

    # create the desired library
    native.cc_binary(
        name = out,  # this name must be correct, it will be the filename
        linkshared = 1,
        deps = deps,
        win_def_file = filtered_def_file,
        target_compatible_with = ["@platforms//os:windows"],
    )

    # however, the created cc_library (a shared library) cannot be correctly
    # consumed by other cc_*...
    interface_library_file = out + ".if.lib"
    native.filegroup(
        name = interface_library_file,
        srcs = [out],
        output_group = "interface_library",
        target_compatible_with = ["@platforms//os:windows"],
    )

    # but this one can be correctly consumed, this is our final product
    native.cc_import(
        name = name,
        interface_library = interface_library_file,
        shared_library = out,
        target_compatible_with = ["@platforms//os:windows"],
    )

ALL_BACKENDS = ["cpu", "gpu", "tpu"]

def if_building_jaxlib(
        if_building,
        if_not_building = [
            "@pypi_jaxlib//:pkg",
            "@pypi_jax_cuda12_plugin//:pkg",
            "@pypi_jax_cuda12_pjrt//:pkg",
        ],
        if_not_building_for_cpu = ["@pypi_jaxlib//:pkg"]):
    """Adds jaxlib and jaxlib cuda plugin wheels as dependencies instead of depending on sources. 

    This allows us to test prebuilt versions of jaxlib wheels against the rest of the JAX codebase.

    Args:
      if_building: the source code targets to depend on in case we don't depend on the jaxlib wheels
      if_not_building: the jaxlib wheels to depend on including gpu-specific plugins in case of
                       gpu-enabled builds
      if_not_building_for_cpu: the jaxlib wheels to depend on in case of cpu-only builds
    """

    return select({
        "//jax:enable_jaxlib_build": if_building,
        "//jax_plugins/cuda:disable_jaxlib_for_cpu_build": if_not_building_for_cpu,
        "//jax_plugins/cuda:disable_jaxlib_for_cuda12_build": if_not_building,
    })

# buildifier: disable=function-docstring
def jax_multiplatform_test(
        name,
        srcs,
        args = [],
        env = {},
        shard_count = None,
        deps = [],
        data = [],
        enable_backends = None,
        backend_variant_args = {},  # buildifier: disable=unused-variable
        backend_tags = {},  # buildifier: disable=unused-variable
        disable_configs = None,  # buildifier: disable=unused-variable
        enable_configs = [],
        config_tags_overrides = None,  # buildifier: disable=unused-variable
        tags = [],
        main = None,
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

    for backend in ALL_BACKENDS:
        if shard_count == None or type(shard_count) == type(0):
            test_shards = shard_count
        else:
            test_shards = shard_count.get(backend, 1)
        test_args = list(args) + [
            "--jax_test_dut=" + backend,
            "--jax_platform_name=" + backend,
        ]
        test_tags = list(tags) + ["jax_test_%s" % backend] + backend_tags.get(backend, [])
        if enable_backends != None and backend not in enable_backends and not any([config.startswith(backend) for config in enable_configs]):
            test_tags.append("manual")
        if backend == "gpu":
            test_tags += tf_cuda_tests_tags()
        native.py_test(
            name = name + "_" + backend,
            srcs = srcs,
            args = test_args,
            env = env,
            deps = [
                "//jax",
                "//jax:test_util",
            ] + deps + if_building_jaxlib([
                "//jaxlib/cuda:gpu_only_test_deps",
                "//jaxlib/rocm:gpu_only_test_deps",
                "//jax_plugins:gpu_plugin_only_test_deps",
            ]),
            data = data,
            shard_count = test_shards,
            tags = test_tags,
            main = main,
            exec_properties = tf_exec_properties({"tags": test_tags}),
        )

def jax_generate_backend_suites(backends = []):
    """Generates test suite targets named cpu_tests, gpu_tests, etc.

    Args:
      backends: the set of backends for which rules should be generated. Defaults to all backends.
    """
    if not backends:
        backends = ALL_BACKENDS
    for backend in backends:
        native.test_suite(
            name = "%s_tests" % backend,
            tags = ["jax_test_%s" % backend, "-manual"],
        )
    native.test_suite(
        name = "backend_independent_tests",
        tags = ["-jax_test_%s" % backend for backend in backends] + ["-manual"],
    )

def _get_wheel_platform_name(platform_name, cpu_name):
    platform = ""
    cpu = ""
    if platform_name == "linux":
        platform = "manylinux2014"
        cpu = cpu_name
    elif platform_name == "macosx":
        if cpu_name == "arm64":
            cpu = "arm64"
            platform = "macosx_11_0"
        else:
            cpu = "x86_64"
            platform = "macosx_10_14"
    elif platform_name == "win":
        platform = "win"
        cpu = "amd64"
    return "{platform}_{cpu}".format(
        platform = platform,
        cpu = cpu,
    )

def _get_cpu(platform_name, platform_tag):
    # Following the convention in jax/tools/build_utils.py.
    if platform_name == "macosx" and platform_tag == "arm64":
        return "arm64"
    if platform_name == "win" and platform_tag == "x86_64":
        return "AMD64"
    return "aarch64" if platform_tag == "arm64" else platform_tag

def _get_full_wheel_name(rule_name, platform_name, cpu_name, major_cuda_version, wheel_version):
    if "pjrt" in rule_name:
        wheel_name_template = "{package_name}-{wheel_version}-py{major_python_version}-none-{wheel_platform_tag}.whl"
    else:
        wheel_name_template = "{package_name}-{wheel_version}-cp{python_version}-cp{python_version}-{wheel_platform_tag}.whl"
    python_version = HERMETIC_PYTHON_VERSION.replace(".", "")
    package_name = rule_name.replace("_wheel", "").replace(
        "cuda",
        "cuda{}".format(major_cuda_version),
    )
    return wheel_name_template.format(
        package_name = package_name,
        python_version = python_version,
        major_python_version = python_version[0],
        wheel_version = wheel_version,
        wheel_platform_tag = _get_wheel_platform_name(platform_name, cpu_name),
    )

def _jax_wheel_impl(ctx):
    include_cuda_libs = ctx.attr.include_cuda_libs[BuildSettingInfo].value
    override_include_cuda_libs = ctx.attr.override_include_cuda_libs[BuildSettingInfo].value
    output_path = ctx.attr.output_path[BuildSettingInfo].value
    git_hash = ctx.attr.git_hash[BuildSettingInfo].value
    executable = ctx.executable.wheel_binary

    if include_cuda_libs and not override_include_cuda_libs:
        fail("JAX wheel shouldn't be built with CUDA dependencies." +
             " Please provide `--config=no_cuda_libs` for bazel build command." +
             " If you absolutely need to add CUDA dependencies, provide" +
             " `--@local_config_cuda//cuda:override_include_cuda_libs=true`.")

    env = {}
    args = ctx.actions.args()

    full_wheel_version = (WHEEL_VERSION + WHEEL_VERSION_SUFFIX)
    env["WHEEL_VERSION_SUFFIX"] = WHEEL_VERSION_SUFFIX
    if BUILD_TAG:
        env["WHEEL_BUILD_TAG"] = BUILD_TAG
        args.add("--build-tag", BUILD_TAG)
        full_wheel_version += "-{}".format(BUILD_TAG)
    if not WHEEL_VERSION_SUFFIX and not BUILD_TAG:
        env["JAX_RELEASE"] = "1"

    cpu = _get_cpu(ctx.attr.platform_name, ctx.attr.platform_tag)
    wheel_name = _get_full_wheel_name(
        ctx.label.name,
        ctx.attr.platform_name,
        cpu,
        ctx.attr.platform_version,
        full_wheel_version,
    )
    output_file = ctx.actions.declare_file(output_path +
                                           "/" + wheel_name)
    wheel_dir = output_file.path[:output_file.path.rfind("/")]

    args.add("--output_path", wheel_dir)  # required argument
    args.add("--cpu", cpu)  # required argument
    args.add("--jaxlib_git_hash", git_hash)  # required argument

    if ctx.attr.enable_cuda:
        args.add("--enable-cuda", "True")
        if ctx.attr.platform_version == "":
            fail("platform_version must be set to a valid cuda version for cuda wheels")
        args.add("--platform_version", ctx.attr.platform_version)  # required for gpu wheels
    if ctx.attr.enable_rocm:
        args.add("--enable-rocm", "True")
        if ctx.attr.platform_version == "":
            fail("platform_version must be set to a valid rocm version for rocm wheels")
        args.add("--platform_version", ctx.attr.platform_version)  # required for gpu wheels
    if ctx.attr.skip_gpu_kernels:
        args.add("--skip_gpu_kernels")

    args.set_param_file_format("flag_per_line")
    args.use_param_file("@%s", use_always = False)
    ctx.actions.run(
        arguments = [args],
        inputs = [],
        outputs = [output_file],
        executable = executable,
        env = env,
    )

    return [DefaultInfo(files = depset(direct = [output_file]))]

_jax_wheel = rule(
    attrs = {
        "wheel_binary": attr.label(
            default = Label("//jaxlib/tools:build_wheel"),
            executable = True,
            # b/365588895 Investigate cfg = "exec" for multi platform builds
            cfg = "target",
        ),
        "platform_tag": attr.string(mandatory = True),
        "platform_name": attr.string(mandatory = True),
        "git_hash": attr.label(default = Label("//jaxlib/tools:jaxlib_git_hash")),
        "output_path": attr.label(default = Label("//jaxlib/tools:output_path")),
        "enable_cuda": attr.bool(default = False),
        # A cuda/rocm version is required for gpu wheels; for cpu wheels, it can be an empty string.
        "platform_version": attr.string(mandatory = True, default = ""),
        "skip_gpu_kernels": attr.bool(default = False),
        "enable_rocm": attr.bool(default = False),
        "include_cuda_libs": attr.label(default = Label("@local_config_cuda//cuda:include_cuda_libs")),
        "override_include_cuda_libs": attr.label(default = Label("@local_config_cuda//cuda:override_include_cuda_libs")),
    },
    implementation = _jax_wheel_impl,
    executable = False,
)

def jax_wheel(name, wheel_binary, enable_cuda = False, platform_version = ""):
    """Create jax artifact wheels.

    Common artifact attributes are grouped within a single macro.

    Args:
      name: the name of the wheel
      wheel_binary: the binary to use to build the wheel
      enable_cuda: whether to build a cuda wheel
      platform_version: the cuda version to use for the wheel

    Returns:
      A directory containing the wheel
    """
    _jax_wheel(
        name = name,
        wheel_binary = wheel_binary,
        enable_cuda = enable_cuda,
        platform_version = platform_version,
        # git_hash is empty by default. Use `--//jaxlib/tools:jaxlib_git_hash=$(git rev-parse HEAD)`
        # flag in bazel command to pass the git hash for nightly or release builds.
        platform_name = select({
            "@platforms//os:osx": "macosx",
            "@platforms//os:macos": "macosx",
            "@platforms//os:windows": "win",
            "@platforms//os:linux": "linux",
        }),
        # TODO(kanglan) Add @platforms//cpu:ppc64le once JAX Bazel is upgraded > 6.5.0.
        platform_tag = select({
            "@platforms//cpu:aarch64": "arm64",
            "@platforms//cpu:arm64": "arm64",
            "@platforms//cpu:x86_64": "x86_64",
        }),
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
    if "PYTHONWARNINGS" not in env:
        env["PYTHONWARNINGS"] = "error"
    py_test(name = name, env = env, **kwargs)
