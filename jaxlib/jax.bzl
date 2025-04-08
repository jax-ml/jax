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
mosaic_gpu_internal_users = []
mosaic_internal_users = []
pallas_gpu_internal_users = []
pallas_tpu_internal_users = []
pallas_fuser_users = []
mosaic_extension_deps = []
serialize_executable_internal_users = []

jax_internal_export_back_compat_test_util_visibility = []
jax_internal_test_harnesses_visibility = []
jax_test_util_visibility = []
loops_visibility = []

PLATFORM_TAGS_DICT = {
    ("Linux", "x86_64"): ("manylinux2014", "x86_64"),
    ("Linux", "aarch64"): ("manylinux2014", "aarch64"),
    ("Linux", "ppc64le"): ("manylinux2014", "ppc64le"),
    ("Darwin", "x86_64"): ("macosx_11_0", "x86_64"),
    ("Darwin", "arm64"): ("macosx_11_0", "arm64"),
    ("Windows", "AMD64"): ("win", "amd64"),
}

_GPU_PYPI_WHEEL_DEPS = [
    "//:jax_wheel_with_internal_test_util",
    "@pypi_jaxlib//:pkg",
    "@pypi_jax_cuda12_plugin//:pkg",
    "@pypi_jax_cuda12_pjrt//:pkg",
]

_CPU_PYPI_WHEEL_DEPS = [
    "//:jax_wheel_with_internal_test_util",
    "@pypi_jaxlib//:pkg",
]

# TODO(vam): remove this once zstandard builds against Python >3.13
def get_zstandard():
    if HERMETIC_PYTHON_VERSION in ("3.13", "3.13-ft", "3.14", "3.14-ft"):
        return []
    return ["@pypi_zstandard//:pkg"]

def get_optional_dep(package, excluded_py_versions = ["3.14", "3.14-ft"]):
    if HERMETIC_PYTHON_VERSION in excluded_py_versions:
        return []
    return [package]

_py_deps = {
    "absl/logging": ["@pypi_absl_py//:pkg"],
    "absl/testing": ["@pypi_absl_py//:pkg"],
    "absl/flags": ["@pypi_absl_py//:pkg"],
    "cloudpickle": get_optional_dep("@pypi_cloudpickle//:pkg"),
    "colorama": get_optional_dep("@pypi_colorama//:pkg"),
    "epath": get_optional_dep("@pypi_etils//:pkg"),  # etils.epath
    "filelock": get_optional_dep("@pypi_filelock//:pkg"),
    "flatbuffers": ["@pypi_flatbuffers//:pkg"],
    "hypothesis": ["@pypi_hypothesis//:pkg"],
    "magma": [],
    "matplotlib": get_optional_dep("@pypi_matplotlib//:pkg"),
    "mpmath": [],
    "opt_einsum": ["@pypi_opt_einsum//:pkg"],
    "pil": get_optional_dep("@pypi_pillow//:pkg"),
    "portpicker": get_optional_dep("@pypi_portpicker//:pkg"),
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
jax_gpu_support_deps = []
jax2tf_deps = []

def pytype_library(name, pytype_srcs = None, **kwargs):
    _ = pytype_srcs  # @unused
    native.py_library(name = name, **kwargs)

def pytype_strict_library(name, pytype_srcs = [], **kwargs):
    data = pytype_srcs + (kwargs["data"] if "data" in kwargs else [])
    new_kwargs = {k: v for k, v in kwargs.items() if k != "data"}
    native.py_library(name = name, data = data, **new_kwargs)

py_strict_library = native.py_library
py_strict_test = native.py_test

def py_library_providing_imports_info(*, name, lib_rule = native.py_library, pytype_srcs = [], **kwargs):
    data = pytype_srcs + (kwargs["data"] if "data" in kwargs else [])
    new_kwargs = {k: v for k, v in kwargs.items() if k != "data"}
    lib_rule(name = name, data = data, **new_kwargs)

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
        if_not_building = _GPU_PYPI_WHEEL_DEPS,
        if_not_building_for_cpu = _CPU_PYPI_WHEEL_DEPS):
    """Adds jaxlib and jaxlib cuda plugin wheels as dependencies instead of depending on sources.

    This allows us to test prebuilt versions of jaxlib wheels against the rest of the JAX codebase.

    Args:
      if_building: the source code targets to depend on in case we don't depend on the jaxlib wheels
      if_not_building: the wheels to depend on including gpu-specific plugins in case of
                       gpu-enabled builds
      if_not_building_for_cpu: the wheels to depend on in case of cpu-only builds
    """

    return select({
        "//jax:enable_jaxlib_build": if_building,
        "//jax_plugins/cuda:disable_jaxlib_for_cpu_build": if_not_building_for_cpu,
        "//jax_plugins/cuda:disable_jaxlib_for_cuda12_build": if_not_building,
        "//conditions:default": [],
    })

def _get_test_deps(deps, backend_independent):
    gpu_build_deps = [
        "//jaxlib/cuda:gpu_only_test_deps",
        "//jaxlib/rocm:gpu_only_test_deps",
        "//jax_plugins:gpu_plugin_only_test_deps",
    ]

    gpu_py_imports = [
        "//:jax_py_import",
        "//jaxlib/tools:jaxlib_py_import",
        "//jaxlib/tools:jax_cuda_plugin_py_import",
        "//jaxlib/tools:jax_cuda_pjrt_py_import",
    ]
    cpu_py_imports = [
        "//:jax_py_import",
        "//jaxlib/tools:jaxlib_py_import",
    ]

    if backend_independent:
        jaxlib_build_deps = deps
        gpu_pypi_wheel_deps = _CPU_PYPI_WHEEL_DEPS
        gpu_py_import_deps = cpu_py_imports
    else:
        jaxlib_build_deps = gpu_build_deps + deps
        gpu_pypi_wheel_deps = _GPU_PYPI_WHEEL_DEPS
        gpu_py_import_deps = gpu_py_imports

    return select({
        "//jax:enable_jaxlib_build": jaxlib_build_deps,
        "//jax_plugins/cuda:disable_jaxlib_for_cpu_build": _CPU_PYPI_WHEEL_DEPS,
        "//jax_plugins/cuda:disable_jaxlib_for_cuda12_build": gpu_pypi_wheel_deps,
        "//jax_plugins/cuda:enable_py_import_for_cpu_build": cpu_py_imports,
        "//jax_plugins/cuda:enable_py_import_for_cuda12_build": gpu_py_import_deps,
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
            deps = _get_test_deps([
                "//jax",
                "//jax:test_util",
            ] + deps, backend_independent = False),
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
            default = Label("//jaxlib/tools:build_wheel"),
            executable = True,
            # b/365588895 Investigate cfg = "exec" for multi platform builds
            cfg = "target",
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
    if "PYTHONWARNINGS" not in env:
        env["PYTHONWARNINGS"] = "error"
    deps = kwargs.get("deps", [])
    test_deps = _get_test_deps(deps, backend_independent = True)
    kwargs["deps"] = test_deps
    py_test(name = name, env = env, **kwargs)

def pytype_test(name, **kwargs):
    deps = kwargs.get("deps", [])
    test_deps = _get_test_deps(deps, backend_independent = True)
    kwargs["deps"] = test_deps
    native.py_test(name = name, **kwargs)

def if_oss(oss_value, google_value = []):
    """Returns one of the arguments based on the non-configurable build env.

    Specifically, it does not return a `select`, and can be used to e.g.
    compute elements of list attributes.
    """
    _ = (google_value, oss_value)  # buildifier: disable=unused-variable
    return oss_value
