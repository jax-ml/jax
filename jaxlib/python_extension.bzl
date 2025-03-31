load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
# load(
#     "@xla//third_party/py/rules_pywrap:pywrap.default.bzl",
#     "pybind_extension",
# )

load("@symbol_locations//rules_pywrap:pywrap.bzl", "pybind_extension")

def pywrap_extension(
        name,
        module_name = None,
        srcs = [],
        deps = [],
        pytype_srcs = [],
        pytype_deps = [],
        copts = [],
        linkopts = [],
        visibility = None,
    ):
    module_name = name if module_name == None else module_name
    lib_name = name + "_pywrap_library"
    src_cc_name = name + "_pywrap_stub.c"
    native.cc_library(
        name = lib_name,
        srcs = srcs,
        copts = copts,
        deps = deps,
        local_defines = [
            "PyInit_{}=Wrapped_PyInit_{}".format(module_name, module_name),
        ],
        visibility = ["//visibility:private"],
    )
    expand_template(
        name = name + "_pywrap_stub",
        testonly = True,
        out = src_cc_name,
        substitutions = {
            "@MODULE_NAME@": module_name,
        },
        template = "//jaxlib:pyinit_stub.c",
        visibility = ["//visibility:private"],
    )
    pybind_extension(
        name = name,
        # module_name = module_name,
        srcs = [src_cc_name],
        deps = [":" + lib_name],
        linkopts = linkopts,
        visibility = visibility,
    )
