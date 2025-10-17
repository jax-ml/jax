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

"""Macros for symlinking files into certain directories at build time.

This appeases rules that require certain directory structures (e.g. Bazel
Python rules) while allowing the use of filegroups and globs. This doesn't use
Fileset because that creates entire directories and therefore prevents multiple
rules from writing into the same directory (necessary for tests, among other
things). Basic usage:

```build
# foo/bar/BUILD

filegroup(
  name = "all_bar_files",
  srcs = glob(["*"]),
)
```

```build
biz/baz/BUILD

symlink_files(
  name = "all_bar_files",
  dst = "bar",
  srcs = ["//foo/bar:all_bar_files"],
  flatten = True,
)

py_library(
  name = "bar",
  srcs = [":all_bar_files"]
)
```

Or if you want to preserve the directory structure of the origin:

```build
# foo/bar/BUILD

filegroup(
  name = "bar_tree",
  srcs = glob(["**/*"]),
)
```

```build
biz/baz/BUILD

symlink_files(
  name = "bar_tree",
  dst = "bar",
  srcs = ["//foo/bar:bar_tree"],
  strip_prefix = "foo/bar",
)

py_library(
  name = "bar",
  srcs = [":bar_tree"]
)
```

A single macro `symlink_inputs` can also be used to wrap an arbitrary rule and
remap any of its inputs that takes a list of labels to be symlinked into some
directory relative to the current one, flattening all the files into a single
directory (as with the `flatten` option to symlink_files).

symlink_inputs(
  name = "bar"
  rule = py_library,
  symlinked_inputs = {"srcs", {"bar": ["//foo/bar:all_bar_files"]}},
)
"""

visibility(["//jaxlib/..."])

def _symlink_files_impl(ctx):
    flatten = ctx.attr.flatten
    strip_prefix = ctx.attr.strip_prefix
    mapping = ctx.attr.mapping
    outputs = []
    for src in ctx.files.srcs:
        src_path = src.short_path
        if src_path in mapping:
            file_dst = mapping[src_path]
        else:
            file_dst = src.basename if flatten else src_path
            if not file_dst.startswith(strip_prefix):
                fail(("File {} has destination {} that does not begin with" +
                      " strip_prefix {}").format(
                    src,
                    file_dst,
                    strip_prefix,
                ))
            file_dst = file_dst[len(strip_prefix):]
        outfile = ctx.attr.dst + "/" + file_dst
        out = ctx.actions.declare_file(outfile)
        outputs.append(out)
        ctx.actions.symlink(output = out, target_file = src)
    outputs = depset(outputs)
    return [DefaultInfo(
        files = outputs,
        runfiles = ctx.runfiles(transitive_files = outputs),
    )]

symlink_files = rule(
    implementation = _symlink_files_impl,
    attrs = {
        "dst": attr.string(
            default = ".",
            doc = "Destination directory into which to symlink `srcs`." +
                  " Relative to current directory.",
        ),
        "srcs": attr.label_list(
            allow_files = True,
            doc = "Files to symlink into `dst`.",
        ),
        "flatten": attr.bool(
            default = False,
            doc = "Whether files in `srcs` should all be flattened to be" +
                  " direct children of `dst` or preserve their existing" +
                  " directory structure.",
        ),
        "strip_prefix": attr.string(
            default = "",
            doc = "Literal string prefix to strip from the paths of all files" +
                  " in `srcs`. All files in `srcs` must begin with this" +
                  " prefix or be present mapping. Generally they would not be" +
                  " used together, but prefix stripping happens after flattening.",
        ),
        "mapping": attr.string_dict(
            default = {},
            doc = "Dictionary indicating where individual files in `srcs`" +
                  " should be mapped to under `dst`. Keys are the origin" +
                  " path of the file (relative to the build system root) and" +
                  " values are the destination relative to `dst`. Files" +
                  " present in `mapping` ignore the `flatten` and" +
                  " `strip_prefix` attributes: their destination is based" +
                  " only on `dst` and the value for their key in `mapping`.",
        ),
    },
)

def symlink_inputs(name, rule, symlinked_inputs, **kwargs):
    """Wraps a rule and symlinks input files into the current directory tree.

    Args:
      rule: the rule (or macro) being wrapped.
      name: name for the generated rule.
      symlinked_inputs: a dictionary of dictionaries indicating label-list
        arguments labels that should be passed to the generated rule after
        being symlinked into the specified directory. For example:
        {"srcs": {"bar": ["//foo/bar:bar.txt"]}}
      **kwargs: additional keyword arguments to forward to the generated rule.
    """
    for kwarg, mapping in symlinked_inputs.items():
        for dst, files in mapping.items():
            if kwarg in kwargs:
                fail(
                    "key %s is already present in this rule" % (kwarg,),
                    attr = "symlinked_inputs",
                )
            if dst == None:
                kwargs[kwarg] = files
            else:
                symlinked_target_name = "_{}_{}".format(name, kwarg)
                symlink_files(
                    name = symlinked_target_name,
                    dst = dst,
                    srcs = files,
                    flatten = True,
                )
                kwargs[kwarg] = [":" + symlinked_target_name]
    rule(
        name = name,
        **kwargs
    )
