# Copyright 2021 Google LLC
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

This appeases rules that require certain directory structured while allowing
the use of filegroups and globs. This doesn't use Fileset because that creates
entire directories and therefore prevents multiple rules from writing into the
same directory. Basic usage:

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
)

py_library(
  name = "bar",
  srcs = [":all_bar_files"]
)
```

A single macro `symlink_inputs` can also be used to wrap an arbitrary rule and
remap any of its inputs that takes a list of labels to be symlinked into some
directory relative to the current one.

symlink_inputs(
  name = "bar"
  rule = py_library,
  symlinked_inputs = {"srcs", {"bar": ["//foo/bar:all_bar_files"]}},
)
"""

def _symlink_files(ctx):
    outputs = []
    for src in ctx.files.srcs:
        out = ctx.actions.declare_file(ctx.attr.dst + "/" + src.basename)
        outputs.append(out)
        ctx.actions.symlink(output = out, target_file = src)
    outputs = depset(outputs)
    return [DefaultInfo(
        files = outputs,
        data_runfiles = ctx.runfiles(transitive_files = outputs),
    )]

# Symlinks srcs into the specified directory.
#
# Args:
#   name: name for the rule.
#   dst: directory to symlink srcs into. Relative the current package.
#   srcs: list of labels that should be symlinked into dst.
symlink_files = rule(
    implementation = _symlink_files,
    attrs = {
        "dst": attr.string(),
        "srcs": attr.label_list(allow_files = True),
    },
)

def symlink_inputs(rule, name, symlinked_inputs, *args, **kwargs):
    """Wraps a rule and symlinks input files into the current directory tree.

    Args:
      rule: the rule (or macro) being wrapped.
      name: name for the generated rule.
      symlinked_inputs: a dictionary of dictionaries indicating label-list
        arguments labels that should be passed to the generated rule after
        being symlinked into the specified directory. For example:
        {"srcs": {"bar": ["//foo/bar:bar.txt"]}}
      *args: additional arguments to forward to the generated rule.
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
                )
                kwargs[kwarg] = [":" + symlinked_target_name]
    rule(
        name = name,
        *args,
        **kwargs
    )
