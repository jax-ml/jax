load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# To update XLA to a new revision,
# a) update URL and strip_prefix to the new git commit hash
# b) get the sha256 hash of the commit by running:
#    curl -L https://github.com/openxla/xla/archive/<git hash>.tar.gz | sha256sum
#    and update the sha256 with the result.
http_archive(
    name = "xla",
    sha256 = "4ec16aff3862c5a243db956ce558d7a62eb79f5e20747b0e80802a3b0d12e419",
    strip_prefix = "xla-12de6ec958419b57be248d0acd2d9f757e71748c",
    urls = [
        "https://github.com/openxla/xla/archive/12de6ec958419b57be248d0acd2d9f757e71748c.tar.gz",
    ],
)

# For development, one often wants to make changes to the TF repository as well
# as the JAX repository. You can override the pinned repository above with a
# local checkout by either:
# a) overriding the TF repository on the build.py command line by passing a flag
#    like:
#    python build/build.py --bazel_options=--override_repository=xla=/path/to/xla
#    or
# b) by commenting out the http_archive above and uncommenting the following:
# local_repository(
#    name = "xla",
#    path = "/path/to/xla",
# )

load("//third_party/ducc:workspace.bzl", ducc = "repo")
ducc()

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
